#include "vmc_minco_nav/vmc_planner.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>

namespace vmc_minco_nav
{

VMCPlanner::VMCPlanner(const VMCParameters& params, rclcpp::Logger logger)
    : params_(params), logger_(logger)
{
    // Seed the random number generator
    rng_.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

std::vector<Eigen::Vector2d> VMCPlanner::plan(const Eigen::Vector2d& start, const Eigen::Vector2d& goal, const std::vector<Eigen::Vector3d>& obstacles)
{
    RCLCPP_INFO(logger_, "VMC-Q Planner: Starting to generate initial path.");
    auto plan_start_time = std::chrono::high_resolution_clock::now();

    // Step 1: Generate virtual prey path
    auto prey_path = generateVirtualPreyPath(start, goal);
    if (prey_path.size() < 2) {
        RCLCPP_WARN(logger_, "Prey path is too short to plan.");
        return {};
    }
    params_.N = prey_path.size(); // Update N based on actual generated points

    // Step 2: Select reference point based on obstacle configuration
    auto ref_point = selectReferencePoint(prey_path, obstacles);
    RCLCPP_INFO(logger_, "Selected reference point: (%.2f, %.2f)", ref_point.x(), ref_point.y());

    // Step 3: Train Q-Table
    RCLCPP_INFO(logger_, "Starting Q-table training for %d episodes...", params_.train_number_max);
    trainQTable(prey_path, ref_point, obstacles, goal);
    RCLCPP_INFO(logger_, "Training finished.");

    // Step 4: Generate final path from trained Q-Table
    auto initial_path = generateFinalRouteFromQ(prey_path, ref_point);
    
    auto plan_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = plan_end_time - plan_start_time;
    RCLCPP_INFO(logger_, "VMC-Q front-end finished in %.4f s, generated %zu waypoints.", elapsed.count(), initial_path.size());

    if (initial_path.empty()) {
        RCLCPP_ERROR(logger_, "VMC-Q planner failed to generate a valid initial path.");
    }
    
    return initial_path;
}

/**
 * @brief 生成虚拟猎物路径(起点到终点的一条直线)，N点数量由prey_path_spacing决定
 * @param start 起点
 * @param goal 终点
*/
std::vector<Eigen::Vector2d> VMCPlanner::generateVirtualPreyPath(const Eigen::Vector2d& start, const Eigen::Vector2d& goal)
{
    std::vector<Eigen::Vector2d> prey_path;
    double total_dist = (goal - start).norm();
    int num_points = static_cast<int>(std::round(total_dist / params_.prey_path_spacing));
    if (num_points < 2) num_points = 2;

    for (int i = 0; i < num_points; ++i) {
        double t = static_cast<double>(i) / (num_points - 1);
        prey_path.push_back(start + t * (goal - start));
    }
    return prey_path;
}

/**
 * @brief 计算障碍物权重，选择参考点，在prey_path中垂线上设置xr
*/
Eigen::Vector2d VMCPlanner::selectReferencePoint(const std::vector<Eigen::Vector2d>& prey_path, const std::vector<Eigen::Vector3d>& obstacles)
{
    double omega_left = 0.0, omega_right = 0.0;

    double Denv_sqrt = 100.0; 

    const auto& p_start = prey_path.front();
    const auto& p_end = prey_path.back();
    Eigen::Vector2d path_vec = p_end - p_start;
    Eigen::Vector2d path_dir = path_vec.normalized();

    for (const auto& obs : obstacles) {
        Eigen::Vector2d obs_center = obs.head<2>();
        double obs_radius = obs.z();
        
        // Project obstacle center onto the path line to find closest point and distance `di`
        double t = path_dir.dot(obs_center - p_start);
        Eigen::Vector2d closest_point_on_path;
        if (t < 0) {
            closest_point_on_path = p_start;
        } else if (t > path_vec.norm()) {
            closest_point_on_path = p_end;
        } else {
            closest_point_on_path = p_start + t * path_dir;
        }
        double di = (obs_center - closest_point_on_path).norm();

        double Si = M_PI * obs_radius * obs_radius;
        double coeff = std::exp(-di / Denv_sqrt) * Si;
        
        // Use cross product to determine if obstacle is left or right of the path vector
        double cross_product = path_vec.x() * (obs_center.y() - p_start.y()) - path_vec.y() * (obs_center.x() - p_start.x());

        if (cross_product > 0) omega_left += coeff;
        else omega_right += coeff;
    }
    
    bool select_left = omega_left <= omega_right;
    RCLCPP_INFO(logger_, "Obstacle weights: left=%.2f, right=%.2f. Selecting %s side.", omega_left, omega_right, select_left ? "left" : "right");

    double path_len = path_vec.norm();
    // A heuristic distance for the reference point, similar to the paper's examples
    double dynamic_distance = std::max(20.0, path_len / 2.0); 

    Eigen::Vector2d mid_path_point = p_start + 0.5 * path_vec;
    Eigen::Vector2d perp_vec(-path_vec.y(), path_vec.x());
    perp_vec.normalize();

    if (!select_left) {
        perp_vec = -perp_vec;
    }

    return mid_path_point + dynamic_distance * perp_vec;
}

/**
 * @brief Q-learning的主流程
*/
void VMCPlanner::trainQTable(const std::vector<Eigen::Vector2d>& prey_path, const Eigen::Vector2d& ref_point, const std::vector<Eigen::Vector3d>& obstacles, const Eigen::Vector2d& goal)
{
    int v_states = static_cast<int>(params_.v_limit * 2 / params_.v_range) + 1;
    int N = prey_path.size();
    Q_table_.assign(N, std::vector<std::vector<double>>(v_states, std::vector<double>(params_.act_num, 0.0)));

    int successful_episodes = 0;
    
    for (int episode = 0; episode < params_.train_number_max; ++episode) {
        bool episode_ended_successfully = false;
        
        // Start of a single training episode
        double v = 1.0;
        Eigen::Vector2d p_curr = stateToWorldCoords(0, v, prey_path, ref_point);
        Eigen::Vector2d p_prev = p_curr; // For the first step, assume no previous movement
        int point_idx = 0;

        while (point_idx < N - 1) {
            // Discretize v to get state index
            int v_idx = static_cast<int>(std::round((v - (1.0 - params_.v_limit)) / params_.v_range));
            v_idx = std::max(0, std::min(v_states - 1, v_idx));

            // Dynamic Epsilon-Greedy action selection
            double epsilon = calculateEpsilon(episode, successful_episodes);
            int action_idx;
            std::uniform_real_distribution<double> dis(0.0, 1.0);
            if (dis(rng_) < epsilon) {
                std::uniform_int_distribution<int> act_dis(0, params_.act_num - 1);
                action_idx = act_dis(rng_);
            } else {
                const auto& q_actions = Q_table_[point_idx][v_idx];
                action_idx = std::distance(q_actions.begin(), std::max_element(q_actions.begin(), q_actions.end()));
            }

            // State transition based on action
            // Action 0: v-range, Action 1: 0, Action 2: v+range
            double v_change = (action_idx - 1) * params_.v_range;
            double v_next = v + v_change;
            
            int next_point_idx = point_idx + 1;
            Eigen::Vector2d p_next = stateToWorldCoords(next_point_idx, v_next, prey_path, ref_point);

            bool is_terminal = false;
            double reward = calculateReward(next_point_idx, v_next, p_curr, p_next, p_prev, obstacles, goal, is_terminal, N);

            // Q-table update
            int next_v_idx = static_cast<int>(std::round((v_next - (1.0 - params_.v_limit)) / params_.v_range));
            next_v_idx = std::max(0, std::min(v_states - 1, next_v_idx));

            double old_q = Q_table_[point_idx][v_idx][action_idx];
            double next_max_q = 0.0;
            if (!is_terminal && next_point_idx < N - 1) {
                const auto& next_q_actions = Q_table_[next_point_idx][next_v_idx];
                next_max_q = *std::max_element(next_q_actions.begin(), next_q_actions.end());
            }
            Q_table_[point_idx][v_idx][action_idx] = old_q + params_.alpha * (reward + params_.gamma * next_max_q - old_q);

            if (is_terminal) {
                if (reward > params_.w_v_bonus_final / 2.0) { // If we got a high positive reward, it was successful
                    episode_ended_successfully = true;
                }
                break;
            }

            // Update state for next step
            p_prev = p_curr;
            p_curr = p_next;
            v = v_next;
            point_idx = next_point_idx;
        }

        if (episode_ended_successfully) {
            successful_episodes++;
        }
        if (episode > 0 && episode % 5000 == 0) {
            RCLCPP_INFO(logger_, "Training episode %d/%d, Success count: %d, Epsilon: %.3f", episode, params_.train_number_max, successful_episodes, calculateEpsilon(episode, successful_episodes));
        }
    }
}

/**
 * @brief 计算动态贪婪策略的概率
*/
double VMCPlanner::calculateEpsilon(int episode_num, int success_count)
{
    double term1 = params_.eta1 * (1.0 - std::tanh(static_cast<double>(episode_num) / params_.xi_train));
    double term2 = params_.eta2 * (1.0 - std::tanh(static_cast<double>(success_count) / params_.xi_reach));
    return std::max(0.05, term1 + term2); // Ensure a minimum exploration rate
}

/**
 * @brief 计算每一次动作获取的奖励
*/
double VMCPlanner::calculateReward(int point_idx, double v_next, const Eigen::Vector2d& p_curr, const Eigen::Vector2d& p_next, const Eigen::Vector2d& p_prev, const std::vector<Eigen::Vector3d>& obstacles, const Eigen::Vector2d& goal, bool& is_terminal, int N)
{
    is_terminal = false;
    
    for (const auto& obs : obstacles) {
        if ((p_next - obs.head<2>()).squaredNorm() < obs.z() * obs.z()) {
            is_terminal = true;
            return params_.w_collision; // Penalty for collision
        }
    }

    // R2: Out of bounds check
    if (v_next > 1.0 + params_.v_limit || v_next < 1.0 - params_.v_limit) {
        is_terminal = true;
        return -1.0; 
    }
    
    // Check for reaching the final prey point
    if (point_idx >= N - 1) {
        is_terminal = true;
        // R1: Goal reached reward
        if (std::abs(v_next - 1.0) < 1e-6) {
             return params_.w_v_bonus_final; 
        }
        // R3: End point orientation penalty
        else {
            return params_.w_v_deviation_fail * std::abs(v_next - 1.0);
        }
    }

    // R4: Yaw angle limit penalty
    Eigen::Vector2d v1 = p_curr - p_prev;
    Eigen::Vector2d v2 = p_next - p_curr;
    if (v1.norm() > 1e-6 && v2.norm() > 1e-6) {
        double cos_angle = v1.dot(v2) / (v1.norm() * v2.norm());
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        double angle_diff = std::acos(cos_angle);
        if (angle_diff > params_.max_yaw_angle) {
            return params_.w_yaw; 
        }
    }
    
    return 0.0; // No reward or penalty for normal steps
}

/**
 * @brief 根据VMC公式计算出v实际对应的坐标
*/
Eigen::Vector2d VMCPlanner::stateToWorldCoords(int point_idx, double v, const std::vector<Eigen::Vector2d>& prey_path, const Eigen::Vector2d& ref_point)
{
    if (point_idx < 0 || point_idx >= prey_path.size()) {
        RCLCPP_ERROR(logger_, "State point_idx out of bounds: %d. Clamping to nearest valid index.", point_idx);
        point_idx = std::max(0, std::min((int)prey_path.size() - 1, point_idx));
    }
    const auto& prey_point = prey_path[point_idx];
    return ref_point + v * (prey_point - ref_point);
}

/**
 * @brief 根据Q表生成最终的执行路径route
*/
std::vector<Eigen::Vector2d> VMCPlanner::generateFinalRouteFromQ(const std::vector<Eigen::Vector2d>& prey_path, const Eigen::Vector2d& ref_point)
{
    std::vector<Eigen::Vector2d> route;
    route.push_back(prey_path.front());
    
    double v = 1.0;
    int N = prey_path.size();
    int v_states = static_cast<int>(params_.v_limit * 2 / params_.v_range) + 1;

    for (int point_idx = 0; point_idx < N - 1; ++point_idx) {
        int v_idx = static_cast<int>(std::round((v - (1.0 - params_.v_limit)) / params_.v_range));
        v_idx = std::max(0, std::min(v_states - 1, v_idx));
        
        const auto& q_actions = Q_table_[point_idx][v_idx];
        int action_idx = std::distance(q_actions.begin(), std::max_element(q_actions.begin(), q_actions.end()));

        // Apply action to get next v
        // Action 0: v-range, Action 1: 0, Action 2: v+range
        double v_change = (action_idx - 1) * params_.v_range;
        double v_next = v + v_change;

        // Clip v_next to be within bounds
        v_next = std::max(1.0 - params_.v_limit, std::min(1.0 + params_.v_limit, v_next));

        route.push_back(stateToWorldCoords(point_idx + 1, v_next, prey_path, ref_point));
        v = v_next;
    }

    // Ensure the last point is exactly the goal
    route.back() = prey_path.back();

    return route;
}

} // namespace vmc_minco_nav