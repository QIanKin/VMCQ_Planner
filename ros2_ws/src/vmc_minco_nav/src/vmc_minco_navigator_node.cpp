#include "vmc_minco_nav/vmc_minco_navigator_node.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "nav2_util/node_utils.hpp"
#include <queue>
#include <algorithm>

namespace vmc_minco_nav
{

VmcMincoNavigatorNode::VmcMincoNavigatorNode(const rclcpp::NodeOptions & options)
    : nav2_util::LifecycleNode("vmc_minco_navigator", "", options),
      costmap_(nullptr)
{
    RCLCPP_INFO(this->get_logger(), "Creating VMC-MINCO Navigator Node...");
}

VmcMincoNavigatorNode::~VmcMincoNavigatorNode() {}

nav2_util::CallbackReturn VmcMincoNavigatorNode::on_configure(const rclcpp_lifecycle::State &)
{
    RCLCPP_INFO(this->get_logger(), "Configuring VMC-MINCO Navigator Node...");
    auto node = shared_from_this();

    // --- 声明并获取参数 ---
    nav2_util::declare_parameter_if_not_declared(node, "global_frame", rclcpp::ParameterValue("map"));
    this->get_parameter("global_frame", global_frame_);
    nav2_util::declare_parameter_if_not_declared(node, "robot_base_frame", rclcpp::ParameterValue("base_link"));
    this->get_parameter("robot_base_frame", robot_base_frame_);
    
    // VMC Parameters
    nav2_util::declare_parameter_if_not_declared(node, "vmc.train_number_max", rclcpp::ParameterValue(15000));
    this->get_parameter("vmc.train_number_max", vmc_params_.train_number_max);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.v_range", rclcpp::ParameterValue(0.01));
    this->get_parameter("vmc.v_range", vmc_params_.v_range);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.v_limit", rclcpp::ParameterValue(0.5));
    this->get_parameter("vmc.v_limit", vmc_params_.v_limit);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.max_yaw_angle", rclcpp::ParameterValue(M_PI / 3.0));
    this->get_parameter("vmc.max_yaw_angle", vmc_params_.max_yaw_angle);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.alpha", rclcpp::ParameterValue(0.5));
    this->get_parameter("vmc.alpha", vmc_params_.alpha);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.gamma", rclcpp::ParameterValue(0.95));
    this->get_parameter("vmc.gamma", vmc_params_.gamma);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.eta1", rclcpp::ParameterValue(0.8));
    this->get_parameter("vmc.eta1", vmc_params_.eta1);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.eta2", rclcpp::ParameterValue(0.2));
    this->get_parameter("vmc.eta2", vmc_params_.eta2);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.xi_train", rclcpp::ParameterValue(1000.0));
    this->get_parameter("vmc.xi_train", vmc_params_.xi_train);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.xi_reach", rclcpp::ParameterValue(10.0));
    this->get_parameter("vmc.xi_reach", vmc_params_.xi_reach);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.goal_radius", rclcpp::ParameterValue(1.0));
    this->get_parameter("vmc.goal_radius", vmc_params_.goal_radius);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.prey_path_spacing", rclcpp::ParameterValue(1.0));
    this->get_parameter("vmc.prey_path_spacing", vmc_params_.prey_path_spacing);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.rewards.goal", rclcpp::ParameterValue(1.0));
    this->get_parameter("vmc.rewards.goal", vmc_params_.w_goal);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.rewards.collision", rclcpp::ParameterValue(-5.0));
    this->get_parameter("vmc.rewards.collision", vmc_params_.w_collision);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.rewards.yaw", rclcpp::ParameterValue(-1.0));
    this->get_parameter("vmc.rewards.yaw", vmc_params_.w_yaw);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.rewards.v_bonus_final", rclcpp::ParameterValue(10.0));
    this->get_parameter("vmc.rewards.v_bonus_final", vmc_params_.w_v_bonus_final);
    nav2_util::declare_parameter_if_not_declared(node, "vmc.rewards.v_deviation_fail", rclcpp::ParameterValue(-1.0));
    this->get_parameter("vmc.rewards.v_deviation_fail", vmc_params_.w_v_deviation_fail);

    // MINCO Parameters
    nav2_util::declare_parameter_if_not_declared(node, "minco.v_max", rclcpp::ParameterValue(1.5));
    this->get_parameter("minco.v_max", minco_params_.v_max);
    nav2_util::declare_parameter_if_not_declared(node, "minco.a_max", rclcpp::ParameterValue(5.0));
    this->get_parameter("minco.a_max", minco_params_.a_max);
    nav2_util::declare_parameter_if_not_declared(node, "minco.drone_radius", rclcpp::ParameterValue(1.0));
    this->get_parameter("minco.drone_radius", minco_params_.drone_radius);
    nav2_util::declare_parameter_if_not_declared(node, "minco.weights.energy", rclcpp::ParameterValue(1.0));
    this->get_parameter("minco.weights.energy", minco_params_.w_energy);
    nav2_util::declare_parameter_if_not_declared(node, "minco.weights.time", rclcpp::ParameterValue(20.0));
    this->get_parameter("minco.weights.time", minco_params_.w_time);
    nav2_util::declare_parameter_if_not_declared(node, "minco.weights.distance", rclcpp::ParameterValue(20.0));
    this->get_parameter("minco.weights.distance", minco_params_.w_distance);
    nav2_util::declare_parameter_if_not_declared(node, "minco.weights.feasibility", rclcpp::ParameterValue(1000.0));
    this->get_parameter("minco.weights.feasibility", minco_params_.w_feasibility);
    nav2_util::declare_parameter_if_not_declared(node, "minco.weights.obstacle", rclcpp::ParameterValue(1000.0));
    this->get_parameter("minco.weights.obstacle", minco_params_.w_obstacle);
    nav2_util::declare_parameter_if_not_declared(node, "minco.kappa", rclcpp::ParameterValue(10));
    this->get_parameter("minco.kappa", minco_params_.kappa);
    nav2_util::declare_parameter_if_not_declared(node, "minco.n_segments", rclcpp::ParameterValue(20));
    this->get_parameter("minco.n_segments", minco_params_.n_segments);

    // PID 和控制器参数
    nav2_util::declare_parameter_if_not_declared(node, "controller.position_pid.p", rclcpp::ParameterValue(1.0));
    this->get_parameter("controller.position_pid.p", pid_pos_x_.p_gain);
    nav2_util::declare_parameter_if_not_declared(node, "controller.position_pid.i", rclcpp::ParameterValue(0.0));
    this->get_parameter("controller.position_pid.i", pid_pos_x_.i_gain);
    nav2_util::declare_parameter_if_not_declared(node, "controller.position_pid.d", rclcpp::ParameterValue(0.1));
    this->get_parameter("controller.position_pid.d", pid_pos_x_.d_gain);
    nav2_util::declare_parameter_if_not_declared(node, "controller.position_pid.output_limit", rclcpp::ParameterValue(minco_params_.v_max));
    this->get_parameter("controller.position_pid.output_limit", pid_pos_x_.output_limit);
    pid_pos_y_ = pid_pos_x_;

    nav2_util::declare_parameter_if_not_declared(node, "controller.velocity_pid.p", rclcpp::ParameterValue(2.0));
    this->get_parameter("controller.velocity_pid.p", pid_vel_x_.p_gain);
    nav2_util::declare_parameter_if_not_declared(node, "controller.velocity_pid.i", rclcpp::ParameterValue(0.1));
    this->get_parameter("controller.velocity_pid.i", pid_vel_x_.i_gain);
    nav2_util::declare_parameter_if_not_declared(node, "controller.velocity_pid.d", rclcpp::ParameterValue(0.0));
    this->get_parameter("controller.velocity_pid.d", pid_vel_x_.d_gain);
    nav2_util::declare_parameter_if_not_declared(node, "controller.velocity_pid.output_limit", rclcpp::ParameterValue(minco_params_.v_max));
    this->get_parameter("controller.velocity_pid.output_limit", pid_vel_x_.output_limit);
    pid_vel_y_ = pid_vel_x_;

    nav2_util::declare_parameter_if_not_declared(node, "controller.lookahead_distance", rclcpp::ParameterValue(0.5));
    this->get_parameter("controller.lookahead_distance", lookahead_distance_);
    nav2_util::declare_parameter_if_not_declared(node, "controller.goal_tolerance", rclcpp::ParameterValue(0.2));
    this->get_parameter("controller.goal_tolerance", goal_tolerance_);


    // --- 初始化规划器组件 ---
    vmc_planner_ = std::make_unique<VMCPlanner>(vmc_params_, this->get_logger());
    minco_optimizer_ = std::make_unique<MincoOptimizer>();

    // --- 设置ROS接口 ---
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    costmap_ros_ = std::make_shared<nav2_costmap_2d::Costmap2DROS>(
        "global_costmap", std::string{get_namespace()}, "global_costmap");
    
    // costmap_ros_ is a LifecycleNode, so we need to configure and activate it.
    costmap_ros_->on_configure(this->get_current_state());
    costmap_ = costmap_ros_->getCostmap();
    
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10, std::bind(&VmcMincoNavigatorNode::goalCallback, this, std::placeholders::_1));
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", rclcpp::QoS(10).best_effort(), std::bind(&VmcMincoNavigatorNode::odomCallback, this, std::placeholders::_1));
        
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn VmcMincoNavigatorNode::on_activate(const rclcpp_lifecycle::State &)
{
    RCLCPP_INFO(this->get_logger(), "Activating VMC-MINCO Navigator Node...");
    path_pub_->on_activate();
    cmd_vel_pub_->on_activate();
    costmap_ros_->on_activate(this->get_current_state());

    // 创建控制定时器
    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(50), // 20 Hz 控制循环
        std::bind(&VmcMincoNavigatorNode::controlLoop, this));
    last_control_time_ = this->now();

    return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn VmcMincoNavigatorNode::on_deactivate(const rclcpp_lifecycle::State &)
{
    RCLCPP_INFO(this->get_logger(), "Deactivating VMC-MINCO Navigator Node...");
    path_pub_->on_deactivate();
    cmd_vel_pub_->on_deactivate();
    costmap_ros_->on_deactivate(this->get_current_state());
    control_timer_->cancel();
    control_timer_.reset();
    return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn VmcMincoNavigatorNode::on_cleanup(const rclcpp_lifecycle::State &)
{
    RCLCPP_INFO(this->get_logger(), "Cleaning up VMC-MINCO Navigator Node...");
    vmc_planner_.reset();
    minco_optimizer_.reset();
    tf_listener_.reset();
    tf_buffer_.reset();
    costmap_ros_->on_cleanup(this->get_current_state());
    costmap_ros_.reset();
    costmap_ = nullptr;
    path_pub_.reset();
    cmd_vel_pub_.reset();
    goal_sub_.reset();
    odom_sub_.reset();
    return nav2_util::CallbackReturn::SUCCESS;
}

nav2_util::CallbackReturn VmcMincoNavigatorNode::on_shutdown(const rclcpp_lifecycle::State &)
{
    RCLCPP_INFO(this->get_logger(), "Shutting down VMC-MINCO Navigator Node...");
    return nav2_util::CallbackReturn::SUCCESS;
}

void VmcMincoNavigatorNode::goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
        RCLCPP_WARN(this->get_logger(), "Node is not active, ignoring goal.");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Received a new goal pose.");
    current_path_.poses.clear();
    
    geometry_msgs::msg::TransformStamped transform;
    try {
        transform = tf_buffer_->lookupTransform(global_frame_, robot_base_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
        RCLCPP_ERROR(this->get_logger(), "Could not transform from %s to %s: %s", robot_base_frame_.c_str(), global_frame_.c_str(), ex.what());
        return;
    }
    
    geometry_msgs::msg::Pose start_pose;
    start_pose.position.x = transform.transform.translation.x;
    start_pose.position.y = transform.transform.translation.y;
    start_pose.orientation = transform.transform.rotation;
    
    runPlanning(start_pose, msg->pose);
}

void VmcMincoNavigatorNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(odom_mutex_);
    // 里程计的速度是在 odom 坐标系下的，通常 odom 和 map 之间只有平移
    // 如果有旋转，需要进行坐标变换
    current_velocity_global_.x() = msg->twist.twist.linear.x;
    current_velocity_global_.y() = msg->twist.twist.linear.y;
}

void VmcMincoNavigatorNode::runPlanning(const geometry_msgs::msg::Pose& start, const geometry_msgs::msg::Pose& goal)
{
    auto planning_start_time = this->get_clock()->now();
    RCLCPP_INFO(this->get_logger(), "Starting VMC-MINCO planning pipeline...");

    Eigen::Vector2d start_point(start.position.x, start.position.y);
    Eigen::Vector2d goal_point(goal.position.x, goal.position.y);

    // === Step 0: 从代价地图提取障碍物 ===
    extractObstaclesFromCostmap(minco_params_.obstacles);
    RCLCPP_INFO(this->get_logger(), "Extracted %zu circular obstacles from costmap.", minco_params_.obstacles.size());

    // === Step 1: VMC 前端规划 ===
    auto initial_waypoints_2d = vmc_planner_->plan(start_point, goal_point, minco_params_.obstacles);
    if (initial_waypoints_2d.empty()) {
        RCLCPP_ERROR(this->get_logger(), "VMC front-end failed. Aborting planning.");
        return;
    }
    
    // === Step 2: MINCO 后端优化 ===
    std::vector<Eigen::Vector2d> minco_initial_waypoints;
    for (int i = 0; i <= minco_params_.n_segments; ++i) {
        int index = static_cast<int>(std::round(i * (initial_waypoints_2d.size() - 1.0) / minco_params_.n_segments));
        minco_initial_waypoints.push_back(initial_waypoints_2d[index]);
    }
    
    minco_params_.start_waypoint = minco_initial_waypoints.front();
    minco_params_.end_waypoint = minco_initial_waypoints.back();
    minco_params_.initial_waypoints = minco_initial_waypoints;
    
    MincoTrajectory optimized_trajectory;
    bool success = minco_optimizer_->optimize(minco_params_, optimized_trajectory, this->get_logger());
    
    if (!success || !optimized_trajectory.isValid()) {
        RCLCPP_ERROR(this->get_logger(), "MINCO back-end optimization failed. Aborting.");
        return;
    }

    // === Step 3: 采样、发布并存储最终轨迹和速度 ===
    current_path_ = sampleTrajectory(optimized_trajectory.coeffs, optimized_trajectory.T, minco_params_, current_path_velocities_);
    current_path_.header.stamp = this->get_clock()->now();
    current_path_.header.frame_id = global_frame_;
    path_pub_->publish(current_path_);
    
    // 重置控制器和路径跟随状态
    current_path_segment_ = 0;
    pid_pos_x_.reset(); pid_pos_y_.reset();
    pid_vel_x_.reset(); pid_vel_y_.reset();
    last_control_time_ = this->now();

    auto planning_end_time = this->get_clock()->now();
    RCLCPP_INFO(this->get_logger(), "VMC-MINCO planning finished in %.4f s. Stored path with %zu poses for execution.", (planning_end_time - planning_start_time).seconds(), current_path_.poses.size());
}

void VmcMincoNavigatorNode::extractObstaclesFromCostmap(std::vector<Eigen::Vector3d>& obstacles)
{
    obstacles.clear();
    unsigned int size_x = costmap_->getSizeInCellsX();
    unsigned int size_y = costmap_->getSizeInCellsY();
    
    std::vector<std::vector<bool>> visited(size_x, std::vector<bool>(size_y, false));
  
    for (unsigned int r = 0; r < size_y; ++r) {
        for (unsigned int c = 0; c < size_x; ++c) {
            if (visited[c][r] || costmap_->getCost(c, r) < nav2_costmap_2d::LETHAL_OBSTACLE) {
                continue;
            }
            
            std::vector<std::pair<unsigned int, unsigned int>> cluster_cells;
            std::queue<std::pair<unsigned int, unsigned int>> q;
            
            q.push({c, r});
            visited[c][r] = true;
            double sum_x = 0.0, sum_y = 0.0;
            
            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();
                cluster_cells.push_back({cx, cy});
                sum_x += cx;
                sum_y += cy;
                
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        if (dr == 0 && dc == 0) continue;
                        unsigned int nc = cx + dc;
                        unsigned int nr = cy + dr;
                        if (nc < size_x && nr < size_y && !visited[nc][nr] &&
                            costmap_->getCost(nc, nr) >= nav2_costmap_2d::LETHAL_OBSTACLE) {
                            visited[nc][nr] = true;
                            q.push({nc, nr});
                        }
                    }
                }
            }
            
            if (cluster_cells.size() < 3) continue;

            double center_x_cell = sum_x / cluster_cells.size();
            double center_y_cell = sum_y / cluster_cells.size();
            
            double max_dist_sq = 0.0;
            for (const auto& p : cluster_cells) {
                double dist_sq = std::pow(p.first - center_x_cell, 2) + std::pow(p.second - center_y_cell, 2);
                max_dist_sq = std::max(max_dist_sq, dist_sq);
            }
            
            double center_x_world, center_y_world;
            costmap_->mapToWorld(static_cast<unsigned int>(center_x_cell), static_cast<unsigned int>(center_y_cell), center_x_world, center_y_world);
            double radius_world = std::sqrt(max_dist_sq) * costmap_->getResolution();
            
            obstacles.emplace_back(center_x_world, center_y_world, radius_world);
        }
    }
}

nav_msgs::msg::Path VmcMincoNavigatorNode::sampleTrajectory(const Eigen::MatrixXd& coeffs, const Eigen::VectorXd& T, const MincoParameters& params, std::vector<Eigen::Vector2d>& velocities)
{
    nav_msgs::msg::Path path;
    velocities.clear();
    
    for (int i = 0; i < params.n_segments; ++i) {
        Eigen::MatrixXd c_i = coeffs.block(i * params.n_coeffs, 0, params.n_coeffs, params.dims);
        
        double segment_duration = T(i);
        for (double t = 0.0; t < segment_duration; t += 0.02) {
            Eigen::VectorXd B0 = MincoOptimizer::getPolyBasis(t, params.n_order, 0);
            Eigen::VectorXd B1 = MincoOptimizer::getPolyBasis(t, params.n_order, 1);
            
            Eigen::RowVector2d pos = B0.transpose() * c_i;
            Eigen::RowVector2d vel = B1.transpose() * c_i;
            
            geometry_msgs::msg::PoseStamped pose;
            pose.pose.position.x = pos.x();
            pose.pose.position.y = pos.y();
            path.poses.push_back(pose);
            velocities.emplace_back(vel.x(), vel.y());
        }
    }
    
    // 添加最后一个点
    Eigen::MatrixXd c_final = coeffs.block((params.n_segments - 1) * params.n_coeffs, 0, params.n_coeffs, params.dims);
    Eigen::VectorXd B0_final = MincoOptimizer::getPolyBasis(T(params.n_segments - 1), params.n_order, 0);
    Eigen::RowVector2d pos_final = B0_final.transpose() * c_final;
    geometry_msgs::msg::PoseStamped pose_final;
    pose_final.pose.position.x = pos_final.x();
    pose_final.pose.position.y = pos_final.y();
    path.poses.push_back(pose_final);
    velocities.emplace_back(0.0, 0.0);

    // 设置所有位姿的朝向
    for (size_t i = 0; i < path.poses.size() - 1; ++i) {
        double dx = path.poses[i+1].pose.position.x - path.poses[i].pose.position.x;
        double dy = path.poses[i+1].pose.position.y - path.poses[i].pose.position.y;
        double yaw = std::atan2(dy, dx);
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        path.poses[i].pose.orientation = tf2::toMsg(q);
    }
    if (path.poses.size() > 1) {
        path.poses.back().pose.orientation = path.poses[path.poses.size()-2].pose.orientation;
    }
    
    return path;
}

void VmcMincoNavigatorNode::controlLoop()
{
    if (current_path_.poses.empty()) { return; }

    geometry_msgs::msg::TransformStamped transform;
    try {
        transform = tf_buffer_->lookupTransform(global_frame_, robot_base_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(this->get_logger(), "Control loop: Could not get robot pose: %s", ex.what());
        return;
    }
    
    Eigen::Vector2d current_pos(transform.transform.translation.x, transform.transform.translation.y);
    
    // 检查是否到达目标
    const auto& goal_pos_geom = current_path_.poses.back().pose.position;
    Eigen::Vector2d goal_pos(goal_pos_geom.x, goal_pos_geom.y);
    if ((current_pos - goal_pos).norm() < goal_tolerance_) {
        RCLCPP_INFO(this->get_logger(), "Goal reached!");
        current_path_.poses.clear();
        cmd_vel_pub_->publish(geometry_msgs::msg::Twist());
        return;
    }
    
    // 寻找前瞻点
    size_t lookahead_idx = current_path_segment_;
    while (lookahead_idx < current_path_.poses.size() - 1) {
        const auto& p_geom = current_path_.poses[lookahead_idx].pose.position;
        Eigen::Vector2d p(p_geom.x, p_geom.y);
        if ((current_pos - p).norm() > lookahead_distance_) {
            break;
        }
        lookahead_idx++;
    }
    current_path_segment_ = lookahead_idx;
    
    const auto& target_pos_geom = current_path_.poses[lookahead_idx].pose.position;
    Eigen::Vector2d target_pos(target_pos_geom.x, target_pos_geom.y);
    const auto& target_vel_ff = current_path_velocities_[lookahead_idx];

    // --- 双环PID控制逻辑 ---
    double dt = (this->now() - last_control_time_).seconds();
    last_control_time_ = this->now();
    if (dt < 1e-6) return;

    // **外环：位置控制** -> 输出期望速度修正量 (全局坐标系)
    Eigen::Vector2d pos_error_global = target_pos - current_pos;
    Eigen::Vector2d desired_vel_from_pos_pid;
    desired_vel_from_pos_pid.x() = pid_pos_x_.calculate(pos_error_global.x(), dt);
    desired_vel_from_pos_pid.y() = pid_pos_y_.calculate(pos_error_global.y(), dt);

    // **内环：速度控制** -> 输出最终的速度指令 (全局坐标系)
    Eigen::Vector2d current_vel_global;
    {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        current_vel_global = current_velocity_global_;
    }
    
    // 目标速度 = 位置环期望速度 + 轨迹前馈速度
    Eigen::Vector2d total_desired_vel = desired_vel_from_pos_pid + target_vel_ff;
    
    // 速度误差 = 目标速度 - 当前实际速度
    Eigen::Vector2d vel_error = total_desired_vel - current_vel_global;

    Eigen::Vector2d final_vel_global;
    final_vel_global.x() = pid_vel_x_.calculate(vel_error.x(), dt);
    final_vel_global.y() = pid_vel_y_.calculate(vel_error.y(), dt);
    
    // 将最终的全局速度指令转换到机器人局部坐标系 (for Mecanum wheels)
    tf2::Quaternion q_robot;
    tf2::fromMsg(transform.transform.rotation, q_robot);
    double yaw = tf2::getYaw(q_robot);
    
    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = final_vel_global.x() * std::cos(yaw) + final_vel_global.y() * std::sin(yaw);
    cmd.linear.y = -final_vel_global.x() * std::sin(yaw) + final_vel_global.y() * std::cos(yaw);
    cmd.angular.z = 0.0;
    
    cmd_vel_pub_->publish(cmd);
}

} // namespace vmc_minco_nav

// Keep the main function in vmc_minco_navigator_node.cpp, not here.
// This file will be compiled as part of the verification_navigator executable.
// Let's create a separate main for that.