#pragma once

#include <vector>
#include <random>
#include <string>
#include <Eigen/Eigen>
#include "rclcpp/rclcpp.hpp"

namespace vmc_minco_nav
{

/**
 * @struct VMCParameters
 * @brief Stores all parameters for the VMC-Q learning algorithm.
 * @details This structure directly corresponds to the VMC parameter declarations
 * in test_v3.m.
 */
struct VMCParameters
{
    int train_number_max;
    double v_range;
    double v_limit; // Corresponds to v_limit in MATLAB
    int N; // Number of points for prey path
    double max_yaw_angle;
    double alpha; // Learning rate (gamma in matlab)
    double gamma; // Discount factor (lamda in matlab)
    double eta1;
    double eta2;
    double xi_train;
    double xi_reach;
    double goal_radius;
    double w_goal;
    double w_collision;
    double w_yaw;
    double w_v_bonus_final;
    double w_v_deviation_fail;
    double prey_path_spacing;
    int act_num = 3;
};


/**
 * @class VMCPlanner
 * @brief Front-end path generator using Virtual Motion Camouflage (VMC) and Q-Learning.
 * @details This class is a C++ replica of the VMC-Q learning logic from the first
 * part of test_v3.m and the concepts from the paper "A Motion Camouflage-Inspired Path Planning
 * Method for UAVs based on Reinforcement Learning".
 */
class VMCPlanner
{
public:
    /**
     * @brief Construct a new VMCPlanner object.
     * @param params The configuration parameters for the VMC algorithm.
     * @param logger A shared pointer to the rclcpp logger for logging messages.
     */
    VMCPlanner(const VMCParameters& params, rclcpp::Logger logger);

    /**
     * @brief Main function to generate an initial feasible path.
     * @param start The starting 2D point.
     * @param goal The goal 2D point.
     * @param obstacles A vector of obstacles, where each obstacle is [x, y, radius].
     * @return A vector of 2D waypoints forming the initial path. Returns an empty vector on failure.
     */
    std::vector<Eigen::Vector2d> plan(const Eigen::Vector2d& start, const Eigen::Vector2d& goal, const std::vector<Eigen::Vector3d>& obstacles);

private:
    /**
     * @brief Generates the virtual prey's path, a straight line from start to goal. 
     * @param start The starting point.
     * @param goal The goal point.
     * @return A vector of points representing the prey path.
     */
    std::vector<Eigen::Vector2d> generateVirtualPreyPath(const Eigen::Vector2d& start, const Eigen::Vector2d& goal);

    /**
     * @brief Selects a reference point based on obstacle distribution. 
     * @details The point is chosen on the side of the prey path with fewer/smaller obstacles to create a more effective state space.
     * @param prey_path The generated virtual prey path.
     * @param obstacles The list of obstacles.
     * @return The selected 2D reference point.
     */
    Eigen::Vector2d selectReferencePoint(const std::vector<Eigen::Vector2d>& prey_path, const std::vector<Eigen::Vector3d>& obstacles);

    /**
     * @brief The main Q-table training loop. [cite: 1618, 1634]
     * @param prey_path The virtual prey path.
     * @param ref_point The selected reference point.
     * @param obstacles The list of obstacles.
     * @param goal The final goal point.
     */
    void trainQTable(const std::vector<Eigen::Vector2d>& prey_path, const Eigen::Vector2d& ref_point, const std::vector<Eigen::Vector3d>& obstacles, const Eigen::Vector2d& goal);
    
    /**
     * @brief Calculates the exploration parameter epsilon using a dynamic greedy strategy. [cite: 1601, 1604]
     * @param episode_num The current training episode number.
     * @param success_count The number of times the agent has successfully reached the goal.
     * @return The calculated epsilon value.
     */
    double calculateEpsilon(int episode_num, int success_count);

    /**
     * @brief Calculates the reward for a state transition. [cite: 1495]
     * @details Corresponds to the reward calculation logic in test_v3.m, including penalties for boundary violation,
     * collision, and excessive yaw angle, and rewards for reaching the goal.
     * @param point_idx Current index on the prey path.
     * @param v Current v-value.
     * @param v_next Next v-value after an action.
     * @param p_curr Current world coordinates of the UAV.
     * @param p_next Next world coordinates of the UAV.
     * @param p_prev Previous world coordinates of the UAV.
     * @param obstacles The list of obstacles.
     * @param goal The final goal.
     * @param is_terminal Output flag, set to true if the state is terminal.
     * @param N Total number of points in the prey path.
     * @return The calculated reward value.
     */
    double calculateReward(int point_idx, double v_next, const Eigen::Vector2d& p_curr, const Eigen::Vector2d& p_next, const Eigen::Vector2d& p_prev, const std::vector<Eigen::Vector3d>& obstacles, const Eigen::Vector2d& goal, bool& is_terminal, int N);
    
    /**
     * @brief Converts a VMC state (index, v-value) to world coordinates. [cite: 1449, 1451]
     * @param point_idx The index on the prey path.
     * @param v The path control parameter.
     * @param prey_path The virtual prey path.
     * @param ref_point The reference point.
     * @return The corresponding 2D world coordinates.
     */
    Eigen::Vector2d stateToWorldCoords(int point_idx, double v, const std::vector<Eigen::Vector2d>& prey_path, const Eigen::Vector2d& ref_point);
    
    /**
     * @brief Generates the final path by following the optimal policy in the trained Q-table.
     * @param prey_path The virtual prey path.
     * @param ref_point The selected reference point.
     * @return A vector of 2D waypoints for the final path.
     */
    std::vector<Eigen::Vector2d> generateFinalRouteFromQ(const std::vector<Eigen::Vector2d>& prey_path, const Eigen::Vector2d& ref_point);

    VMCParameters params_;
    std::vector<std::vector<std::vector<double>>> Q_table_;
    std::mt19937 rng_;
    rclcpp::Logger logger_;
};

} // namespace vmc_minco_nav