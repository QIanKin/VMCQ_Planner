#pragma once

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp" // 新增：用于获取当前速度
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_util/lifecycle_node.hpp"

#include "vmc_minco_nav/vmc_planner.hpp"
#include "vmc_minco_nav/minco_optimizer.hpp"
#include "vmc_minco_nav/minco_types.hpp"

namespace vmc_minco_nav
{

// 简单的PID控制器结构体
struct PIDController
{
    double p_gain = 0.0, i_gain = 0.0, d_gain = 0.0;
    double integral = 0.0, prev_error = 0.0;
    double output_limit = 1.0; // 输出限幅

    double calculate(double error, double dt)
    {
        integral += error * dt;
        // 可以增加积分抗饱和逻辑
        // integral = std::clamp(integral, -limit, limit);
        double derivative = (dt > 1e-6) ? (error - prev_error) / dt : 0.0;
        prev_error = error;
        double output = p_gain * error + i_gain * integral + d_gain * derivative;
        return std::clamp(output, -output_limit, output_limit);
    }

    void reset() {
        integral = 0.0;
        prev_error = 0.0;
    }
};

class VmcMincoNavigatorNode : public nav2_util::LifecycleNode
{
public:
    VmcMincoNavigatorNode(const rclcpp::NodeOptions & options);
    ~VmcMincoNavigatorNode();

protected:
    // --- Lifecycle Methods ---
    nav2_util::CallbackReturn on_configure(const rclcpp_lifecycle::State & state) override;
    nav2_util::CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override;
    nav2_util::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override;
    nav2_util::CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override;
    nav2_util::CallbackReturn on_shutdown(const rclcpp_lifecycle::State & state) override;

private:
    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg); // 新增：里程计回调
    void runPlanning(const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal);
    void extractObstaclesFromCostmap(std::vector<Eigen::Vector3d>& obstacles);
    nav_msgs::msg::Path sampleTrajectory(const Eigen::MatrixXd& coeffs, const Eigen::VectorXd& T, const MincoParameters& params, std::vector<Eigen::Vector2d>& velocities);
    void controlLoop();

    // ROS 2 Interfaces
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_; // 新增
    rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    nav2_costmap_2d::Costmap2D* costmap_;
    
    rclcpp::TimerBase::SharedPtr control_timer_;

    // Core Planner Components
    std::unique_ptr<VMCPlanner> vmc_planner_;
    std::unique_ptr<MincoOptimizer> minco_optimizer_;

    // State & Parameters
    std::string global_frame_;
    std::string robot_base_frame_;
    VMCParameters vmc_params_;
    MincoParameters minco_params_;

    // Trajectory Following
    nav_msgs::msg::Path current_path_;
    std::vector<Eigen::Vector2d> current_path_velocities_; // 新增：存储轨迹对应的速度
    size_t current_path_segment_;
    
    // 双环PID控制器
    PIDController pid_pos_x_, pid_pos_y_; // 外环：位置PID
    PIDController pid_vel_x_, pid_vel_y_; // 内环：速度PID
    
    Eigen::Vector2d current_velocity_global_; // 新增：存储机器人当前在全局坐标系下的速度
    std::mutex odom_mutex_; // 新增：保护里程计数据

    double lookahead_distance_;
    double goal_tolerance_;
    rclcpp::Time last_control_time_;
};

} // namespace vmc_minco_nav