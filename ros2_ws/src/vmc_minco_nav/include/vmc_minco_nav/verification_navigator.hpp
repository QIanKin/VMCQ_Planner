#pragma once

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// 引用您项目中已有的类
#include "vmc_minco_nav/vmc_planner.hpp"
#include "vmc_minco_nav/minco_optimizer.hpp"
#include "vmc_minco_nav/minco_types.hpp"

namespace vmc_minco_nav
{

class VerificationNavigator : public rclcpp::Node
{
public:
    VerificationNavigator(const rclcpp::NodeOptions & options);

private:
    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void obstaclesCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);

    void runPlanning(const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal);
    
    // 从您原有代码中借用的采样函数签名
    nav_msgs::msg::Path sampleTrajectory(const Eigen::MatrixXd& coeffs, const Eigen::VectorXd& T, const MincoParameters& params);

    void publishWaypoints(const std::vector<Eigen::Vector2d>& waypoints, 
                          const std::string& ns,
                          double r, double g, double b);

    // ROS 2 接口
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_sub_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr planned_path_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr vmc_path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr waypoints_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // 规划器核心组件
    std::unique_ptr<VMCPlanner> vmc_planner_;
    std::unique_ptr<MincoOptimizer> minco_optimizer_;

    // 状态和参数
    std::string global_frame_;
    std::string robot_base_frame_;
    VMCParameters vmc_params_;
    MincoParameters minco_params_;
    
    bool obstacles_received_ = false;
};

} // namespace vmc_minco_nav