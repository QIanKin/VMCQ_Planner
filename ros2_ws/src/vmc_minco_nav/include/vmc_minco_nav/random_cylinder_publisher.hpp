#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <random>

namespace vmc_minco_nav
{

class RandomCylinderPublisher : public rclcpp::Node
{
public:
    RandomCylinderPublisher(const rclcpp::NodeOptions & options);

private:
    void publish_obstacles();

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacles_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 参数
    double map_x_size_;
    double map_y_size_;
    int num_obstacles_;
    double min_radius_;
    double max_radius_;
    double map_height_;
    std::string frame_id_;

    std::mt19937 rng_; // 随机数生成器
};

} // namespace vmc_minco_nav