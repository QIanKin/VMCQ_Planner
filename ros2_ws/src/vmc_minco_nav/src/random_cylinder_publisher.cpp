#include "vmc_minco_nav/random_cylinder_publisher.hpp"

namespace vmc_minco_nav
{

RandomCylinderPublisher::RandomCylinderPublisher(const rclcpp::NodeOptions & options)
    : Node("random_cylinder_publisher", options),
      rng_(std::random_device{}())
{
    RCLCPP_INFO(this->get_logger(), "Initializing Random Cylinder Publisher...");

    // 声明并获取参数
    map_x_size_ = this->declare_parameter<double>("map_size.x", 20.0);
    map_y_size_ = this->declare_parameter<double>("map_size.y", 20.0);
    map_height_ = this->declare_parameter<double>("map_size.z", 2.0);
    num_obstacles_ = this->declare_parameter<int>("obstacles.number", 15);
    min_radius_ = this->declare_parameter<double>("obstacles.radius_min", 0.5);
    max_radius_ = this->declare_parameter<double>("obstacles.radius_max", 1.5);
    frame_id_ = this->declare_parameter<std::string>("frame_id", "map");
    
    // 创建发布者
    obstacles_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/obstacles", rclcpp::QoS(1).transient_local());

    // 创建一个定时器，仅执行一次来发布静态障碍物
    timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        [this]() -> void {
            this->publish_obstacles();
            this->timer_->cancel(); // 发布一次后就取消定时器
        });
}

void RandomCylinderPublisher::publish_obstacles()
{
    visualization_msgs::msg::MarkerArray marker_array;
    std::uniform_real_distribution<double> x_dist(0.0, map_x_size_);
    std::uniform_real_distribution<double> y_dist(0.0, map_y_size_);
    std::uniform_real_distribution<double> r_dist(min_radius_, max_radius_);

    for (int i = 0; i < num_obstacles_; ++i)
    {
        double x = x_dist(rng_);
        double y = y_dist(rng_);
        double r = r_dist(rng_);
        
        // 避免在地图角落生成障碍物，给起始点留出空间
        if (x < 3.0 && y < 3.0) continue;
        if (x > map_x_size_ - 3.0 && y > map_y_size_ - 3.0) continue;

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "obstacles";
        marker.id = i;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = map_height_ / 2.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = r * 2.0; // 直径
        marker.scale.y = r * 2.0; // 直径
        marker.scale.z = map_height_;

        marker.color.r = 0.5f;
        marker.color.g = 0.5f;
        marker.color.b = 0.5f;
        marker.color.a = 1.0f;
        
        marker.lifetime = rclcpp::Duration(0, 0); // 永久显示

        marker_array.markers.push_back(marker);
    }

    obstacles_pub_->publish(marker_array);
    RCLCPP_INFO(this->get_logger(), "Published %zu obstacles.", marker_array.markers.size());
}

} // namespace vmc_minco_nav

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<vmc_minco_nav::RandomCylinderPublisher>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}