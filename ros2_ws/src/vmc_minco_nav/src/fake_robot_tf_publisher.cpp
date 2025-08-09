#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("fake_robot_tf_publisher");

    node->declare_parameter<double>("start.x", 1.0);
    node->declare_parameter<double>("start.y", 1.0);
    double start_x = node->get_parameter("start.x").as_double();
    double start_y = node->get_parameter("start.y").as_double();
    
    RCLCPP_INFO(node->get_logger(), "Publishing static TF for fake robot at (%.2f, %.2f)", start_x, start_y);

    auto static_broadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
    
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = node->get_clock()->now(); // Corrected: use get_clock()->now()
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    transform.transform.translation.x = start_x;
    transform.transform.translation.y = start_y;
    transform.transform.translation.z = 0.0;
    transform.transform.rotation.w = 1.0;

    static_broadcaster->sendTransform(transform);
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}