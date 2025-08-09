#include "vmc_minco_nav/verification_navigator.hpp"
#include "nav2_util/node_utils.hpp" // 用于参数声明
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp" // 新增，用于TF转换

namespace vmc_minco_nav
{

VerificationNavigator::VerificationNavigator(const rclcpp::NodeOptions & options)
    : Node("verification_navigator", options)
{
    RCLCPP_INFO(this->get_logger(), "Initializing Verification Navigator Node...");
    
    // --- 声明和获取参数 ---
    nav2_util::declare_parameter_if_not_declared(this, "global_frame", rclcpp::ParameterValue("map"));
    this->get_parameter("global_frame", global_frame_);
    nav2_util::declare_parameter_if_not_declared(this, "robot_base_frame", rclcpp::ParameterValue("base_link"));
    this->get_parameter("robot_base_frame", robot_base_frame_);
    
    // VMC & MINCO 参数
    nav2_util::declare_parameter_if_not_declared(this, "vmc.train_number_max", rclcpp::ParameterValue(15000));
    this->get_parameter("vmc.train_number_max", vmc_params_.train_number_max);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.prey_path_spacing", rclcpp::ParameterValue(0.5));
    this->get_parameter("vmc.prey_path_spacing", vmc_params_.prey_path_spacing);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.v_range", rclcpp::ParameterValue(0.01));
    this->get_parameter("vmc.v_range", vmc_params_.v_range);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.v_limit", rclcpp::ParameterValue(0.5));
    this->get_parameter("vmc.v_limit", vmc_params_.v_limit);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.max_yaw_angle", rclcpp::ParameterValue(M_PI / 3.0));
    this->get_parameter("vmc.max_yaw_angle", vmc_params_.max_yaw_angle);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.alpha", rclcpp::ParameterValue(0.5));
    this->get_parameter("vmc.alpha", vmc_params_.alpha);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.gamma", rclcpp::ParameterValue(0.95));
    this->get_parameter("vmc.gamma", vmc_params_.gamma);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.eta1", rclcpp::ParameterValue(0.8));
    this->get_parameter("vmc.eta1", vmc_params_.eta1);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.eta2", rclcpp::ParameterValue(0.2));
    this->get_parameter("vmc.eta2", vmc_params_.eta2);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.xi_train", rclcpp::ParameterValue(1000.0));
    this->get_parameter("vmc.xi_train", vmc_params_.xi_train);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.xi_reach", rclcpp::ParameterValue(10.0));
    this->get_parameter("vmc.xi_reach", vmc_params_.xi_reach);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.goal_radius", rclcpp::ParameterValue(1.0));
    this->get_parameter("vmc.goal_radius", vmc_params_.goal_radius);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.rewards.goal", rclcpp::ParameterValue(1.0));
    this->get_parameter("vmc.rewards.goal", vmc_params_.w_goal);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.rewards.collision", rclcpp::ParameterValue(-5.0));
    this->get_parameter("vmc.rewards.collision", vmc_params_.w_collision);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.rewards.yaw", rclcpp::ParameterValue(-1.0));
    this->get_parameter("vmc.rewards.yaw", vmc_params_.w_yaw);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.rewards.v_bonus_final", rclcpp::ParameterValue(10.0));
    this->get_parameter("vmc.rewards.v_bonus_final", vmc_params_.w_v_bonus_final);
    nav2_util::declare_parameter_if_not_declared(this, "vmc.rewards.v_deviation_fail", rclcpp::ParameterValue(-1.0));
    this->get_parameter("vmc.rewards.v_deviation_fail", vmc_params_.w_v_deviation_fail);

    nav2_util::declare_parameter_if_not_declared(this, "minco.v_max", rclcpp::ParameterValue(1.5));
    this->get_parameter("minco.v_max", minco_params_.v_max);
    nav2_util::declare_parameter_if_not_declared(this, "minco.n_segments", rclcpp::ParameterValue(10));
    this->get_parameter("minco.n_segments", minco_params_.n_segments);
    nav2_util::declare_parameter_if_not_declared(this, "minco.drone_radius", rclcpp::ParameterValue(1.0));
    this->get_parameter("minco.drone_radius", minco_params_.drone_radius);


    // --- 初始化规划器组件 ---
    vmc_planner_ = std::make_unique<VMCPlanner>(vmc_params_, this->get_logger());
    minco_optimizer_ = std::make_unique<MincoOptimizer>();

    // --- 设置ROS接口 ---
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10, std::bind(&VerificationNavigator::goalCallback, this, std::placeholders::_1));
    
    obstacles_sub_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        "/obstacles", rclcpp::QoS(1).transient_local(), std::bind(&VerificationNavigator::obstaclesCallback, this, std::placeholders::_1));

    planned_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
    vmc_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/vmc_path", 10);
    waypoints_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/waypoints", 10);
}

void VerificationNavigator::obstaclesCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg)
{
    minco_params_.obstacles.clear();
    for (const auto& marker : msg->markers) {
        minco_params_.obstacles.emplace_back(
            marker.pose.position.x,
            marker.pose.position.y,
            marker.scale.x / 2.0 // Marker scale is diameter, we need radius
        );
    }
    obstacles_received_ = true;
    RCLCPP_INFO(this->get_logger(), "Received and stored %zu obstacles.", minco_params_.obstacles.size());
}

void VerificationNavigator::goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!obstacles_received_) {
        RCLCPP_WARN(this->get_logger(), "No obstacles received yet. Ignoring goal.");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Received a new goal pose for verification.");

    geometry_msgs::msg::TransformStamped transform;
    try {
        // Use a timeout to wait for the transform to become available
        transform = tf_buffer_->lookupTransform(global_frame_, robot_base_frame_, tf2::TimePointZero, std::chrono::seconds(1));
    } catch (const tf2::TransformException & ex) {
        RCLCPP_ERROR(this->get_logger(), "Could not get start pose TF: %s", ex.what());
        return;
    }
    
    geometry_msgs::msg::Pose start_pose;
    start_pose.position.x = transform.transform.translation.x;
    start_pose.position.y = transform.transform.translation.y;
    
    runPlanning(start_pose, msg->pose);
}

void VerificationNavigator::runPlanning(const geometry_msgs::msg::Pose& start, const geometry_msgs::msg::Pose& goal)
{
    RCLCPP_INFO(this->get_logger(), "Starting verification planning pipeline...");

    Eigen::Vector2d start_point(start.position.x, start.position.y);
    Eigen::Vector2d goal_point(goal.position.x, goal.position.y);

    RCLCPP_INFO(this->get_logger(), "VMC Start: (%.2f, %.2f), Goal: (%.2f, %.2f)", start_point.x(), start_point.y(), goal_point.x(), goal_point.y());

    auto initial_waypoints_2d = vmc_planner_->plan(start_point, goal_point, minco_params_.obstacles);
    if (initial_waypoints_2d.empty()) {
        RCLCPP_ERROR(this->get_logger(), "VMC front-end failed. Aborting.");
        return;
    }

    nav_msgs::msg::Path vmc_path_msg;
    vmc_path_msg.header.stamp = this->get_clock()->now();
    vmc_path_msg.header.frame_id = global_frame_;
    for(const auto& p : initial_waypoints_2d) {
        geometry_msgs::msg::PoseStamped pose;
        pose.pose.position.x = p.x();
        pose.pose.position.y = p.y();
        pose.pose.orientation.w = 1.0;
        vmc_path_msg.poses.push_back(pose);
    }
    vmc_path_pub_->publish(vmc_path_msg);
    
    std::vector<Eigen::Vector2d> minco_initial_waypoints;
    for (int i = 0; i <= minco_params_.n_segments; ++i) {
        int index = static_cast<int>(std::round(i * (initial_waypoints_2d.size() - 1.0) / minco_params_.n_segments));
        index = std::max(0, std::min((int)initial_waypoints_2d.size() - 1, index));
        minco_initial_waypoints.push_back(initial_waypoints_2d[index]);
    }
    
    RCLCPP_INFO(this->get_logger(), "Waypoints for MINCO (%zu):", minco_initial_waypoints.size());
    std::cout << std::fixed << std::setprecision(4);
    for(const auto& p : minco_initial_waypoints) {
        std::cout << "  ( " << p.x() << ", " << p.y() << " )" << std::endl;
    }
    
    publishWaypoints(minco_initial_waypoints, "initial_waypoints", 1.0, 0.0, 0.0); // Red

    minco_params_.start_waypoint = minco_initial_waypoints.front();
    minco_params_.end_waypoint = minco_initial_waypoints.back();
    minco_params_.initial_waypoints = minco_initial_waypoints;
    
    MincoTrajectory optimized_trajectory;
    bool success = minco_optimizer_->optimize(minco_params_, optimized_trajectory, this->get_logger());
    
    if (!success || !optimized_trajectory.isValid()) {
        RCLCPP_ERROR(this->get_logger(), "MINCO back-end optimization failed.");
        return;
    }
    
    nav_msgs::msg::Path final_path = sampleTrajectory(optimized_trajectory.coeffs, optimized_trajectory.T, minco_params_);
    final_path.header.stamp = this->get_clock()->now();
    final_path.header.frame_id = global_frame_;
    planned_path_pub_->publish(final_path);

    RCLCPP_INFO(this->get_logger(), "Verification planning finished. Published visualization paths.");
}

void VerificationNavigator::publishWaypoints(
    const std::vector<Eigen::Vector2d>& waypoints, 
    const std::string& ns,
    double r, double g, double b)
{
    visualization_msgs::msg::MarkerArray markers;
    // 先发送一个DELETEALL指令，清除旧的标记
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header.frame_id = global_frame_;
    delete_marker.header.stamp = this->get_clock()->now();
    delete_marker.ns = ns;
    delete_marker.id = 0;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(delete_marker);
    waypoints_pub_->publish(markers);

    markers.markers.clear();
    int id = 0;
    for (const auto& wp : waypoints) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = global_frame_;
        marker.header.stamp = this->get_clock()->now();
        marker.ns = ns;
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position.x = wp.x();
        marker.pose.position.y = wp.y();
        marker.pose.position.z = 0.5; // 让标记点浮在空中，更显眼
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;
        marker.color.a = 0.8;
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.lifetime = rclcpp::Duration(0, 0); // 永久
        markers.markers.push_back(marker);
    }
    waypoints_pub_->publish(markers);
}

nav_msgs::msg::Path VerificationNavigator::sampleTrajectory(const Eigen::MatrixXd& coeffs, const Eigen::VectorXd& T, const MincoParameters& params)
{
    nav_msgs::msg::Path path;
    
    for (int i = 0; i < params.n_segments; ++i) {
        Eigen::MatrixXd c_i = coeffs.block(i * params.n_coeffs, 0, params.n_coeffs, params.dims);
        
        double segment_duration = T(i);
        for (double t = 0.0; t < segment_duration; t += 0.02) { // 20ms sample rate
            Eigen::VectorXd B0 = MincoOptimizer::getPolyBasis(t, params.n_order, 0);
            Eigen::RowVector2d pos = B0.transpose() * c_i;
            
            geometry_msgs::msg::PoseStamped pose;
            pose.pose.position.x = pos.x();
            pose.pose.position.y = pos.y();
            path.poses.push_back(pose);
        }
    }
    
    Eigen::MatrixXd c_final = coeffs.block((params.n_segments - 1) * params.n_coeffs, 0, params.n_coeffs, params.dims);
    Eigen::VectorXd B0_final = MincoOptimizer::getPolyBasis(T(params.n_segments - 1), params.n_order, 0);
    Eigen::RowVector2d pos_final = B0_final.transpose() * c_final;
    geometry_msgs::msg::PoseStamped pose_final;
    pose_final.pose.position.x = pos_final.x();
    pose_final.pose.position.y = pos_final.y();
    path.poses.push_back(pose_final);

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

} // namespace vmc_minco_nav

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    auto node = std::make_shared<vmc_minco_nav::VerificationNavigator>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}