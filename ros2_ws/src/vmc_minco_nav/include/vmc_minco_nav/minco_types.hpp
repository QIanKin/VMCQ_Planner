#pragma once

#include <vector>
#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace vmc_minco_nav
{

// 存储所有MINCO相关参数的结构体
// 与您的 matlab params 结构体完全对应
struct MincoParameters
{
    // 动力学和几何约束
    double v_max = 12.0;
    double a_max = 5.0;
    double drone_radius = 1.0;
    // 障碍物: [x, y, radius]
    std::vector<Eigen::Vector3d> obstacles;

    // MINCO 轨迹参数 (固定为 minimum jerk)
    const int dims = 2;
    const int s = 3;
    const int n_order = 2 * s - 1; // 5
    const int n_coeffs = n_order + 1; // 6

    // 优化权重
    double w_energy = 1.0;
    double w_time = 20.0;
    double w_distance = 20.0;
    double w_feasibility = 1000.0;
    double w_obstacle = 1000.0;

    // 惩罚函数采样密度
    int kappa = 10;

    // 起点和终点状态
    Eigen::Vector2d start_waypoint;
    Eigen::Vector2d end_waypoint;
    Eigen::Vector2d start_vel = Eigen::Vector2d::Zero();
    Eigen::Vector2d start_acc = Eigen::Vector2d::Zero();
    Eigen::Vector2d end_vel = Eigen::Vector2d::Zero();
    Eigen::Vector2d end_acc = Eigen::Vector2d::Zero();

    // 航点信息
    int n_segments;
    std::vector<Eigen::Vector2d> initial_waypoints;
};

// 用于NLopt代价函数传递数据的结构体
struct NloptData
{
    const MincoParameters* params;
    // 添加成员以在 costFunction 内部传递/缓存数据
    Eigen::MatrixXd M_banded;
    Eigen::SparseMatrix<double> M_full;
};

// 存储最终轨迹的结构体
struct MincoTrajectory
{
    Eigen::MatrixXd coeffs; // (n_coeffs * n_segments) x dims
    Eigen::VectorXd T;      // n_segments x 1
    double total_duration;

    bool isValid() const {
        return T.size() > 0 && coeffs.size() > 0;
    }
};

} // namespace vmc_minco_nav