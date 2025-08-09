#pragma once

#include "minco_types.hpp"
#include <nlopt.hpp>
#include <rclcpp/logger.hpp> // 新增，用于接收Logger对象

namespace vmc_minco_nav
{

class MincoOptimizer
{
public:
    MincoOptimizer() = default;

    /**
     * @brief 运行完整的MINCO优化流程
     * @param params 包含所有配置参数的结构体
     * @param trajectory 输出的优化后的轨迹
     * @param logger ROS 2 logger object for logging within the optimization process.
     * @return true 如果优化成功
     */
    bool optimize(const MincoParameters& params, MincoTrajectory& trajectory, rclcpp::Logger logger);

    // ### 多项式基向量 (设为public，以便外部采样轨迹) ###
    static Eigen::VectorXd getPolyBasis(double t, int n_order, int derivative_order);
    static void getPolyBases(double t, int n_order, Eigen::VectorXd& B0, Eigen::VectorXd& B1, Eigen::VectorXd& B2, Eigen::VectorXd& B3);

private:
    // ### 核心代价函数 (供NLopt调用) ###
    static double costFunction(const std::vector<double>& x, std::vector<double>& grad, void* data);

    // ### 时间变量转换 (对应 forwardT.m, backwardT.m, backwardGradT.m) ###
    static Eigen::VectorXd forwardT(const Eigen::VectorXd& tau);
    static Eigen::VectorXd backwardT(const Eigen::VectorXd& T);
    static Eigen::VectorXd backwardGradT(const Eigen::VectorXd& tau, const Eigen::VectorXd& gradT);

    // ### MINCO 矩阵构建与求解 (对应 build_minco_matrix.m, solve_banded_system.m) ###
    static void buildMincoMatrix(const std::vector<Eigen::Vector2d>& waypoints, const Eigen::VectorXd& T, const MincoParameters& params, Eigen::MatrixXd& M_banded, Eigen::MatrixXd& b, Eigen::SparseMatrix<double>& M_full);
    static Eigen::MatrixXd solveBandedSystem(Eigen::MatrixXd M_banded, const Eigen::MatrixXd& b, int p, int q);
    
    // ### 成本与梯度计算 (对应 calculate_*.m) ###
    static void calculateEnergyAndGradient(const Eigen::MatrixXd& coeffs, const Eigen::VectorXd& T, const MincoParameters& params, double& energy, Eigen::MatrixXd& grad_c, Eigen::VectorXd& grad_T);
    static void calculatePenaltyAndGradient(const Eigen::MatrixXd& coeffs, const Eigen::VectorXd& T, const MincoParameters& params, double& penalty, Eigen::MatrixXd& grad_c, Eigen::VectorXd& grad_T);

    // ### 梯度反向传播 (对应 backpropagate_minco_gradient.m) ###
    static void backpropagateMincoGradient(const Eigen::VectorXd& T, const Eigen::MatrixXd& total_grad_c, const Eigen::VectorXd& total_grad_T_direct, const MincoParameters& params, const Eigen::SparseMatrix<double>& M_full, const Eigen::MatrixXd& coeffs, Eigen::MatrixXd& grad_q, Eigen::VectorXd& grad_T_total);
};

} // namespace vmc_minco_nav