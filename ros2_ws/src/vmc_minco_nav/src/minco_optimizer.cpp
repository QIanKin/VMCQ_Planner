#include "vmc_minco_nav/minco_optimizer.hpp"
#include "rclcpp/rclcpp.hpp" 
#include <iostream>
#include <chrono>

namespace vmc_minco_nav
{

// #################################################################################################
//                              主优化流程                           
// #################################################################################################

bool MincoOptimizer::optimize(const MincoParameters& params, MincoTrajectory& trajectory, rclcpp::Logger logger)
{
    int n_segments = params.n_segments;
    int n_intermediate_waypoints = n_segments - 1;

    // --- 1. 初始化优化变量 x = [tau_vec; q_vec] ---
    double initial_total_time = (params.start_waypoint - params.end_waypoint).norm() / (params.v_max * 0.5);
    Eigen::VectorXd T_initial = Eigen::VectorXd::Ones(n_segments) * (initial_total_time / n_segments);
    Eigen::VectorXd tau_initial = backwardT(T_initial);

    Eigen::MatrixXd q_initial(n_intermediate_waypoints, params.dims);
    for(int i = 0; i < n_intermediate_waypoints; ++i) {
        q_initial.row(i) = params.initial_waypoints[i+1];
    }

    int tau_len = n_segments;
    int q_len = n_intermediate_waypoints * params.dims;
    std::vector<double> x(tau_len + q_len);
    Eigen::Map<Eigen::VectorXd>(x.data(), tau_len) = tau_initial;
    Eigen::Map<Eigen::MatrixXd>(x.data() + tau_len, q_initial.rows(), q_initial.cols()) = q_initial;

    // --- 2. 配置 NLopt 优化器 ---
    nlopt::opt opt(nlopt::LD_LBFGS, tau_len + q_len);
    NloptData data_wrapper{&params}; // 初始化，M_banded 和 M_full 将在 costFunction 内部填充
    
    opt.set_min_objective(MincoOptimizer::costFunction, &data_wrapper);
    opt.set_xtol_rel(1e-4);
    opt.set_maxeval(100);

    // --- 3. 运行优化 ---
    double final_cost;
    try {
        RCLCPP_INFO(logger, "Starting MINCO optimization...");
        auto start_time = std::chrono::high_resolution_clock::now();
        nlopt::result result = opt.optimize(x, final_cost);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        RCLCPP_INFO(logger, "Optimization finished in %.4fs. Final cost: %.4f", elapsed.count(), final_cost);

    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "NLopt failed: %s", e.what());
        return false;
    }

    // --- 4. 解码优化结果 ---
    Eigen::VectorXd tau_optimized = Eigen::Map<Eigen::VectorXd>(x.data(), tau_len);
    Eigen::MatrixXd q_optimized = Eigen::Map<Eigen::MatrixXd>(x.data() + tau_len, q_initial.rows(), q_initial.cols());
    
    trajectory.T = forwardT(tau_optimized);
    trajectory.total_duration = trajectory.T.sum();

    std::vector<Eigen::Vector2d> final_waypoints;
    final_waypoints.push_back(params.start_waypoint);
    for(int i=0; i < n_intermediate_waypoints; ++i) final_waypoints.push_back(q_optimized.row(i));
    final_waypoints.push_back(params.end_waypoint);
    
    // --- 5. 最终求解系数 ---
    Eigen::MatrixXd M_banded_final, b_final;
    Eigen::SparseMatrix<double> M_full_final;
    buildMincoMatrix(final_waypoints, trajectory.T, params, M_banded_final, b_final, M_full_final);
    trajectory.coeffs = solveBandedSystem(M_banded_final, b_final, params.n_coeffs, params.n_coeffs);

    return true;
}

/**
 * @brief 总代价函数，同时实现解析梯度的返回 
*/
double MincoOptimizer::costFunction(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    NloptData* d = reinterpret_cast<NloptData*>(data);
    const MincoParameters* params = d->params;
    
    // --- 1. 解码优化变量 ---
    int n_segments = params->n_segments;
    int n_intermediate_waypoints = n_segments - 1;
    int tau_len = n_segments;
    int q_len = n_intermediate_waypoints * params->dims;

    Eigen::VectorXd tau = Eigen::Map<const Eigen::VectorXd>(x.data(), tau_len);
    Eigen::MatrixXd q_intermediate = Eigen::Map<const Eigen::MatrixXd>(x.data() + tau_len, n_intermediate_waypoints, params->dims);
    
    Eigen::VectorXd T = forwardT(tau);
    
    std::vector<Eigen::Vector2d> waypoints;
    waypoints.push_back(params->start_waypoint);
    for(int i=0; i < n_intermediate_waypoints; ++i) waypoints.push_back(q_intermediate.row(i));
    waypoints.push_back(params->end_waypoint);

    // --- 2. 构建并求解MINCO问题 ---
    Eigen::MatrixXd b;
    buildMincoMatrix(waypoints, T, *params, d->M_banded, b, d->M_full);
    
    Eigen::MatrixXd coeffs = solveBandedSystem(d->M_banded, b, params->n_coeffs, params->n_coeffs);

    if (coeffs.hasNaN()) {
        if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0);
        return 1e9;
    }

    // --- 3. 计算总成本和显式梯度 ---
    double energy_cost, penalty_cost;
    Eigen::MatrixXd grad_E_c, grad_P_c;
    Eigen::VectorXd grad_E_T, grad_P_T;

    calculateEnergyAndGradient(coeffs, T, *params, energy_cost, grad_E_c, grad_E_T);
    calculatePenaltyAndGradient(coeffs, T, *params, penalty_cost, grad_P_c, grad_P_T);
    
    double time_cost = T.sum();
    Eigen::VectorXd grad_T_time = Eigen::VectorXd::Ones(n_segments);

    // --- 4. 聚合总成本和显式梯度 ---
    double total_cost = params->w_energy * energy_cost + params->w_time * time_cost + penalty_cost;
    
    // --- 5. 梯度反向传播 ---
    if (!grad.empty()) {
        Eigen::MatrixXd total_grad_c = params->w_energy * grad_E_c + grad_P_c;
        Eigen::VectorXd total_grad_T_direct = params->w_energy * grad_E_T + params->w_time * grad_T_time + grad_P_T;
        
        Eigen::MatrixXd grad_q;
        Eigen::VectorXd total_grad_T;
        backpropagateMincoGradient(T, total_grad_c, total_grad_T_direct, *params, d->M_full, coeffs, grad_q, total_grad_T);

        // --- 6. 组装最终梯度向量 ---
        Eigen::VectorXd grad_tau = backwardGradT(tau, total_grad_T);
        
        Eigen::Map<Eigen::VectorXd>(grad.data(), tau_len) = grad_tau;
        Eigen::Map<Eigen::MatrixXd>(grad.data() + tau_len, grad_q.rows(), grad_q.cols()) = grad_q;
    }
    
    return total_cost;
}


// #################################################################################################
//                          辅助函数实现                        
// #################################################################################################

/**
 * @brief 将无约束变量tau转换为有约束变量T
 * @return T 有约束变量
*/
Eigen::VectorXd MincoOptimizer::forwardT(const Eigen::VectorXd& tau) {
    Eigen::VectorXd T(tau.size());
    for (int i = 0; i < tau.size(); ++i) {
        if (tau(i) > 0) {
            T(i) = (0.5 * tau(i) + 1.0) * tau(i) + 1.0;
        } else {
            T(i) = 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
        }
    }
    return T;
}

/**
 * @brief 将有约束变量T转换为无约束变量tau
 * @return tau 无约束变量
*/
Eigen::VectorXd MincoOptimizer::backwardT(const Eigen::VectorXd& T) {
    Eigen::VectorXd tau(T.size());
    for (int i = 0; i < T.size(); ++i) {
        if (T(i) > 1.0) {
            tau(i) = sqrt(2.0 * T(i) - 1.0) - 1.0;
        } else {
            double t_val = std::max(T(i), 1e-9);
            tau(i) = 1.0 - sqrt(2.0 / t_val - 1.0);
        }
    }
    return tau;
}

/**
 * @brief 对T的梯度转换为对tau的梯度（grad_T -> grad_Tau）
 * @param tau 无约束变量(用于确定T的范围)
 * @param gradT 对T的梯度
 * @return gradTau 对tau的梯度 
*/
Eigen::VectorXd MincoOptimizer::backwardGradT(const Eigen::VectorXd& tau, const Eigen::VectorXd& gradT) {
    Eigen::VectorXd gradTau(tau.size());
    for (int i = 0; i < tau.size(); ++i) {
        double dT_dtau = 0;
        if (tau(i) > 0) {
            dT_dtau = tau(i) + 1.0;
        } else {
            double den = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
            dT_dtau = -(tau(i) - 1.0) / (den * den);
        }
        gradTau(i) = gradT(i) * dT_dtau;
    }
    return gradTau;
}

/**
 * @brief 返回多项式的基向量
 * @param t 需要确定的基向量时间t
 * @param n_order minico优化的阶数2s-1(jerk = 2 * 3 - 1 =5)
 * @return B0 & B1 & B2 & B3 位置基向量 & 速度基向量 & 加速度基向量 & jerk基向量
*/
void MincoOptimizer::getPolyBases(double t, int n_order, Eigen::VectorXd& B0, Eigen::VectorXd& B1, Eigen::VectorXd& B2, Eigen::VectorXd& B3) {
    int n_coeffs = n_order + 1;
    B0.resize(n_coeffs); B1.setZero(n_coeffs); B2.setZero(n_coeffs); B3.setZero(n_coeffs);
    
    for (int i = 0; i <= n_order; ++i) B0(i) = std::pow(t, i);
    for (int i = 1; i <= n_order; ++i) B1(i) = i * std::pow(t, i - 1);
    for (int i = 2; i <= n_order; ++i) B2(i) = i * (i - 1) * std::pow(t, i - 2);
    for (int i = 3; i <= n_order; ++i) B3(i) = i * (i - 1) * (i - 2) * std::pow(t, i - 3);
}

/**
 * @brief 返回特定阶数的多项式的基向量
 * @param t 需要确定的基向量时间t
 * @param n_order minico优化的阶数2s-1(jerk = 2 * 3 - 1 =5)
 * @param derivative_order 指定阶数的基向量(位置 速度 加速度 jerk)
 * @return 对应阶数的基向量
*/
Eigen::VectorXd MincoOptimizer::getPolyBasis(double t, int n_order, int derivative_order) {
    int n_coeffs = n_order + 1;
    Eigen::VectorXd basis = Eigen::VectorXd::Zero(n_coeffs);

    if (derivative_order > n_order) {
        return basis;
    }

    for (int i = derivative_order; i <= n_order; ++i) {
        double coeff = 1.0;
        // 计算 i! / (i - k)!
        for (int j = 0; j < derivative_order; ++j) {
            coeff *= (i - j);
        }
        
        if (i - derivative_order >= 0) {
           basis(i) = coeff * std::pow(t, i - derivative_order);
        }
    }
    return basis;
}


/**
 * @brief 构建MINCO问题的约束矩阵M和向量b。
 * @param waypoints 中间航点 (n_segments-1个)。
 * @param T         每段轨迹的时长 (n_segments个)。
 * @param params    MINCO问题参数。
 * @param M_banded  (输出) 紧凑格式的带状矩阵M。
 * @param b         (输出) 约束向量b。
 * @param M_full    (输出) 完整的稀疏矩阵M，用于梯度反向传播。
 * @note 此实现遵循minco.hpp中的矩阵布局，确保与梯度反向传播函数联动正确。
 * 对于s=3(min-jerk)，每个中间点产生6个约束: d3,d4连续, 航点位置, d0,d1,d2连续。
 */
void MincoOptimizer::buildMincoMatrix(const std::vector<Eigen::Vector2d>& waypoints,
                                      const Eigen::VectorXd& T,
                                      const MincoParameters& params, // 使用您的结构体
                                      Eigen::MatrixXd& M_banded,
                                      Eigen::MatrixXd& b,
                                      Eigen::SparseMatrix<double>& M_full)
{
    int n_segments = params.n_segments;
    int n_coeffs = params.n_coeffs;
    int total_vars = n_segments * n_coeffs;
    int s = params.s;

    int p = n_coeffs, q = n_coeffs;
    M_banded.setZero(p + q + 1, total_vars);
    b.setZero(total_vars, params.dims);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(10 * total_vars);

    auto set_val = [&](int r, int c, double val) {
        if (val != 0.0) {
            M_banded(r - c + q, c) = val;
            triplets.emplace_back(r, c, val);
        }
    };

    // --- 头部约束 (0到s-1阶导数) ---
    int r_base = 0;
    for (int k = 0; k < s; ++k) {
        Eigen::VectorXd basis = getPolyBasis(0, params.n_order, k);
        for (int j = 0; j < n_coeffs; ++j) {
            set_val(r_base + k, j, basis(j));
        }
    }
    // ****** 从结构体直接赋值 ******
    b.row(0) = params.start_waypoint;
    b.row(1) = params.start_vel;
    b.row(2) = params.start_acc;

    r_base += s;

    // --- 中间约束 (每个中间点产生 2*s 个约束) ---
    for (int i = 0; i < n_segments - 1; ++i) {
        int c_base = i * n_coeffs;
        double t_i = T(i);
        
        // s=3时, 约束顺序: d3, d4, p, d0, d1, d2 连续/约束
        for (int k = s; k < 2 * s - 1; ++k) {
            Eigen::VectorXd basis1 = getPolyBasis(t_i, params.n_order, k);
            Eigen::VectorXd basis2 = getPolyBasis(0, params.n_order, k);
            for (int j = 0; j < n_coeffs; ++j) {
                set_val(r_base, c_base + j, basis1(j));
                set_val(r_base, c_base + n_coeffs + j, -basis2(j));
            }
            r_base++;
        }

        Eigen::VectorXd basis_p = getPolyBasis(t_i, params.n_order, 0);
        for (int j = 0; j < n_coeffs; ++j) {
            set_val(r_base, c_base + j, basis_p(j));
        }
        b.row(r_base) = waypoints[i+1];
        r_base++;

        for (int k = 0; k < s; ++k) {
            Eigen::VectorXd basis1 = getPolyBasis(t_i, params.n_order, k);
            Eigen::VectorXd basis2 = getPolyBasis(0, params.n_order, k);
            for (int j = 0; j < n_coeffs; ++j) {
                set_val(r_base, c_base + j, basis1(j));
                set_val(r_base, c_base + n_coeffs + j, -basis2(j));
            }
            r_base++;
        }
    }

    // --- 尾部约束 (0到s-1阶导数) ---
    int c_base = (n_segments - 1) * n_coeffs;
    double t_M = T.tail(1)(0);
    for (int k = 0; k < s; ++k) {
        Eigen::VectorXd basis = getPolyBasis(t_M, params.n_order, k);
        for (int j = 0; j < n_coeffs; ++j) {
            set_val(r_base + k, c_base + j, basis(j));
        }
    }
    // ****** 从您的结构体直接赋值 ******
    b.row(r_base + 0) = params.end_waypoint;
    b.row(r_base + 1) = params.end_vel;
    b.row(r_base + 2) = params.end_acc;

    M_full.resize(total_vars, total_vars);
    M_full.setFromTriplets(triplets.begin(), triplets.end());
}

/**
 * @brief 使用带状LU分解高效求解线性系统 Mc=b。
 * @param M_banded 紧凑格式存储的带状矩阵M。
 * @param b        右侧向量b。
 * @param p        M的下带宽。
 * @param q        M的上带宽。
 * @return         方程的解x (即系数矩阵c)。
 */
Eigen::MatrixXd MincoOptimizer::solveBandedSystem(Eigen::MatrixXd M_banded,
                                                  const Eigen::MatrixXd& b,
                                                  int p, int q)
{
    int N = b.rows();
    if (N == 0) return Eigen::MatrixXd(0, b.cols());

    // In-place LU factorization (修正了循环边界 k < N - 1)
    for (int k = 0; k < N - 1; ++k) {
        double pivot = M_banded(q, k);
        if (std::abs(pivot) < 1e-12) {
            // 矩阵接近奇异，通常由不佳的时间分配导致 (T_i -> 0)
            // 返回NaN以向优化器发出失败信号
            return Eigen::MatrixXd::Constant(b.rows(), b.cols(), std::numeric_limits<double>::quiet_NaN());
        }

        for (int i = k + 1; i < std::min(k + p + 1, N); ++i) {
            M_banded(i - k + q, k) /= pivot;
        }

        for (int j = k + 1; j < std::min(k + q + 1, N); ++j) {
            double val = M_banded(k - j + q, j);
            if (val != 0.0) {
                for (int i = k + 1; i < std::min(k + p + 1, N); ++i) {
                    M_banded(i - j + q, j) -= M_banded(i - k + q, k) * val;
                }
            }
        }
    }

    // 前向替换: 解 Ly = b
    Eigen::MatrixXd y = b;
    for (int k = 0; k < N; ++k) {
        for (int i = k + 1; i < std::min(k + p + 1, N); ++i) {
            y.row(i) -= M_banded(i - k + q, k) * y.row(k);
        }
    }

    // 反向替换: 解 Ux = y
    Eigen::MatrixXd x = y;
    for (int k = N - 1; k >= 0; --k) {
        x.row(k) /= M_banded(q, k);
        for (int i = std::max(0, k - q); i < k; ++i) {
            x.row(i) -= M_banded(i - k + q, k) * x.row(k);
        }
    }

    return x;
}

/**
 * @brief 计算平滑度能量energy 及其梯度(grad_c, grad_T)。
 * @note 此函数为s=3(minimum jerk)的高效硬编码实现，与MATLAB和minco.hpp版本对齐。
 */
void MincoOptimizer::calculateEnergyAndGradient(const Eigen::MatrixXd& coeffs,
                                                const Eigen::VectorXd& T,
                                                const MincoParameters& params,
                                                double& energy,
                                                Eigen::MatrixXd& grad_c,
                                                Eigen::VectorXd& grad_T)
{
    // --- 初始化 ---
    energy = 0.0;
    grad_c.setZero(coeffs.rows(), coeffs.cols());
    grad_T.setZero(T.size());

    const int n_coeffs = params.n_coeffs;
    const int s = params.s;

    // --- 遍历每一段轨迹 ---
    for (int i = 0; i < params.n_segments; ++i) {
        // --- 提取与能量相关的系数 (c3, c4, c5) ---
        // 对应 t^3, t^4, t^5 的系数向量 (c_i(3), c_i(4), c_i(5))
        const auto& c3 = coeffs.row(i * n_coeffs + s);
        const auto& c4 = coeffs.row(i * n_coeffs + s + 1);
        const auto& c5 = coeffs.row(i * n_coeffs + s + 2);

        // --- 预计算T的各次幂 ---
        double t1 = T(i);
        double t2 = t1 * t1;
        double t3 = t2 * t1;
        double t4 = t3 * t1;
        double t5 = t4 * t1;

        // ===================================================================
        // 1. 计算能量 (Jc) - 对应 minco.hpp -> getEnergy
        // ===================================================================
        energy += 36.0 * c3.squaredNorm() * t1 +
                  144.0 * c3.dot(c4) * t2 +
                  192.0 * c4.squaredNorm() * t3 +
                  240.0 * c3.dot(c5) * t3 +
                  720.0 * c4.dot(c5) * t4 +
                  720.0 * c5.squaredNorm() * t5;

        // ===================================================================
        // 2. 计算能量对系数的梯度 (∂Jc/∂c) - 对应 minco.hpp -> getEnergyPartialGradByCoeffs
        // ===================================================================
        // 对 c3 的梯度
        grad_c.row(i * n_coeffs + s) = 72.0 * c3 * t1 + 144.0 * c4 * t2 + 240.0 * c5 * t3;
        // 对 c4 的梯度
        grad_c.row(i * n_coeffs + s + 1) = 144.0 * c3 * t2 + 384.0 * c4 * t3 + 720.0 * c5 * t4;
        // 对 c5 的梯度
        grad_c.row(i * n_coeffs + s + 2) = 240.0 * c3 * t3 + 720.0 * c4 * t4 + 1440.0 * c5 * t5;
        
        // ===================================================================
        // 3. 计算能量对时间的梯度 (∂Jc/∂T) - 对应 minco.hpp -> getEnergyPartialGradByTimes
        // ===================================================================
        grad_T(i) = 36.0 * c3.squaredNorm() +
                    288.0 * c3.dot(c4) * t1 +
                    576.0 * c4.squaredNorm() * t2 +
                    720.0 * c3.dot(c5) * t2 +
                    2880.0 * c4.dot(c5) * t3 +
                    3600.0 * c5.squaredNorm() * t4;
    }
}

/**
 * @brief 计算惩罚项(动力学 避障) 和 梯度（grad_c 和 grad_T）
          目前没有考虑距离惩罚，仅考虑可行性，后续需要实现别的约束，可通过梯度推导引入
*/
void MincoOptimizer::calculatePenaltyAndGradient(const Eigen::MatrixXd& coeffs, const Eigen::VectorXd& T, const MincoParameters& params, double& penalty, Eigen::MatrixXd& grad_c, Eigen::VectorXd& grad_T) {
    penalty = 0.0;
    grad_c.setZero(coeffs.rows(), coeffs.cols());
    grad_T.setZero(T.size());
    
    // PART 2: FEASIBILITY AND OBSTACLE COST
    for (int i = 0; i < params.n_segments; ++i) {
        Eigen::MatrixXd c_i = coeffs.block(i * params.n_coeffs, 0, params.n_coeffs, params.dims);
        double h = T(i) / params.kappa;
        
        for (int j = 0; j <= params.kappa; ++j) {
            double t = (double)j / params.kappa * T(i);
            Eigen::VectorXd B0, B1, B2, B3;
            getPolyBases(t, params.n_order, B0, B1, B2, B3);

            Eigen::RowVector2d pos = (B0.transpose() * c_i);
            Eigen::RowVector2d vel = (B1.transpose() * c_i);
            Eigen::RowVector2d acc = (B2.transpose() * c_i);
            Eigen::RowVector2d jer = (B3.transpose() * c_i);
            
            double omg = (j == 0 || j == params.kappa) ? 0.5 : 1.0;
            
            // Obstacle Penalty
            for (const auto& obs : params.obstacles) {
                Eigen::RowVector2d obs_center = obs.head<2>().transpose();
                double safe_dist_sq = std::pow(obs.z() + params.drone_radius, 2);
                Eigen::RowVector2d dist_vec = pos - obs_center;
                double dist_sq = dist_vec.squaredNorm();
                double coll_vio = safe_dist_sq - dist_sq;

                if (coll_vio > 0) {
                    double cost_obs = params.w_obstacle * std::pow(coll_vio, 3);
                    Eigen::RowVector2d grad_p = params.w_obstacle * (-6) * std::pow(coll_vio, 2) * dist_vec;
                    
                    penalty += omg * h * cost_obs;
                    grad_c.block(i * params.n_coeffs, 0, params.n_coeffs, params.dims) += omg * h * (B0 * grad_p);
                    
                    double alpha = (double)j / params.kappa;
                    double grad_T_obs_direct = alpha * grad_p * vel.transpose();
                    grad_T(i) += omg * (cost_obs / params.kappa + h * grad_T_obs_direct);
                }
            }

            // Velocity Penalty
            double v_pen = vel.squaredNorm() - params.v_max * params.v_max;
            if (v_pen > 0) {
                double cost_v = params.w_feasibility * std::pow(v_pen, 3);
                Eigen::RowVector2d grad_v = params.w_feasibility * 6 * std::pow(v_pen, 2) * vel;
                
                penalty += omg * h * cost_v;
                grad_c.block(i * params.n_coeffs, 0, params.n_coeffs, params.dims) += omg * h * (B1 * grad_v);

                double alpha = (double)j / params.kappa;
                double grad_T_v_direct = alpha * grad_v * acc.transpose();
                grad_T(i) += omg * (cost_v / params.kappa + h * grad_T_v_direct);
            }

            // Acceleration Penalty
            double a_pen = acc.squaredNorm() - params.a_max * params.a_max;
            if (a_pen > 0) {
                double cost_a = params.w_feasibility * std::pow(a_pen, 3);
                Eigen::RowVector2d grad_a = params.w_feasibility * 6 * std::pow(a_pen, 2) * acc;

                penalty += omg * h * cost_a;
                grad_c.block(i * params.n_coeffs, 0, params.n_coeffs, params.dims) += omg * h * (B2 * grad_a);
                
                double alpha = (double)j / params.kappa;
                double grad_T_a_direct = alpha * grad_a * jer.transpose();
                grad_T(i) += omg * (cost_a / params.kappa + h * grad_T_a_direct);
            }
        }
    }
}


/**
 * @brief 实现MINCO梯度反向传播，将成本对c和T的梯度传播回q和T。
 * @param T                  各段时长向量。
 * @param total_grad_c       成本K对系数c的梯度 (∂K/∂c)。
 * @param total_grad_T_direct 成本K对时长T的直接梯度 (∂K/∂T)。
 * @param params             MINCO问题参数。
 * @param M_full             完整稀疏矩阵M。
 * @param coeffs             当前的多项式系数c。
 * @param grad_q             (输出) 成本W对中间航点q的梯度 (∂W/∂q)。
 * @param grad_T_total       (输出) 成本W对时长T的总梯度 (∂W/∂T)。
 */
void MincoOptimizer::backpropagateMincoGradient(const Eigen::VectorXd& T,
                                                const Eigen::MatrixXd& total_grad_c,
                                                const Eigen::VectorXd& total_grad_T_direct,
                                                const MincoParameters& params,
                                                const Eigen::SparseMatrix<double>& M_full,
                                                const Eigen::MatrixXd& coeffs,
                                                Eigen::MatrixXd& grad_q,
                                                Eigen::VectorXd& grad_T_total)
{
    int n_segments = params.n_segments;
    int s = params.s;
    int n_coeffs = params.n_coeffs;
    int dims = params.dims;

    // --- 步骤 1: 求解伴随系统 G = M^{-T} * total_grad_c ---
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(M_full.transpose());
    if (solver.info() != Eigen::Success) {
        grad_q.setZero(n_segments > 1 ? n_segments - 1 : 0, dims);
        grad_T_total = total_grad_T_direct; // 求解失败时，仅返回直接梯度
        return;
    }
    Eigen::MatrixXd G = solver.solve(total_grad_c);

    // --- 步骤 2: 传播梯度到 q (∂W/∂q) ---
    // 根据 buildMincoMatrix 的布局，航点约束行的索引是 s + i*(2s) + (s-1)
    grad_q.resize(n_segments > 1 ? n_segments - 1 : 0, dims);
    for (int i = 0; i < n_segments - 1; ++i) {
        int r_waypoint = s + i * (2 * s) + (s - 1);
        grad_q.row(i) = G.row(r_waypoint);
    }

    // --- 步骤 3: 传播梯度到 T (计算间接梯度 ∂W/∂T_indirect) ---
    Eigen::VectorXd grad_T_indirect = Eigen::VectorXd::Zero(n_segments);

    // 中间段的梯度
    for (int i = 0; i < n_segments - 1; ++i) {
        double t_i = T(i);
        // E_i * c_i 块对 T_i 的导数, 这是一个 (2s) x dims 的矩阵
        Eigen::MatrixXd dEi_dc = Eigen::MatrixXd::Zero(2 * s, dims);
        const auto& c_i = coeffs.block(i * n_coeffs, 0, n_coeffs, dims);
        
        // 高阶连续性部分 (s-1个约束)
        for (int k = s; k < 2 * s - 1; ++k) {
            dEi_dc.row(k - s) = getPolyBasis(t_i, params.n_order, k + 1).transpose() * c_i;
        }
        // 航点位置部分 (1个约束)
        dEi_dc.row(s - 1) = getPolyBasis(t_i, params.n_order, 1).transpose() * c_i;
        // 低阶连续性部分 (s个约束)
        for (int k = 0; k < s; ++k) {
            dEi_dc.row(s + k) = getPolyBasis(t_i, params.n_order, k + 1).transpose() * c_i;
        }

        // 提取G中对应的块并计算内积
        int r_block_start = s + i * (2 * s);
        grad_T_indirect(i) = (G.block(r_block_start, 0, 2 * s, dims).array() * dEi_dc.array()).sum();
    }

    // 最后一段的梯度
    if (n_segments > 0) {
        double t_M = T(n_segments - 1);
        // E_M * c_M 块对 T_M 的导数, 这是一个 s x dims 的矩阵
        Eigen::MatrixXd dEM_dc = Eigen::MatrixXd::Zero(s, dims);
        const auto& c_M = coeffs.block((n_segments - 1) * n_coeffs, 0, n_coeffs, dims);

        for (int k = 0; k < s; ++k) {
            dEM_dc.row(k) = getPolyBasis(t_M, params.n_order, k + 1).transpose() * c_M;
        }
        
        grad_T_indirect(n_segments - 1) = (G.bottomRows(s).array() * dEM_dc.array()).sum();
    }
    
    // --- 步骤 4: 组合得到最终的时间梯度 ---
    grad_T_total = total_grad_T_direct - grad_T_indirect;
}

} // namespace vmc_minco_nav