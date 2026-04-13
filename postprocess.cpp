#include "field_structures.h"
#include "ns3d_func.h"
#include <cmath>
#include <iostream>

// -----------------------------------------------------------------------------
// 中间诊断函数，计算总能量、残差、RMS
// -----------------------------------------------------------------------------
void compute_diagnostics(Field3D &F, const SolverParams &P, const GridDesc &G)
{
    const LocalDesc &L = F.L;
    double dx = G.dx;
    double dx3 = dx * dx * dx;
	// per-variable accumulators for true residual calculation
    // double sum_abs_res_rho = 0.0, sum_abs_res_rhou = 0.0, sum_abs_res_rhov = 0.0, sum_abs_res_rhow = 0.0, sum_abs_res_E = 0.0;
    double sum_sq_res_rho = 0.0, sum_sq_res_rhou = 0.0, sum_sq_res_rhov = 0.0, sum_sq_res_rhow = 0.0, sum_sq_res_E = 0.0;
	double max_abs_rho = 0.0, max_abs_rhou = 0.0, max_abs_rhov = 0.0, max_abs_rhow = 0.0, max_abs_E = 0.0;
    double max_abs_u = 0.0, max_abs_v = 0.0, max_abs_w = 0.0;
    double max_rho = -1e300, min_rho = 1e300;
    double max_p = -1e300, min_p = 1e300;
    int count = 0;

    F.updateResiduals();

    for (int k = L.ngz; k < L.ngz + L.nz; ++k)
    for (int j = L.ngy; j < L.ngy + L.ny; ++j)
    for (int i = L.ngx; i < L.ngx + L.nx; ++i)
    {
        int id = F.I(i,j,k);
        double rho = F.rho[id];
        double u = F.u[id], v = F.v[id], w = F.w[id];
        double E = F.E[id];
        double p = F.p[id];
        double rhou = F.rhou[id];
        double rhov = F.rhov[id];
        double rhow = F.rhow[id];

        max_abs_u = std::max(max_abs_u, std::abs(u));
        max_abs_v = std::max(max_abs_v, std::abs(v));
        max_abs_w = std::max(max_abs_w, std::abs(w));
        max_rho = std::max(max_rho, rho);
        min_rho = std::min(min_rho, rho);
        max_p = std::max(max_p, p);
        min_p = std::min(min_p, p);

        // max absolute values for normalization
        max_abs_rho = std::max(max_abs_rho, std::abs(rho));
        max_abs_rhou = std::max(max_abs_rhou, std::abs(rhou));
        max_abs_rhov = std::max(max_abs_rhov, std::abs(rhov));
        max_abs_rhow = std::max(max_abs_rhow, std::abs(rhow));
        max_abs_E = std::max(max_abs_E, std::abs(E));

		// residual
    
        double res_rho = F.res_rho[id];;
        double res_rhou = F.res_rhou[id];
        double res_rhov = F.res_rhov[id];
        double res_rhow = F.res_rhow[id];
        double res_E = F.res_E[id];

        /*
        sum_abs_res_rho += std::abs(res_rho);
        sum_abs_res_rhou += std::abs(res_rhou);
        sum_abs_res_rhov += std::abs(res_rhov);
        sum_abs_res_rhow += std::abs(res_rhow);
        sum_abs_res_E += std::abs(res_E);
        */

		sum_sq_res_rho += res_rho * res_rho * dx3;
        sum_sq_res_rhou += res_rhou * res_rhou * dx3;
        sum_sq_res_rhov += res_rhov * res_rhov * dx3;
        sum_sq_res_rhow += res_rhow * res_rhow * dx3;
        sum_sq_res_E += res_E * res_E * dx3;
        ++count;
    }

	// global reductions
    // double g_sum_abs_res_rho = 0.0, g_sum_abs_res_rhou = 0.0, g_sum_abs_res_rhov = 0.0, g_sum_abs_res_rhow = 0.0, g_sum_abs_res_E = 0.0;
    double g_sum_sq_res_rho = 0.0, g_sum_sq_res_rhou = 0.0, g_sum_sq_res_rhov = 0.0, g_sum_sq_res_rhow = 0.0, g_sum_sq_res_E = 0.0;
    double g_max_abs_rho = 0.0, g_max_abs_rhou = 0.0, g_max_abs_rhov = 0.0, g_max_abs_rhow = 0.0, g_max_abs_E = 0.0;
    double g_max_abs_u = 0.0, g_max_abs_v = 0.0, g_max_abs_w = 0.0;
    double g_max_rho = 0.0, g_min_rho = 0.0;
    double g_max_p = 0.0, g_min_p = 0.0;
    int g_N = 0;

    // MPI_Allreduce(&sum_abs_res_rho, &g_sum_abs_res_rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&sum_abs_res_rhou, &g_sum_abs_res_rhou, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&sum_abs_res_rhov, &g_sum_abs_res_rhov, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&sum_abs_res_rhow, &g_sum_abs_res_rhow, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(&sum_abs_res_E, &g_sum_abs_res_E, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&sum_sq_res_rho, &g_sum_sq_res_rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_res_rhou, &g_sum_sq_res_rhou, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_res_rhov, &g_sum_sq_res_rhov, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_res_rhow, &g_sum_sq_res_rhow, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&sum_sq_res_E, &g_sum_sq_res_E, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(&max_abs_rho, &g_max_abs_rho, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_rhou, &g_max_abs_rhou, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_rhov, &g_max_abs_rhov, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_rhow, &g_max_abs_rhow, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_E, &g_max_abs_E, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_u, &g_max_abs_u, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_v, &g_max_abs_v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_abs_w, &g_max_abs_w, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&max_rho, &g_max_rho, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&min_rho, &g_min_rho, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&max_p, &g_max_p, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&min_p, &g_min_p, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    MPI_Allreduce(&count, &g_N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // input diagnostics
    // residual = (g_sum_abs_res_rho / g_N) / g_max_abs_rho;
    F.global_res_rho = std::sqrt( (g_sum_sq_res_rho ) ) / g_max_abs_rho;
    F.global_res_rhou = std::sqrt( (g_sum_sq_res_rhou) ) / g_max_abs_rhou;
    F.global_res_rhov = std::sqrt( (g_sum_sq_res_rhov) ) / g_max_abs_rhov;
    F.global_res_rhow = std::sqrt( (g_sum_sq_res_rhow) ) / g_max_abs_rhow;
    F.global_res_E = std::sqrt( (g_sum_sq_res_E) ) / g_max_abs_E;

    F.global_max_abs_u = g_max_abs_u;
    F.global_max_abs_v = g_max_abs_v;
    F.global_max_abs_w = g_max_abs_w;
    F.global_max_rho = g_max_rho;
    F.global_min_rho = g_min_rho;
    F.global_max_p = g_max_p;
    F.global_min_p = g_min_p;

}

// -----------------------------------------------------------------------------
// 计算总能量（全局积分）：动能 + 内能；结果存入 F.global_Etot
// -----------------------------------------------------------------------------
void compute_total_energy(Field3D &F, const GridDesc &G, const CartDecomp &C, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double cell_vol = G.dx * G.dy * G.dz;

    double local_sum = 0.0;
    for (int k = L.ngz; k < L.ngz + L.nz; ++k)
    for (int j = L.ngy; j < L.ngy + L.ny; ++j)
    for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
        int id = F.I(i,j,k);
        double rho = F.rho[id];
        double u = F.u[id];
        double v = F.v[id];
        double w = F.w[id];
        double p = F.p[id];

        double kinetic = 0.5 * rho * (u*u + v*v + w*w);
        // double eint = p / (P.gamma - 1.0);
        double eint = 0;
        local_sum += (kinetic + eint) * cell_vol;
    }

    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, C.cart_comm);
    F.global_Etot = global_sum;
}

// -----------------------------------------------------------------------------
// 监控流向速度: 在 x=0 与 x=pi 截面统计 u 的截面平均值（全局 MPI 汇总）
// -----------------------------------------------------------------------------
void monitor_mean_u_sections(Field3D &F,
                             const GridDesc &G,
                             const CartDecomp &C,
                             double current_time)
{
    const LocalDesc &L = F.L;
    if (G.global_nx <= 0 || G.dx <= 0.0) {
        if (C.rank == 0) {
            std::cout << "[Monitor U-section] invalid grid spacing in x; skip\n";
        }
        return;
    }

    const auto nearest_global_i = [&](double x_target) -> int {
        long long gi = static_cast<long long>(std::llround((x_target - G.x0) / G.dx));
        if (C.periods[0]) {
            const long long n = static_cast<long long>(G.global_nx);
            gi = ((gi % n) + n) % n;
            return static_cast<int>(gi);
        }
        if (gi < 0) gi = 0;
        if (gi > static_cast<long long>(G.global_nx - 1)) gi = static_cast<long long>(G.global_nx - 1);
        return static_cast<int>(gi);
    };

    const int gi_x0 = nearest_global_i(0.0);
    const int gi_xpi = nearest_global_i(M_PI);

    double local_sum_x0 = 0.0;
    double local_sum_xpi = 0.0;
    double local_cnt_x0 = 0.0;
    double local_cnt_xpi = 0.0;

    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int gi = L.ox + (i - L.ngx);
                const int id = F.I(i, j, k);
                if (gi == gi_x0) {
                    local_sum_x0 += F.u[id] * F.Ja[id]; // weight by local cell volume for better accuracy
                    local_cnt_x0 += F.Ja[id]; // count weighted by local cell volume for better accuracy
                }
                if (gi == gi_xpi) {
                    local_sum_xpi += F.u[id] * F.Ja[id]; // weight by local cell volume for better accuracy
                    local_cnt_xpi += F.Ja[id]; // count weighted by local cell volume for better accuracy
                }
            }
        }
    }

    double global_sum_x0 = 0.0;
    double global_sum_xpi = 0.0;
    double global_cnt_x0 = 0.0;
    double global_cnt_xpi = 0.0;

    MPI_Allreduce(&local_sum_x0, &global_sum_x0, 1, MPI_DOUBLE, MPI_SUM, C.cart_comm);
    MPI_Allreduce(&local_sum_xpi, &global_sum_xpi, 1, MPI_DOUBLE, MPI_SUM, C.cart_comm);
    MPI_Allreduce(&local_cnt_x0, &global_cnt_x0, 1, MPI_DOUBLE, MPI_SUM, C.cart_comm);
    MPI_Allreduce(&local_cnt_xpi, &global_cnt_xpi, 1, MPI_DOUBLE, MPI_SUM, C.cart_comm);

    const double mean_u_x0 = (global_cnt_x0 > 0) ? (global_sum_x0 / global_cnt_x0) : 0.0;
    const double mean_u_xpi = (global_cnt_xpi > 0) ? (global_sum_xpi / global_cnt_xpi) : 0.0;

    F.mean_u_x0 = mean_u_x0;
    F.mean_u_xpi = mean_u_xpi;
    if (C.rank == 0) {
        const double x0_actual = G.x0 + gi_x0 * G.dx;
        const double xpi_actual = G.x0 + gi_xpi * G.dx;
        std::cout << "[Monitor U-section] t=" << current_time
                  << "  <u>|x=0(" << x0_actual << ")=" << mean_u_x0
                  << "  <u>|x=pi(" << xpi_actual << ")=" << mean_u_xpi
                  << "\n";
    }
}

// -----------------------------------------------------------------------------
// 各向同性湍流后处理函数
// -----------------------------------------------------------------------------
void isotropic_post_process(Field3D &F, const GridDesc &G, const CartDecomp &C,const SolverParams &P, const double current_time)
{
    // 计算并输出能量谱
    std::stringstream ss;
    ss << "output/output_time_" << std::fixed << std::setprecision(5) << current_time;
    compute_energy_spectrum(F, G, C, ss.str() + "_spectrum.dat");


    compute_turbulence_statistics(F, G, P, C, current_time);
    // 计算并输出湍流统计量
    // Taylor 微尺度、雷诺数等
    // 这里可以添加更多的统计量计算和输出
}