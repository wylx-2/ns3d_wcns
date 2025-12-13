#include "field_structures.h"
#include "ns3d_func.h"

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

    MPI_Allreduce(&count, &g_N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // input diagnostics
    // residual = (g_sum_abs_res_rho / g_N) / g_max_abs_rho;
    F.global_res_rho = std::sqrt( (g_sum_sq_res_rho ) ) / g_max_abs_rho;
    F.global_res_rhou = std::sqrt( (g_sum_sq_res_rhou) ) / g_max_abs_rhou;
    F.global_res_rhov = std::sqrt( (g_sum_sq_res_rhov) ) / g_max_abs_rhov;
    F.global_res_rhow = std::sqrt( (g_sum_sq_res_rhow) ) / g_max_abs_rhow;
    F.global_res_E = std::sqrt( (g_sum_sq_res_E) ) / g_max_abs_E;

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