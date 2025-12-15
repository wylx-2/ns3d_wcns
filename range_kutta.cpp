#include "field_structures.h"
#include "ns3d_func.h"
#include <cmath>
#include <mpi.h>

//==================================================================
// 三阶 Runge-Kutta 时间推进主循环模块
//==================================================================

// RHS 计算占位符函数（用户需在此定义具体的通量差分或高阶算子）
void compute_rhs(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs)
{
    LocalDesc &L = F.L;
    const double idx = 1.0 / G.dx;
    const double idy = 1.0 / G.dy;
    const double idz = 1.0 / G.dz;

    // 清空 RHS
    std::fill(F.rhs_rho.begin(), F.rhs_rho.end(), 0.0);
    std::fill(F.rhs_rhou.begin(), F.rhs_rhou.end(), 0.0);
    std::fill(F.rhs_rhov.begin(), F.rhs_rhov.end(), 0.0);
    std::fill(F.rhs_rhow.begin(), F.rhs_rhow.end(), 0.0);
    std::fill(F.rhs_E.begin(), F.rhs_E.end(), 0.0);

    // FVS计算无粘通量
    compute_invis_flux(F, P);
    // 交换半节点的通量
    apply_boundary_halfnode_flux(F, G, C, P); //还需要额外边界处理！
    // The cell-centered compact schemes
    compute_invis_dflux(F, P, G);

    // 计算空间导数
    compute_gradients(F, G);
    // 同步不同进程的ghost区域空间导数
    // exchange_halos_gradients(F, C, L, out_reqs); //还需要额外边界处理！
    // 计算粘性通量
    compute_viscous_flux(F, P);
    // 应该在这里交换粘性通量的halo区域，并处理周期边界
    exchange_halos_viscous_flux(F, C, L, out_reqs); //还需要额外边界处理！

    // 粘性通量的导数，注意这个函数直接在RHS上累加
    compute_vis_flux(F, G);
}

// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs, double dt)
{
    LocalDesc &L = F.L;
    const int N = F.rho.size();

    // Stage 1
    compute_rhs(F, C, G, P, out_reqs);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = F.rho0[n] + dt * F.rhs_rho[n];
        F.rhou[n] = F.rhou0[n] + dt * F.rhs_rhou[n];
        F.rhov[n] = F.rhov0[n] + dt * F.rhs_rhov[n];
        F.rhow[n] = F.rhow0[n] + dt * F.rhs_rhow[n];
        F.E[n] = F.E0[n] + dt * F.rhs_E[n];
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);

    // Stage 2
    compute_rhs(F, C, G, P, out_reqs);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = 0.75 * F.rho0[n] + 0.25 * (F.rho[n] + dt * F.rhs_rho[n]);
        F.rhou[n] = 0.75 * F.rhou0[n] + 0.25 * (F.rhou[n] + dt * F.rhs_rhou[n]);
        F.rhov[n] = 0.75 * F.rhov0[n] + 0.25 * (F.rhov[n] + dt * F.rhs_rhov[n]);
        F.rhow[n] = 0.75 * F.rhow0[n] + 0.25 * (F.rhow[n] + dt * F.rhs_rhow[n]);
        F.E[n] = 0.75 * F.E0[n] + 0.25 * (F.E[n] + dt * F.rhs_E[n]);
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);

    // Stage 3
    compute_rhs(F, C, G, P, out_reqs);
    for (int n = 0; n < N; ++n)
    {
        F.rho[n] = (1.0 / 3.0) * F.rho0[n] + (2.0 / 3.0) * (F.rho[n] + dt * F.rhs_rho[n]);
        F.rhou[n] = (1.0 / 3.0) * F.rhou0[n] + (2.0 / 3.0) * (F.rhou[n] + dt * F.rhs_rhou[n]);
        F.rhov[n] = (1.0 / 3.0) * F.rhov0[n] + (2.0 / 3.0) * (F.rhov[n] + dt * F.rhs_rhov[n]);
        F.rhow[n] = (1.0 / 3.0) * F.rhow0[n] + (2.0 / 3.0) * (F.rhow[n] + dt * F.rhs_rhow[n]);
        F.E[n] = (1.0 / 3.0) * F.E0[n] + (2.0 / 3.0) * (F.E[n] + dt * F.rhs_E[n]);
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);
}
