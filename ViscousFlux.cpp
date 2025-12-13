#include "ns3d_func.h"
#include "field_structures.h"
#include <mpi.h>
#include <algorithm>
#include <iostream>

// -----------------------------------------------------------------
// ---------   计算粘性通量模块 -------------------------------------
// -----------------------------------------------------------------   

// 计算空间导数
// ==============================
// 通用差分模板
// ==============================
inline double diff_2nd_forward(const std::vector<double> &f, int i, double dx, int flag) {
    return (-3.0*f[i] + 4.0*f[i+flag] - f[i+2*flag]) / (2.0*dx);
}

inline double diff_2nd_central(const std::vector<double> &f, int i, double dx) {
    return (f[i+1] - f[i-1]) / (2.0*dx);
}

inline double diff_4th_central(const std::vector<double> &f, int i, double dx) {
    return (-f[i+2] + 8.0*f[i+1] - 8.0*f[i-1] + f[i-2]) / (12.0*dx);
}

inline double diff_6th_central(const std::vector<double> &f, int i, double dx) {
    return (f[i+3] - 9.0*f[i+2] + 45.0*f[i+1] - 45.0*f[i-1] + 9.0*f[i-2] - f[i-3]) / (60.0*dx);
}


// C4th/C6th 改这个函数
inline int choose_scheme(int idx, int nstart, int nend, bool periodic)
{
    if (periodic)
        return 6; // always use 6th order for periodic
    int dist_left = idx - nstart;
    if (dist_left == 0)
        return 1; // forward 2nd order for left boundary (flag == +1)
    int dist_right = (nend - 1) - idx;
    if (dist_right == 0)
        return -1; // backward 2nd order for right boundary (flag == -1)
    int dmin = std::min(dist_left, dist_right);
    if (dmin == 1)
        return 2; // 2nd order central for sub-boundary points
    else if (dmin == 2)
        return 4; // 4th order central for sub-sub-boundary points
    else
        return 6; // 6th order central for inner points
}

inline double diff_x(const std::vector<double> &f, int i, int j, int k, double dx, int order_x, const LocalDesc &L)
{
    std::vector<double> dummy(7);
    for (int ii = 0; ii < 7; ++ii) 
    {
        dummy[ii] = f[idx3(i + ii - 3, j, k, L)];
    }

    if (order_x == 1 || order_x == -1)
        return diff_2nd_forward(dummy, 3, dx, order_x);
    else if (order_x == 2)
        return diff_2nd_central(dummy, 3, dx);
    else if (order_x == 4)
        return diff_4th_central(dummy, 3, dx);
    else
        return diff_6th_central(dummy, 3, dx);
}

inline double diff_y(const std::vector<double> &f, int i, int j, int k, double dy, int order_y, const LocalDesc &L)
{
    std::vector<double> dummy(7);
    for (int jj = 0; jj < 7; ++jj)
    {
        dummy[jj] = f[idx3(i, j + jj - 3, k, L)];
    }

    if (order_y == 1 || order_y == -1)
        return diff_2nd_forward(dummy, 3, dy, order_y);
    else if (order_y == 2)
        return diff_2nd_central(dummy, 3, dy);
    else if (order_y == 4)
        return diff_4th_central(dummy, 3, dy);
    else
        return diff_6th_central(dummy, 3, dy);
}

inline double diff_z(const std::vector<double> &f, int i, int j, int k, double dz, int order_z, const LocalDesc &L)
{
    std::vector<double> dummy(7);
    for (int kk = 0; kk < 7; ++kk)
    {
        dummy[kk] = f[idx3(i, j, k + kk - 3, L)];
    }

    if (order_z == 1 || order_z == -1)
        return diff_2nd_forward(dummy, 3, dz, order_z);
    else if (order_z == 2)
        return diff_2nd_central(dummy, 3, dz);
    else if (order_z == 4)
        return diff_4th_central(dummy, 3, dz);
    else
        return diff_6th_central(dummy, 3, dz);
}
// ==================================================
// 主函数：自适应阶数梯度计算, 根据边界距离选择差分格式
// 计算内点的6阶中心差分，边界处根据距离选择低阶格式
// ==================================================
void compute_gradients(Field3D &F, const GridDesc &G)
{
    const LocalDesc &L = F.L;
    const double dx = G.dx, dy = G.dy, dz = G.dz;
    int nx=L.nx, ny=L.ny, nz=L.nz;
    int ngx=L.ngx, ngy=L.ngy, ngz=L.ngz;

    // 周期判断
    bool periodic_x = (L.nbr_xm != MPI_PROC_NULL && L.nbr_xp != MPI_PROC_NULL);
    bool periodic_y = (L.nbr_ym != MPI_PROC_NULL && L.nbr_yp != MPI_PROC_NULL);
    bool periodic_z = (L.nbr_zm != MPI_PROC_NULL && L.nbr_zp != MPI_PROC_NULL);

    for (int k = ngz; k < ngz+nz; ++k)
    for (int j = ngy; j < ngy+ny; ++j)
    for (int i = ngx; i < ngx+nx; ++i)
    {
        int id = F.I(i,j,k);

        int order_x = choose_scheme(i, ngx, ngx+nx, periodic_x);
        int order_y = choose_scheme(j, ngy, ngy+ny, periodic_y);
        int order_z = choose_scheme(k, ngz, ngz+nz, periodic_z);

        // Compute gradients
        F.du_dx[id] = diff_x(F.u, i, j, k, dx, order_x, L);
        F.du_dy[id] = diff_y(F.u, i, j, k, dy, order_y, L);
        F.du_dz[id] = diff_z(F.u, i, j, k, dz, order_z, L);

        F.dv_dx[id] = diff_x(F.v, i, j, k, dx, order_x, L);
        F.dv_dy[id] = diff_y(F.v, i, j, k, dy, order_y, L);
        F.dv_dz[id] = diff_z(F.v, i, j, k, dz, order_z, L);

        F.dw_dx[id] = diff_x(F.w, i, j, k, dx, order_x, L);
        F.dw_dy[id] = diff_y(F.w, i, j, k, dy, order_y, L);
        F.dw_dz[id] = diff_z(F.w, i, j, k, dz, order_z, L);

        F.dT_dx[id] = diff_x(F.T, i, j, k, dx, order_x, L);
        F.dT_dy[id] = diff_y(F.T, i, j, k, dy, order_y, L);
        F.dT_dz[id] = diff_z(F.T, i, j, k, dz, order_z, L);
    }
}

// 计算内点粘性通量
void compute_viscous_flux(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double gamma = P.gamma;
    const double Rgas = P.Rgas;
    const double Pr = P.Pr;
    const double Cp = P.Cp;
    const int nx = L.nx, ny = L.ny, nz = L.nz;
    const int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;

    for (int k = ngz; k < ngz+nz; ++k)
    for (int j = ngy; j < ngy+ny; ++j)
    for (int i = ngx; i < ngx+nx; ++i)
    {
        int id = F.I(i, j, k);
        double mu = P.get_mu(F.T[id]);

        // 速度散度
        double divU = F.du_dx[id] + F.dv_dy[id] + F.dw_dz[id];

        // 应力分量
        double tau_xx = 2.0 * mu * F.du_dx[id] - (2.0/3.0) * mu * divU;
        double tau_yy = 2.0 * mu * F.dv_dy[id] - (2.0/3.0) * mu * divU;
        double tau_zz = 2.0 * mu * F.dw_dz[id] - (2.0/3.0) * mu * divU;

        double tau_xy = mu * (F.du_dy[id] + F.dv_dx[id]);
        double tau_xz = mu * (F.du_dz[id] + F.dw_dx[id]);
        double tau_yz = mu * (F.dv_dz[id] + F.dw_dy[id]);

        // 热通量
        double qx = -mu * Cp / Pr * F.dT_dx[id];
        double qy = -mu * Cp / Pr * F.dT_dy[id];
        double qz = -mu * Cp / Pr * F.dT_dz[id];

        double u = F.u[id], v = F.v[id], w = F.w[id];

        // X方向粘性通量
        F.Fvflux_mass[id] = 0.0;
        F.Fvflux_momx[id] = tau_xx;
        F.Fvflux_momy[id] = tau_xy;
        F.Fvflux_momz[id] = tau_xz;
        F.Fvflux_E[id]    = u*tau_xx + v*tau_xy + w*tau_xz + qx;

        // Y方向粘性通量
        F.Hvflux_mass[id] = 0.0;
        F.Hvflux_momx[id] = tau_xy;
        F.Hvflux_momy[id] = tau_yy;
        F.Hvflux_momz[id] = tau_yz;
        F.Hvflux_E[id]    = u*tau_xy + v*tau_yy + w*tau_yz + qy;

        // Z方向粘性通量
        F.Gvflux_mass[id] = 0.0;
        F.Gvflux_momx[id] = tau_xz;
        F.Gvflux_momy[id] = tau_yz;
        F.Gvflux_momz[id] = tau_zz;
        F.Gvflux_E[id]    = u*tau_xz + v*tau_yz + w*tau_zz + qz;
    }
}

void compute_vis_flux(Field3D &F, const GridDesc &G)
{
    const LocalDesc &L = F.L;
    const double dx = G.dx, dy = G.dy, dz = G.dz;
    int nx=L.nx, ny=L.ny, nz=L.nz;
    int ngx=L.ngx, ngy=L.ngy, ngz=L.ngz;

    bool periodic_x = (L.nbr_xm != MPI_PROC_NULL && L.nbr_xp != MPI_PROC_NULL);
    bool periodic_y = (L.nbr_ym != MPI_PROC_NULL && L.nbr_yp != MPI_PROC_NULL);
    bool periodic_z = (L.nbr_zm != MPI_PROC_NULL && L.nbr_zp != MPI_PROC_NULL);

    for (int k = ngz; k < ngz+nz; ++k)
    for (int j = ngy; j < ngy+ny; ++j)
    for (int i = ngx; i < ngx+nx; ++i)
    {
        int id = F.I(i,j,k);

        int order_x = choose_scheme(i, ngx, ngx+nx, periodic_x);
        int order_y = choose_scheme(j, ngy, ngy+ny, periodic_y);
        int order_z = choose_scheme(k, ngz, ngz+nz, periodic_z);

        // Compute viscous flux gradients
        // rho 通量为0，不计算
        /*
        F.rhs_rho[id] += diff_x(F.Fvflux_mass, i, j, k, dx, order_x, L);
        F.rhs_rho[id] += diff_y(F.Hvflux_mass, i, j, k, dy, order_y, L);
        F.rhs_rho[id] += diff_z(F.Gvflux_mass, i, j, k, dz, order_z, L);
        */

        F.rhs_rhou[id] += diff_x(F.Fvflux_momx, i, j, k, dx, order_x, L);
        F.rhs_rhou[id] += diff_y(F.Hvflux_momx, i, j, k, dy, order_y, L);
        F.rhs_rhou[id] += diff_z(F.Gvflux_momx, i, j, k, dz, order_z, L);

        F.rhs_rhov[id] += diff_x(F.Fvflux_momy, i, j, k, dx, order_x, L);
        F.rhs_rhov[id] += diff_y(F.Hvflux_momy, i, j, k, dy, order_y, L);
        F.rhs_rhov[id] += diff_z(F.Gvflux_momy, i, j, k, dz, order_z, L);

        F.rhs_rhow[id] += diff_x(F.Fvflux_momz, i, j, k, dx, order_x, L);
        F.rhs_rhow[id] += diff_y(F.Hvflux_momz, i, j, k, dy, order_y, L);
        F.rhs_rhow[id] += diff_z(F.Gvflux_momz, i, j, k, dz, order_z, L);

        F.rhs_E[id] += diff_x(F.Fvflux_E, i, j, k, dx, order_x, L);
        F.rhs_E[id] += diff_y(F.Hvflux_E, i, j, k, dy, order_y, L);
        F.rhs_E[id] += diff_z(F.Gvflux_E, i, j, k, dz, order_z, L);
    }
}

// function to compute only du/dx for isotropic turbulence analysis
void compute_gradients_dudx(Field3D &F, const GridDesc &G)
{
    const LocalDesc &L = F.L;
    const double dx = G.dx, dy = G.dy, dz = G.dz;
    int nx=L.nx, ny=L.ny, nz=L.nz;
    int ngx=L.ngx, ngy=L.ngy, ngz=L.ngz;

    // 周期判断
    bool periodic_x = (L.nbr_xm != MPI_PROC_NULL && L.nbr_xp != MPI_PROC_NULL);

    for (int k = ngz; k < ngz+nz; ++k)
    for (int j = ngy; j < ngy+ny; ++j)
    for (int i = ngx; i < ngx+nx; ++i)
    {
        int id = F.I(i,j,k);
        int order_x = choose_scheme(i, ngx, ngx+nx, periodic_x);
        
        // Compute gradients
        F.du_dx[id] = diff_x(F.u, i, j, k, dx, order_x, L);
    }
    
}