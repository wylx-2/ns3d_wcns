#include "ns3d_func.h"
#include "field_structures.h"
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

namespace {
void write_viscous_xface_debug_tecplot(
    const LocalDesc &L,
    const CartDecomp &C,
    const std::vector<double> &u_fx,
    const std::vector<double> &v_fx,
    const std::vector<double> &w_fx,
    const std::vector<double> &du_dx_fx,
    const std::vector<double> &du_dy_fx,
    const std::vector<double> &du_dz_fx,
    const std::vector<double> &dv_dx_fx,
    const std::vector<double> &dv_dy_fx,
    const std::vector<double> &dv_dz_fx,
    const std::vector<double> &dw_dx_fx,
    const std::vector<double> &dw_dy_fx,
    const std::vector<double> &dw_dz_fx,
    const std::vector<double> &dT_dx_fx,
    const std::vector<double> &dT_dy_fx,
    const std::vector<double> &dT_dz_fx,
    const std::vector<double> &T_fx,
    const std::vector<double> &xi_x_fx,
    const std::vector<double> &xi_y_fx,
    const std::vector<double> &xi_z_fx)
{
    std::filesystem::create_directories("output");

    std::ostringstream oss;
    oss << "output/viscous_xface_debug_rank" << std::setw(4) << std::setfill('0') << C.rank << ".dat";
    std::ofstream ofs(oss.str(), std::ofstream::out);
    if (!ofs) {
        std::cerr << "Failed to open viscous x-face debug file: " << oss.str() << "\n";
        return;
    }

    ofs << "TITLE = \"Viscous Flux X-Face Debug\"\n";
    ofs << "VARIABLES = "
        << "\"i\" \"j\" \"k\" \"gi\" \"gj\" \"gk\" "
        << "\"u_fx\" \"v_fx\" \"w_fx\" "
        << "\"du_dx_fx\" \"du_dy_fx\" \"du_dz_fx\" "
        << "\"dv_dx_fx\" \"dv_dy_fx\" \"dv_dz_fx\" "
        << "\"dw_dx_fx\" \"dw_dy_fx\" \"dw_dz_fx\" "
        << "\"dT_dx_fx\" \"dT_dy_fx\" \"dT_dz_fx\" "
        << "\"T_fx\" "
        << "\"xi_x_fx\" \"xi_y_fx\" \"xi_z_fx\"\n";
    ofs << "ZONE T=\"rank_" << C.rank << "\", I=" << (L.sx - 1) << ", J=" << L.sy << ", K=" << L.sz
        << ", DATAPACKING=POINT\n";

    ofs << std::scientific << std::setprecision(12);
    for (int k = 0; k < L.sz; ++k) {
        for (int j = 0; j < L.sy; ++j) {
            for (int i = 0; i < L.sx - 1; ++i) {
                const int idf = idx_fx(i, j, k, L);
                const int gi = L.ox + (i - L.ngx);
                const int gj = L.oy + (j - L.ngy);
                const int gk = L.oz + (k - L.ngz);
                ofs << i << " " << j << " " << k << " "
                    << gi << " " << gj << " " << gk << " "
                    << u_fx[idf] << " " << v_fx[idf] << " " << w_fx[idf] << " "
                    << du_dx_fx[idf] << " " << du_dy_fx[idf] << " " << du_dz_fx[idf] << " "
                    << dv_dx_fx[idf] << " " << dv_dy_fx[idf] << " " << dv_dz_fx[idf] << " "
                    << dw_dx_fx[idf] << " " << dw_dy_fx[idf] << " " << dw_dz_fx[idf] << " "
                    << dT_dx_fx[idf] << " " << dT_dy_fx[idf] << " " << dT_dz_fx[idf] << " "
                    << T_fx[idf] << " "
                    << xi_x_fx[idf] << " " << xi_y_fx[idf] << " " << xi_z_fx[idf] << "\n";
            }
        }
    }
}
} // namespace

// -----------------------------------------------------------------
// ---------   计算粘性通量模块 -------------------------------------
// -----------------------------------------------------------------   

// 计算空间导数
// ==============================
// 通用差分模板
// ==============================
/*
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
*/

void compute_gradients_dudx(Field3D &F, const GridDesc &G)
{
    const LocalDesc &L = F.L;
    const double idx = 1.0 / G.dx;

    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int id = F.I(i, j, k);

                if (i == L.ngx && L.nbr_xm == MPI_PROC_NULL) {
                    F.du_dx[id] = idx * (-1.5 * F.u[F.I(i, j, k)] + 2.0 * F.u[F.I(i + 1, j, k)] - 0.5 * F.u[F.I(i + 2, j, k)]);
                } else if (i == L.ngx + L.nx - 1 && L.nbr_xp == MPI_PROC_NULL) {
                    F.du_dx[id] = idx * (1.5 * F.u[F.I(i, j, k)] - 2.0 * F.u[F.I(i - 1, j, k)] + 0.5 * F.u[F.I(i - 2, j, k)]);
                } else {
                    F.du_dx[id] = 0.5 * idx * (F.u[F.I(i + 1, j, k)] - F.u[F.I(i - 1, j, k)]);
                }
            }
        }
    }
}

// ==================================================
// 主函数：自适应阶数梯度计算, 根据边界距离选择差分格式
// 计算内点的6阶中心差分，边界处根据距离选择低阶格式
// ==================================================

// 计算半节点粘性通量
void compute_viscous_flux(Field3D &F, const CartDecomp &C, const GridDesc &G, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double dxi = G.dx;
    const double deta = G.dy;
    const double dzeta = G.dz;
    const double idx = 1.0 / dxi;
    const double idy = 1.0 / deta;
    const double idz = 1.0 / dzeta;

    std::vector<double> du_dxi, du_deta, du_dzeta;
    std::vector<double> dv_dxi, dv_deta, dv_dzeta;
    std::vector<double> dw_dxi, dw_deta, dw_dzeta;
    std::vector<double> dT_dxi, dT_deta, dT_dzeta;

    const int tot = L.sx * L.sy * L.sz;
    const int fx_count = (L.sx - 1) * L.sy * L.sz;
    const int fy_count = L.sx * (L.sy - 1) * L.sz;
    const int fz_count = L.sx * L.sy * (L.sz - 1);

    auto compute_metric_weighted_grad = [&](const std::vector<double> &phi,
                                            const std::vector<double> &xi_comp_fx,
                                            const std::vector<double> &eta_comp_fy,
                                            const std::vector<double> &zeta_comp_fz,
                                            std::vector<double> &dphi_dcomp,
                                            std::vector<double> &dphi_dxi_out,
                                            std::vector<double> &dphi_deta_out,
                                            std::vector<double> &dphi_dzeta_out,
                                            int tag_base) {
        std::vector<double> phi_fx(fx_count, 0.0), phi_fy(fy_count, 0.0), phi_fz(fz_count, 0.0);
        std::vector<double> prod_fx(fx_count, 0.0), prod_fy(fy_count, 0.0), prod_fz(fz_count, 0.0);

        dphi_dxi_out.assign(tot, 0.0);
        dphi_deta_out.assign(tot, 0.0);
        dphi_dzeta_out.assign(tot, 0.0);
        dphi_dcomp.assign(tot, 0.0);

        interp_half_x(phi, phi_fx, L);
        interp_half_y(phi, phi_fy, L);
        interp_half_z(phi, phi_fz, L);
        exchange_half_halo_x(phi_fx, L, C, L.ngx, tag_base + 0);
        exchange_half_halo_y(phi_fy, L, C, L.ngy, tag_base + 10);
        exchange_half_halo_z(phi_fz, L, C, L.ngz, tag_base + 20);
        interp_half_x_boundary(phi, phi_fx, L);
        interp_half_y_boundary(phi, phi_fy, L);
        interp_half_z_boundary(phi, phi_fz, L);

        for (int k = 0; k < L.sz; ++k) {
            for (int j = 0; j < L.sy; ++j) {
                for (int i = 0; i < L.sx - 1; ++i) {
                    const int idf = idx_fx(i, j, k, L);
                    prod_fx[idf] = phi_fx[idf] * xi_comp_fx[idf];
                }
            }
        }
        for (int k = 0; k < L.sz; ++k) {
            for (int j = 0; j < L.sy - 1; ++j) {
                for (int i = 0; i < L.sx; ++i) {
                    const int idf = idx_fy(i, j, k, L);
                    prod_fy[idf] = phi_fy[idf] * eta_comp_fy[idf];
                }
            }
        }
        for (int k = 0; k < L.sz - 1; ++k) {
            for (int j = 0; j < L.sy; ++j) {
                for (int i = 0; i < L.sx; ++i) {
                    const int idf = idx_fz(i, j, k, L);
                    prod_fz[idf] = phi_fz[idf] * zeta_comp_fz[idf];
                }
            }
        }

        diff_x_half(prod_fx, dphi_dxi_out, idx, L);
        diff_y_half(prod_fy, dphi_deta_out, idy, L);
        diff_z_half(prod_fz, dphi_dzeta_out, idz, L);
        exchange_node_halo_x(dphi_dxi_out, L, C, L.ngx, tag_base + 30);
        exchange_node_halo_y(dphi_deta_out, L, C, L.ngy, tag_base + 40);
        exchange_node_halo_z(dphi_dzeta_out, L, C, L.ngz, tag_base + 50);
        diff_x_half_boundary(prod_fx, dphi_dxi_out, idx, L);
        diff_y_half_boundary(prod_fy, dphi_deta_out, idy, L);
        diff_z_half_boundary(prod_fz, dphi_dzeta_out, idz, L);

        for (int k = 0; k < L.sz; ++k) {
            for (int j = 0; j < L.sy; ++j) {
                for (int i = 0; i < L.sx; ++i) {
                    const int id = idx3(i, j, k, L);
                    const double Ja = F.Ja[id];
                    if(Ja == 0.0) continue; // 避免除以零
                    dphi_dcomp[id] = (dphi_dxi_out[id] + dphi_deta_out[id] + dphi_dzeta_out[id]) / Ja;
                }
            }
        }
    };

    auto interp_full_x = [&](const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L) {
        interp_half_x(node, face, L);
        exchange_half_halo_x(face, L, C, L.ngx, 1000);
        interp_half_x_boundary(node, face, L);
    };
    auto interp_full_y = [&](const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L) {
        interp_half_y(node, face, L);
        exchange_half_halo_y(face, L, C, L.ngy, 1010);
        interp_half_y_boundary(node, face, L);
    };
    auto interp_full_z = [&](const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L) {
        interp_half_z(node, face, L);
        exchange_half_halo_z(face, L, C, L.ngz, 1020);
        interp_half_z_boundary(node, face, L);
    };

    compute_metric_weighted_grad(F.u, F.xi_x_fx, F.eta_x_fy, F.zeta_x_fz,
                                 F.du_dx, du_dxi, du_deta, du_dzeta, 8100);
    compute_metric_weighted_grad(F.u, F.xi_y_fx, F.eta_y_fy, F.zeta_y_fz,
                                 F.du_dy, du_dxi, du_deta, du_dzeta, 8200);
    compute_metric_weighted_grad(F.u, F.xi_z_fx, F.eta_z_fy, F.zeta_z_fz,
                                 F.du_dz, du_dxi, du_deta, du_dzeta, 8300);

    compute_metric_weighted_grad(F.v, F.xi_x_fx, F.eta_x_fy, F.zeta_x_fz,
                                 F.dv_dx, dv_dxi, dv_deta, dv_dzeta, 8400);
    compute_metric_weighted_grad(F.v, F.xi_y_fx, F.eta_y_fy, F.zeta_y_fz,
                                 F.dv_dy, dv_dxi, dv_deta, dv_dzeta, 8500);
    compute_metric_weighted_grad(F.v, F.xi_z_fx, F.eta_z_fy, F.zeta_z_fz,
                                 F.dv_dz, dv_dxi, dv_deta, dv_dzeta, 8600);

    compute_metric_weighted_grad(F.w, F.xi_x_fx, F.eta_x_fy, F.zeta_x_fz,
                                 F.dw_dx, dw_dxi, dw_deta, dw_dzeta, 8700);
    compute_metric_weighted_grad(F.w, F.xi_y_fx, F.eta_y_fy, F.zeta_y_fz,
                                 F.dw_dy, dw_dxi, dw_deta, dw_dzeta, 8800);
    compute_metric_weighted_grad(F.w, F.xi_z_fx, F.eta_z_fy, F.zeta_z_fz,
                                 F.dw_dz, dw_dxi, dw_deta, dw_dzeta, 8900);

    compute_metric_weighted_grad(F.T, F.xi_x_fx, F.eta_x_fy, F.zeta_x_fz,
                                 F.dT_dx, dT_dxi, dT_deta, dT_dzeta, 9000);
    compute_metric_weighted_grad(F.T, F.xi_y_fx, F.eta_y_fy, F.zeta_y_fz,
                                 F.dT_dy, dT_dxi, dT_deta, dT_dzeta, 9100);
    compute_metric_weighted_grad(F.T, F.xi_z_fx, F.eta_z_fy, F.zeta_z_fz,
                                 F.dT_dz, dT_dxi, dT_deta, dT_dzeta, 9200);

    std::vector<double> u_fx(fx_count, 0.0), v_fx(fx_count, 0.0), w_fx(fx_count, 0.0);
    std::vector<double> du_dx_fx(fx_count, 0.0), du_dy_fx(fx_count, 0.0), du_dz_fx(fx_count, 0.0);
    std::vector<double> dv_dx_fx(fx_count, 0.0), dv_dy_fx(fx_count, 0.0), dv_dz_fx(fx_count, 0.0);
    std::vector<double> dw_dx_fx(fx_count, 0.0), dw_dy_fx(fx_count, 0.0), dw_dz_fx(fx_count, 0.0);
    std::vector<double> dT_dx_fx(fx_count, 0.0), dT_dy_fx(fx_count, 0.0), dT_dz_fx(fx_count, 0.0);
    std::vector<double> T_fx(fx_count, 0.0);

    std::vector<double> u_fy(fy_count, 0.0), v_fy(fy_count, 0.0), w_fy(fy_count, 0.0);
    std::vector<double> du_dx_fy(fy_count, 0.0), du_dy_fy(fy_count, 0.0), du_dz_fy(fy_count, 0.0);
    std::vector<double> dv_dx_fy(fy_count, 0.0), dv_dy_fy(fy_count, 0.0), dv_dz_fy(fy_count, 0.0);
    std::vector<double> dw_dx_fy(fy_count, 0.0), dw_dy_fy(fy_count, 0.0), dw_dz_fy(fy_count, 0.0);
    std::vector<double> dT_dx_fy(fy_count, 0.0), dT_dy_fy(fy_count, 0.0), dT_dz_fy(fy_count, 0.0);
    std::vector<double> T_fy(fy_count, 0.0);

    std::vector<double> u_fz(fz_count, 0.0), v_fz(fz_count, 0.0), w_fz(fz_count, 0.0);
    std::vector<double> du_dx_fz(fz_count, 0.0), du_dy_fz(fz_count, 0.0), du_dz_fz(fz_count, 0.0);
    std::vector<double> dv_dx_fz(fz_count, 0.0), dv_dy_fz(fz_count, 0.0), dv_dz_fz(fz_count, 0.0);
    std::vector<double> dw_dx_fz(fz_count, 0.0), dw_dy_fz(fz_count, 0.0), dw_dz_fz(fz_count, 0.0);
    std::vector<double> dT_dx_fz(fz_count, 0.0), dT_dy_fz(fz_count, 0.0), dT_dz_fz(fz_count, 0.0);
    std::vector<double> T_fz(fz_count, 0.0);

    interp_full_x(F.u, u_fx, L); interp_full_x(F.v, v_fx, L); interp_full_x(F.w, w_fx, L);
    interp_full_x(F.du_dx, du_dx_fx, L); interp_full_x(F.du_dy, du_dy_fx, L); interp_full_x(F.du_dz, du_dz_fx, L);
    interp_full_x(F.dv_dx, dv_dx_fx, L); interp_full_x(F.dv_dy, dv_dy_fx, L); interp_full_x(F.dv_dz, dv_dz_fx, L);
    interp_full_x(F.dw_dx, dw_dx_fx, L); interp_full_x(F.dw_dy, dw_dy_fx, L); interp_full_x(F.dw_dz, dw_dz_fx, L);
    interp_full_x(F.dT_dx, dT_dx_fx, L); interp_full_x(F.dT_dy, dT_dy_fx, L); interp_full_x(F.dT_dz, dT_dz_fx, L);
    interp_full_x(F.T, T_fx, L);

    // Debug dump requested: output x-face viscous variables and stop.
    /*
    write_viscous_xface_debug_tecplot(
        L, C,
        u_fx, v_fx, w_fx,
        du_dx_fx, du_dy_fx, du_dz_fx,
        dv_dx_fx, dv_dy_fx, dv_dz_fx,
        dw_dx_fx, dw_dy_fx, dw_dz_fx,
        dT_dx_fx, dT_dy_fx, dT_dz_fx,
        T_fx,
        F.xi_x_fx, F.xi_y_fx, F.xi_z_fx);
    MPI_Barrier(C.cart_comm);
    if (C.rank == 0) {
        std::cout << "X-face Tecplot debug files written to output/, stopping program as requested." << std::endl;
    }
    MPI_Abort(C.cart_comm, 0);
    */

    interp_full_y(F.u, u_fy, L); interp_full_y(F.v, v_fy, L); interp_full_y(F.w, w_fy, L);
    interp_full_y(F.du_dx, du_dx_fy, L); interp_full_y(F.du_dy, du_dy_fy, L); interp_full_y(F.du_dz, du_dz_fy, L);
    interp_full_y(F.dv_dx, dv_dx_fy, L); interp_full_y(F.dv_dy, dv_dy_fy, L); interp_full_y(F.dv_dz, dv_dz_fy, L);
    interp_full_y(F.dw_dx, dw_dx_fy, L); interp_full_y(F.dw_dy, dw_dy_fy, L); interp_full_y(F.dw_dz, dw_dz_fy, L);
    interp_full_y(F.dT_dx, dT_dx_fy, L); interp_full_y(F.dT_dy, dT_dy_fy, L); interp_full_y(F.dT_dz, dT_dz_fy, L);
    interp_full_y(F.T, T_fy, L);

    interp_full_z(F.u, u_fz, L); interp_full_z(F.v, v_fz, L); interp_full_z(F.w, w_fz, L);
    interp_full_z(F.du_dx, du_dx_fz, L); interp_full_z(F.du_dy, du_dy_fz, L); interp_full_z(F.du_dz, du_dz_fz, L);
    interp_full_z(F.dv_dx, dv_dx_fz, L); interp_full_z(F.dv_dy, dv_dy_fz, L); interp_full_z(F.dv_dz, dv_dz_fz, L);
    interp_full_z(F.dw_dx, dw_dx_fz, L); interp_full_z(F.dw_dy, dw_dy_fz, L); interp_full_z(F.dw_dz, dw_dz_fz, L);
    interp_full_z(F.dT_dx, dT_dx_fz, L); interp_full_z(F.dT_dy, dT_dy_fz, L); interp_full_z(F.dT_dz, dT_dz_fz, L);
    interp_full_z(F.T, T_fz, L);

    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = 0; i < L.sx - 1; ++i) {
                const int idf = idx_fx(i, j, k, L);
                const double mu = P.get_mu(T_fx[idf]);
                const double kappa = mu * P.Cp / P.Pr;
                const double div = du_dx_fx[idf] + dv_dy_fx[idf] + dw_dz_fx[idf];

                const double tau_xx = mu * (2.0 * du_dx_fx[idf] - 2.0 * div / 3.0);
                const double tau_xy = mu * (du_dy_fx[idf] + dv_dx_fx[idf]);
                const double tau_xz = mu * (du_dz_fx[idf] + dw_dx_fx[idf]);

                const double tau_yy = mu * (2.0 * dv_dy_fx[idf] - 2.0 * div / 3.0);
                const double tau_yz = mu * (dv_dz_fx[idf] + dw_dy_fx[idf]);
                const double tau_zz = mu * (2.0 * dw_dz_fx[idf] - 2.0 * div / 3.0);

                const double b_x = u_fx[idf] * tau_xx + v_fx[idf] * tau_xy + w_fx[idf] * tau_xz + kappa * dT_dx_fx[idf];
                const double b_y = u_fx[idf] * tau_xy + v_fx[idf] * tau_yy + w_fx[idf] * tau_yz + kappa * dT_dy_fx[idf];
                const double b_z = u_fx[idf] * tau_xz + v_fx[idf] * tau_yz + w_fx[idf] * tau_zz + kappa * dT_dz_fx[idf];


                F.vis_flux_fx_momx[idf] = tau_xx * F.xi_x_fx[idf] + tau_xy * F.xi_y_fx[idf] + tau_xz * F.xi_z_fx[idf];
                F.vis_flux_fx_momy[idf] = tau_xy * F.xi_x_fx[idf] + tau_yy * F.xi_y_fx[idf] + tau_yz * F.xi_z_fx[idf];
                F.vis_flux_fx_momz[idf] = tau_xz * F.xi_x_fx[idf] + tau_yz * F.xi_y_fx[idf] + tau_zz * F.xi_z_fx[idf];
                F.vis_flux_fx_E[idf] = b_x * F.xi_x_fx[idf] + b_y * F.xi_y_fx[idf] + b_z * F.xi_z_fx[idf];
            }
        }
    }

    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = 0; j < L.sy - 1; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idf = idx_fy(i, j, k, L);
                const double mu = P.get_mu(T_fy[idf]);
                const double kappa = mu * P.Cp / P.Pr;
                const double div = du_dx_fy[idf] + dv_dy_fy[idf] + dw_dz_fy[idf];

                const double tau_yy = mu * (2.0 * dv_dy_fy[idf] - 2.0 * div / 3.0);
                const double tau_xy = mu * (du_dy_fy[idf] + dv_dx_fy[idf]);
                const double tau_yz = mu * (dv_dz_fy[idf] + dw_dy_fy[idf]);

                const double tau_xx = mu * (2.0 * du_dx_fy[idf] - 2.0 * div / 3.0);
                const double tau_xz = mu * (du_dz_fy[idf] + dw_dx_fy[idf]);
                const double tau_zz = mu * (2.0 * dw_dz_fy[idf] - 2.0 * div / 3.0);

                const double b_x = u_fy[idf] * tau_xx + v_fy[idf] * tau_xy + w_fy[idf] * tau_xz + kappa * dT_dx_fy[idf];
                const double b_y = u_fy[idf] * tau_xy + v_fy[idf] * tau_yy + w_fy[idf] * tau_yz + kappa * dT_dy_fy[idf];
                const double b_z = u_fy[idf] * tau_xz + v_fy[idf] * tau_yz + w_fy[idf] * tau_zz + kappa * dT_dz_fy[idf];

                F.vis_flux_fy_momx[idf] = tau_xx * F.eta_x_fy[idf] + tau_xy * F.eta_y_fy[idf] + tau_xz * F.eta_z_fy[idf];
                F.vis_flux_fy_momy[idf] = tau_xy * F.eta_x_fy[idf] + tau_yy * F.eta_y_fy[idf] + tau_yz * F.eta_z_fy[idf];
                F.vis_flux_fy_momz[idf] = tau_xz * F.eta_x_fy[idf] + tau_yz * F.eta_y_fy[idf] + tau_zz * F.eta_z_fy[idf];
                F.vis_flux_fy_E[idf] = b_x * F.eta_x_fy[idf] + b_y * F.eta_y_fy[idf] + b_z * F.eta_z_fy[idf];
            }
        }
    }

    for (int k = 0; k < L.sz - 1; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idf = idx_fz(i, j, k, L);
                const double mu = P.get_mu(T_fz[idf]);
                const double kappa = mu * P.Cp / P.Pr;
                const double div = du_dx_fz[idf] + dv_dy_fz[idf] + dw_dz_fz[idf];

                const double tau_zz = mu * (2.0 * dw_dz_fz[idf] - 2.0 * div / 3.0);
                const double tau_xz = mu * (du_dz_fz[idf] + dw_dx_fz[idf]);
                const double tau_yz = mu * (dv_dz_fz[idf] + dw_dy_fz[idf]);

                const double tau_xx = mu * (2.0 * du_dx_fz[idf] - 2.0 * div / 3.0);
                const double tau_xy = mu * (du_dy_fz[idf] + dv_dx_fz[idf]);
                const double tau_yy = mu * (2.0 * dv_dy_fz[idf] - 2.0 * div / 3.0);

                const double b_x = u_fz[idf] * tau_xx + v_fz[idf] * tau_xy + w_fz[idf] * tau_xz + kappa * dT_dx_fz[idf];
                const double b_y = u_fz[idf] * tau_xy + v_fz[idf] * tau_yy + w_fz[idf] * tau_yz + kappa * dT_dy_fz[idf];
                const double b_z = u_fz[idf] * tau_xz + v_fz[idf] * tau_yz + w_fz[idf] * tau_zz + kappa * dT_dz_fz[idf];

                F.vis_flux_fz_momx[idf] = tau_xx * F.zeta_x_fz[idf] + tau_xy * F.zeta_y_fz[idf] + tau_xz * F.zeta_z_fz[idf];
                F.vis_flux_fz_momy[idf] = tau_xy * F.zeta_x_fz[idf] + tau_yy * F.zeta_y_fz[idf] + tau_yz * F.zeta_z_fz[idf];
                F.vis_flux_fz_momz[idf] = tau_xz * F.zeta_x_fz[idf] + tau_yz * F.zeta_y_fz[idf] + tau_zz * F.zeta_z_fz[idf];
                F.vis_flux_fz_E[idf] = b_x * F.zeta_x_fz[idf] + b_y * F.zeta_y_fz[idf] + b_z * F.zeta_z_fz[idf];
            }
        }
    }
}


/*
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
        
        //F.rhs_rho[id] += diff_x(F.Fvflux_mass, i, j, k, dx, order_x, L);
        //F.rhs_rho[id] += diff_y(F.Hvflux_mass, i, j, k, dy, order_y, L);
        //F.rhs_rho[id] += diff_z(F.Gvflux_mass, i, j, k, dz, order_z, L);
        

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
*/