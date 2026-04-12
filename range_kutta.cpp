#include "field_structures.h"
#include "ns3d_func.h"
#include <cmath>
#include <mpi.h>

namespace {
inline bool is_wall_boundary_point(int i, int j, int k, const LocalDesc &L, const SolverParams &P)
{
    if (L.nbr_xm == MPI_PROC_NULL && P.bc_xmin == SolverParams::BCType::Wall && i == L.ngx) return true;
    if (L.nbr_xp == MPI_PROC_NULL && P.bc_xmax == SolverParams::BCType::Wall && i == (L.ngx + L.nx - 1)) return true;
    if (L.nbr_ym == MPI_PROC_NULL && P.bc_ymin == SolverParams::BCType::Wall && j == L.ngy) return true;
    if (L.nbr_yp == MPI_PROC_NULL && P.bc_ymax == SolverParams::BCType::Wall && j == (L.ngy + L.ny - 1)) return true;
    if (L.nbr_zm == MPI_PROC_NULL && P.bc_zmin == SolverParams::BCType::Wall && k == L.ngz) return true;
    if (L.nbr_zp == MPI_PROC_NULL && P.bc_zmax == SolverParams::BCType::Wall && k == (L.ngz + L.nz - 1)) return true;
    return false;
}
}

//==================================================================
// 三阶 Runge-Kutta 时间推进主循环模块
//==================================================================

// 输入全场包括ghost的节点值，返回gh-1：gh+nx-1的半节点值，边界则返回gh层的半节点值
void diff_x_half(const std::vector<double> &flux_fx, std::vector<double> &rhs, double idx, const LocalDesc &L)
{
    const int ib = L.ngx;
	const int ie = L.ngx + L.nx - 1;

    for (int k = 0; k < L.sz; ++k) {
        for (int j = 0; j < L.sy; ++j) {
            for (int i = ib; i <= ie; ++i) {
                int id = idx3(i, j, k, L);
                int id1 = idx_fx(i, j, k, L), id2 = idx_fx(i + 1, j, k, L), id3 = idx_fx(i + 2, j, k, L);
                int id_1 = idx_fx(i - 1, j, k, L), id_2 = idx_fx(i - 2, j, k, L), id_3 = idx_fx(i - 3, j, k, L);
                rhs[id] = idx * ((75.0 / 64.0)  * (flux_fx[id1] - flux_fx[id_1])
				  - (25.0 / 384.0) * (flux_fx[id2] - flux_fx[id_2])
				  + ( 3.0 / 640.0) * (flux_fx[id3] - flux_fx[id_3]));
			}
        }
    }
}

void diff_x_half_boundary(const std::vector<double> &flux_fx, std::vector<double> &rhs, double idx, const LocalDesc &L)
{
    const int ib = 0;
	const int ie = L.sx - 1;

    if (L.nbr_xm != MPI_PROC_NULL && L.nbr_xp != MPI_PROC_NULL) return;

    for (int k = 0; k < L.sz; ++k) {
        for (int j = 0; j < L.sy; ++j) {
            if (L.nbr_xm == MPI_PROC_NULL) {
				rhs[idx3(ib + 2, j, k, L)] = (idx / 24.0) * (
                  flux_fx[idx_fx(ib , j, k, L)] - 27.0 * flux_fx[idx_fx(ib + 1, j, k, L)]
                  + 27.0 * flux_fx[idx_fx(ib + 2, j, k, L)] - flux_fx[idx_fx(ib + 3, j, k, L)]);
				rhs[idx3(ib + 1, j, k, L)] = (idx / 24.0) * (
				  - 22.0 * flux_fx[idx_fx(ib, j, k, L)] + 17.0 * flux_fx[idx_fx(ib + 1, j, k, L)]
				  + 9.0 * flux_fx[idx_fx(ib + 2, j, k, L)] - 5.0 * flux_fx[idx_fx(ib + 3, j, k, L)]
                  + flux_fx[idx_fx(ib + 4, j, k, L)]);
            }

            if (L.nbr_xp == MPI_PROC_NULL) {
                rhs[idx3(ie - 2, j, k, L)] = (idx / 24.0) * (
                  - flux_fx[idx_fx(ie - 1, j, k, L)] + 27.0 * flux_fx[idx_fx(ie - 2, j, k, L)]
                  - 27.0 * flux_fx[idx_fx(ie - 3, j, k, L)] + flux_fx[idx_fx(ie - 4, j, k, L)]);
                rhs[idx3(ie - 1, j, k, L)] = (idx / 24.0) * (
                  22.0 * flux_fx[idx_fx(ie - 1, j, k, L)] - 17.0 * flux_fx[idx_fx(ie - 2, j, k, L)]
                  - 9.0 * flux_fx[idx_fx(ie - 3, j, k, L)] + 5.0 * flux_fx[idx_fx(ie - 4, j, k, L)]
                  - flux_fx[idx_fx(ie - 5, j, k, L)]);
            }
        }
    }
}

void diff_y_half(const std::vector<double> &flux_fy, std::vector<double> &rhs, double idy, const LocalDesc &L)
{
    const int jb = L.ngy;
	const int je = L.ngy + L.ny - 1;

    for (int k = 0; k < L.sz; ++k) {
        for (int i = 0; i < L.sx; ++i)  {
            for (int j = jb; j <= je; ++j) {
                int id = idx3(i, j, k, L);
                int id1 = idx_fy(i, j, k, L), id2 = idx_fy(i, j + 1, k, L), id3 = idx_fy(i, j + 2, k, L);
                int id_1 = idx_fy(i, j - 1, k, L), id_2 = idx_fy(i, j - 2, k, L), id_3 = idx_fy(i, j - 3, k, L);
                rhs[id] = idy * ((75.0 / 64.0)  * (flux_fy[id1] - flux_fy[id_1])
				  - (25.0 / 384.0) * (flux_fy[id2] - flux_fy[id_2])
				  + ( 3.0 / 640.0) * (flux_fy[id3] - flux_fy[id_3]));
			}
        }
    }
}

void diff_y_half_boundary(const std::vector<double> &flux_fy, std::vector<double> &rhs, double idy, const LocalDesc &L)
{
    const int jb = 0;
	const int je = L.sy - 1;

    if (L.nbr_ym != MPI_PROC_NULL && L.nbr_yp != MPI_PROC_NULL) return;

    for (int k = 0; k < L.sz; ++k) {
        for (int i = 0; i < L.sx; ++i) {
            if (L.nbr_ym == MPI_PROC_NULL) {
                rhs[idx3(i, jb + 2, k, L)] = (idy / 24.0) * (
                  flux_fy[idx_fy(i, jb, k, L)] - 27.0 * flux_fy[idx_fy(i, jb + 1, k, L)]
                  + 27.0 * flux_fy[idx_fy(i, jb + 2, k, L)] - flux_fy[idx_fy(i, jb + 3, k, L)]);
                rhs[idx3(i, jb + 1, k, L)] = (idy / 24.0) * (
                  - 22.0 * flux_fy[idx_fy(i, jb, k, L)] + 17.0 * flux_fy[idx_fy(i, jb + 1, k, L)]
                  + 9.0 * flux_fy[idx_fy(i, jb + 2, k, L)] - 5.0 * flux_fy[idx_fy(i, jb + 3, k, L)]
                  + flux_fy[idx_fy(i, jb + 4, k, L)]);
            }

            if (L.nbr_yp == MPI_PROC_NULL) {
                rhs[idx3(i, je - 2, k, L)] = (idy / 24.0) * (
                  - flux_fy[idx_fy(i, je - 1, k, L)] + 27.0 * flux_fy[idx_fy(i, je - 2, k, L)]
                  - 27.0 * flux_fy[idx_fy(i, je - 3, k, L)] + flux_fy[idx_fy(i, je - 4, k, L)]);
                rhs[idx3(i, je - 1, k, L)] = (idy / 24.0) * (
                  22.0 * flux_fy[idx_fy(i, je - 1, k, L)] - 17.0 * flux_fy[idx_fy(i, je - 2, k, L)]
                  - 9.0 * flux_fy[idx_fy(i, je - 3, k, L)] + 5.0 * flux_fy[idx_fy(i, je - 4, k, L)]
                  - flux_fy[idx_fy(i, je - 5, k, L)]);
            }
        }
    }
}

void diff_z_half(const std::vector<double> &flux_fz, std::vector<double> &rhs, double idz, const LocalDesc &L)
{
    const int kb = L.ngz;
	const int ke = L.ngz + L.nz - 1;

    for (int j = 0; j < L.sy; ++j) {
        for (int i = 0; i < L.sx; ++i) {
            for (int k = kb; k <= ke; ++k) {
                int id = idx3(i, j, k, L);
                int id1 = idx_fz(i, j, k, L), id2 = idx_fz(i, j, k + 1, L), id3 = idx_fz(i, j, k + 2, L);
                int id_1 = idx_fz(i, j, k - 1, L), id_2 = idx_fz(i, j, k - 2, L), id_3 = idx_fz(i, j, k - 3, L);
                rhs[id] = idz * ((75.0 / 64.0)  * (flux_fz[id1] - flux_fz[id_1])
				  - (25.0 / 384.0) * (flux_fz[id2] - flux_fz[id_2])
				  + ( 3.0 / 640.0) * (flux_fz[id3] - flux_fz[id_3]));
			}
        }
    }
}

void diff_z_half_boundary(const std::vector<double> &flux_fz, std::vector<double> &rhs, double idz, const LocalDesc &L)
{
    const int kb = 0;
	const int ke = L.sz - 1;

    if (L.nbr_zm != MPI_PROC_NULL && L.nbr_zp != MPI_PROC_NULL) return;

    for (int j = 0; j < L.sy; ++j) {
        for (int i = 0; i < L.sx; ++i) {
            if (L.nbr_zm == MPI_PROC_NULL) {
                rhs[idx3(i, j, kb + 2, L)] = (idz / 24.0) * (
                  flux_fz[idx_fz(i, j, kb, L)] - 27.0 * flux_fz[idx_fz(i, j, kb + 1, L)]
                  + 27.0 * flux_fz[idx_fz(i, j, kb + 2, L)] - flux_fz[idx_fz(i, j, kb + 3, L)]);
                rhs[idx3(i, j, kb + 1, L)] = (idz / 24.0) * (
                  - 22.0 * flux_fz[idx_fz(i, j, kb, L)] + 17.0 * flux_fz[idx_fz(i, j, kb + 1, L)]
                  + 9.0 * flux_fz[idx_fz(i, j, kb + 2, L)] - 5.0 * flux_fz[idx_fz(i, j, kb + 3, L)]
                  + flux_fz[idx_fz(i, j, kb + 4, L)]);
            }

            if (L.nbr_zp == MPI_PROC_NULL) {
                rhs[idx3(i, j, ke - 2, L)] = (idz / 24.0) * (
                  - flux_fz[idx_fz(i, j, ke - 1, L)] + 27.0 * flux_fz[idx_fz(i, j, ke - 2, L)]
                  - 27.0 * flux_fz[idx_fz(i, j, ke - 3, L)] + flux_fz[idx_fz(i, j, ke - 4, L)]);
                rhs[idx3(i, j, ke - 1, L)] = (idz / 24.0) * (
                  22.0 * flux_fz[idx_fz(i, j, ke - 1, L)] - 17.0 * flux_fz[idx_fz(i, j, ke - 2, L)]
                  - 9.0 * flux_fz[idx_fz(i, j, ke - 3, L)] + 5.0 * flux_fz[idx_fz(i, j, ke - 4, L)]
                  - flux_fz[idx_fz(i, j, ke - 5, L)]);
            }
        }
    }
}

void compute_rhs(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs)
{
    LocalDesc &L = F.L;
    const double idx = 1.0 / G.dx;
    const double idy = 1.0 / G.dy;
    const double idz = 1.0 / G.dz;
    static unsigned long long invis_flux_dump_seq = 0;

    // 清空 RHS
    std::fill(F.rhs_rho.begin(), F.rhs_rho.end(), 0.0);
    std::fill(F.rhs_rhou.begin(), F.rhs_rhou.end(), 0.0);
    std::fill(F.rhs_rhov.begin(), F.rhs_rhov.end(), 0.0);
    std::fill(F.rhs_rhow.begin(), F.rhs_rhow.end(), 0.0);
    std::fill(F.rhs_E.begin(), F.rhs_E.end(), 0.0);

    // 计算无粘通量
    compute_invis_flux(F, P, C);
    // 填充边界半节点的无粘通量
    compute_invis_flux_boundary(F, P);
    // 计算粘性通量
    compute_viscous_flux(F, C, G, P);

    // 组装并交换半节点的通量，然后求rhs
    std::vector<double> temp(L.sx * L.sy * L.sz, 0.0);

    // x方向：组装通量并交换halo
    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = 0; i < L.sx - 1; ++i) {
                int id = idx_fx(i, j, k, L);
                F.flux_fx_momx[id] = F.flux_fx_momx[id] - F.vis_flux_fx_momx[id];
                F.flux_fx_momy[id] = F.flux_fx_momy[id] - F.vis_flux_fx_momy[id];
                F.flux_fx_momz[id] = F.flux_fx_momz[id] - F.vis_flux_fx_momz[id];
                F.flux_fx_E[id]    = F.flux_fx_E[id]    - F.vis_flux_fx_E[id];
            }
        }
    }
    // x方向差分
    diff_x_half(F.flux_fx_mass, temp, idx, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rho[n]  += temp[n];
    diff_x_half(F.flux_fx_momx, temp, idx, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhou[n] += temp[n];
    diff_x_half(F.flux_fx_momy, temp, idx, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhov[n] += temp[n];
    diff_x_half(F.flux_fx_momz, temp, idx, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhow[n] += temp[n];
    diff_x_half(F.flux_fx_E, temp, idx, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_E[n]    += temp[n];

    // y方向：组装通量并交换halo
    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
            for (int j = 0; j < L.sy - 1; ++j) {
                int id = idx_fy(i, j, k, L);
                F.flux_fy_momx[id] = F.flux_fy_momx[id] - F.vis_flux_fy_momx[id];
                F.flux_fy_momy[id] = F.flux_fy_momy[id] - F.vis_flux_fy_momy[id];
                F.flux_fy_momz[id] = F.flux_fy_momz[id] - F.vis_flux_fy_momz[id];
                F.flux_fy_E[id]    = F.flux_fy_E[id]    - F.vis_flux_fy_E[id];
            }
        }
    }

    // y方向差分
    diff_y_half(F.flux_fy_mass, temp, idy, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rho[n]  += temp[n];
    diff_y_half(F.flux_fy_momx, temp, idy, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhou[n] += temp[n];
    diff_y_half(F.flux_fy_momy, temp, idy, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhov[n] += temp[n];
    diff_y_half(F.flux_fy_momz, temp, idy, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhow[n] += temp[n];
    diff_y_half(F.flux_fy_E, temp, idy, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_E[n]    += temp[n];

    // z方向：组装通量并交换halo
    for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
        for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
            for (int k = 0; k < L.sz - 1; ++k) {
                int id = idx_fz(i, j, k, L);
                F.flux_fz_momx[id] = F.flux_fz_momx[id] - F.vis_flux_fz_momx[id];
                F.flux_fz_momy[id] = F.flux_fz_momy[id] - F.vis_flux_fz_momy[id];
                F.flux_fz_momz[id] = F.flux_fz_momz[id] - F.vis_flux_fz_momz[id];
                F.flux_fz_E[id]    = F.flux_fz_E[id]    - F.vis_flux_fz_E[id];
            }
        }
    }

    // z方向差分
    diff_z_half(F.flux_fz_mass, temp, idz, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rho[n]  += temp[n];
    diff_z_half(F.flux_fz_momx, temp, idz, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhou[n] += temp[n];
    diff_z_half(F.flux_fz_momy, temp, idz, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhov[n] += temp[n];
    diff_z_half(F.flux_fz_momz, temp, idz, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_rhow[n] += temp[n];
    diff_z_half(F.flux_fz_E, temp, idz, L);
    for (size_t n = 0; n < temp.size(); ++n) F.rhs_E[n]    += temp[n];

    // 体积力源项
    if (P.use_body_force) 
    {
        const double fx = P.body_force_x;
        const double fy = P.body_force_y;
        const double fz = P.body_force_z;
        for (int k = 0; k < L.sz; ++k) {
            for (int j = 0; j < L.sy; ++j) {
                for (int i = 0; i < L.sx; ++i) {
                    const int id = F.I(i, j, k);
                    const double Ja = F.Ja[id];
                    const double rho = F.rho[id];
                    F.rhs_rhou[id] -= Ja * rho * fx;
                    F.rhs_rhov[id] -= Ja * rho * fy;
                    F.rhs_rhow[id] -= Ja * rho * fz;
                    F.rhs_E[id]    -= Ja * (F.rhou[id] * fx + F.rhov[id] * fy + F.rhow[id] * fz);
                }
            }
        }
    }

    /*
    write_halfnode_invis_flux_tecplot_rank(F, C, 'x', invis_flux_dump_seq, "halfnode_invis_flux");
    write_halfnode_invis_flux_tecplot_rank(F, C, 'y', invis_flux_dump_seq, "halfnode_invis_flux");
    write_halfnode_invis_flux_tecplot_rank(F, C, 'z', invis_flux_dump_seq, "halfnode_invis_flux");
    write_rhs_tecplot_rank(F, C, invis_flux_dump_seq, "rhs_values");
    ++invis_flux_dump_seq;

    // 固壁边界点不参与推进，RHS 置零，由 apply_wall_bc 统一更新其状态
    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                if (!is_wall_boundary_point(i, j, k, L, P)) continue;
                const int id = F.I(i, j, k);
                F.rhs_rho[id] = 0.0;
                F.rhs_rhou[id] = 0.0;
                F.rhs_rhov[id] = 0.0;
                F.rhs_rhow[id] = 0.0;
                F.rhs_E[id] = 0.0;
            }
        }
    }

    */
}

// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs, double dt)
{
    LocalDesc &L = F.L;

    // Stage 1
    compute_rhs(F, C, G, P, out_reqs);
    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int n = F.I(i, j, k);
                F.rho[n] = F.rho0[n] - dt * F.rhs_rho[n] / F.Ja[n];
                F.rhou[n] = F.rhou0[n] - dt * F.rhs_rhou[n] / F.Ja[n];
                F.rhov[n] = F.rhov0[n] - dt * F.rhs_rhov[n] / F.Ja[n];
                F.rhow[n] = F.rhow0[n] - dt * F.rhs_rhow[n] / F.Ja[n];
                F.E[n] = F.E0[n] - dt * F.rhs_E[n] / F.Ja[n];
            }
        }
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);

    // Stage 2
    compute_rhs(F, C, G, P, out_reqs);
    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int n = F.I(i, j, k);
                F.rho[n] = 0.75 * F.rho0[n] + 0.25 * (F.rho[n] - dt * F.rhs_rho[n] / F.Ja[n]);
                F.rhou[n] = 0.75 * F.rhou0[n] + 0.25 * (F.rhou[n] - dt * F.rhs_rhou[n] / F.Ja[n]);
                F.rhov[n] = 0.75 * F.rhov0[n] + 0.25 * (F.rhov[n] - dt * F.rhs_rhov[n] / F.Ja[n]);
                F.rhow[n] = 0.75 * F.rhow0[n] + 0.25 * (F.rhow[n] - dt * F.rhs_rhow[n] / F.Ja[n]);
                F.E[n] = 0.75 * F.E0[n] + 0.25 * (F.E[n] - dt * F.rhs_E[n] / F.Ja[n]);
            }
        }
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);

    // Stage 3
    compute_rhs(F, C, G, P, out_reqs);
    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int n = F.I(i, j, k);
                F.rho[n] = (1.0 / 3.0) * F.rho0[n] + (2.0 / 3.0) * (F.rho[n] - dt * F.rhs_rho[n] / F.Ja[n]);
                F.rhou[n] = (1.0 / 3.0) * F.rhou0[n] + (2.0 / 3.0) * (F.rhou[n] - dt * F.rhs_rhou[n] / F.Ja[n]);
                F.rhov[n] = (1.0 / 3.0) * F.rhov0[n] + (2.0 / 3.0) * (F.rhov[n] - dt * F.rhs_rhov[n] / F.Ja[n]);
                F.rhow[n] = (1.0 / 3.0) * F.rhow0[n] + (2.0 / 3.0) * (F.rhow[n] - dt * F.rhs_rhow[n] / F.Ja[n]);
                F.E[n] = (1.0 / 3.0) * F.E0[n] + (2.0 / 3.0) * (F.E[n] - dt * F.rhs_E[n] / F.Ja[n]);
            }
        }
    }
    F.conservedToPrimitive(P);
    apply_boundary(F, G, C, P);
    F.primitiveToConserved(P);
}
