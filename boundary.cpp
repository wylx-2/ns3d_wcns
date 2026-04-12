#include "field_structures.h"
#include "ns3d_func.h"
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

enum FaceID { XMIN=0, XMAX=1, YMIN=2, YMAX=3, ZMIN=4, ZMAX=5 };
struct NeighborInfo { int nbr; SolverParams::BCType face; FaceID id; };

// static void apply_edge_corner_fix(Field3D &F, const LocalDesc &L, const SolverParams &P);
//------------------------------------------------------------
// 边界更新核心函数
//------------------------------------------------------------
void apply_boundary(Field3D &F, GridDesc &G, CartDecomp &C,
                    const SolverParams &P)
{
    LocalDesc &L = F.L;
    // Step 1: Halo exchange for periodic boundaries
    HaloRequests reqs;
    exchange_halos_physical(F, C, L, reqs);

    // Step 2: 对每个方向检查是否需要本地边界
    // Map neighbor -> that side's BC type and FaceID
    NeighborInfo dirs[6] = {
        {L.nbr_xm, P.bc_xmin, XMIN}, {L.nbr_xp, P.bc_xmax, XMAX},
        {L.nbr_ym, P.bc_ymin, YMIN}, {L.nbr_yp, P.bc_ymax, YMAX},
        {L.nbr_zm, P.bc_zmin, ZMIN}, {L.nbr_zp, P.bc_zmax, ZMAX}
    };

    for (auto &d : dirs)
    {
        if (d.nbr != MPI_PROC_NULL) continue; // 有邻居 → 已由通信完成
        switch (d.face)
        {
            case SolverParams::BCType::Wall:
                apply_wall_bc(F, G, L, P, d.id);
                break;
            case SolverParams::BCType::Symmetry:
                apply_symmetry_bc(F, L, d.id);
                break;
            case SolverParams::BCType::Outflow:
                apply_outflow_bc(F, L, d.id);
                break;
            case SolverParams::BCType::Inflow:
                apply_inflow_bc(F, L, d.id);
                break;
            case SolverParams::BCType::Periodic:
                // 周期边界已由通信处理，无需额外操作
                break;
        }
    }

    // apply_edge_corner_fix(F, L, P);
    
    MPI_Barrier(MPI_COMM_WORLD);
}

static void apply_edge_corner_fix(Field3D &F, const LocalDesc &L, const SolverParams &P)
{
    auto is_active = [](int nbr, SolverParams::BCType bc) {
        return nbr == MPI_PROC_NULL && bc != SolverParams::BCType::Periodic;
    };

    const bool xmin = is_active(L.nbr_xm, P.bc_xmin);
    const bool xmax = is_active(L.nbr_xp, P.bc_xmax);
    const bool ymin = is_active(L.nbr_ym, P.bc_ymin);
    const bool ymax = is_active(L.nbr_yp, P.bc_ymax);
    const bool zmin = is_active(L.nbr_zm, P.bc_zmin);
    const bool zmax = is_active(L.nbr_zp, P.bc_zmax);

    auto fix_point = [&](int i, int j, int k,
                        bool on_xmin, bool on_xmax,
                        bool on_ymin, bool on_ymax,
                        bool on_zmin, bool on_zmax) {
        const int active_faces = (on_xmin ? 1 : 0) + (on_xmax ? 1 : 0)
                               + (on_ymin ? 1 : 0) + (on_ymax ? 1 : 0)
                               + (on_zmin ? 1 : 0) + (on_zmax ? 1 : 0);
        if (active_faces < 2) return;

        double rho_sum = 0.0, p_sum = 0.0, T_sum = 0.0;
        double u_sum = 0.0, v_sum = 0.0, w_sum = 0.0;
        int sample_count = 0;

        auto add_sample = [&](int ii, int jj, int kk) {
            const int id = F.I(ii, jj, kk);
            rho_sum += F.rho[id];
            p_sum   += F.p[id];
            T_sum   += F.T[id];
            u_sum   += F.u[id];
            v_sum   += F.v[id];
            w_sum   += F.w[id];
            ++sample_count;
        };

        if (on_xmin) add_sample(i + 1, j, k);
        if (on_xmax) add_sample(i - 1, j, k);
        if (on_ymin) add_sample(i, j + 1, k);
        if (on_ymax) add_sample(i, j - 1, k);
        if (on_zmin) add_sample(i, j, k + 1);
        if (on_zmax) add_sample(i, j, k - 1);

        if (sample_count == 0) return;

        const int idg = F.I(i, j, k);
        F.rho[idg] = rho_sum / static_cast<double>(sample_count);
        F.p[idg]   = p_sum   / static_cast<double>(sample_count);
        F.T[idg]   = T_sum   / static_cast<double>(sample_count);

        const bool has_wall = (on_xmin && P.bc_xmin == SolverParams::BCType::Wall)
                           || (on_xmax && P.bc_xmax == SolverParams::BCType::Wall)
                           || (on_ymin && P.bc_ymin == SolverParams::BCType::Wall)
                           || (on_ymax && P.bc_ymax == SolverParams::BCType::Wall)
                           || (on_zmin && P.bc_zmin == SolverParams::BCType::Wall)
                           || (on_zmax && P.bc_zmax == SolverParams::BCType::Wall);

        if (has_wall) {
            F.u[idg] = 0.0;
            F.v[idg] = 0.0;
            F.w[idg] = 0.0;
        } else {
            F.u[idg] = u_sum / static_cast<double>(sample_count);
            F.v[idg] = v_sum / static_cast<double>(sample_count);
            F.w[idg] = w_sum / static_cast<double>(sample_count);

            if ((on_xmin && P.bc_xmin == SolverParams::BCType::Symmetry) ||
                (on_xmax && P.bc_xmax == SolverParams::BCType::Symmetry)) {
                F.u[idg] = 0.0;
            }
            if ((on_ymin && P.bc_ymin == SolverParams::BCType::Symmetry) ||
                (on_ymax && P.bc_ymax == SolverParams::BCType::Symmetry)) {
                F.v[idg] = 0.0;
            }
            if ((on_zmin && P.bc_zmin == SolverParams::BCType::Symmetry) ||
                (on_zmax && P.bc_zmax == SolverParams::BCType::Symmetry)) {
                F.w[idg] = 0.0;
            }
        }

        F.E[idg] = F.p[idg] / (P.gamma - 1.0)
                 + 0.5 * F.rho[idg] * (F.u[idg] * F.u[idg] + F.v[idg] * F.v[idg] + F.w[idg] * F.w[idg]);
    };

    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const bool on_xmin = xmin && (i == L.ngx);
                const bool on_xmax = xmax && (i == L.ngx + L.nx - 1);
                const bool on_ymin = ymin && (j == L.ngy);
                const bool on_ymax = ymax && (j == L.ngy + L.ny - 1);
                const bool on_zmin = zmin && (k == L.ngz);
                const bool on_zmax = zmax && (k == L.ngz + L.nz - 1);
                fix_point(i, j, k,
                          on_xmin, on_xmax,
                          on_ymin, on_ymax,
                          on_zmin, on_zmax);
            }
        }
    }
}

// new No-slip wall boundary condition with ghost

void apply_wall_bc(Field3D &F, GridDesc &G, const LocalDesc &L,const SolverParams &P, int face)
{
    (void)G;

    const double gamma = P.gamma;

    auto fill_cell = [&](int ig, int jg, int kg, int ir, int jr, int kr) {
        const int idg = F.I(ig, jg, kg);
        const int idr = F.I(ir, jr, kr);

        // no-slip: all velocity components are odd reflection
        F.u[idg] = -F.u[idr];
        F.v[idg] = -F.v[idr];
        F.w[idg] = -F.w[idr];

        // scalar quantities use even reflection (zero normal gradient)
        F.p[idg] = F.p[idr];
        F.T[idg] = F.T[idr];
        F.rho[idg] = F.rho[idr];
    };

    if (face == XMIN) {
        for (int k = 0; k < L.sz; ++k) {
        for (int j = 0; j < L.sy; ++j) {
            const int ii = F.I(L.ngx, j, k);
            F.u[ii] = 0.0;
            F.v[ii] = 0.0;
            F.w[ii] = 0.0;

            for (int layer = 1; layer <= L.ngx; ++layer) {
                const int ig = L.ngx - layer;
                const int ir = L.ngx + layer;
                fill_cell(ig, j, k, ir, j, k);
            }
        }}
        return;
    }

    if (face == XMAX) {
        for (int k = 0; k < L.sz; ++k) {
        for (int j = 0; j < L.sy; ++j) {
            const int ii = F.I(L.ngx + L.nx - 1, j, k);
            F.u[ii] = 0.0;
            F.v[ii] = 0.0;
            F.w[ii] = 0.0;
                
            for (int layer = 1; layer <= L.ngx; ++layer) {
                const int ig = L.ngx + L.nx - 1 + layer;
                const int ir = L.ngx + L.nx - 1 - layer;
                fill_cell(ig, j, k, ir, j, k);
            }
        }}
        return;
    }

    if (face == YMIN) {
        for(int k = 0; k < L.sz; ++k) {
        for(int i = 0; i < L.sx; ++i) {
            const int jj = F.I(i, L.ngy, k);
            F.u[jj] = 0.0;
            F.v[jj] = 0.0;
            F.w[jj] = 0.0;

            for (int layer = 1; layer <= L.ngy; ++layer) {
                const int jg = L.ngy - layer;
                const int jr = L.ngy + layer;
                fill_cell(i, jg, k, i, jr, k);
            }
        }}
        return;
    }

    if (face == YMAX) {
        for(int k = 0; k < L.sz; ++k) {
        for(int i = 0; i < L.sx; ++i) {
            const int jj = F.I(i, L.ngy + L.ny - 1, k);
            F.u[jj] = 0.0;
            F.v[jj] = 0.0;
            F.w[jj] = 0.0;

            for (int layer = 1; layer <= L.ngy; ++layer) {
                const int jg = L.ngy + L.ny - 1 + layer;
                const int jr = L.ngy + L.ny - 1 - layer;
                fill_cell(i, jg, k, i, jr, k);
            }
        }}
        return;
    }

    if (face == ZMIN) {
        for (int j = 0; j < L.sy; ++j) {
        for (int i = 0; i < L.sx; ++i) {
            const int kk = F.I(i, j, L.ngz);
            F.u[kk] = 0.0;
            F.v[kk] = 0.0;
            F.w[kk] = 0.0;

            for (int layer = 1; layer <= L.ngz; ++layer) {
                const int kg = L.ngz - layer;
                const int kr = L.ngz + layer;
                fill_cell(i, j, kg, i, j, kr);
            }
        }}
        return;
    }

    if (face == ZMAX) {
        for (int j = 0; j < L.sy; ++j) {
        for (int i = 0; i < L.sx; ++i) {
            const int kk = F.I(i, j, L.ngz + L.nz - 1);
            F.u[kk] = 0.0;
            F.v[kk] = 0.0;
            F.w[kk] = 0.0;
        
            for (int layer = 1; layer <= L.ngz; ++layer) {
                const int kg = L.ngz + L.nz - 1 + layer;
                const int kr = L.ngz + L.nz - 1 - layer;
                fill_cell(i, j, kg, i, j, kr);
            }
        }}
        return;
    }
}


// Wall boundary condition implementation
// No-slip wall 存在问题
/*
void apply_wall_bc(Field3D &F, GridDesc &G, const LocalDesc &L,const SolverParams &P, int face)
{
    const std::vector<double> p_old = F.p;
    const std::vector<double> T_old = F.T;

    auto update_pt = [&](int i, int j, int k,
                         int i1, int j1, int k1,
                         int i2, int j2, int k2,
                         int iep, int jep, int kep,
                         int iem, int jem, int kem,
                         int izp, int jzp, int kzp,
                         int izm, int jzm, int kzm,
                         double a, double b,
                         double sign_n) {
        const int idg = F.I(i, j, k);
        const int id1 = F.I(i1, j1, k1);
        const int id2 = F.I(i2, j2, k2);
        const int id_ep = F.I(iep, jep, kep);
        const int id_em = F.I(iem, jem, kem);
        const int id_zp = F.I(izp, jzp, kzp);
        const int id_zm = F.I(izm, jzm, kzm);

        const double dp_t1 = p_old[id_ep] - p_old[id_em];
        const double dp_t2 = p_old[id_zp] - p_old[id_zm];
        const double dT_t1 = T_old[id_ep] - T_old[id_em];
        const double dT_t2 = T_old[id_zp] - T_old[id_zm];
        
        sign_n = 0.0; // 强制不考虑非正交校正
        F.p[idg] = F.p[id1];//(4.0 * F.p[id1] - F.p[id2] + sign_n * (a * dp_t1 + b * dp_t2)) / 3.0;
        F.T[idg] = F.T[id1];//(4.0 * F.T[id1] - F.T[id2] + sign_n * (a * dT_t1 + b * dT_t2)) / 3.0;
        F.u[idg] = 0.0;
        F.v[idg] = 0.0;
        F.w[idg] = 0.0;
        F.rho[idg] = F.p[idg] / (P.Rgas * F.T[idg]);
    };

    if (face == XMIN) {
        const int i = L.ngx;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
            for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
                const int idg = F.I(i, j, k);
                const double gnn = F.xi_x[idg] * F.xi_x[idg] + F.xi_y[idg] * F.xi_y[idg] + F.xi_z[idg] * F.xi_z[idg];
                const double a = (F.xi_x[idg] * F.eta_x[idg] + F.xi_y[idg] * F.eta_y[idg] + F.xi_z[idg] * F.eta_z[idg]) / gnn * (G.dx / G.dy);
                const double b = (F.xi_x[idg] * F.zeta_x[idg] + F.xi_y[idg] * F.zeta_y[idg] + F.xi_z[idg] * F.zeta_z[idg]) / gnn * (G.dx / G.dz);
                update_pt(i, j, k,
                          i + 1, j, k,
                          i + 2, j, k,
                          i, j + 1, k,
                          i, j - 1, k,
                          i, j, k + 1,
                          i, j, k - 1,
                          a, b, +1.0);
            }
        }
    } else if (face == XMAX) {
        const int i = L.ngx + L.nx - 1;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
            for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
                const int idg = F.I(i, j, k);
                const double gnn = F.xi_x[idg] * F.xi_x[idg] + F.xi_y[idg] * F.xi_y[idg] + F.xi_z[idg] * F.xi_z[idg];
                const double a = (F.xi_x[idg] * F.eta_x[idg] + F.xi_y[idg] * F.eta_y[idg] + F.xi_z[idg] * F.eta_z[idg]) / gnn * (G.dx / G.dy);
                const double b = (F.xi_x[idg] * F.zeta_x[idg] + F.xi_y[idg] * F.zeta_y[idg] + F.xi_z[idg] * F.zeta_z[idg]) / gnn * (G.dx / G.dz);
                update_pt(i, j, k,
                          i - 1, j, k,
                          i - 2, j, k,
                          i, j + 1, k,
                          i, j - 1, k,
                          i, j, k + 1,
                          i, j, k - 1,
                          a, b, -1.0);
            }
        }
    } else if (face == YMIN) {
        const int j = L.ngy;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idg = F.I(i, j, k);
                const double gnn = F.eta_x[idg] * F.eta_x[idg] + F.eta_y[idg] * F.eta_y[idg] + F.eta_z[idg] * F.eta_z[idg];
                const double a = (F.eta_x[idg] * F.xi_x[idg] + F.eta_y[idg] * F.xi_y[idg] + F.eta_z[idg] * F.xi_z[idg]) / gnn * (G.dy / G.dx);
                const double b = (F.eta_x[idg] * F.zeta_x[idg] + F.eta_y[idg] * F.zeta_y[idg] + F.eta_z[idg] * F.zeta_z[idg]) / gnn * (G.dy / G.dz);
                update_pt(i, j, k,
                          i, j + 1, k,
                          i, j + 2, k,
                          i + 1, j, k,
                          i - 1, j, k,
                          i, j, k + 1,
                          i, j, k - 1,
                          a, b, +1.0);
            }
        }
    } else if (face == YMAX) {
        const int j = L.ngy + L.ny - 1;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idg = F.I(i, j, k);
                const double gnn = F.eta_x[idg] * F.eta_x[idg] + F.eta_y[idg] * F.eta_y[idg] + F.eta_z[idg] * F.eta_z[idg];
                const double a = (F.eta_x[idg] * F.xi_x[idg] + F.eta_y[idg] * F.xi_y[idg] + F.eta_z[idg] * F.xi_z[idg]) / gnn * (G.dy / G.dx);
                const double b = (F.eta_x[idg] * F.zeta_x[idg] + F.eta_y[idg] * F.zeta_y[idg] + F.eta_z[idg] * F.zeta_z[idg]) / gnn * (G.dy / G.dz);
                update_pt(i, j, k,
                          i, j - 1, k,
                          i, j - 2, k,
                          i + 1, j, k,
                          i - 1, j, k,
                          i, j, k + 1,
                          i, j, k - 1,
                          a, b, -1.0);
            }
        }
    } else if (face == ZMIN) {
        const int k = L.ngz;
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idg = F.I(i, j, k);
                const double gnn = F.zeta_x[idg] * F.zeta_x[idg] + F.zeta_y[idg] * F.zeta_y[idg] + F.zeta_z[idg] * F.zeta_z[idg];
                const double a = (F.zeta_x[idg] * F.xi_x[idg] + F.zeta_y[idg] * F.xi_y[idg] + F.zeta_z[idg] * F.xi_z[idg]) / gnn * (G.dz / G.dx);
                const double b = (F.zeta_x[idg] * F.eta_x[idg] + F.zeta_y[idg] * F.eta_y[idg] + F.zeta_z[idg] * F.eta_z[idg]) / gnn * (G.dz / G.dy);
                update_pt(i, j, k,
                          i, j, k + 1,
                          i, j, k + 2,
                          i + 1, j, k,
                          i - 1, j, k,
                          i, j + 1, k,
                          i, j - 1, k,
                          a, b, +1.0);
            }
        }
    } else if (face == ZMAX) {
        const int k = L.ngz + L.nz - 1;
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idg = F.I(i, j, k);
                const double gnn = F.zeta_x[idg] * F.zeta_x[idg] + F.zeta_y[idg] * F.zeta_y[idg] + F.zeta_z[idg] * F.zeta_z[idg];
                const double a = (F.zeta_x[idg] * F.xi_x[idg] + F.zeta_y[idg] * F.xi_y[idg] + F.zeta_z[idg] * F.xi_z[idg]) / gnn * (G.dz / G.dx);
                const double b = (F.zeta_x[idg] * F.eta_x[idg] + F.zeta_y[idg] * F.eta_y[idg] + F.zeta_z[idg] * F.eta_z[idg]) / gnn * (G.dz / G.dy);
                update_pt(i, j, k,
                          i, j, k - 1,
                          i, j, k - 2,
                          i + 1, j, k,
                          i - 1, j, k,
                          i, j + 1, k,
                          i, j - 1, k,
                          a, b, -1.0);
            }
        }
    }
}
*/
// Symmetry boundary condition implementation
void apply_symmetry_bc(Field3D &F, const LocalDesc &L, int face)
{
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;

    if (face == XMIN) {
        const int i = ngx;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k)
            for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
                int id = F.I(i, j, k);
                F.u[id] = 0.0;
            }
    }
    if (face == XMAX) {
        const int i = ngx + L.nx - 1;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k)
            for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
                int id = F.I(i, j, k);
                F.u[id] = 0.0;
            }
    }
    if (face == YMIN) {
        const int j = ngy;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k)
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                int id = F.I(i, j, k);
                F.v[id] = 0.0;
            }
    }
    if (face == YMAX) {
        const int j = ngy + L.ny - 1;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k)
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                int id = F.I(i, j, k);
                F.v[id] = 0.0;
            }
    }
    if (face == ZMIN) {
        const int k = ngz;
        for (int j = L.ngy; j < L.ngy + L.ny; ++j)
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                int id = F.I(i, j, k);
                F.w[id] = 0.0;
            }
    }
    if (face == ZMAX) {
        const int k = ngz + L.nz - 1;
        for (int j = L.ngy; j < L.ngy + L.ny; ++j)
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                int id = F.I(i, j, k);
                F.w[id] = 0.0;
            }
    }
}

// Outflow boundary condition implementation
// simple zero-gradient extrapolation for all variables
// need to be corrected !!!
void apply_outflow_bc(Field3D &F, const LocalDesc &L, int face)
{
    if (face == XMIN) {
        const int i = L.ngx;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
            for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
                const int idg = F.I(i, j, k);
                F.rho[idg] = F.rho[F.I(i + 1, j, k)];
                F.u[idg] = F.u[F.I(i + 1, j, k)];
                F.v[idg] = F.v[F.I(i + 1, j, k)];
                F.w[idg] = F.w[F.I(i + 1, j, k)];
                F.p[idg] = F.p[F.I(i + 1, j, k)];
            }
        }
    } else if (face == XMAX) {
        const int i = L.ngx + L.nx - 1;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
            for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
                const int idg = F.I(i, j, k);
                F.rho[idg] = F.rho[F.I(i - 1, j, k)];
                F.u[idg] = F.u[F.I(i - 1, j, k)];
                F.v[idg] = F.v[F.I(i - 1, j, k)];
                F.w[idg] = F.w[F.I(i - 1, j, k)];
                F.p[idg] = F.p[F.I(i - 1, j, k)];
            }
        }
    } else if (face == YMIN) {
        const int j = L.ngy;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idg = F.I(i, j, k);
                F.rho[idg] = F.rho[F.I(i, j + 1, k)];
                F.u[idg] = F.u[F.I(i, j + 1, k)];
                F.v[idg] = F.v[F.I(i, j + 1, k)];
                F.w[idg] = F.w[F.I(i, j + 1, k)];
                F.p[idg] = F.p[F.I(i, j + 1, k)];
            }
        }
    } else if (face == YMAX) {
        const int j = L.ngy + L.ny - 1;
        for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idg = F.I(i, j, k);
                F.rho[idg] = F.rho[F.I(i, j - 1, k)];
                F.u[idg] = F.u[F.I(i, j - 1, k)];
                F.v[idg] = F.v[F.I(i, j - 1, k)];
                F.w[idg] = F.w[F.I(i, j - 1, k)];
                F.p[idg] = F.p[F.I(i, j - 1, k)];
            }
        }
    } else if (face == ZMIN) {
        const int k = L.ngz;
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idg = F.I(i, j, k);
                F.rho[idg] = F.rho[F.I(i, j, k + 1)];
                F.u[idg] = F.u[F.I(i, j, k + 1)];
                F.v[idg] = F.v[F.I(i, j, k + 1)];
                F.w[idg] = F.w[F.I(i, j, k + 1)];
                F.p[idg] = F.p[F.I(i, j, k + 1)];
            }
        }
    } else if (face == ZMAX) {
        const int k = L.ngz + L.nz - 1;
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const int idg = F.I(i, j, k);
                F.rho[idg] = F.rho[F.I(i, j, k - 1)];
                F.u[idg] = F.u[F.I(i, j, k - 1)];
                F.v[idg] = F.v[F.I(i, j, k - 1)];
                F.w[idg] = F.w[F.I(i, j, k - 1)];
                F.p[idg] = F.p[F.I(i, j, k - 1)];
            }
        }
    }

    /*
    auto copy=[&](int i1,int j1,int k1,int i2,int j2,int k2){
        int id1=F.I(i1,j1,k1), id2=F.I(i2,j2,k2);
        F.rho[id1]=F.rho[id2];
        F.u[id1]=F.u[id2];
        F.v[id1]=F.v[id2];
        F.w[id1]=F.w[id2];
        F.p[id1]=F.p[id2];
    };

    // copy boundary values from interior (use per-axis ghost counts)
    if (face==XMIN)
        for(int k=0;k<sz;++k) for(int j=0;j<sy;++j) for(int i=0;i<ngx;++i)
            copy(i,j,k, ngx, j, k);

    if (face==XMAX)
        for(int k=0;k<sz;++k) for(int j=0;j<sy;++j) for(int i=sx-ngx;i<sx;++i)
            copy(i,j,k, sx-ngx-1, j, k);

    if (face==YMIN)
        for(int k=0;k<sz;++k) for(int j=0;j<ngy;++j) for(int i=0;i<sx;++i)
            copy(i,j,k, i, ngy, k);

    if (face==YMAX)
        for(int k=0;k<sz;++k) for(int j=sy-ngy;j<sy;++j) for(int i=0;i<sx;++i)
            copy(i,j,k, i, sy-ngy-1, k);

    if (face==ZMIN)
        for(int k=0;k<ngz;++k) for(int j=0;j<sy;++j) for(int i=0;i<sx;++i)
            copy(i,j,k, i, j, ngz);

    if (face==ZMAX)
        for(int k=sz-ngz;k<sz;++k) for(int j=0;j<sy;++j) for(int i=0;i<sx;++i)
            copy(i,j,k, i, j, sz-ngz-1);

    */
}

// Inflow boundary condition implementation
void apply_inflow_bc(Field3D &F, const LocalDesc &L, int face)
{
    int ng=L.ngx, sx=L.sx, sy=L.sy, sz=L.sz;
    double rho0=1.0, u0=1.0, v0=0.0, w0=0.0, p0=1.0, gamma=1.4;

    double E0 = p0/(gamma-1.0) + 0.5*rho0*(u0*u0+v0*v0+w0*w0);
    auto fill=[&](int i,int j,int k){
        int id=F.I(i,j,k);
        F.rho[id]=rho0;
        F.u[id]=u0; F.v[id]=v0; F.w[id]=w0;
        F.p[id]=p0;
        F.E[id]=E0;
    };

    if (face==XMIN) for(int k=0;k<sz;++k)for(int j=0;j<sy;++j)for(int i=0;i<ng;++i) fill(i,j,k);
    if (face==XMAX) for(int k=0;k<sz;++k)for(int j=0;j<sy;++j)for(int i=sx-ng;i<sx;++i) fill(i,j,k);
    if (face==YMIN) for(int k=0;k<sz;++k)for(int j=0;j<ng;++j)for(int i=0;i<sx;++i) fill(i,j,k);
    if (face==YMAX) for(int k=0;k<sz;++k)for(int j=sy-ng;j<sy;++j)for(int i=0;i<sx;++i) fill(i,j,k);
    if (face==ZMIN) for(int k=0;k<ng;++k)for(int j=0;j<sy;++j)for(int i=0;i<sx;++i) fill(i,j,k);
    if (face==ZMAX) for(int k=sz-ng;k<sz;++k)for(int j=0;j<sy;++j)for(int i=0;i<sx;++i) fill(i,j,k);
}

/*
void apply_boundary_halfnode_flux(Field3D &F, const GridDesc &G, CartDecomp &C,
                    const SolverParams &P)
{
    LocalDesc &L = F.L;
    // Step 1: Halo exchange for periodic boundaries
    HaloRequests reqs;
    exchange_halos_halfnode_flux(F, C, L, reqs);

    // Step 2: 对每个方向检查是否需要本地边界
    // Map neighbor -> that side's BC type and FaceID
    NeighborInfo dirs[6] = {
        {L.nbr_xm, P.bc_xmin, XMIN}, {L.nbr_xp, P.bc_xmax, XMAX},
        {L.nbr_ym, P.bc_ymin, YMIN}, {L.nbr_yp, P.bc_ymax, YMAX},
        {L.nbr_zm, P.bc_zmin, ZMIN}, {L.nbr_zp, P.bc_zmax, ZMAX}
    };

    for (auto &d : dirs)
    {
        if (d.nbr != MPI_PROC_NULL) continue; // 有邻居 → 已由通信完成
        switch (d.face)
        {
            case SolverParams::BCType::Wall:
                break;
            case SolverParams::BCType::Symmetry:
                break;
            case SolverParams::BCType::Outflow:
                apply_outflow_bc_halfnode_flux(F, L, d.id);
                break;
            case SolverParams::BCType::Inflow:
                break;
            case SolverParams::BCType::Periodic:
                // 周期边界已由通信处理，无需额外操作
                break;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// Outflow boundary condition implementation
void apply_outflow_bc_halfnode_flux(Field3D &F, const LocalDesc &L, int face)
{
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;
    int sx = L.sx, sy = L.sy, sz = L.sz;

    auto copy_fx=[&](int i1,int j1,int k1,int i2,int j2,int k2){
        int id1=idx_fx(i1,j1,k1,L), id2=idx_fx(i2,j2,k2,L);
        F.flux_fx_mass[id1]=F.flux_fx_mass[id2];
        F.flux_fx_momx[id1]=F.flux_fx_momx[id2];
        F.flux_fx_momy[id1]=F.flux_fx_momy[id2];
        F.flux_fx_momz[id1]=F.flux_fx_momz[id2];
        F.flux_fx_E[id1]=F.flux_fx_E[id2];
    };
    auto copy_fy=[&](int i1,int j1,int k1,int i2,int j2,int k2){
        int id1=idx_fy(i1,j1,k1,L), id2=idx_fy(i2,j2,k2,L);
        F.flux_fy_mass[id1]=F.flux_fy_mass[id2];
        F.flux_fy_momx[id1]=F.flux_fy_momx[id2];
        F.flux_fy_momy[id1]=F.flux_fy_momy[id2];
        F.flux_fy_momz[id1]=F.flux_fy_momz[id2];
        F.flux_fy_E[id1]=F.flux_fy_E[id2];
    };
    auto copy_fz=[&](int i1,int j1,int k1,int i2,int j2,int k2){
        int id1=idx_fz(i1,j1,k1,L), id2=idx_fz(i2,j2,k2,L);
        F.flux_fz_mass[id1]=F.flux_fz_mass[id2];
        F.flux_fz_momx[id1]=F.flux_fz_momx[id2];
        F.flux_fz_momy[id1]=F.flux_fz_momy[id2];
        F.flux_fz_momz[id1]=F.flux_fz_momz[id2];
        F.flux_fz_E[id1]=F.flux_fz_E[id2];
    };

    // copy boundary values from interior (use per-axis ghost counts)
    if (face==XMIN)
        for(int k=0;k<sz;++k) for(int j=0;j<sy;++j) for(int i=0;i<ngx-1;++i)
            copy_fx(i,j,k, ngx-1, j, k);

    if (face==XMAX)
        for(int k=0;k<sz;++k) for(int j=0;j<sy;++j) for(int i=sx-ngx;i<sx-1;++i)
            copy_fx(i,j,k, sx-ngx-1, j, k);

    if (face==YMIN)
        for(int k=0;k<sz;++k) for(int j=0;j<ngy-1;++j) for(int i=0;i<sx;++i)
            copy_fy(i,j,k, i, ngy-1, k);

    if (face==YMAX)
        for(int k=0;k<sz;++k) for(int j=sy-ngy;j<sy-1;++j) for(int i=0;i<sx;++i)
            copy_fy(i,j,k, i, sy-ngy-1, k);

    if (face==ZMIN)
        for(int k=0;k<ngz-1;++k) for(int j=0;j<sy;++j) for(int i=0;i<sx;++i)
            copy_fz(i,j,k, i, j, ngz-1);

    if (face==ZMAX)
        for(int k=sz-ngz;k<sz-1;++k) for(int j=0;j<sy;++j) for(int i=0;i<sx;++i)
            copy_fz(i,j,k, i, j, sz-ngz-1);
}
*/