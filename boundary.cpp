#include "field_structures.h"
#include "ns3d_func.h"
#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

enum FaceID { XMIN=0, XMAX=1, YMIN=2, YMAX=3, ZMIN=4, ZMAX=5 };
struct NeighborInfo { int nbr; SolverParams::BCType face; FaceID id; };
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
                apply_wall_bc(F, L, d.id);
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
    
    MPI_Barrier(MPI_COMM_WORLD);
}

// Wall boundary condition implementation
// No-slip wall 存在问题
void apply_wall_bc(Field3D &F, const LocalDesc &L, int face)
{
    int ng = L.ngx, sx = L.sx, sy = L.sy, sz = L.sz;

    if (face == XMIN)
        for (int k = 0; k < sz; ++k)
        for (int j = 0; j < sy; ++j)
        for (int i = 0; i < ng; ++i)
        {
            int idg = F.I(i, j, k), idm = F.I(2 * ng - 1 - i, j, k);
            F.rho[idg] = F.rho[idm];
            F.u[idg] = -F.u[idm];
            F.v[idg] = F.v[idm];
            F.w[idg] = F.w[idm];
            F.p[idg] = F.p[idm];
        }
    else if (face == XMAX)
        for (int k = 0; k < sz; ++k)
            for (int j = 0; j < sy; ++j)
                for (int i = sx - ng; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(2 * (sx - ng) - 1 - i, j, k);
                    F.rho[idg] = F.rho[idm];
                    F.u[idg] = -F.u[idm];
                    F.v[idg] = F.v[idm];
                    F.w[idg] = F.w[idm];
                    F.p[idg] = F.p[idm];
                }
    else if (face == YMIN)
        for (int k = 0; k < sz; ++k)
            for (int j = 0; j < ng; ++j)
                for (int i = 0; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(i, 2 * ng - 1 - j, k);
                    F.rho[idg] = F.rho[idm];
                    F.v[idg] = -F.v[idm];
                    F.u[idg] = F.u[idm];
                    F.w[idg] = F.w[idm];
                    F.p[idg] = F.p[idm];
                }
    else if (face == YMAX)
        for (int k = 0; k < sz; ++k)
            for (int j = sy - ng; j < sy; ++j)
                for (int i = 0; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(i, 2 * (sy - ng) - 1 - j, k);
                    F.rho[idg] = F.rho[idm];
                    F.v[idg] = -F.v[idm];
                    F.u[idg] = F.u[idm];
                    F.w[idg] = F.w[idm];
                    F.p[idg] = F.p[idm];
                }
    else if (face == ZMIN)
        for (int k = 0; k < ng; ++k)
            for (int j = 0; j < sy; ++j)
                for (int i = 0; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(i, j, 2 * ng - 1 - k);
                    F.rho[idg] = F.rho[idm];
                    F.w[idg] = -F.w[idm];
                    F.u[idg] = F.u[idm];
                    F.v[idg] = F.v[idm];
                    F.p[idg] = F.p[idm];
                }
    else if (face == ZMAX)
        for (int k = sz - ng; k < sz; ++k)
            for (int j = 0; j < sy; ++j)
                for (int i = 0; i < sx; ++i)
                {
                    int idg = F.I(i, j, k), idm = F.I(i, j, 2 * (sz - ng) - 1 - k);
                    F.rho[idg] = F.rho[idm];
                    F.w[idg] = -F.w[idm];
                    F.u[idg] = F.u[idm];
                    F.v[idg] = F.v[idm];
                    F.p[idg] = F.p[idm];
                }
}

// Symmetry boundary condition implementation
void apply_symmetry_bc(Field3D &F, const LocalDesc &L, int face)
{
    // 实际上与 apply_wall_bc 相同，可以直接调用
    apply_wall_bc(F, L, face);
}

// Outflow boundary condition implementation
void apply_outflow_bc(Field3D &F, const LocalDesc &L, int face)
{
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;
    int sx = L.sx, sy = L.sy, sz = L.sz;

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