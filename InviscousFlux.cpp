#include "ns3d_func.h"
#include <fstream>
#include <cstdlib>

namespace {

inline bool use_one_sided_face_x(int i, const LocalDesc &L)
{
    if (L.nx < 5) return false;
    if (L.nbr_xm == MPI_PROC_NULL && (i == L.ngx - 1 || i == L.ngx || i == L.ngx + 1)) return true;
    if (L.nbr_xp == MPI_PROC_NULL) {
        const int i1 = L.ngx + L.nx - 1;
        if (i == i1 - 2 || i == i1 - 1 || i == i1) return true;
    }
    return false;
}

inline bool use_one_sided_face_y(int j, const LocalDesc &L)
{
    if (L.ny < 5) return false;
    if (L.nbr_ym == MPI_PROC_NULL && (j == L.ngy - 1 || j == L.ngy || j == L.ngy + 1)) return true;
    if (L.nbr_yp == MPI_PROC_NULL) {
        const int j1 = L.ngy + L.ny - 1;
        if (j == j1 - 2 || j == j1 - 1 || j == j1) return true;
    }
    return false;
}

inline bool use_one_sided_face_z(int k, const LocalDesc &L)
{
    if (L.nz < 5) return false;
    if (L.nbr_zm == MPI_PROC_NULL && (k == L.ngz - 1 || k == L.ngz || k == L.ngz + 1)) return true;
    if (L.nbr_zp == MPI_PROC_NULL) {
        const int k1 = L.ngz + L.nz - 1;
        if (k == k1 - 2 || k == k1 - 1 || k == k1) return true;
    }
    return false;
}

inline double interp_half_scalar_x_one_sided(const std::vector<double> &a, int i, int j, int k, const LocalDesc &L)
{
    const int i0 = 0;
    const int i1 = L.sx - 2;
    if (L.nbr_xm == MPI_PROC_NULL) {
        if (i == i0) {
            return (1.0 / 128.0) * (315.0 * a[idx3(i0 + 1, j, k, L)] - 420.0 * a[idx3(i0 + 2, j, k, L)]
				+ 378.0 * a[idx3(i0 + 3, j, k, L)] - 180.0 * a[idx3(i0 + 4, j, k, L)]
				 + 35.0 * a[idx3(i0 + 5, j, k, L)]);
        }
        if (i == i0 + 1) {
            return (1.0 / 128.0) * (35.0 * a[idx3(i0 + 1, j, k, L)] + 140.0 * a[idx3(i0 + 2, j, k, L)]
                - 70.0 * a[idx3(i0 + 3, j, k, L)] + 28.0 * a[idx3(i0 + 4, j, k, L)]
                - 5.0 * a[idx3(i0 + 5, j, k, L)]);
        }
        if (i == i0 + 2) {
            return (1.0 / 128.0) * (- 5.0 * a[idx3(i0 + 1, j, k, L)] + 60.0 * a[idx3(i0 + 2, j, k, L)]
                + 90.0 * a[idx3(i0 + 3, j, k, L)] - 20.0 * a[idx3(i0 + 4, j, k, L)]
                + 3.0 * a[idx3(i0 + 5, j, k, L)]);
        }
    }
    if (L.nbr_xp == MPI_PROC_NULL) {
        if(i == i1) {
            return (1.0 / 128.0) * (315.0 * a[idx3(i1, j, k, L)] - 420.0 * a[idx3(i1 - 1, j, k, L)]
                + 378.0 * a[idx3(i1 - 2, j, k, L)] - 180.0 * a[idx3(i1 - 3, j, k, L)]
                + 35.0 * a[idx3(i1 - 4, j, k, L)]);
        }
        if(i == i1 - 1) {
            return (1.0 / 128.0) * (35.0 * a[idx3(i1, j, k, L)] + 140.0 * a[idx3(i1 - 1, j, k, L)]
                - 70.0 * a[idx3(i1 - 2, j, k, L)] + 28.0 * a[idx3(i1 - 3, j, k, L)]
                - 5.0 * a[idx3(i1 - 4, j, k, L)]);
        }
        if(i == i1 - 2) {
            return (1.0 / 128.0) * (- 5.0 * a[idx3(i1, j, k, L)] + 60.0 * a[idx3(i1 - 1, j, k, L)] 
                + 90.0 * a[idx3(i1 - 2, j, k, L)]- 20.0 * a[idx3(i1 - 3, j, k, L)] + 3.0 * a[idx3(i1 - 4, j, k, L)]);
        }
    }
    std::cerr << "Error: interp_half_scalar_x_one_sided called with i=" << i << " which does not require one-sided interpolation." << std::endl;
    return 0.0; // should not reach here
}

inline double interp_half_scalar_y_one_sided(const std::vector<double> &a, int i, int j, int k, const LocalDesc &L)
{
    const int j0 = 0;
    const int j1 = L.sy - 2;
    if(L.nbr_ym == MPI_PROC_NULL) {
        if (j == j0) {
            return (1.0 / 128.0) * (315.0 * a[idx3(i, j0 + 1, k, L)] - 420.0 * a[idx3(i, j0 + 2, k, L)]
                + 378.0 * a[idx3(i, j0 + 3, k, L)] - 180.0 * a[idx3(i, j0 + 4, k, L)]
                + 35.0 * a[idx3(i, j0 + 5, k, L)]);
        }
        if (j == j0 + 1) {
            return (1.0 / 128.0) * (35.0 * a[idx3(i, j0 + 1, k, L)] + 140.0 * a[idx3(i, j0 + 2, k, L)]
                - 70.0 * a[idx3(i, j0 + 3, k, L)] + 28.0 * a[idx3(i, j0 + 4, k, L)]
                - 5.0 * a[idx3(i, j0 + 5, k, L)]);
        }
        if (j == j0 + 2) {
            return (1.0 / 128.0) * (- 5.0 * a[idx3(i, j0 + 1, k, L)] + 60.0 * a[idx3(i, j0 + 2, k, L)]
                + 90.0 * a[idx3(i, j0 + 3, k, L)] - 20.0 * a[idx3(i, j0 + 4, k, L)]
                + 3.0 * a[idx3(i, j0 + 5, k, L)]);
        }
    }
    if (L.nbr_yp == MPI_PROC_NULL) {
        if (j == j1) {
            return (1.0 / 128.0) * (315.0 * a[idx3(i, j1, k, L)] - 420.0 * a[idx3(i, j1 - 1, k, L)] 
            + 378.0 * a[idx3(i, j1 - 2, k, L)] - 180.0 * a[idx3(i, j1 - 3, k, L)] + 35.0 * a[idx3(i, j1 - 4, k, L)]);
        }
        if (j == j1 - 1) {
            return (1.0 / 128.0) * (35.0 * a[idx3(i, j1, k, L)] + 140.0 * a[idx3(i, j1 - 1, k, L)]
                - 70.0 * a[idx3(i, j1 - 2, k, L)] + 28.0 * a[idx3(i, j1 - 3, k, L)]
                - 5.0 * a[idx3(i, j1 - 4, k, L)]);
        }
        if (j == j1 - 2) {
            return (1.0 / 128.0) * (- 5.0 * a[idx3(i, j1, k, L)] + 60.0 * a[idx3(i, j1 - 1, k, L)] + 90.0 * a[idx3(i, j1 - 2, k, L)]
                - 20.0 * a[idx3(i, j1 - 3, k, L)] + 3.0 * a[idx3(i, j1 - 4, k, L)]);
        }
    }
    std::cerr << "Error: interp_half_scalar_y_one_sided called with j=" << j << " which does not require one-sided interpolation." << std::endl;
    return 0.0; // should not reach here
}

inline double interp_half_scalar_z_one_sided(const std::vector<double> &a, int i, int j, int k, const LocalDesc &L)
{
    const int k0 = 0;
    const int k1 = L.sz - 2;
    if(L.nbr_zm == MPI_PROC_NULL) {
        if (k == k0) {
            return (1.0 / 128.0) * (315.0 * a[idx3(i, j, k0 + 1, L)] - 420.0 * a[idx3(i, j, k0 + 2, L)]
                + 378.0 * a[idx3(i, j, k0 + 3, L)] - 180.0 * a[idx3(i, j, k0 + 4, L)]
                + 35.0 * a[idx3(i, j, k0 + 5, L)]);
        }
        if (k == k0 + 1) {
            return (1.0 / 128.0) * (35.0 * a[idx3(i, j, k0 + 1, L)] + 140.0 * a[idx3(i, j, k0 + 2, L)]
                - 70.0 * a[idx3(i, j, k0 + 3, L)] + 28.0 * a[idx3(i, j, k0 + 4, L)]
                - 5.0 * a[idx3(i, j, k0 + 5, L)]);
        }
        if (k == k0 + 2) {
            return (1.0 / 128.0) * (- 5.0 * a[idx3(i, j, k0 + 1, L)] + 60.0 * a[idx3(i, j, k0 + 2, L)]
                + 90.0 * a[idx3(i, j, k0 + 3, L)] - 20.0 * a[idx3(i, j, k0 + 4, L)]
                + 3.0 * a[idx3(i, j, k0 + 5, L)]);
        }
    }
    if (L.nbr_zp == MPI_PROC_NULL) {
        if (k == k1) {
            return (1.0 / 128.0) * (315.0 * a[idx3(i, j, k1, L)] - 420.0 * a[idx3(i, j, k1 - 1, L)] 
            + 378.0 * a[idx3(i, j, k1 - 2, L)] - 180.0 * a[idx3(i, j, k1 - 3, L)] + 35.0 * a[idx3(i, j, k1 - 4, L)]);
        }
        if (k == k1 - 1) {
            return (1.0 / 128.0) * (35.0 * a[idx3(i, j, k1, L)] + 140.0 * a[idx3(i, j, k1 - 1, L)]
                - 70.0 * a[idx3(i, j, k1 - 2, L)] + 28.0 * a[idx3(i, j, k1 - 3, L)]
                - 5.0 * a[idx3(i, j, k1 - 4, L)]);
        }
        if (k == k1 - 2) {
            return (1.0 / 128.0) * (- 5.0 * a[idx3(i, j, k1, L)] + 60.0 * a[idx3(i, j, k1 - 1, L)] + 90.0 * a[idx3(i, j, k1 - 2, L)]
                - 20.0 * a[idx3(i, j, k1 - 3, L)] + 3.0 * a[idx3(i, j, k1 - 4, L)]);
        }
    }
    std::cerr << "Error: interp_half_scalar_z_one_sided called with k=" << k << " which does not require one-sided interpolation." << std::endl;
    return 0.0; // should not reach here
}

inline void build_face_state_U(const Field3D &F, int i, int j, int k, char dir, std::array<double,5> &Uf)
{
    const LocalDesc &L = F.L;
    if (dir == 'x') {
        Uf[0] = interp_half_scalar_x_one_sided(F.rho,  i, j, k, L);
        Uf[1] = interp_half_scalar_x_one_sided(F.rhou, i, j, k, L);
        Uf[2] = interp_half_scalar_x_one_sided(F.rhov, i, j, k, L);
        Uf[3] = interp_half_scalar_x_one_sided(F.rhow, i, j, k, L);
        Uf[4] = interp_half_scalar_x_one_sided(F.E,    i, j, k, L);
    } else if (dir == 'y') {
        Uf[0] = interp_half_scalar_y_one_sided(F.rho,  i, j, k, L);
        Uf[1] = interp_half_scalar_y_one_sided(F.rhou, i, j, k, L);
        Uf[2] = interp_half_scalar_y_one_sided(F.rhov, i, j, k, L);
        Uf[3] = interp_half_scalar_y_one_sided(F.rhow, i, j, k, L);
        Uf[4] = interp_half_scalar_y_one_sided(F.E,    i, j, k, L);
    } else {
        Uf[0] = interp_half_scalar_z_one_sided(F.rho,  i, j, k, L);
        Uf[1] = interp_half_scalar_z_one_sided(F.rhou, i, j, k, L);
        Uf[2] = interp_half_scalar_z_one_sided(F.rhov, i, j, k, L);
        Uf[3] = interp_half_scalar_z_one_sided(F.rhow, i, j, k, L);
        Uf[4] = interp_half_scalar_z_one_sided(F.E,    i, j, k, L);
    }
}

inline void inviscid_flux_from_U(std::vector<double> &Fface,
                                 const std::array<double,5> &U,
                                 double nx, double ny, double nz,
                                 double gamma)
{
    const double rho = std::max(U[0], 1e-12);
    const double u = U[1] / rho;
    const double v = U[2] / rho;
    const double w = U[3] / rho;
    const double kinetic = 0.5 * rho * (u*u + v*v + w*w);
    const double p = std::max((gamma - 1.0) * (U[4] - kinetic), 1e-12);
    const double vn = u*nx + v*ny + w*nz;
    Fface[0] = rho * vn;
    Fface[1] = rho * u * vn + p * nx;
    Fface[2] = rho * v * vn + p * ny;
    Fface[3] = rho * w * vn + p * nz;
    Fface[4] = (U[4] + p) * vn;
}

} // namespace

// -----------------------------------------------------------------
// ---------   Flux Vector Splitting (FVS) -------------------------
// -----------------------------------------------------------------

// 边界和近边界面通量
void compute_invis_flux_boundary(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    int nx = L.nx, ny = L.ny, nz = L.nz;
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;
    int sz = L.sz, sy = L.sy, sx = L.sx;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    
    // i方向边界面
    for (int k = ngz; k < ngz+nz; ++k) {
    for (int j = ngy; j < ngy+ny; ++j) {
        if(L.nbr_xm == MPI_PROC_NULL) {
            for (int i = 0; i < ngx; ++i) {
                std::array<double, VAR> Uf;
                std::vector<double> Fface(VAR, 0.0);
                build_face_state_U(F, i, j, k, 'x', Uf);
                int fid = idx_fx(i, j, k, L);
                double nx_face = F.xi_x_fx[fid], ny_face = F.xi_y_fx[fid], nz_face = F.xi_z_fx[fid];
                inviscid_flux_from_U(Fface, Uf, nx_face, ny_face, nz_face, P.gamma);
                F.flux_fx_mass[fid] = Fface[0];
                F.flux_fx_momx[fid] = Fface[1];
                F.flux_fx_momy[fid] = Fface[2];
                F.flux_fx_momz[fid] = Fface[3];
                F.flux_fx_E[fid]    = Fface[4];
            }
        }
        if(L.nbr_xp == MPI_PROC_NULL) {
            for (int i = ngx + nx - 1; i < sx - 1; ++i) {
                std::array<double, VAR> Uf;
                std::vector<double> Fface(VAR, 0.0);
                build_face_state_U(F, i, j, k, 'x', Uf);
                int fid = idx_fx(i, j, k, L);
                double nx_face = F.xi_x_fx[fid], ny_face = F.xi_y_fx[fid], nz_face = F.xi_z_fx[fid];
                inviscid_flux_from_U(Fface, Uf, nx_face, ny_face, nz_face, P.gamma);
                F.flux_fx_mass[fid] = Fface[0];
                F.flux_fx_momx[fid] = Fface[1];
                F.flux_fx_momy[fid] = Fface[2];
                F.flux_fx_momz[fid] = Fface[3];
                F.flux_fx_E[fid]    = Fface[4];
            }
        }
    }}

    // j方向边界面
    for (int k = ngz; k < ngz+nz; ++k) {
    for (int i = ngx; i < ngx+nx; ++i) {
        if(L.nbr_ym == MPI_PROC_NULL) {
            for (int j = 0; j < ngy; ++j) {
                std::array<double, VAR> Uf;
                std::vector<double> Fface(VAR, 0.0);
                build_face_state_U(F, i, j, k, 'y', Uf);
                int fid = idx_fy(i, j, k, L);
                double nx_face = F.eta_x_fy[fid], ny_face = F.eta_y_fy[fid], nz_face = F.eta_z_fy[fid];
                inviscid_flux_from_U(Fface, Uf, nx_face, ny_face, nz_face, P.gamma);
                F.flux_fy_mass[fid] = Fface[0];
                F.flux_fy_momx[fid] = Fface[1];
                F.flux_fy_momy[fid] = Fface[2];
                F.flux_fy_momz[fid] = Fface[3];
                F.flux_fy_E[fid]    = Fface[4];
            }
        }
        if(L.nbr_yp == MPI_PROC_NULL) {
            for (int j = ngy + ny - 1; j < sy - 1; ++j) {
                std::array<double, VAR> Uf;
                std::vector<double> Fface(VAR, 0.0);
                build_face_state_U(F, i, j, k, 'y', Uf);
                int fid = idx_fy(i, j, k, L);
                double nx_face = F.eta_x_fy[fid], ny_face = F.eta_y_fy[fid], nz_face = F.eta_z_fy[fid];
                inviscid_flux_from_U(Fface, Uf, nx_face, ny_face, nz_face, P.gamma);
                F.flux_fy_mass[fid] = Fface[0];
                F.flux_fy_momx[fid] = Fface[1];
                F.flux_fy_momy[fid] = Fface[2];
                F.flux_fy_momz[fid] = Fface[3];
                F.flux_fy_E[fid]    = Fface[4];
            }
        }
    }}

    // z方向边界面
    for (int j = ngy; j < ngy+ny; ++j) {
    for (int i = ngx; i < ngx+nx; ++i) {
        if(L.nbr_zm == MPI_PROC_NULL) {
            for (int k = 0; k < ngz; ++k) {
                std::array<double, VAR> Uf;
                std::vector<double> Fface(VAR, 0.0);
                build_face_state_U(F, i, j, k, 'z', Uf);
                int fid = idx_fz(i, j, k, L);
                double nx_face = F.zeta_x_fz[fid], ny_face = F.zeta_y_fz[fid], nz_face = F.zeta_z_fz[fid];
                inviscid_flux_from_U(Fface, Uf, nx_face, ny_face, nz_face, P.gamma);
                F.flux_fz_mass[fid] = Fface[0];
                F.flux_fz_momx[fid] = Fface[1];
                F.flux_fz_momy[fid] = Fface[2];
                F.flux_fz_momz[fid] = Fface[3];
                F.flux_fz_E[fid]    = Fface[4];
            }
        }
        if(L.nbr_zp == MPI_PROC_NULL) {
            for (int k = ngz + nz - 1; k < sz - 1; ++k) {
                std::array<double, VAR> Uf;
                std::vector<double> Fface(VAR, 0.0);
                build_face_state_U(F, i, j, k, 'z', Uf);
                int fid = idx_fz(i, j, k, L);
                double nx_face = F.zeta_x_fz[fid], ny_face = F.zeta_y_fz[fid], nz_face = F.zeta_z_fz[fid];
                inviscid_flux_from_U(Fface, Uf, nx_face, ny_face, nz_face, P.gamma);
                F.flux_fz_mass[fid] = Fface[0];
                F.flux_fz_momx[fid] = Fface[1];
                F.flux_fz_momy[fid] = Fface[2];
                F.flux_fz_momz[fid] = Fface[3];
                F.flux_fz_E[fid]    = Fface[4];
            }
        }
    }}
}

// 采用wcns方法计算无粘通量导数
void compute_invis_flux(Field3D &F, const SolverParams &P, const CartDecomp &C)
{
    const LocalDesc &L = F.L;
    int nx = L.nx, ny = L.ny, nz = L.nz;
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;
    int sz = L.sz, sy = L.sy, sx = L.sx;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    int stencil = P.stencil;
    
    if (stencil < 2) {
        std::cerr << "Stencil must be >= 2\n";
        return;
    }
    
    // center offset for mapping stencil indices m -> cell indices ii
    // use (stencil-1)/2 so that for even stencil (e.g. 6) m indices map to i-2..i+3
    int mid = (stencil - 1) / 2;
    // quick sanity: require domain size to contain stencil
    if (sx < stencil || sy < stencil || sz < stencil) {
        std::cerr << "computeFVSFluxes: local array too small for stencil\n";
        return;
    }

    // i方向通量重构(xi方向)
    for (int k = ngz; k < ngz+nz; ++k) {
    for (int j = ngy; j < ngy+ny; ++j) {
    for (int i = ngx - 1; i < ngx + nx; ++i) {
        std::vector<double> Fface(VAR, 0.0);
        int fid = idx_fx(i, j, k, L);
        double nx_face = F.xi_x_fx[fid], ny_face = F.xi_y_fx[fid], nz_face = F.xi_z_fx[fid]; // i方向法向量

        // dynamic 2D arrays: VAR x stencil
        std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
        std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

        for (int m = 0; m < stencil; ++m) {
            int ii = i + (m - mid); // 以i为中心的stencil(6点模板为i-2到i+3) when mid=(stencil-1)/2
            int id = F.I(ii, j, k);

            Ut[0][m] = F.rho[id];
            Ut[1][m] = F.rhou[id];
            Ut[2][m] = F.rhov[id];
            Ut[3][m] = F.rhow[id];
            Ut[4][m] = F.E[id];
            ut[0][m] = F.rho[id];
            ut[1][m] = F.u[id];
            ut[2][m] = F.v[id];
            ut[3][m] = F.w[id];
            ut[4][m] = F.p[id];
        }

        WCNS_Riemann_InviscidFlux(Fface, Ut, ut, P, nx_face, ny_face, nz_face);

                
        F.flux_fx_mass[fid] = Fface[0];
        F.flux_fx_momx[fid] = Fface[1];
        F.flux_fx_momy[fid] = Fface[2];
        F.flux_fx_momz[fid] = Fface[3];
        F.flux_fx_E[fid]    = Fface[4];
    }}}

    // j方向通量重构(eta方向)
    for (int k = ngz; k < ngz+nz; ++k) {
        for (int i = ngx; i < ngx+nx; ++i) {
            for (int j = ngy - 1; j < ngy + ny; ++j) {
                std::vector<double> Fface(VAR, 0.0);
                int fid = idx_fy(i, j, k, L);
                double nx_face = F.eta_x_fy[fid], ny_face = F.eta_y_fy[fid], nz_face = F.eta_z_fy[fid]; // j方向法向量

                // dynamic 2D arrays: VAR x stencil
                std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

                for (int m = 0; m < stencil; ++m) {
                    int jj = j + (m - mid); // 以j为中心的stencil(6点模板为j-2到j+3) when mid=(stencil-1)/2
                    int id = F.I(i, jj, k);

                    Ut[0][m] = F.rho[id];
                    Ut[1][m] = F.rhou[id];
                    Ut[2][m] = F.rhov[id];
                    Ut[3][m] = F.rhow[id];
                    Ut[4][m] = F.E[id];
                    ut[0][m] = F.rho[id];
                    ut[1][m] = F.u[id];
                    ut[2][m] = F.v[id];
                    ut[3][m] = F.w[id];
                    ut[4][m] = F.p[id];
                }

                WCNS_Riemann_InviscidFlux(Fface, Ut, ut, P, nx_face, ny_face, nz_face);

                F.flux_fy_mass[fid] = Fface[0];
                F.flux_fy_momx[fid] = Fface[1];
                F.flux_fy_momy[fid] = Fface[2];
                F.flux_fy_momz[fid] = Fface[3];
                F.flux_fy_E[fid]    = Fface[4];
            }
        }
    }

    // k方向通量重构(zeta方向)
    for (int i = ngx; i < ngx+nx; ++i) {
        for (int j = ngy; j < ngy+ny; ++j) {
            for (int k = ngz - 1; k < ngz + nz; ++k) {
                std::vector<double> Fface(VAR, 0.0);
                int fid = idx_fz(i, j, k, L);
                double nx_face = F.zeta_x_fz[fid], ny_face = F.zeta_y_fz[fid], nz_face = F.zeta_z_fz[fid]; // k方向法向量

                // dynamic 2D arrays: VAR x stencil
                std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

                for (int m = 0; m < stencil; ++m) {
                    int kk = k + (m - mid); // 以k为中心的stencil(6点模板为k-2到k+3) when mid=(stencil-1)/2
                    int id = F.I(i, j, kk);

                    Ut[0][m] = F.rho[id];
                    Ut[1][m] = F.rhou[id];
                    Ut[2][m] = F.rhov[id];
                    Ut[3][m] = F.rhow[id];
                    Ut[4][m] = F.E[id];
                    ut[0][m] = F.rho[id];
                    ut[1][m] = F.u[id];
                    ut[2][m] = F.v[id];
                    ut[3][m] = F.w[id];
                    ut[4][m] = F.p[id];
                }

                WCNS_Riemann_InviscidFlux(Fface, Ut, ut, P, nx_face, ny_face, nz_face);


                F.flux_fz_mass[fid] = Fface[0];
                F.flux_fz_momx[fid] = Fface[1];
                F.flux_fz_momy[fid] = Fface[2];
                F.flux_fz_momz[fid] = Fface[3];
                F.flux_fz_E[fid]    = Fface[4];
            }
        }
    }

    // 计算无粘通量后立即交换半节点的无粘通量，以便边界差分使用
    exchange_half_halo_x(F.flux_fx_mass, L, C, L.ngx, 500);
    exchange_half_halo_x(F.flux_fx_momx, L, C, L.ngx, 510);
    exchange_half_halo_x(F.flux_fx_momy, L, C, L.ngx, 520);
    exchange_half_halo_x(F.flux_fx_momz, L, C, L.ngx, 530);
    exchange_half_halo_x(F.flux_fx_E,    L, C, L.ngx, 540);
    exchange_half_halo_y(F.flux_fy_mass, L, C, L.ngy, 545);
    exchange_half_halo_y(F.flux_fy_momx, L, C, L.ngy, 550);
    exchange_half_halo_y(F.flux_fy_momy, L, C, L.ngy, 560);
    exchange_half_halo_y(F.flux_fy_momz, L, C, L.ngy, 570);
    exchange_half_halo_y(F.flux_fy_E,    L, C, L.ngy, 580);
    exchange_half_halo_z(F.flux_fz_mass, L, C, L.ngz, 585);
    exchange_half_halo_z(F.flux_fz_momx, L, C, L.ngz, 590);
    exchange_half_halo_z(F.flux_fz_momy, L, C, L.ngz, 595);
    exchange_half_halo_z(F.flux_fz_momz, L, C, L.ngz, 600);
    exchange_half_halo_z(F.flux_fz_E,    L, C, L.ngz, 605);
}

void WCNS_Riemann_InviscidFlux(std::vector<double> &Fface,
                             const std::vector<std::vector<double>> &Ut,
                             const std::vector<std::vector<double>> &ut,
                             const SolverParams &P, double nx, double ny, double nz)
{
    // alias
    double gamma = P.gamma;
    bool sigma = P.char_recon;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    int stencil = P.stencil;

    // 1) Interpolation steps:
    std::vector<double> UL(VAR, 0.0);
    std::vector<double> UR(VAR, 0.0);
    if(sigma)
    {
        // characteristic-wise interpolation
        // 1.a) compute Roe-averaged state from leftmost and rightmost states in stencil
        std::vector<std::vector<double>> wchar(VAR, std::vector<double>(stencil));
        std::vector<std::vector<double>> LU(VAR, std::vector<double>(stencil));

        double Lmat[VAR][VAR], Rmat[VAR][VAR], lambar[VAR];
        const double ul_L[5] = { ut[0][2], ut[1][2], ut[2][2], ut[3][2], ut[4][2] };
        const double ur_L[5] = { ut[0][2], ut[1][2], ut[2][2], ut[3][2], ut[4][2] };
        build_eigen_matrices(ul_L, ur_L, nx, ny, nz, gamma, Lmat, Rmat, lambar);
        for (int m = 0; m < stencil; ++m) {
            for (int n = 0; n < VAR; ++n) {
                double sumLU = 0.0;
                for (int r = 0; r < VAR; ++r) {
                    sumLU += Lmat[n][r] * Ut[r][m];
                }
                LU[n][m] = sumLU;
            }
        }

        std::vector<double> Q_char(VAR, 0.0);
        for (int n = 0; n < VAR; ++n) {
            std::vector<double> Qt(stencil);
            for (int m = 0; m < stencil; ++m) {
                Qt[m] = LU[n][m];
            }
            Q_char[n] = interpolate_select(Qt, +1.0, P);
        }

        // transform back to conservative flux via Fflux = R * wflux_char
        for (int n = 0; n < VAR; ++n) {
            double sum = 0.0;
            for (int r = 0; r < VAR; ++r) sum += Rmat[n][r] * Q_char[r];
            UL[n] = sum;
        }

        // repeat for right state
        const double ul_R[5] = { ut[0][3], ut[1][3], ut[2][3], ut[3][3], ut[4][3] };
        const double ur_R[5] = { ut[0][3], ut[1][3], ut[2][3], ut[3][3], ut[4][3] };
        build_eigen_matrices(ul_R, ur_R, nx, ny, nz, gamma, Lmat, Rmat, lambar); // 这里有冗余，WCNS需要采用所在单元的特征矩阵
        for (int m = 0; m < stencil; ++m) {
            for (int n = 0; n < VAR; ++n) {
                double sumLU = 0.0;
                for (int r = 0; r < VAR; ++r) {
                    sumLU += Lmat[n][r] * Ut[r][m];
                }
                LU[n][m] = sumLU;
            }
        }

        for (int n = 0; n < VAR; ++n) {
            std::vector<double> Qt(stencil);
            for (int m = 0; m < stencil; ++m) {
                Qt[m] = LU[n][m];
            }
            Q_char[n] = interpolate_select(Qt, -1.0, P);
        }

        // transform back to conservative flux via Fflux = R * wflux_char
        for (int n = 0; n < VAR; ++n) {
            double sum = 0.0;
            for (int r = 0; r < VAR; ++r) sum += Rmat[n][r] * Q_char[r];
            UR[n] = sum;
        }
    }
    else
    {
        // component-wise interpolation
        for (int n = 0; n < VAR; ++n) {
            UL[n] = interpolate_select(Ut[n], +1.0, P);
            UR[n] = interpolate_select(Ut[n], -1.0, P);
        }
    }

    // 2) Riemann solver to get Fface from UL, UR
    std::vector<double> FL(VAR, 0.0), FR(VAR, 0.0);

    // Riemann solver
    switch (P.riemann_solver) {
        case SolverParams::RiemannSolver::Roe:
            Roe_Riemann_solver(Fface, UL, UR, nx, ny, nz, gamma);
            break;
        case SolverParams::RiemannSolver::Rusanov:
            Rusanov_Riemann_solver(Fface, UL, UR, nx, ny, nz, gamma);
            break;
        case SolverParams::RiemannSolver::HLLC:
            HLLC_Riemann_solver(Fface, UL, UR, nx, ny, nz, gamma);
            break;
        case SolverParams::RiemannSolver::HLL:
            HLL_Riemann_solver(Fface, UL, UR, nx, ny, nz, gamma);
            break;
        case SolverParams::RiemannSolver::HLLC_p:
            HLLC_p_Riemann_solver(Fface, UL, UR, nx, ny, nz, gamma);
            break;
        case SolverParams::RiemannSolver::AUSM:
            AUSM_Riemann_solver(Fface, UL, UR, nx, ny, nz, gamma);
            break;
        default:
            std::cerr << "Unknown Riemann solver\n";
            break;
    }
}

void Rusanov_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma)
{
    double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    // Rusanov (Local Lax-Friedrichs) Riemann solver
    double rho_L = UL[0];
    double u_L = UL[1]/rho_L;
    double v_L = UL[2]/rho_L;
    double w_L = UL[3]/rho_L;
    double p_L = (UL[4] - 0.5*rho_L*(u_L*u_L + v_L*v_L + w_L*w_L)) * (gamma - 1.0);
    double a_L = std::sqrt(gamma * p_L / rho_L);
    double rho_R = UR[0];
    double u_R = UR[1]/rho_R;
    double v_R = UR[2]/rho_R;
    double w_R = UR[3]/rho_R;
    double p_R = (UR[4] - 0.5*rho_R*(u_R*u_R + v_R*v_R + w_R*w_R)) * (gamma - 1.0);
    double a_R = std::sqrt(gamma * p_R / rho_R);

    // fluxes
    double FL[5], FR[5];
    double V_n = u_L*nx + v_L*ny + w_L*nz;
    FL[0] = rho_L * V_n;
    FL[1] = rho_L * u_L * V_n + p_L * nx;
    FL[2] = rho_L * v_L * V_n + p_L * ny;
    FL[3] = rho_L * w_L * V_n + p_L * nz;
    FL[4] = (UL[4] + p_L) * V_n;
    V_n = u_R*nx + v_R*ny + w_R*nz;
    FR[0] = rho_R * V_n;
    FR[1] = rho_R * u_R * V_n + p_R * nx;
    FR[2] = rho_R * v_R * V_n + p_R * ny;
    FR[3] = rho_R * w_R * V_n + p_R * nz;
    FR[4] = (UR[4] + p_R) * V_n;

    // 最大特征值
    double smax = std::max( std::abs(u_L*nx + v_L*ny + w_L*nz) + a_L * norm,
                            std::abs(u_R*nx + v_R*ny + w_R*nz) + a_R * norm );
    // compute flux
    for (int n = 0; n < 5; ++n) {
        Fface[n] = 0.5 * (FL[n] + FR[n]) - 0.5 * smax * (UR[n] - UL[n]);
    }
}

void Roe_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma)
{
    // Roe Riemann solver
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    double rho_L = UL[0];
    double u_L = UL[1]/rho_L;
    double v_L = UL[2]/rho_L;
    double w_L = UL[3]/rho_L;
    double p_L = (UL[4] - 0.5*rho_L*(u_L*u_L + v_L*v_L + w_L*w_L)) * (gamma - 1.0);
    double a_L = std::sqrt(gamma * p_L / rho_L);
    double rho_R = UR[0];
    double u_R = UR[1]/rho_R;
    double v_R = UR[2]/rho_R;
    double w_R = UR[3]/rho_R;
    double p_R = (UR[4] - 0.5*rho_R*(u_R*u_R + v_R*v_R + w_R*w_R)) * (gamma - 1.0);
    double a_R = std::sqrt(gamma * p_R / rho_R);

    // flux
    double FL[5], FR[5];
    double V_n = u_L*nx + v_L*ny + w_L*nz;
    FL[0] = rho_L * V_n;
    FL[1] = rho_L * u_L * V_n + p_L * nx;
    FL[2] = rho_L * v_L * V_n + p_L * ny;
    FL[3] = rho_L * w_L * V_n + p_L * nz;
    FL[4] = (UL[4] + p_L) * V_n;
    V_n = u_R*nx + v_R*ny + w_R*nz;
    FR[0] = rho_R * V_n;
    FR[1] = rho_R * u_R * V_n + p_R * nx;
    FR[2] = rho_R * v_R * V_n + p_R * ny;
    FR[3] = rho_R * w_R * V_n + p_R * nz;
    FR[4] = (UR[4] + p_R) * V_n;

    // Roe average states
    double rho_bar, u_bar, v_bar, w_bar, h_bar, a_bar;
    const double ul[5] = {rho_L,u_L,v_L,w_L,p_L};
    const double ur[5] = {rho_R,u_R,v_R,w_R,p_R};
    double Lmat[VAR][VAR], Rmat[VAR][VAR], lambar[VAR];
    build_eigen_matrices(ul, ur, nx, ny, nz, gamma, Lmat, Rmat, lambar);

    // 计算波强度
    double alpha[VAR];
    for (int m = 0; m < VAR; ++m) {
        alpha[m] = 0.0;
        for (int n = 0; n < VAR; ++n) {
            alpha[m] += Lmat[m][n] * (UR[n] - UL[n]);
        }
    }

    // entropy fix for eigenvalues
    // Use a robust local threshold based on eigenvalue magnitude.
    double max_abs_lambda = 0.0;
    for (int m = 0; m < VAR; ++m) max_abs_lambda = std::max(max_abs_lambda, std::abs(lambar[m]));
    double delta = 0.1 * max_abs_lambda;
    for (int m = 0; m < VAR; ++m) {
        if (std::abs(lambar[m]) < delta) {
            lambar[m] = (lambar[m]*lambar[m] + delta*delta) / (2.0*delta);
        }   
    }

    // compute flux
    for (int n = 0; n < VAR; ++n) {
        Fface[n] = 0.5 * (FL[n] + FR[n]);
        for (int m = 0; m < VAR; ++m) {
            Fface[n] -= 0.5 * std::abs(lambar[m]) * alpha[m] * Rmat[n][m];
        }
    }
}

void HLL_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma)
{
    const double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    // 提取左右原始变量
    double rhoL = UL[0];
    double uL   = UL[1] / rhoL;
    double vL   = UL[2] / rhoL;
    double wL   = UL[3] / rhoL;
    double EL   = UL[4];
    double pL   = (EL - 0.5 * rhoL * (uL*uL + vL*vL + wL*wL)) * (gamma - 1.0);
    double aL   = std::sqrt(gamma * pL / rhoL);

    double rhoR = UR[0];
    double uR   = UR[1] / rhoR;
    double vR   = UR[2] / rhoR;
    double wR   = UR[3] / rhoR;
    double ER   = UR[4];
    double pR   = (ER - 0.5 * rhoR * (uR*uR + vR*vR + wR*wR)) * (gamma - 1.0);
    double aR   = std::sqrt(gamma * pR / rhoR);

    // 法向速度
    double unL = uL*nx + vL*ny + wL*nz;
    double unR = uR*nx + vR*ny + wR*nz;

    // 计算左右通量 FL, FR
    double FL[5], FR[5];
    FL[0] = rhoL * unL;
    FL[1] = rhoL * uL * unL + pL * nx;
    FL[2] = rhoL * vL * unL + pL * ny;
    FL[3] = rhoL * wL * unL + pL * nz;
    FL[4] = (EL + pL) * unL;

    FR[0] = rhoR * unR;
    FR[1] = rhoR * uR * unR + pR * nx;
    FR[2] = rhoR * vR * unR + pR * ny;
    FR[3] = rhoR * wR * unR + pR * nz;
    FR[4] = (ER + pR) * unR;

    // 估计最快波速 SL, SR (Davis 估计)
    double SL = std::min(unL - aL * norm, unR - aR * norm);
    double SR = std::max(unL + aL * norm, unR + aR * norm);

    // 根据波速位置确定通量
    if (SL >= 0.0) {
        // 超音速向右：全部来自左侧
        for (int i = 0; i < 5; ++i) Fface[i] = FL[i];
    }
    else if (SR <= 0.0) {
        // 超音速向左：全部来自右侧
        for (int i = 0; i < 5; ++i) Fface[i] = FR[i];
    }
    else {
        // 亚音速：使用 HLL 公式
        double denom = SR - SL;
        double fac1  = SR / denom;
        double fac2  = SL / denom;
        double fac3  = SL * SR / denom;
        for (int i = 0; i < 5; ++i) {
            Fface[i] = (fac1 * FL[i] - fac2 * FR[i] + fac3 * (UR[i] - UL[i]));
        }
    }
}

void HLLC_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma)
{
    const double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    const double norm_safe = std::max(norm, 1e-14);
    const double norm2 = norm_safe * norm_safe;
    const double nxh = nx / norm_safe;
    const double nyh = ny / norm_safe;
    const double nzh = nz / norm_safe;
    // 提取左右原始变量
    double rhoL = UL[0];
    double uL   = UL[1] / rhoL;
    double vL   = UL[2] / rhoL;
    double wL   = UL[3] / rhoL;
    double EL   = UL[4];
    double pL   = (EL - 0.5 * rhoL * (uL*uL + vL*vL + wL*wL)) * (gamma - 1.0);
    double aL   = std::sqrt(gamma * pL / rhoL);

    double rhoR = UR[0];
    double uR   = UR[1] / rhoR;
    double vR   = UR[2] / rhoR;
    double wR   = UR[3] / rhoR;
    double ER   = UR[4];
    double pR   = (ER - 0.5 * rhoR * (uR*uR + vR*vR + wR*wR)) * (gamma - 1.0);
    double aR   = std::sqrt(gamma * pR / rhoR);

    // 法向速度
    double unL = uL*nx + vL*ny + wL*nz;
    double unR = uR*nx + vR*ny + wR*nz;
    double qnL = unL / norm_safe;
    double qnR = unR / norm_safe;

    // 计算左右通量 FL, FR
    double FL[5], FR[5];
    FL[0] = rhoL * unL;
    FL[1] = rhoL * uL * unL + pL * nx;
    FL[2] = rhoL * vL * unL + pL * ny;
    FL[3] = rhoL * wL * unL + pL * nz;
    FL[4] = (EL + pL) * unL;

    FR[0] = rhoR * unR;
    FR[1] = rhoR * uR * unR + pR * nx;
    FR[2] = rhoR * vR * unR + pR * ny;
    FR[3] = rhoR * wR * unR + pR * nz;
    FR[4] = (ER + pR) * unR;

    // 估计最快波速 SL, SR (Davis 估计)
    double SL_hat = std::min(qnL - aL, qnR - aR);
    double SR_hat = std::max(qnL + aL, qnR + aR);
    double SL = SL_hat * norm_safe;
    double SR = SR_hat * norm_safe;

    // 如果界面处于超音速区域，直接采用迎风通量
    if (SL >= 0.0) {
        for (int i = 0; i < 5; ++i) Fface[i] = FL[i];
        return;
    }
    if (SR <= 0.0) {
        for (int i = 0; i < 5; ++i) Fface[i] = FR[i];
        return;
    }

    // -----------------------------------------------------------------
    // 亚音速情况：需要计算接触波速度 SM 和中间通量
    // 采用 Batten 等人的估计公式 (AIAA Journal, 1997)
    // -----------------------------------------------------------------
    double rho_sqrtL = std::sqrt(rhoL);
    double rho_sqrtR = std::sqrt(rhoR);
    double rho_sum   = rho_sqrtL + rho_sqrtR;

    // Roe 平均速度 (仅用于计算 SM，也可用其他方式)
    double u_roe = (rho_sqrtL * uL + rho_sqrtR * uR) / rho_sum;
    double v_roe = (rho_sqrtL * vL + rho_sqrtR * vR) / rho_sum;
    double w_roe = (rho_sqrtL * wL + rho_sqrtR * wR) / rho_sum;
    double un_roe = u_roe * nxh + v_roe * nyh + w_roe * nzh;

    // 接触波速度 SM 取 Roe 平均法向速度（简单近似，鲁棒性好）
    double SM_hat = un_roe;

    // 更精确的 SM 可用下式（当分母不为零时），此处为可选，但为了简洁保留简单平均
    double num = rhoR*qnR*(SR_hat - qnR) - rhoL*qnL*(SL_hat - qnL) + pL - pR;
    double den = rhoR*(SR_hat - qnR) - rhoL*(SL_hat - qnL);
    if (std::fabs(den) > 1e-12) {
        SM_hat = num / den;
    } else {
        SM_hat = un_roe;
    }
    double SM = SM_hat * norm_safe;

    // 计算中间压力 p* (取左右平均值以提高对称性)
    // double p_starL = pL + rhoL * (unL - SL) * (unL - SM);
    // double p_starR = pR + rhoR * (unR - SR) * (unR - SM);
    // double p_star  = 0.5 * (p_starL + p_starR);

    // -----------------------------------------------------------------
    // 根据接触波位置判断区域，并构造相应的中间通量 F*L 或 F*R
    // -----------------------------------------------------------------
    if (SM >= 0.0) {
        // 左波与接触波之间 (SL <= 0 <= SM)：采用 F*L
        // 计算左侧中间状态 U*L
        double rho_star_L = rhoL * (SL_hat - qnL) / (SL_hat - SM_hat);

        // 左侧切向速度保持不变
        double utxL = uL - qnL * nxh;
        double utyL = vL - qnL * nyh;
        double utzL = wL - qnL * nzh;

        double u_starL_x = SM_hat * nxh + utxL;
        double u_starL_y = SM_hat * nyh + utyL;
        double u_starL_z = SM_hat * nzh + utzL;

        // 中间状态总能量 (源自 Rankine-Hugoniot 关系)
        // double E_star_L = ((SL - unL) * EL - pL * unL + p_star * SM) / (SL - SM);
        double E_star_L = EL/rhoL + (SM_hat - qnL) * (SM_hat + pL/(rhoL*(SL_hat - qnL)));

        double U_star_L[5] = {rho_star_L,
                              rho_star_L * u_starL_x,
                              rho_star_L * u_starL_y,
                              rho_star_L * u_starL_z,
                              rho_star_L * E_star_L};

        // F*L = FL + SL * (U*L - UL)
        for (int i = 0; i < 5; ++i) {
            Fface[i] = FL[i] + SL * (U_star_L[i] - UL[i]);
        }
    }
    else { // SM < 0.0
        // 接触波与右波之间 (SM <= 0 <= SR)：采用 F*R
        double rho_star_R = rhoR * (SR_hat - qnR) / (SR_hat - SM_hat);

        double utxR = uR - qnR * nxh;
        double utyR = vR - qnR * nyh;
        double utzR = wR - qnR * nzh;

        double u_starR_x = SM_hat * nxh + utxR;
        double u_starR_y = SM_hat * nyh + utyR;
        double u_starR_z = SM_hat * nzh + utzR;

        // double E_star_R = ((SR - unR) * ER - pR * unR + p_star * SM) / (SR - SM);
        double E_star_R = ER/rhoR + (SM_hat - qnR) * (SM_hat + pR/(rhoR*(SR_hat - qnR)));

        double U_star_R[5] = {rho_star_R,
                              rho_star_R * u_starR_x,
                              rho_star_R * u_starR_y,
                              rho_star_R * u_starR_z,
                              rho_star_R * E_star_R};

        // F*R = FR + SR * (U*R - UR)
        for (int i = 0; i < 5; ++i) {
            Fface[i] = FR[i] + SR * (U_star_R[i] - UR[i]);
        }
    }
}

void HLLC_p_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma)
{
    const double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    const double norm_safe = std::max(norm, 1e-14);
    const double norm2 = norm_safe * norm_safe;
    const double nxh = nx / norm_safe;
    const double nyh = ny / norm_safe;
    const double nzh = nz / norm_safe;
    // HLLC Riemann solver with pressure-based contact wave speed estimation
    // 该函数实现与 HLLC_Riemann_solver 类似，但接触波速度 SM 的估计改为基于压力的公式
    // 提取左右原始变量
    double rhoL = UL[0];
    double uL   = UL[1] / rhoL;
    double vL   = UL[2] / rhoL;
    double wL   = UL[3] / rhoL;
    double EL   = UL[4];
    double pL   = (EL - 0.5 * rhoL * (uL*uL + vL*vL + wL*wL)) * (gamma - 1.0);
    double aL   = std::sqrt(gamma * pL / rhoL);

    double rhoR = UR[0];
    double uR   = UR[1] / rhoR;
    double vR   = UR[2] / rhoR;
    double wR   = UR[3] / rhoR;
    double ER   = UR[4];
    double pR   = (ER - 0.5 * rhoR * (uR*uR + vR*vR + wR*wR)) * (gamma - 1.0);
    double aR   = std::sqrt(gamma * pR / rhoR);

    // 法向速度
    double unL = uL*nx + vL*ny + wL*nz;
    double unR = uR*nx + vR*ny + wR*nz;
    double qnL = unL / norm_safe;
    double qnR = unR / norm_safe;

    // 计算左右通量 FL, FR
    double FL[5], FR[5];
    FL[0] = rhoL * unL;
    FL[1] = rhoL * uL * unL + pL * nx;
    FL[2] = rhoL * vL * unL + pL * ny;
    FL[3] = rhoL * wL * unL + pL * nz;
    FL[4] = (EL + pL) * unL;

    FR[0] = rhoR * unR;
    FR[1] = rhoR * uR * unR + pR * nx;
    FR[2] = rhoR * vR * unR + pR * ny;
    FR[3] = rhoR * wR * unR + pR * nz;
    FR[4] = (ER + pR) * unR;

    // 压强估计
    double p_star = 0.5 * (pL + pR) - 0.125 * (qnR - qnL) * (rhoL + rhoR) * (aL + aR);
    p_star = std::max(0.0, p_star); // 保持非负压力

    // 波速估计(Toro)
    double SL, SR;
    if (p_star <= pL) {
        SL = qnL - aL;
    } else {
        SL = qnL - aL * std::sqrt((gamma+1.0)*p_star/(2.0*gamma*pL) + (gamma-1.0)/(2.0*gamma));
    }
    if (p_star <= pR) {
        SR = qnR + aR;
    } else {
        SR = qnR + aR * std::sqrt((gamma+1.0)*p_star/(2.0*gamma*pR) + (gamma-1.0)/(2.0*gamma));
    }
    double SL_hat = SL;
    double SR_hat = SR;
    SL *= norm_safe;
    SR *= norm_safe;

    if (SL >= 0.0) {
        for (int i = 0; i < 5; ++i) Fface[i] = FL[i];
        return;
    }
    if (SR <= 0.0) {
        for (int i = 0; i < 5; ++i) Fface[i] = FR[i];
        return;
    }

    double SM_hat = (pR - pL + rhoL*qnL*(SL_hat - qnL) - rhoR*qnR*(SR_hat - qnR)) /
                    (rhoL*(SL_hat - qnL) - rhoR*(SR_hat - qnR));
    double SM = SM_hat * norm_safe;
    if (SM >= 0.0) {
        double rho_star_L = rhoL * (SL_hat - qnL) / (SL_hat - SM_hat);

        // 左侧切向速度保持不变
        double utxL = uL - qnL * nxh;
        double utyL = vL - qnL * nyh;
        double utzL = wL - qnL * nzh;

        double u_starL_x = SM_hat * nxh + utxL;
        double u_starL_y = SM_hat * nyh + utyL;
        double u_starL_z = SM_hat * nzh + utzL;

        // 中间状态总能量 (源自 Rankine-Hugoniot 关系)
        // double E_star_L = ((SL - unL) * EL - pL * unL + p_star * SM) / (SL - SM);
        double E_star_L = EL/rhoL + (SM_hat - qnL) * (SM_hat + pL/(rhoL*(SL_hat - qnL)));

        double U_star_L[5] = {rho_star_L,
                              rho_star_L * u_starL_x,
                              rho_star_L * u_starL_y,
                              rho_star_L * u_starL_z,
                              rho_star_L * E_star_L};

        // F*L = FL + SL * (U*L - UL)
        for (int i = 0; i < 5; ++i) {
            Fface[i] = FL[i] + SL * (U_star_L[i] - UL[i]);
        }
    }
    else { // SM < 0.0
        // 接触波与右波之间 (SM <= 0 <= SR)：采用 F*R
        double rho_star_R = rhoR * (SR_hat - qnR) / (SR_hat - SM_hat);

        double utxR = uR - qnR * nxh;
        double utyR = vR - qnR * nyh;
        double utzR = wR - qnR * nzh;

        double u_starR_x = SM_hat * nxh + utxR;
        double u_starR_y = SM_hat * nyh + utyR;
        double u_starR_z = SM_hat * nzh + utzR;

        // double E_star_R = ((SR - unR) * ER - pR * unR + p_star * SM) / (SR - SM);
        double E_star_R = ER/rhoR + (SM_hat - qnR) * (SM_hat + pR/(rhoR*(SR_hat - qnR)));

        double U_star_R[5] = {rho_star_R,
                              rho_star_R * u_starR_x,
                              rho_star_R * u_starR_y,
                              rho_star_R * u_starR_z,
                              rho_star_R * E_star_R};

        // F*R = FR + SR * (U*R - UR)
        for (int i = 0; i < 5; ++i) {
            Fface[i] = FR[i] + SR * (U_star_R[i] - UR[i]);
        }
    }

}

void AUSM_Riemann_solver(std::vector<double> &Fface,
                    const std::vector<double> &UL,
                    const std::vector<double> &UR,
                    double nx, double ny, double nz,
                    double gamma)
{
    const int VAR = 5;
    const double Kp = 0.25;
    const double Ku = 0.75;
    const double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    const double eps_norm = 1e-14;
    const double norm_safe = std::max(norm, eps_norm);

    // =========================
    // 1. 左右状态
    // =========================
    double rho_L = UL[0];
    double u_L = UL[1] / rho_L;
    double v_L = UL[2] / rho_L;
    double w_L = UL[3] / rho_L;
    double E_L = UL[4];

    double p_L = (E_L - 0.5 * rho_L * (u_L*u_L + v_L*v_L + w_L*w_L)) * (gamma - 1.0);
    p_L = std::max(p_L, 1e-12);

    double a_L = std::sqrt(gamma * p_L / rho_L);
    double H_L = (E_L + p_L) / rho_L;

    double rho_R = UR[0];
    double u_R = UR[1] / rho_R;
    double v_R = UR[2] / rho_R;
    double w_R = UR[3] / rho_R;
    double E_R = UR[4];

    double p_R = (E_R - 0.5 * rho_R * (u_R*u_R + v_R*v_R + w_R*w_R)) * (gamma - 1.0);
    p_R = std::max(p_R, 1e-12);

    double a_R = std::sqrt(gamma * p_R / rho_R);
    double H_R = (E_R + p_R) / rho_R;

    // =========================
    // 2. 法向速度
    // =========================
    double un_L = u_L*nx + v_L*ny + w_L*nz;
    double un_R = u_R*nx + v_R*ny + w_R*nz;

    // =========================
    // 3. 界面平均声速
    // =========================
    double a_bar = 0.5 * (a_L + a_R);
    double rho_bar = 0.5 * (rho_L + rho_R);

    // =========================
    // 马赫数
    // =========================
    double a_n = a_bar * norm_safe;
    double ML = un_L / a_n;
    double MR = un_R / a_n;
    double M_bar2 = (un_L*un_L + un_R*un_R) / (2.0*a_n*a_n); // 平均马赫数的平方
    double M_o2 = std::min(1.0, std::max(M_bar2, 0.5));  // 原始公式还包括远场马赫数
    double f_a = M_o2 * (2.0 - M_o2); // 平滑函数，控制亚音速区域的修正强度

    auto M4_plus = [](double M) {
        if (std::abs(M) >= 1.0)
            return 0.5 * (M + std::abs(M));
        else
            return 0.25 * (M + 1.0)*(M + 1.0); // *(1 + 0.5 * (M - 1.0)*(M - 1.0)); 
    };

    auto M4_minus = [](double M) {
        if (std::abs(M) >= 1.0)
            return 0.5 * (M - std::abs(M));
        else
            return -0.25 * (M - 1.0)*(M - 1.0); //*(1 + 0.5 * (M + 1.0)*(M + 1.0));
    };

    double Mp = M4_plus(ML);
    double Mm = M4_minus(MR);

    // =========================
    // 质量通量
    // =========================
    double M_half = Mp + Mm - Kp/f_a*std::max(1.0 - M_bar2, 0.0) * ((p_R - p_L) / (rho_bar * a_n * a_n));
    double m_dot = a_n * M_half * (M_half >= 0.0 ? rho_L : rho_R); // 质量通量，迎风选择

    // =========================
    // 压力分裂
    // =========================
    auto P5_plus = [](double M) {
        if (std::abs(M) >= 1.0)
            return 0.5* (1.0 + (M > 0 ? 1.0 : -1.0));
        else
            return 0.25 * (M + 1.0)*(M + 1.0)*(2.0 - M);
    };

    auto P5_minus = [](double M) {
        if (std::abs(M) >= 1.0)
            return 0.5* (1.0 - (M > 0 ? 1.0 : -1.0));
        else
            return 0.25 * (M - 1.0)*(M - 1.0)*(2.0 + M);
    };

    double p_half_plus = P5_plus(ML);
    double p_half_minus = P5_minus(MR);
    double p_half = p_half_plus * p_L + p_half_minus * p_R - Ku * p_half_minus * p_half_plus * (rho_L + rho_R) * f_a * (u_R - u_L) * a_bar;

    // =========================
    // 8. 上风选择
    // =========================
    double u_face, v_face, w_face, H_face;

    if (m_dot >= 0.0) {
        u_face = u_L;
        v_face = v_L;
        w_face = w_L;
        H_face = H_L;
    } else {
        u_face = u_R;
        v_face = v_R;
        w_face = w_R;
        H_face = H_R;
    }

    // =========================
    // 9. 构造通量
    // =========================
    Fface[0] = m_dot;
    Fface[1] = m_dot * u_face + p_half * nx;
    Fface[2] = m_dot * v_face + p_half * ny;
    Fface[3] = m_dot * w_face + p_half * nz;
    Fface[4] = m_dot * H_face;
}

// Roe平均
void computeRoeAveragedState(double &rho_bar, double &u_bar, double &v_bar, double &w_bar,
                             double &h_bar, double &a_bar,
                             const double ul[5], const double ur[5],
                             double gamma)
{
    // 提取左状态变量
    double rho_L = ul[0];
    double u_L = ul[1];
    double v_L = ul[2];
    double w_L = ul[3];
    double p_L = ul[4];
    double E_L = rho_L * (0.5 * (u_L*u_L + v_L*v_L + w_L*w_L) + p_L / ((gamma - 1.0) * rho_L));
    // 提取右状态变量
    double rho_R = ur[0];
    double u_R = ur[1];
    double v_R = ur[2];
    double w_R = ur[3];
    double p_R = ur[4];
    double E_R = rho_R * (0.5 * (u_R*u_R + v_R*v_R + w_R*w_R) + p_R / ((gamma - 1.0) * rho_R));

    // 计算Roe平均态
    double sqrt_rho_L = std::sqrt(rho_L);
    double sqrt_rho_R = std::sqrt(rho_R);
    rho_bar = sqrt_rho_L * sqrt_rho_R;
    double denom = 1.0 / (sqrt_rho_L + sqrt_rho_R);
    u_bar = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * denom;
    v_bar = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) * denom;
    w_bar = (sqrt_rho_L * w_L + sqrt_rho_R * w_R) * denom;
    // Total specific enthalpy H = (E + p) / rho. Here Ul[4] and Ur[4] are the total energy (conserved E).
    double H_L = (E_L + p_L) / rho_L;
    double H_R = (E_R + p_R) / rho_R;
    h_bar = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) * denom;
    double kinetic_bar = 0.5 * (u_bar*u_bar + v_bar*v_bar + w_bar*w_bar);
    a_bar = std::sqrt((gamma - 1.0) * (h_bar - kinetic_bar));

}

// 计算左/右 特征向量矩阵 L (左) 与 R (右) 对任意法向量 (nx,ny,nz)
// using Blazek-style formula from your snippet
static void build_eigen_matrices(const double ul[5], const double ur[5],
                                 double nx, double ny, double nz,
                                 double gamma,
                                 double Lmat[5][5], double Rmat[5][5],
                                 double lambar[5])
{
    // first compute Roe averaged quantities
    double rhobar, ubar, vbar, wbar, Hbar, abar, pbar;
    computeRoeAveragedState(rhobar, ubar, vbar, wbar, Hbar, abar, ul, ur, gamma);

    double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
    double inv_norm = (norm > 1e-14) ? 1.0 / norm : 0.0;
    double nxh = nx * inv_norm;
    double nyh = ny * inv_norm;
    double nzh = nz * inv_norm;
    double Vhat = nxh * ubar + nyh * vbar + nzh * wbar;
    double V = norm * Vhat;
    double c = abar;

    lambar[0] = V - c * norm;
    lambar[1] = V;
    lambar[2] = V;
    lambar[3] = V;
    lambar[4] = V + c * norm;

    double phi = 0.5 * (gamma - 1.0) * (ubar*ubar + vbar*vbar + wbar*wbar);

    double a1 = gamma - 1.0;
    double a2 = 1.0 / (std::sqrt(2.0) * rhobar * c);
    double a3 = rhobar / (std::sqrt(2.0) * c);
    double a4 = (phi + c*c) / (gamma - 1.0);
    double a5 = 1.0 - phi / (c*c);
    double a6 = phi / (gamma - 1.0);

    // Left eigenvectors L (rows)
    // L[0][:]
    Lmat[0][0] = a2 * (phi + c * Vhat);
    Lmat[0][1] = -a2 * (a1 * ubar + nxh * c);
    Lmat[0][2] = -a2 * (a1 * vbar + nyh * c);
    Lmat[0][3] = -a2 * (a1 * wbar + nzh * c);
    Lmat[0][4] = a1 * a2;

    // L[1][:]
    Lmat[1][0] = nxh * a5 - (nzh * vbar - nyh * wbar) / rhobar;
    Lmat[1][1] = nxh * a1 * ubar / (c*c);
    Lmat[1][2] = nxh * a1 * vbar / (c*c) + nzh / rhobar;
    Lmat[1][3] = nxh * a1 * wbar / (c*c) - nyh / rhobar;
    Lmat[1][4] = -nxh * a1 / (c*c);

    // L[2][:]
    Lmat[2][0] = nzh * a5 - (nyh * ubar - nxh * vbar) / rhobar;
    Lmat[2][1] = nzh * a1 * ubar / (c*c) + nyh / rhobar;
    Lmat[2][2] = nzh * a1 * vbar / (c*c) - nxh / rhobar;
    Lmat[2][3] = nzh * a1 * wbar / (c*c);
    Lmat[2][4] = -nzh * a1 / (c*c);

    // L[3][:]
    Lmat[3][0] = nyh * a5 - (nxh * wbar - nzh * ubar) / rhobar;
    Lmat[3][1] = nyh * a1 * ubar / (c*c) - nzh / rhobar;
    Lmat[3][2] = nyh * a1 * vbar / (c*c);
    Lmat[3][3] = nyh * a1 * wbar / (c*c) + nxh / rhobar;
    Lmat[3][4] = -nyh * a1 / (c*c);

    // L[4][:]
    Lmat[4][0] = a2 * (phi - c * Vhat);
    Lmat[4][1] = -a2 * (a1 * ubar - nxh * c);
    Lmat[4][2] = -a2 * (a1 * vbar - nyh * c);
    Lmat[4][3] = -a2 * (a1 * wbar - nzh * c);
    Lmat[4][4] = a1 * a2;

    // Right eigenvectors R (columns)
    // R[:,0]
    Rmat[0][0] = a3;
    Rmat[1][0] = a3 * (ubar - nxh*c);
    Rmat[2][0] = a3 * (vbar - nyh*c);
    Rmat[3][0] = a3 * (wbar - nzh*c);
    Rmat[4][0] = a3 * (a4 - c * Vhat);

    // R[:,1]
    Rmat[0][1] = nxh;
    Rmat[1][1] = nxh * ubar;
    Rmat[2][1] = nxh * vbar + nzh * rhobar;
    Rmat[3][1] = nxh * wbar - nyh * rhobar;
    Rmat[4][1] = nxh * a6 + rhobar * (nzh * vbar - nyh * wbar);

    // R[:,2]
    Rmat[0][2] = nzh;
    Rmat[1][2] = nzh * ubar + nyh * rhobar;
    Rmat[2][2] = nzh * vbar - nxh * rhobar;
    Rmat[3][2] = nzh * wbar;
    Rmat[4][2] = nzh * a6 + rhobar * (nyh * ubar - nxh * vbar);

    // R[:,3]
    Rmat[0][3] = nyh;
    Rmat[1][3] = nyh * ubar - nzh * rhobar;
    Rmat[2][3] = nyh * vbar;
    Rmat[3][3] = nyh * wbar + nxh * rhobar;
    Rmat[4][3] = nyh * a6 + rhobar * (nxh * wbar - nzh * ubar);

    // R[:,4]
    Rmat[0][4] = a3;
    Rmat[1][4] = a3 * (ubar + nxh*c);
    Rmat[2][4] = a3 * (vbar + nyh*c);
    Rmat[3][4] = a3 * (wbar + nzh*c);
    Rmat[4][4] = a3 * (a4 + c * Vhat);
}


inline double diff_6th_central_half(const std::vector<double> &f, int i, double dx) {
    return ((f[3]-f[2])*(75.0/64.0) + (f[4]-f[1])*(-25.0/384.0) + (f[5]-f[0])*(3.0/640.0))/dx;
}
inline double diff_4th_central_half(const std::vector<double> &f, int i, double dx) {
    return ((f[3]-f[2])*(9.0/8.0) + (f[4]-f[1])*(-1.0/24.0))/dx;
}
inline double diff_x_half(const std::vector<double> &f, int i, int j, int k, double dx, const LocalDesc &L)
{
    std::vector<double> dummy(6);
    for (int ii = 0; ii < 6; ++ii) 
    {
        dummy[ii] = f[idx_fx(i + ii - 3, j, k, L)];
    }

    return diff_6th_central_half(dummy, 3, dx);
}
inline double diff_y_half(const std::vector<double> &f, int i, int j, int k, double dy, const LocalDesc &L)
{
    std::vector<double> dummy(6);
    for (int ii = 0; ii < 6; ++ii) 
    {
        dummy[ii] = f[idx_fy(i, j + ii - 3, k, L)];
    }

    return diff_6th_central_half(dummy, 3, dy);
}
inline double diff_z_half(const std::vector<double> &f, int i, int j, int k, double dz, const LocalDesc &L)
{
    std::vector<double> dummy(6);
    for (int ii = 0; ii < 6; ++ii) 
    {
        dummy[ii] = f[idx_fz(i, j, k + ii - 3, L)];
    }

    return diff_6th_central_half(dummy, 3, dz);
}

void compute_invis_dflux(Field3D &F, const SolverParams &P, const GridDesc &G)
{
    const LocalDesc &L = F.L;

    // 半节点中心差分
    for (int k = L.ngz; k < L.ngz + L.nz; ++k){
    for (int j = L.ngy; j < L.ngy + L.ny; ++j){
    for (int i = L.ngx; i < L.ngx + L.nx; ++i){
        // mass
        int id = F.I(i, j, k);
        F.rhs_rho[id] -= diff_x_half(F.flux_fx_mass, i, j, k, G.dx, L);
        F.rhs_rho[id] -= diff_y_half(F.flux_fy_mass, i, j, k, G.dy, L);
        F.rhs_rho[id] -= diff_z_half(F.flux_fz_mass, i, j, k, G.dz, L);
        // momx
        F.rhs_rhou[id] -= diff_x_half(F.flux_fx_momx, i, j, k, G.dx, L);
        F.rhs_rhou[id] -= diff_y_half(F.flux_fy_momx, i, j, k, G.dy, L);
        F.rhs_rhou[id] -= diff_z_half(F.flux_fz_momx, i, j, k, G.dz, L);
        // momy
        F.rhs_rhov[id] -= diff_x_half(F.flux_fx_momy, i, j, k, G.dx, L);
        F.rhs_rhov[id] -= diff_y_half(F.flux_fy_momy, i, j, k, G.dy, L);
        F.rhs_rhov[id] -= diff_z_half(F.flux_fz_momy, i, j, k, G.dz, L);
        // momz
        F.rhs_rhow[id] -= diff_x_half(F.flux_fx_momz, i, j, k, G.dx, L);
        F.rhs_rhow[id] -= diff_y_half(F.flux_fy_momz, i, j, k, G.dy, L);
        F.rhs_rhow[id] -= diff_z_half(F.flux_fz_momz, i, j, k, G.dz, L);
        // energy
        F.rhs_E[id] -= diff_x_half(F.flux_fx_E, i, j, k, G.dx, L);
        F.rhs_E[id] -= diff_y_half(F.flux_fy_E, i, j, k, G.dy, L);
        F.rhs_E[id] -= diff_z_half(F.flux_fz_E, i, j, k, G.dz, L);
    }}}
}