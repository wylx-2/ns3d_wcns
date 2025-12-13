#include "ns3d_func.h"

// -----------------------------------------------------------------
// ---------   Flux Vector Splitting (FVS) -------------------------
// -----------------------------------------------------------------

// 计算无粘通量
void compute_flux(Field3D &F, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double gamma = P.gamma;

    for (int k = 0; k < L.sz; ++k) {
        for (int j = 0; j < L.sy; ++j) {
            for (int i = 0; i < L.sx; ++i) {

                int id = F.I(i, j, k);
                double rho = F.rho[id];
                double u = F.u[id];
                double v = F.v[id];
                double w = F.w[id];
                double E = F.E[id];
                double p = F.p[id];

                // X方向物理通量
                F.Fflux_mass[id] = rho * u;
                F.Fflux_momx[id] = rho * u * u + p;
                F.Fflux_momy[id] = rho * u * v;
                F.Fflux_momz[id] = rho * u * w;
                F.Fflux_E[id]    = (E + p) * u;

                // Y方向物理通量
                F.Hflux_mass[id] = rho * v;
                F.Hflux_momx[id] = rho * u * v;
                F.Hflux_momy[id] = rho * v * v + p;
                F.Hflux_momz[id] = rho * v * w;
                F.Hflux_E[id]    = (E + p) * v;

                // Z方向物理通量
                F.Gflux_mass[id] = rho * w;
                F.Gflux_momx[id] = rho * u * w;
                F.Gflux_momy[id] = rho * v * w;
                F.Gflux_momz[id] = rho * w * w + p;
                F.Gflux_E[id]    = (E + p) * w;
            }
        }
    }
}

// FVS main 计算通量的重构值
void computeFVSFluxes(Field3D &F, const SolverParams &P)
{
    compute_flux(F, P);

    const LocalDesc &L = F.L;
    int nx = L.nx, ny = L.ny, nz = L.nz;
    int ngx = L.ngx, ngy = L.ngy, ngz = L.ngz;
    int sz = L.sz, sy = L.sy, sx = L.sx;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    // Use runtime stencil size from SolverParams so different reconstructions
    // (WENO5, C6th, ...) can be selected at runtime.
    int stencil = P.stencil;

    
    if (stencil < 2) {
        std::cerr << "computeFVSFluxes: stencil must be >= 2\n";
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

    // X方向通量重构
    for (int k = ngz; k < ngz+nz; ++k) {
        for (int j = ngy; j < ngy+ny; ++j) {
            // choose i range so that ii = i + (m-mid) stays inside [0, L.sx-1]
            for (int i = ngx - 1; i < ngx + nx; ++i) {
                // dynamic 2D arrays: VAR x stencil
                std::vector<std::vector<double>> Ft(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

                for (int m = 0; m < stencil; ++m) {
                    int ii = i + (m - mid); // 以i为中心的stencil(6点模板为i-2到i+3) when mid=(stencil-1)/2
                    int id = F.I(ii, j, k);

                    Ft[0][m] = F.Fflux_mass[id];
                    Ft[1][m] = F.Fflux_momx[id];
                    Ft[2][m] = F.Fflux_momy[id];
                    Ft[3][m] = F.Fflux_momz[id];
                    Ft[4][m] = F.Fflux_E[id];
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

                std::vector<double> Fface(VAR, 0.0);
                reconstructInviscidFlux(Fface, Ft, Ut, ut, P, /*dim=*/0);

                int fid = idx_fx(i, j, k, L);
                F.flux_fx_mass[fid] = Fface[0];
                F.flux_fx_momx[fid] = Fface[1];
                F.flux_fx_momy[fid] = Fface[2];
                F.flux_fx_momz[fid] = Fface[3];
                F.flux_fx_E[fid]    = Fface[4];
            }
        }
    }

    // Y方向通量重构
    for (int k = ngz; k < ngz+nz; ++k) {
        for (int i = ngx; i < ngx+nx; ++i) {
            for (int j = ngy - 1; j < ngy + ny; ++j) {
                // dynamic 2D arrays: VAR x stencil
                std::vector<std::vector<double>> Ft(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

                for (int m = 0; m < stencil; ++m) {
                    int jj = j + (m - mid);
                    int id = F.I(i, jj, k);

                    Ft[0][m] = F.Hflux_mass[id];
                    Ft[1][m] = F.Hflux_momx[id];
                    Ft[2][m] = F.Hflux_momy[id];
                    Ft[3][m] = F.Hflux_momz[id];
                    Ft[4][m] = F.Hflux_E[id];
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

                std::vector<double> Fface(VAR, 0.0);
                reconstructInviscidFlux(Fface, Ft, Ut, ut, P, /*dim=*/1);

                int fid = idx_fy(i, j, k, L);
                F.flux_fy_mass[fid] = Fface[0];
                F.flux_fy_momx[fid] = Fface[1];
                F.flux_fy_momy[fid] = Fface[2];
                F.flux_fy_momz[fid] = Fface[3];
                F.flux_fy_E[fid]    = Fface[4];
            }
        }
    }

    // Z方向通量重构
    for (int j = ngy; j < ngy+ny; ++j) {
        for (int i = ngx; i < ngx+nx; ++i) {
            for (int k = ngz -1; k < ngz + nz; ++k) {
                // dynamic 2D arrays: VAR x stencil
                std::vector<std::vector<double>> Ft(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> Ut(VAR, std::vector<double>(stencil));
                std::vector<std::vector<double>> ut(VAR, std::vector<double>(stencil));

                for (int m = 0; m < stencil; ++m) {
                    int kk = k + (m - mid);
                    int id = F.I(i, j, kk);

                    Ft[0][m] = F.Gflux_mass[id];
                    Ft[1][m] = F.Gflux_momx[id];
                    Ft[2][m] = F.Gflux_momy[id];
                    Ft[3][m] = F.Gflux_momz[id];
                    Ft[4][m] = F.Gflux_E[id];
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

                std::vector<double> Fface(VAR, 0.0);
                reconstructInviscidFlux(Fface, Ft, Ut, ut, P, /*dim=*/2);

                int fid = idx_fz(i, j, k, L);
                F.flux_fz_mass[fid] = Fface[0];
                F.flux_fz_momx[fid] = Fface[1];
                F.flux_fz_momy[fid] = Fface[2];
                F.flux_fz_momz[fid] = Fface[3];
                F.flux_fz_E[fid]    = Fface[4];
            }
        }
    }

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

    double V = nx * ubar + ny * vbar + nz * wbar;
    double c = abar;

    lambar[0] = V - c;
    lambar[1] = V;
    lambar[2] = V;
    lambar[3] = V;
    lambar[4] = V + c;

    double phi = 0.5 * (gamma - 1.0) * (ubar*ubar + vbar*vbar + wbar*wbar);

    double a1 = gamma - 1.0;
    double a2 = 1.0 / (std::sqrt(2.0) * rhobar * c);
    double a3 = rhobar / (std::sqrt(2.0) * c);
    double a4 = (phi + c*c) / (gamma - 1.0);
    double a5 = 1.0 - phi / (c*c);
    double a6 = phi / (gamma - 1.0);

    // Left eigenvectors L (rows)
    // L[0][:]
    Lmat[0][0] = a2 * (phi + c * V);
    Lmat[0][1] = -a2 * (a1 * ubar + nx * c);
    Lmat[0][2] = -a2 * (a1 * vbar + ny * c);
    Lmat[0][3] = -a2 * (a1 * wbar + nz * c);
    Lmat[0][4] = a1 * a2;

    // L[1][:]
    Lmat[1][0] = nx * a5 - (nz * vbar - ny * wbar) / rhobar;
    Lmat[1][1] = nx * a1 * ubar / (c*c);
    Lmat[1][2] = nx * a1 * vbar / (c*c) + nz / rhobar;
    Lmat[1][3] = nx * a1 * wbar / (c*c) - ny / rhobar;
    Lmat[1][4] = -nx * a1 / (c*c);

    // L[2][:]
    Lmat[2][0] = nz * a5 - (ny * ubar - nx * vbar) / rhobar;
    Lmat[2][1] = nz * a1 * ubar / (c*c) + ny / rhobar;
    Lmat[2][2] = nz * a1 * vbar / (c*c) - nx / rhobar;
    Lmat[2][3] = nz * a1 * wbar / (c*c);
    Lmat[2][4] = -nz * a1 / (c*c);

    // L[3][:]
    Lmat[3][0] = ny * a5 - (nx * wbar - nz * ubar) / rhobar;
    Lmat[3][1] = ny * a1 * ubar / (c*c) - nz / rhobar;
    Lmat[3][2] = ny * a1 * vbar / (c*c);
    Lmat[3][3] = ny * a1 * wbar / (c*c) + nx / rhobar;
    Lmat[3][4] = -ny * a1 / (c*c);

    // L[4][:]
    Lmat[4][0] = a2 * (phi - c * V);
    Lmat[4][1] = -a2 * (a1 * ubar - nx * c);
    Lmat[4][2] = -a2 * (a1 * vbar - ny * c);
    Lmat[4][3] = -a2 * (a1 * wbar - nz * c);
    Lmat[4][4] = a1 * a2;

    // Right eigenvectors R (columns)
    // R[:,0]
    Rmat[0][0] = a3;
    Rmat[1][0] = a3 * (ubar - nx*c);
    Rmat[2][0] = a3 * (vbar - ny*c);
    Rmat[3][0] = a3 * (wbar - nz*c);
    Rmat[4][0] = a3 * (a4 - c * V);

    // R[:,1]
    Rmat[0][1] = nx;
    Rmat[1][1] = nx * ubar;
    Rmat[2][1] = nx * vbar + nz * rhobar;
    Rmat[3][1] = nx * wbar - ny * rhobar;
    Rmat[4][1] = nx * a6 + rhobar * (nz * vbar - ny * wbar);

    // R[:,2]
    Rmat[0][2] = nz;
    Rmat[1][2] = nz * ubar + ny * rhobar;
    Rmat[2][2] = nz * vbar - nx * rhobar;
    Rmat[3][2] = nz * wbar;
    Rmat[4][2] = nz * a6 + rhobar * (ny * ubar - nx * vbar);

    // R[:,3]
    Rmat[0][3] = ny;
    Rmat[1][3] = ny * ubar - nz * rhobar;
    Rmat[2][3] = ny * vbar;
    Rmat[3][3] = ny * wbar + nx * rhobar;
    Rmat[4][3] = ny * a6 + rhobar * (nx * wbar - nz * ubar);

    // R[:,4]
    Rmat[0][4] = a3;
    Rmat[1][4] = a3 * (ubar + nx*c);
    Rmat[2][4] = a3 * (vbar + ny*c);
    Rmat[3][4] = a3 * (wbar + nz*c);
    Rmat[4][4] = a3 * (a4 + c * V);
}

// 无粘通量重构
// reconstructInviscidFlux: accept dynamic containers (VAR x stencil)
void reconstructInviscidFlux(std::vector<double> &Fface,
                             const std::vector<std::vector<double>> &Ft,
                             const std::vector<std::vector<double>> &Ut,
                             const std::vector<std::vector<double>> &ut,
                             const SolverParams &P, int dim)
{
    // alias
    double gamma = P.gamma;
    bool sigma = P.char_recon;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    // Use runtime stencil size from SolverParams so different reconstructions
    // (WENO5, C6th, ...) can be selected at runtime.
    int stencil = P.stencil;

    // determine normal vector (nx,ny,nz)
    double nx = 0.0, ny = 0.0, nz = 0.0;
    if (dim == 0) { nx = 1.0; ny = 0.0; nz = 0.0; }
    if (dim == 1) { nx = 0.0; ny = 1.0; nz = 0.0; }
    if (dim == 2) { nx = 0.0; ny = 0.0; nz = 1.0; }

    // 1) compute local stencil eigenvalues lamda[n][m] using primitives ut
    std::vector<std::vector<double>> lamda(VAR, std::vector<double>(stencil));
    for (int m = 0; m < stencil; ++m) {
        double rho = ut[0][m];
        double uu  = ut[1][m];
        double vv  = ut[2][m];
        double ww  = ut[3][m];
        double pp  = ut[4][m];
        double c = std::sqrt(gamma * pp / std::max(rho, 1e-12));
        double V = nx * uu + ny * vv + nz * ww;
        lamda[0][m] = V - c;
        lamda[1][m] = V;
        lamda[2][m] = V;
        lamda[3][m] = V;
        lamda[4][m] = V + c;
    }

    // 2) choose interface (left / right) states as stencil center positions.
    // For a runtime stencil length `stencil`, pick the middle split as
    // mid = stencil/2, then left = mid-1, right = mid. This maps correctly
    // for even (e.g. 6 -> left=2,right=3) and odd (e.g. 5 -> left=1,right=2)
    int mid = (stencil - 1) / 2;
    double ul[VAR], ur[VAR];
    int ileft = mid;
    int iright = mid + 1;
    for (int n = 0; n < VAR; ++n) { ul[n] = ut[n][ileft]; ur[n] = ut[n][iright]; }

    // 4) choose dissipation (lamdamax) per characteristic based on P.fvs_type:
    double lamdamax[VAR];
    switch (P.fvs_type) {
        case SolverParams::FVS_Type::LaxFriedrichs: {
            // Global Lax-Friedrichs: choose a global max over stencil and components
            double gmax = 0.0;
            for (int m = 0; m < stencil; ++m)
                for (int n = 0; n < VAR; ++n)
                    gmax = std::max(gmax, std::abs(lamda[n][m]));
            for (int n = 0; n < VAR; ++n) lamdamax[n] = gmax;
        } break;
        case SolverParams::FVS_Type::Rusanov: {
            // Local Lax-Friedrichs: per-component max over stencil
            for (int n = 0; n < VAR; ++n) {
                double mxx = 0.0;
                for (int m = 0; m < stencil; ++m) mxx = std::max(mxx, std::abs(lamda[n][m]));
                lamdamax[n] = mxx;
            }
        } break;
        case SolverParams::FVS_Type::VanLeer: {
            // Van Leer: not finished yet
        } break;
        case SolverParams::FVS_Type::StegerWarming:
        {
            // Steger-Warming: not finished yet
        } break;
        default: {
            // default to get error
            std::cerr << "reconstructInviscidFlux: unknown FVS_Type\n";
            return;
        } break;
    }

    // 5) component-wise cheap option
    if (!sigma) {
        // For each component n, form wtplus = 0.5*(Ft + lamdamax * Ut) across stencil
        std::vector<double> wface(VAR, 0.0);
        for (int n = 0; n < VAR; ++n) {
            std::vector<double> wplus(stencil), wminus(stencil);
            for (int m = 0; m < stencil; ++m) {
                wplus[m]  = 0.5 * (Ft[n][m] + lamdamax[n] * Ut[n][m]);
                wminus[m] = 0.5 * (Ft[n][m] - lamdamax[n] * Ut[n][m]);
            }
            double plus_face = reconstruct_select(wplus, +1.0, P);
            double minus_face = reconstruct_select(wminus, -1.0, P);
            wface[n] = plus_face + minus_face;
        }
        // component-wise result is wface in conservative flux-like variables
        for (int n = 0; n < VAR; ++n) Fface[n] = wface[n];
        return;
    }

    // 6) characteristic-wise reconstruction:
    // Compute characteristic variables w = L * Ft  and LU = L * Ut (L is left-eig matrix)
    std::vector<std::vector<double>> wchar(VAR, std::vector<double>(stencil));
    std::vector<std::vector<double>> LU(VAR, std::vector<double>(stencil));
    // 3) compute Roe averaged eigenvectors and lambar
    double Lmat[VAR][VAR], Rmat[VAR][VAR], lambar[VAR];
    build_eigen_matrices(ul, ur, nx, ny, nz, gamma, Lmat, Rmat, lambar);
    for (int m = 0; m < stencil; ++m) {
        for (int n = 0; n < VAR; ++n) {
            double sumw = 0.0, sumLU = 0.0;
            for (int r = 0; r < VAR; ++r) {
                sumw  += Lmat[n][r] * Ft[r][m];
                sumLU += Lmat[n][r] * Ut[r][m];
            }
            wchar[n][m] = sumw;
            LU[n][m] = sumLU;
        }
    }

    // For each characteristic n, form wtplus = 0.5*(w + lamdamax * LU) and wtminus = ...
    std::vector<double> wflux_char(VAR, 0.0);
    for (int n = 0; n < VAR; ++n) {
        std::vector<double> wtplus(stencil), wtminus(stencil);
        for (int m = 0; m < stencil; ++m) {
            wtplus[m] = 0.5 * (wchar[n][m] + lamdamax[n] * LU[n][m]);
            wtminus[m] = 0.5 * (wchar[n][m] - lamdamax[n] * LU[n][m]);
        }
        double plus_face = reconstruct_select(wtplus, +1.0, P);
        double minus_face = reconstruct_select(wtminus, -1.0, P);
        wflux_char[n] = plus_face + minus_face;
    }

    // transform back to conservative flux via Fflux = R * wflux_char
    for (int n = 0; n < VAR; ++n) {
        double sum = 0.0;
        for (int r = 0; r < VAR; ++r) sum += Rmat[n][r] * wflux_char[r];
        Fface[n] = sum;
    }
}
