#include "ns3d_func.h"
#include <fstream>
#include <cstdlib>

// -----------------------------------------------------------------
// ---------   Flux Vector Splitting (FVS) -------------------------
// -----------------------------------------------------------------

// 采用wcns方法计算无粘通量导数
void compute_invis_flux(Field3D &F, const SolverParams &P)
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

    // X方向通量重构
    for (int k = ngz; k < ngz+nz; ++k) {
        for (int j = ngy; j < ngy+ny; ++j) {
            // choose i range so that ii = i + (m-mid) stays inside [0, L.sx-1]
            for (int i = ngx - 1; i < ngx + nx; ++i) {
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

                std::vector<double> Fface(VAR, 0.0);
                WCNS_Riemann_InviscidFlux(Fface, Ut, ut, P, /*dim=*/0);

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

                std::vector<double> Fface(VAR, 0.0);
                WCNS_Riemann_InviscidFlux(Fface, Ut, ut, P, /*dim=*/1);

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
    for (int i = ngx; i < ngx+nx; ++i) {
        for (int j = ngy; j < ngy+ny; ++j) {
            for (int k = ngz - 1; k < ngz + nz; ++k) {
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

                std::vector<double> Fface(VAR, 0.0);
                WCNS_Riemann_InviscidFlux(Fface, Ut, ut, P, /*dim=*/2);

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

void WCNS_Riemann_InviscidFlux(std::vector<double> &Fface,
                             const std::vector<std::vector<double>> &Ut,
                             const std::vector<std::vector<double>> &ut,
                             const SolverParams &P, int dim)
{
    // alias
    double gamma = P.gamma;
    bool sigma = P.char_recon;
    const int VAR = 5; // 变量个数：rho, rhou, rhov, rhow, E
    int stencil = P.stencil;

    // determine normal vector (nx,ny,nz)
    double nx = 0.0, ny = 0.0, nz = 0.0;
    if (dim == 0) { nx = 1.0; ny = 0.0; nz = 0.0; }
    if (dim == 1) { nx = 0.0; ny = 1.0; nz = 0.0; }
    if (dim == 2) { nx = 0.0; ny = 0.0; nz = 1.0; }

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
    Roe_Riemann_solver(Fface, UL, UR, nx, ny, nz, gamma);
    // Rusanov_Riemann_solver(Fface, UL, UR, nx, ny, nz, gamma);
}

void Rusanov_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma)
{
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
    double smax = std::max( std::abs(u_L*nx + v_L*ny + w_L*nz) + a_L,
                            std::abs(u_R*nx + v_R*ny + w_R*nz) + a_R );
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

    
    // Debug: 输出沿 x 方向 (j=6, k=6) 的 
    bool Debug_Flux_fx_momx = false;
    if (Debug_Flux_fx_momx){
        int j_probe =  6;
        int k_probe =  6;

        bool in_range = (j_probe >= L.ngy && j_probe < L.ngy + L.ny &&
                         k_probe >= L.ngz && k_probe < L.ngz + L.nz);
        if (!in_range) {
            std::cerr << "Debug probe (j=6,k=6) is outside computed range.\n";
            std::exit(EXIT_FAILURE);
        }

        std::ofstream ofs("flux_fx_momx_j6_k6.dat");
        if (!ofs) {
            std::cerr << "Failed to open flux_fx_momx_j6_k6.dat for writing.\n";
            std::exit(EXIT_FAILURE);
        }

        ofs << "# i flux_fx_momx at j=6 k=6\n";
        for (int i = 0; i < L.sx -1; ++i) {
            int id = idx_fx(i, j_probe, k_probe, L);
            ofs << i << ' ' << F.flux_fx_momx[id] << '\n';
        }
        ofs.close();

    // Debug: 输出沿 x 方向 (j=6, k=6) 的 rhs_rhou 并退出
        if (!in_range) {
            std::cerr << "Debug probe (j=6,k=6) is outside computed range.\n";
            std::exit(EXIT_FAILURE);
        }

        std::ofstream ofs2("rhs_rhou_j6_k6.dat");
        if (!ofs2) {
            std::cerr << "Failed to open rhs_rhou_j6_k6.dat for writing.\n";
            std::exit(EXIT_FAILURE);
        }

        ofs2 << "# i rhs_rhou at j=6 k=6\n";
        for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
            int id = F.I(i, j_probe, k_probe);
            ofs2 << i << ' ' << F.rhs_rhou[id] << '\n';
        }
        ofs2.close();
        std::exit(EXIT_SUCCESS);
    }
}