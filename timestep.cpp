#include "field_structures.h"
#include <mpi.h>
#include <cmath>
#include <algorithm>

double compute_timestep(Field3D &F, const GridDesc &G, const SolverParams &P){
    double dt_local = 1e300;
    const int nx = F.L.nx, ny = F.L.ny, nz = F.L.nz;
    const int ngh = F.L.ngx; 
    double dt_global; 

    auto point_distance = [&](int i1, int j1, int k1, int i2, int j2, int k2) -> double {
        const int id1 = F.I(i1, j1, k1);
        const int id2 = F.I(i2, j2, k2);
        const double dx = F.coord_x[id2] - F.coord_x[id1];
        const double dy = F.coord_y[id2] - F.coord_y[id1];
        const double dz = F.coord_z[id2] - F.coord_z[id1];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    auto local_spacing = [&](int i, int j, int k, int di, int dj, int dk, double fallback) -> double {
        constexpr double eps = 1e-14;
        double h = fallback;

        if (di != 0) {
            double hp = point_distance(i, j, k, i + di, j, k);
            double hm = point_distance(i - di, j, k, i, j, k);
            if (hp > eps) h = std::min(h, hp);
            if (hm > eps) h = std::min(h, hm);
        } else if (dj != 0) {
            double hp = point_distance(i, j, k, i, j + dj, k);
            double hm = point_distance(i, j - dj, k, i, j, k);
            if (hp > eps) h = std::min(h, hp);
            if (hm > eps) h = std::min(h, hm);
        } else if (dk != 0) {
            double hp = point_distance(i, j, k, i, j, k + dk);
            double hm = point_distance(i, j, k - dk, i, j, k);
            if (hp > eps) h = std::min(h, hp);
            if (hm > eps) h = std::min(h, hm);
        }

        return (h > eps && std::isfinite(h)) ? h : fallback;
    };

    for(int k=ngh;k<ngh+nz;++k)
    for(int j=ngh;j<ngh+ny;++j)
    for(int i=ngh;i<ngh+nx;++i){
        int id = F.I(i,j,k);
        const double rho = F.rho[id];
        const double rhou = F.rhou[id];
        const double rhov = F.rhov[id];
        const double rhow = F.rhow[id];
        const double E = F.E[id];

        if (!(std::isfinite(rho) && std::isfinite(rhou) && std::isfinite(rhov) &&
              std::isfinite(rhow) && std::isfinite(E)) || rho <= 0.0) {
            dt_local = 0.0;
            continue;
        }

        const double u = rhou / rho;
        const double v = rhov / rho;
        const double w = rhow / rho;
        const double kinetic = 0.5 * rho * (u*u + v*v + w*w);
        const double p = (P.gamma - 1.0) * (E - kinetic);
        if (!(std::isfinite(p)) || p <= 0.0) {
            dt_local = 0.0;
            continue;
        }

        const double a2 = P.gamma * p / rho;
        if (!(std::isfinite(a2)) || a2 <= 0.0) {
            dt_local = 0.0;
            continue;
        }
        const double a = std::sqrt(a2);

        // Local thermodynamic temperature for viscosity/thermal diffusion estimate.
        const double T = p / (rho * P.Rgas);
        if (!(std::isfinite(T)) || T <= 0.0) {
            dt_local = 0.0;
            continue;
        }

        const double hx = local_spacing(i, j, k, 1, 0, 0, G.dx);
        const double hy = local_spacing(i, j, k, 0, 1, 0, G.dy);
        const double hz = local_spacing(i, j, k, 0, 0, 1, G.dz);

        const double inv_dt_conv = (std::fabs(u) + a) / hx
                                 + (std::fabs(v) + a) / hy
                                 + (std::fabs(w) + a) / hz;

        // Explicit diffusion CFL: dt <= O(1) / [ D * (1/hx^2 + 1/hy^2 + 1/hz^2) ]
        // Use the larger of momentum diffusion nu and thermal diffusion alpha.
        const double mu = P.get_mu(T);
        const double nu = (std::isfinite(mu) && mu > 0.0) ? (mu / rho) : 0.0;
        const double alpha = (std::isfinite(mu) && mu > 0.0 && P.Pr > 0.0) ? (mu / (rho * P.Pr)) : 0.0;
        const double diff = std::max(nu, alpha);
        const double inv_h2_sum = 1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz);
        const double inv_dt_visc = 2.0 * diff * inv_h2_sum;

        const double inv_dt = inv_dt_conv + inv_dt_visc;
        if (std::isfinite(inv_dt) && inv_dt > 0.0) {
            dt_local = std::min(dt_local, 1.0 / inv_dt);
        }
    }

    MPI_Allreduce(&dt_local,&dt_global,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    if (!std::isfinite(dt_global) || dt_global <= 0.0) {
        return 0.0;
    }
    return P.cfl*dt_global;
}