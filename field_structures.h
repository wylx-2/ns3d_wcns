/*
 * ns3d_field_structures.cpp
 *
 * Data structures for a 3D compressible Navier-Stokes solver (SoA storage)
 * - Grid and local decomposition descriptions
 * - Field3D: conserved & primitive variables, intermediates, face-centered fluxes
 * - Halo/ghost exchange helpers (MPI non-blocking, packing/unpacking)
 *
 * Requirements: C++17, MPI
 * Build example: mpicxx -O3 -std=c++17 ns3d_field_structures.cpp -o ns3d_fields
 *
 * Notes:
 *  - This file focuses on data layout, allocation and MPI halo mechanics.
 *  - Numerical kernels (RHS, reconstructions, viscous terms, Riemann solvers)
 *    should use the provided accessors and face buffers.
 */

#ifndef NS3D_FIELD_STRUCTURES_H
#define NS3D_FIELD_STRUCTURES_H

#include <mpi.h>
#include <vector>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>

// --------------------------- Grid and decomposition -------------------------

struct GridDesc {
    int global_nx = 0, global_ny = 0, global_nz = 0; // number of physical points (no ghost)
    double x0 = 0., y0 = 0., z0 = 0.;                // origin
    double Lx = 1., Ly = 1., Lz = 1.;                // domain lengths
    double dx = 1., dy = 1., dz = 1.;                // grid spacing (uniform)
};

struct CartDecomp {
    MPI_Comm cart_comm = MPI_COMM_NULL;
    int rank = 0;
    int size = 1;
    int dims[3] = {0,0,0};   // px,py,pz
    int coords[3] = {0,0,0}; // coordinate of this rank
    int periods[3] = {0,0,0};// always non-periodic by default
};

// Local description per MPI rank. nx,ny,nz are local physical cells (without ghosts)
struct LocalDesc {
    int nx=0, ny=0, nz=0;    // local points (physical)
    int ngx=1, ngy=1, ngz=1; // ghost layers per side
    int sx=0, sy=0, sz=0;    // sizes including ghosts: sx = nx + 2*ngx
    int ox=0, oy=0, oz=0;    // global origin indices (starting global index of local physical region)

    // neighbors' ranks (use MPI_Cart_shift to set)
    int nbr_xm=-1, nbr_xp=-1;
    int nbr_ym=-1, nbr_yp=-1;
    int nbr_zm=-1, nbr_zp=-1;
};

// --------------------------- indexing helpers --------------------------------

inline int idx3(int i, int j, int k, const LocalDesc &L) noexcept {
    // i in [0, sx-1], j in [0, sy-1], k in [0, sz-1]
    return (k * L.sy + j) * L.sx + i;
}

// face-centered indices for flux arrays
// For X-faces (between i and i+1), there are (sx-1) faces in i-direction; 
// physical faces count = nx+1, 
// so for i_face used indeed is [ghosts-1, ghosts+nx-1]*[ghosts, ghosts+nx-1]*[ghosts, ghosts+nz-1]
inline int idx_fx(int i_face, int j, int k, const LocalDesc &L) noexcept {
    // i_face in [0, sx-2]
    return (k * L.sy + j) * (L.sx - 1) + i_face;
}
inline int idx_fy(int i, int j_face, int k, const LocalDesc &L) noexcept {
    return (k * (L.sy - 1) + j_face) * L.sx + i;
}
inline int idx_fz(int i, int j, int k_face, const LocalDesc &L) noexcept {
    return (k_face * L.sy + j) * L.sx + i;
}

// --------------------------- Physical & numerical parameters -----------------

struct SolverParams {
    double gamma = 1.4;      // ratio of specific heats
    double Pr = 0.71;        // Prandtl number
    double Ma = 0.4;        // Mach number
    double Re = 500.0;      // Reynolds number
    double cfl = 0.1;
    double dt_fixed = -1.0;   // if >0, use fixed time step

    double mu = 1.0 / Re;      // dynamic viscosity
    double S_ref = 110.4/273.0;   // Sutherland's constant
    double get_mu(double T) const
    {   
        // constant viscosity for now
        // return mu; 

        // Sutherland's law
        return mu * pow(T, 1.5) * (1 + S_ref) / (T + S_ref);

        // power law
        // return pow(T, 0.76);
    }
    double Cv = 1.0/(gamma*(gamma-1.0)*Ma*Ma);
    double Cp = Cv*gamma;
    double Rgas = 1.0/(Ma*Ma*gamma);

    bool use_periodic = false; // convenience
    // Flux Vector Splitting (FVS) selection
    enum class FVS_Type {
        StegerWarming,
        LaxFriedrichs,
        Rusanov,         // Local Lax-Friedrichs
        VanLeer
    };
    FVS_Type fvs_type = FVS_Type::StegerWarming;
    // Reconstruction selection for face reconstruction routines
    enum class Reconstruction {
        WENO5,     // stencil-based WENO5
        WCNS,      // Weighted Compact Nonlinear Scheme
        LINEAR,    // simple linear reconstruction
        MDCD,      // Minimum Dissipation controlled dispersion
    };
    enum class ViscousScheme {
        C6th,      // Sixth-order central difference
        C4th       // Fourth-order central difference
    };
    Reconstruction recon = Reconstruction::WENO5;
    ViscousScheme vis_scheme = ViscousScheme::C6th;
    double mdcd_diss = 0.01;  // MDCD dissipation coefficient
    double mdcd_disp = 0.0463783;  // MDCD dispersion coefficient

    // runtime stencil size for reconstructions
    int stencil = 6;
    int ghost_layers = 3;
    // characteristic-wise or component-wise reconstruction
    bool char_recon = false;

    // boundary types at each side (if neighbor is MPI_PROC_NULL)
    enum class BCType { Periodic, Wall, Symmetry, Outflow, Inflow };
    BCType bc_xmin = BCType::Periodic;
    BCType bc_xmax = BCType::Periodic;
    BCType bc_ymin = BCType::Periodic;
    BCType bc_ymax = BCType::Periodic;
    BCType bc_zmin = BCType::Periodic;
    BCType bc_zmax = BCType::Periodic;

    // simulation control
    int max_steps = 1;
    int monitor_Stepfreq = 1;
    double output_Timefreq = 1.0;
    double TotalTime = 1.0;

    // output/post-processing flags
    bool post_basicfield = true;
    bool isotropic_analyse = true;

    // monitor switches
    bool monitor_res = true;
    bool monitor_energy = true;
};

// --------------------------- Halo exchange requests --------------------------
struct HaloRequests {
    std::vector<MPI_Request> reqs;
    std::vector<MPI_Status> stats;
};

// --------------------------- Field container (SoA) ---------------------------

struct Field3D {
    // sizes
    LocalDesc L;

    // main conserved variables (SoA): shape = sx * sy * sz
    // Conserved variable ordering: rho, rho*u, rho*v, rho*w, E
    std::vector<double> rho, rhou, rhov, rhow, E;
    std::vector<double> rho0, rhou0, rhov0, rhow0, E0;
    std::vector<double> res_rho, res_rhou, res_rhov, res_rhow, res_E;

    // golbal residual accumulators
    double global_res_rho = 0.0;
    double global_res_rhou = 0.0;
    double global_res_rhov = 0.0;
    double global_res_rhow = 0.0;
    double global_res_E = 0.0;
    double global_Etot = 0.0;

    // Primitive variables (optional cache) -- keep for convenience / performance
    std::vector<double> u, v, w, p, T;

    // Intermediate / RHS arrays
    std::vector<double> rhs_rho, rhs_rhou, rhs_rhov, rhs_rhow, rhs_E;

    // flux arrays at cell centers (for FD schemes)
    std::vector<double> Fflux_mass, Fflux_momx, Fflux_momy, Fflux_momz, Fflux_E;
    std::vector<double> Hflux_mass, Hflux_momx, Hflux_momy, Hflux_momz, Hflux_E;
    std::vector<double> Gflux_mass, Gflux_momx, Gflux_momy, Gflux_momz, Gflux_E;

    std::vector<double> Fvflux_mass, Fvflux_momx, Fvflux_momy, Fvflux_momz, Fvflux_E;
    std::vector<double> Hvflux_mass, Hvflux_momx, Hvflux_momy, Hvflux_momz, Hvflux_E;
    std::vector<double> Gvflux_mass, Gvflux_momx, Gvflux_momy, Gvflux_momz, Gvflux_E;

    // Additional intermediate arrays if needed (gradients, viscous stresses, etc.)
    std::vector<double> du_dx, du_dy, du_dz;
    std::vector<double> dv_dx, dv_dy, dv_dz;
    std::vector<double> dw_dx, dw_dy, dw_dz;
    std::vector<double> dT_dx, dT_dy, dT_dz;

    // Face-centered flux arrays (half-node fluxes) - flux components for each face
    // these numerical fluxes are computed by reconstruction
    // For X-face fluxes we store flux components (mass, mom_x, mom_y, mom_z, E) per face
    std::vector<double> flux_fx_mass, flux_fx_momx, flux_fx_momy, flux_fx_momz, flux_fx_E;
    std::vector<double> flux_fy_mass, flux_fy_momx, flux_fy_momy, flux_fy_momz, flux_fy_E;
    std::vector<double> flux_fz_mass, flux_fz_momx, flux_fz_momy, flux_fz_momz, flux_fz_E;

    // constructor
    Field3D() = default;

    void allocate(const LocalDesc &Ld) {
        L = Ld;
        L.sx = L.nx + 2*L.ngx;
        L.sy = L.ny + 2*L.ngy;
        L.sz = L.nz + 2*L.ngz;
        const int tot = L.sx * L.sy * L.sz;

        rho.assign(tot, 1.0);
        rhou.assign(tot, 0.0);
        rhov.assign(tot, 0.0);
        rhow.assign(tot, 0.0);
        E.assign(tot, 0.0);

        rho0.assign(tot, 0.0);
        rhou0.assign(tot, 0.0);
        rhov0.assign(tot, 0.0);
        rhow0.assign(tot, 0.0);
        E0.assign(tot, 0.0);

        res_rho.assign(tot, 0.0);
        res_rhou.assign(tot, 0.0);
        res_rhov.assign(tot, 0.0);
        res_rhow.assign(tot, 0.0);
        res_E.assign(tot, 0.0);

        u.assign(tot, 0.0);
        v.assign(tot, 0.0);
        w.assign(tot, 0.0);
        p.assign(tot, 0.0);
        T.assign(tot, 0.0);

        rhs_rho.assign(tot, 0.0);
        rhs_rhou.assign(tot, 0.0);
        rhs_rhov.assign(tot, 0.0);
        rhs_rhow.assign(tot, 0.0);
        rhs_E.assign(tot, 0.0);

        Fflux_mass.assign(tot, 0.0);
        Fflux_momx.assign(tot, 0.0);
        Fflux_momy.assign(tot, 0.0);
        Fflux_momz.assign(tot, 0.0);
        Fflux_E.assign(tot, 0.0);

        Hflux_mass.assign(tot, 0.0);
        Hflux_momx.assign(tot, 0.0);
        Hflux_momy.assign(tot, 0.0);
        Hflux_momz.assign(tot, 0.0);
        Hflux_E.assign(tot, 0.0);

        Gflux_mass.assign(tot, 0.0);
        Gflux_momx.assign(tot, 0.0);
        Gflux_momy.assign(tot, 0.0);
        Gflux_momz.assign(tot, 0.0);
        Gflux_E.assign(tot, 0.0);

        Fvflux_mass.assign(tot, 0.0);
        Fvflux_momx.assign(tot, 0.0);
        Fvflux_momy.assign(tot, 0.0);
        Fvflux_momz.assign(tot, 0.0);
        Fvflux_E.assign(tot, 0.0);

        Hvflux_mass.assign(tot, 0.0);
        Hvflux_momx.assign(tot, 0.0);
        Hvflux_momy.assign(tot, 0.0);
        Hvflux_momz.assign(tot, 0.0);
        Hvflux_E.assign(tot, 0.0);

        Gvflux_mass.assign(tot, 0.0);
        Gvflux_momx.assign(tot, 0.0);
        Gvflux_momy.assign(tot, 0.0);
        Gvflux_momz.assign(tot, 0.0);
        Gvflux_E.assign(tot, 0.0);

        du_dx.assign(tot, 0.0);
        du_dy.assign(tot, 0.0);
        du_dz.assign(tot, 0.0);
        dv_dx.assign(tot, 0.0);
        dv_dy.assign(tot, 0.0);
        dv_dz.assign(tot, 0.0);
        dw_dx.assign(tot, 0.0);
        dw_dy.assign(tot, 0.0);
        dw_dz.assign(tot, 0.0);
        dT_dx.assign(tot, 0.0);
        dT_dy.assign(tot, 0.0);
        dT_dz.assign(tot, 0.0);

        // allocate face arrays sizes:
        int fx_count = (L.sx - 1) * L.sy * L.sz; // faces between i and i+1
        int fy_count = L.sx * (L.sy - 1) * L.sz;
        int fz_count = L.sx * L.sy * (L.sz - 1);

        flux_fx_mass.assign(fx_count, 0.0);
        flux_fx_momx.assign(fx_count, 0.0);
        flux_fx_momy.assign(fx_count, 0.0);
        flux_fx_momz.assign(fx_count, 0.0);
        flux_fx_E.assign(fx_count, 0.0);

        flux_fy_mass.assign(fy_count, 0.0);
        flux_fy_momx.assign(fy_count, 0.0);
        flux_fy_momy.assign(fy_count, 0.0);
        flux_fy_momz.assign(fy_count, 0.0);
        flux_fy_E.assign(fy_count, 0.0);

        flux_fz_mass.assign(fz_count, 0.0);
        flux_fz_momx.assign(fz_count, 0.0);
        flux_fz_momy.assign(fz_count, 0.0);
        flux_fz_momz.assign(fz_count, 0.0);
        flux_fz_E.assign(fz_count, 0.0);
    }

    inline int I(int i, int j, int k) const noexcept { return idx3(i,j,k,L); }

    // accessors for conserved
    inline double& Rho(int i, int j, int k) noexcept { return rho[I(i,j,k)]; }
    inline double& RhoU(int i, int j, int k) noexcept { return rhou[I(i,j,k)]; }
    inline double& RhoV(int i, int j, int k) noexcept { return rhov[I(i,j,k)]; }
    inline double& RhoW(int i, int j, int k) noexcept { return rhow[I(i,j,k)]; }
    inline double& Eint(int i, int j, int k) noexcept { return E[I(i,j,k)]; }

    // accessors for primitives
    inline double& U(int i, int j, int k) noexcept { return u[I(i,j,k)]; }
    inline double& V(int i, int j, int k) noexcept { return v[I(i,j,k)]; }
    inline double& W(int i, int j, int k) noexcept { return w[I(i,j,k)]; }
    inline double& P(int i, int j, int k) noexcept { return p[I(i,j,k)]; }
    inline double& Temp(int i, int j, int k) noexcept { return T[I(i,j,k)]; }

    // accessors for RHS
    inline double& RHS_Rho(int i, int j, int k) noexcept { return rhs_rho[I(i,j,k)]; }
    inline double& RHS_RhoU(int i, int j, int k) noexcept { return rhs_rhou[I(i,j,k)]; }
    inline double& RHS_RhoV(int i, int j, int k) noexcept { return rhs_rhov[I(i,j,k)]; }
    inline double& RHS_RhoW(int i, int j, int k) noexcept { return rhs_rhow[I(i,j,k)]; }
    inline double& RHS_E(int i, int j, int k) noexcept { return rhs_E[I(i,j,k)]; }

    // Debugging helpers: set/add RHS for mass with logging (prints each modification)
    inline void ShowRHS_Rho(int i, int j, int k, int num) noexcept {
        int id = I(i,j,k);
        std::cout << "[RHS_DEBUG "<< num <<"] RHS_Rho(" << i << "," << j << "," << k << ") = " << rhs_rho[id] << std::endl;
    }


    // accessors for fluxes at cell centers
    inline double& Fflux_Mass(int i, int j, int k) noexcept { return Fflux_mass[I(i,j,k)]; }
    inline double& Fflux_MomX(int i, int j, int k) noexcept { return Fflux_momx[I(i,j,k)]; }
    inline double& Fflux_MomY(int i, int j, int k) noexcept { return Fflux_momy[I(i,j,k)]; }
    inline double& Fflux_MomZ(int i, int j, int k) noexcept { return Fflux_momz[I(i,j,k)]; }
    inline double& Fflux_Energy(int i, int j, int k) noexcept { return Fflux_E[I(i,j,k)]; }

    inline double& Hflux_Mass(int i, int j, int k) noexcept { return Hflux_mass[I(i,j,k)]; }
    inline double& Hflux_MomX(int i, int j, int k) noexcept { return Hflux_momx[I(i,j,k)]; }
    inline double& Hflux_MomY(int i, int j, int k) noexcept { return Hflux_momy[I(i,j,k)]; }
    inline double& Hflux_MomZ(int i, int j, int k) noexcept { return Hflux_momz[I(i,j,k)]; }
    inline double& Hflux_Energy(int i, int j, int k) noexcept { return Hflux_E[I(i,j,k)]; }

    inline double& Gflux_Mass(int i, int j, int k) noexcept { return Gflux_mass[I(i,j,k)]; }
    inline double& Gflux_MomX(int i, int j, int k) noexcept { return Gflux_momx[I(i,j,k)]; }
    inline double& Gflux_MomY(int i, int j, int k) noexcept { return Gflux_momy[I(i,j,k)]; }
    inline double& Gflux_MomZ(int i, int j, int k) noexcept { return Gflux_momz[I(i,j,k)]; }
    inline double& Gflux_Energy(int i, int j, int k) noexcept { return Gflux_E[I(i,j,k)]; }

    // accessors for face-centered fluxes
    inline double& FX_mass(int i_face, int j, int k) noexcept { return flux_fx_mass[idx_fx(i_face,j,k,L)]; }
    inline double& FX_momx(int i_face, int j, int k) noexcept { return flux_fx_momx[idx_fx(i_face,j,k,L)]; }
    inline double& FX_momy(int i_face, int j, int k) noexcept { return flux_fx_momy[idx_fx(i_face,j,k,L)]; }
    inline double& FX_momz(int i_face, int j, int k) noexcept { return flux_fx_momz[idx_fx(i_face,j,k,L)]; }
    inline double& FX_E(int i_face, int j, int k) noexcept { return flux_fx_E[idx_fx(i_face,j,k,L)]; }

    inline double& FY_mass(int i, int j_face, int k) noexcept { return flux_fy_mass[idx_fy(i,j_face,k,L)]; }
    inline double& FY_momx(int i, int j_face, int k) noexcept { return flux_fy_momx[idx_fy(i,j_face,k,L)]; }
    inline double& FY_momy(int i, int j_face, int k) noexcept { return flux_fy_momy[idx_fy(i,j_face,k,L)]; }
    inline double& FY_momz(int i, int j_face, int k) noexcept { return flux_fy_momz[idx_fy(i,j_face,k,L)]; }
    inline double& FY_E(int i, int j_face, int k) noexcept { return flux_fy_E[idx_fy(i,j_face,k,L)]; }

    inline double& FZ_mass(int i, int j, int k_face) noexcept { return flux_fz_mass[idx_fz(i,j,k_face,L)]; }
    inline double& FZ_momx(int i, int j, int k_face) noexcept { return flux_fz_momx[idx_fz(i,j,k_face,L)]; }
    inline double& FZ_momy(int i, int j, int k_face) noexcept { return flux_fz_momy[idx_fz(i,j,k_face,L)]; }
    inline double& FZ_momz(int i, int j, int k_face) noexcept { return flux_fz_momz[idx_fz(i,j,k_face,L)]; }
    inline double& FZ_E(int i, int j, int k_face) noexcept { return flux_fz_E[idx_fz(i,j,k_face,L)]; }

    // convert conserved -> primitive for the inner domain (including ghost)
    void conservedToPrimitive(const SolverParams &par) {
        const double gamma = par.gamma;
        for (int k = 0; k < L.sz; ++k)
            for (int j = 0; j < L.sy; ++j)
                for (int i = 0; i < L.sx; ++i) {
                    int id = I(i,j,k);
                    double rr = rho[id];
                    if (rr <= 0.0) {
                        // avoid dividing by zero; set fallback small positive density
                        rr = 1e-12; 
                        rho[id] = rr;
                    }
                    u[id] = rhou[id] / rr;
                    v[id] = rhov[id] / rr;
                    w[id] = rhow[id] / rr;
                    double kinetic = 0.5 * (u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);
                    double e_internal = E[id] / rr - kinetic; // specific internal energy
                    // pressure using ideal gas
                    p[id] = (gamma - 1.0) * rr * e_internal; // NOTE: here p = (gamma-1) * rho * e
                    T[id] = p[id] / (rr * par.Rgas);
                }
    }

    //convert primitive -> conserved for the inner domain (including ghost)
    void primitiveToConserved(const SolverParams &par) {
        const double gamma = par.gamma;
        for (int k = 0; k < L.sz; ++k)
            for (int j = 0; j < L.sy; ++j)
                for (int i = 0; i < L.sx; ++i) {
                    int id = I(i,j,k);
                    double rr = rho[id];
                    if (rr <= 0.0) {
                        // avoid dividing by zero; set fallback small positive density
                        rr = 1e-12; 
                        rho[id] = rr;
                    }
                    rhou[id] = rr * u[id];
                    rhov[id] = rr * v[id];
                    rhow[id] = rr * w[id];
                    double kinetic = 0.5 * (u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);
                    double e_internal = p[id] / ((gamma - 1.0) * rr);
                    E[id] = rr * (e_internal + kinetic);
                    T[id] = p[id] / (rr * par.Rgas);
                }
    }   

    // record current conserved variables into _0 arrays
    void recordConservedTo0() {
        std::copy(rho.begin(), rho.end(), rho0.begin());
        std::copy(rhou.begin(), rhou.end(), rhou0.begin());
        std::copy(rhov.begin(), rhov.end(), rhov0.begin());
        std::copy(rhow.begin(), rhow.end(), rhow0.begin());
        std::copy(E.begin(), E.end(), E0.begin());
    }

    // residual update: res_var = var - var0
    void updateResiduals() {
        for (size_t i = 0; i < rho.size(); ++i) {
            res_rho[i]  = rho[i]  - rho0[i];
            res_rhou[i] = rhou[i] - rhou0[i];
            res_rhov[i] = rhov[i] - rhov0[i];
            res_rhow[i] = rhow[i] - rhow0[i];
            res_E[i]    = E[i]    - E0[i];
        }
    }
};

// --------------------------- MPI halo exchange utilities ----------------------

// Pack a set of variables for a X-face (left or right) into a contiguous buffer
// We pack multiple variables in one buffer to reduce number of messages
// face side: if send_left==true, pack the left-most 'ngx' physical layers (i=ngx..ngx+ngx-1) to send to left neighbor

inline void pack_x_face_send(const Field3D &F, std::vector<double> &buf, int send_left) {
    const LocalDesc &L = F.L;
    int gx = L.ngx;
    int ny = L.ny, nz = L.nz;
    int sx = L.sx, sy = L.sy, sz = L.sz;
    // buffer size expected = gx * ny * nz * 5
    int p = 0;
    int istart = send_left ? L.ngx : (L.ngx + L.nx - gx);
    for (int k = L.ngz; k < L.ngz + nz; ++k) {
        for (int j = L.ngy; j < L.ngy + ny; ++j) {
            for (int ii = 0; ii < gx; ++ii) {
                int i = istart + ii;
                int id = F.I(i,j,k);
                buf[p++] = F.rho[id];
                buf[p++] = F.u[id];
                buf[p++] = F.v[id];
                buf[p++] = F.w[id];
                buf[p++] = F.p[id];
            }
        }
    }
    assert(p == (int)buf.size());
}

// Unpack received X-face buffer into ghost cells on side recv_left (if recv_left true, unpack into left ghost cells)
inline void unpack_x_face_recv(Field3D &F, const std::vector<double> &buf, int recv_left) {
    const LocalDesc &L = F.L;
    int gx = L.ngx;
    int ny = L.ny, nz = L.nz;
    int istart = recv_left ? 0 : (L.ngx + L.nx);
    int p = 0;
    for (int k = L.ngz; k < L.ngz + nz; ++k) {
        for (int j = L.ngy; j < L.ngy + ny; ++j) {
            for (int ii = 0; ii < gx; ++ii) {
                int i = istart + ii;
                int id = F.I(i,j,k);
                F.rho[id]  = buf[p++];
                F.u[id] = buf[p++];
                F.v[id] = buf[p++];
                F.w[id] = buf[p++];
                F.p[id] = buf[p++];
            }
        }
    }
    assert(p == (int)buf.size());
}

// Similar packing/unpacking for Y and Z faces (omitted full duplication for brevity)
inline void pack_y_face_send(const Field3D &F, std::vector<double> &buf, int send_bottom) {
    const LocalDesc &L = F.L;
    int gy = L.ngy;
    int nx = L.nx, nz = L.nz;
    int p = 0;
    int jstart = send_bottom ? L.ngy : (L.ngy + L.ny - gy);
    for (int k = L.ngz; k < L.ngz + nz; ++k) {
        for (int jj = 0; jj < gy; ++jj) {
            int j = jstart + jj;
            for (int i = L.ngx; i < L.ngx + nx; ++i) {
                int id = F.I(i,j,k);
                buf[p++] = F.rho[id];
                buf[p++] = F.u[id];
                buf[p++] = F.v[id];
                buf[p++] = F.w[id];
                buf[p++] = F.p[id];
            }
        }
    }
    assert(p == (int)buf.size());
}
inline void unpack_y_face_recv(Field3D &F, const std::vector<double> &buf, int recv_bottom) {
    const LocalDesc &L = F.L;
    int gy = L.ngy;
    int nx = L.nx, nz = L.nz;
    int jstart = recv_bottom ? 0 : (L.ngy + L.ny);
    int p = 0;
    for (int k = L.ngz; k < L.ngz + nz; ++k) {
        for (int jj = 0; jj < gy; ++jj) {
            int j = jstart + jj;
            for (int i = L.ngx; i < L.ngx + nx; ++i) {
                int id = F.I(i,j,k);
                F.rho[id]  = buf[p++];
                F.u[id] = buf[p++];
                F.v[id] = buf[p++];
                F.w[id] = buf[p++];
                F.p[id] = buf[p++];
            }
        }
    }
    assert(p == (int)buf.size());
}

inline void pack_z_face_send(const Field3D &F, std::vector<double> &buf, int send_back) {
    const LocalDesc &L = F.L;
    int gz = L.ngz;
    int nx = L.nx, ny = L.ny;
    int p = 0;
    int kstart = send_back ? L.ngz : (L.ngz + L.nz - gz);
    for (int kk = 0; kk < gz; ++kk) {
        int k = kstart + kk;
        for (int j = L.ngy; j < L.ngy + ny; ++j) {
            for (int i = L.ngx; i < L.ngx + nx; ++i) {
                int id = F.I(i,j,k);
                buf[p++] = F.rho[id];
                buf[p++] = F.u[id];
                buf[p++] = F.v[id];
                buf[p++] = F.w[id];
                buf[p++] = F.p[id];
            }
        }
    }
    assert(p == (int)buf.size());
}
inline void unpack_z_face_recv(Field3D &F, const std::vector<double> &buf, int recv_back) {
    const LocalDesc &L = F.L;
    int gz = L.ngz;
    int nx = L.nx, ny = L.ny;
    int kstart = recv_back ? 0 : (L.ngz + L.nz);
    int p = 0;
    for (int kk = 0; kk < gz; ++kk) {
        int k = kstart + kk;
        for (int j = L.ngy; j < L.ngy + ny; ++j) {
            for (int i = L.ngx; i < L.ngx + nx; ++i) {
                int id = F.I(i,j,k);
                F.rho[id]  = buf[p++];
                F.u[id] = buf[p++];
                F.v[id] = buf[p++];
                F.w[id] = buf[p++];
                F.p[id] = buf[p++];
            }
        }
    }
    assert(p == (int)buf.size());
}

// --------------------------- packing/unpacking for viscous flux --------------
inline void pack_x_face_send_vis_flux(const Field3D &F, std::vector<double> &buf, int send_left) {
    const LocalDesc &L = F.L;
    int gx = L.ngx;
    int ny = L.ny, nz = L.nz;
    int p = 0;
    int istart = send_left ? L.ngx : (L.ngx + L.nx - gx);
    for (int k = L.ngz; k < L.ngz + nz; ++k) {
        for (int j = L.ngy; j < L.ngy + ny; ++j) {
            for (int ii = 0; ii < gx; ++ii) {
                int i = istart + ii;
                int id = F.I(i,j,k);
                buf[p++] = F.Fvflux_momx[id]; buf[p++] = F.Fvflux_momy[id]; buf[p++] = F.Fvflux_momz[id];buf[p++] = F.Fvflux_E[id]; 
                buf[p++] = F.Hvflux_momx[id]; buf[p++] = F.Hvflux_momy[id]; buf[p++] = F.Hvflux_momz[id];buf[p++] = F.Hvflux_E[id];
                buf[p++] = F.Gvflux_momx[id]; buf[p++] = F.Gvflux_momy[id]; buf[p++] = F.Gvflux_momz[id];buf[p++] = F.Gvflux_E[id];
            }
        }
    }
    assert(p == (int)buf.size());
}

inline void unpack_x_face_recv_vis_flux(Field3D &F, const std::vector<double> &buf, int recv_left) {
    const LocalDesc &L = F.L;
    int gx = L.ngx;
    int ny = L.ny, nz = L.nz;
    int istart = recv_left ? 0 : (L.ngx + L.nx);
    int p = 0;
    for (int k = L.ngz; k < L.ngz + nz; ++k) {
        for (int j = L.ngy; j < L.ngy + ny; ++j) {
            for (int ii = 0; ii < gx; ++ii) {
                int i = istart + ii;
                int id = F.I(i,j,k);
                F.Fvflux_momx[id] = buf[p++]; F.Fvflux_momy[id] = buf[p++]; F.Fvflux_momz[id] = buf[p++]; F.Fvflux_E[id] = buf[p++];
                F.Hvflux_momx[id] = buf[p++]; F.Hvflux_momy[id] = buf[p++]; F.Hvflux_momz[id] = buf[p++]; F.Hvflux_E[id] = buf[p++];
                F.Gvflux_momx[id] = buf[p++]; F.Gvflux_momy[id] = buf[p++]; F.Gvflux_momz[id] = buf[p++]; F.Gvflux_E[id] = buf[p++];
            }
        }
    }
    assert(p == (int)buf.size());
}

inline void pack_y_face_send_vis_flux(const Field3D &F, std::vector<double> &buf, int send_bottom) {
    const LocalDesc &L = F.L;
    int gy = L.ngy;
    int nx = L.nx, nz = L.nz;
    int p = 0;
    int jstart = send_bottom ? L.ngy : (L.ngy + L.ny - gy);
    for (int k = L.ngz; k < L.ngz + nz; ++k) {
        for (int jj = 0; jj < gy; ++jj) {
            int j = jstart + jj;
            for (int i = L.ngx; i < L.ngx + nx; ++i) {
                int id = F.I(i,j,k);
                buf[p++] = F.Fvflux_momx[id]; buf[p++] = F.Fvflux_momy[id]; buf[p++] = F.Fvflux_momz[id];buf[p++] = F.Fvflux_E[id];
                buf[p++] = F.Hvflux_momx[id]; buf[p++] = F.Hvflux_momy[id]; buf[p++] = F.Hvflux_momz[id];buf[p++] = F.Hvflux_E[id];
                buf[p++] = F.Gvflux_momx[id]; buf[p++] = F.Gvflux_momy[id]; buf[p++] = F.Gvflux_momz[id];buf[p++] = F.Gvflux_E[id];
            }
        }
    }
    assert(p == (int)buf.size());
}

inline void unpack_y_face_recv_vis_flux(Field3D &F, const std::vector<double> &buf, int recv_bottom) {
    const LocalDesc &L = F.L;
    int gy = L.ngy;
    int nx = L.nx, nz = L.nz;
    int jstart = recv_bottom ? 0 : (L.ngy + L.ny);
    int p = 0;
    for (int k = L.ngz; k < L.ngz + nz; ++k) {
        for (int jj = 0; jj < gy; ++jj) {
            int j = jstart + jj;
            for (int i = L.ngx; i < L.ngx + nx; ++i) {
                int id = F.I(i,j,k);
                F.Fvflux_momx[id] = buf[p++]; F.Fvflux_momy[id] = buf[p++]; F.Fvflux_momz[id] = buf[p++]; F.Fvflux_E[id] = buf[p++];
                F.Hvflux_momx[id] = buf[p++]; F.Hvflux_momy[id] = buf[p++]; F.Hvflux_momz[id] = buf[p++]; F.Hvflux_E[id] = buf[p++];
                F.Gvflux_momx[id] = buf[p++]; F.Gvflux_momy[id] = buf[p++]; F.Gvflux_momz[id] = buf[p++]; F.Gvflux_E[id] = buf[p++];
            }
        }
    }
    assert(p == (int)buf.size());
}

inline void pack_z_face_send_vis_flux(const Field3D &F, std::vector<double> &buf, int send_back) {
    const LocalDesc &L = F.L;
    int gz = L.ngz;
    int nx = L.nx, ny = L.ny;
    int p = 0;
    int kstart = send_back ? L.ngz : (L.ngz + L.nz - gz);
    for (int kk = 0; kk < gz; ++kk) {
        int k = kstart + kk;
        for (int j = L.ngy; j < L.ngy + ny; ++j) {
            for (int i = L.ngx; i < L.ngx + nx; ++i) {
                int id = F.I(i,j,k);
                buf[p++] = F.Fvflux_momx[id]; buf[p++] = F.Fvflux_momy[id]; buf[p++] = F.Fvflux_momz[id];buf[p++] = F.Fvflux_E[id];
                buf[p++] = F.Hvflux_momx[id]; buf[p++] = F.Hvflux_momy[id]; buf[p++] = F.Hvflux_momz[id];buf[p++] = F.Hvflux_E[id];
                buf[p++] = F.Gvflux_momx[id]; buf[p++] = F.Gvflux_momy[id]; buf[p++] = F.Gvflux_momz[id];buf[p++] = F.Gvflux_E[id];
            }
        }
    }
    assert(p == (int)buf.size());
}

inline void unpack_z_face_recv_vis_flux(Field3D &F, const std::vector<double> &buf, int recv_back) {
    const LocalDesc &L = F.L;
    int gz = L.ngz;
    int nx = L.nx, ny = L.ny;
    int kstart = recv_back ? 0 : (L.ngz + L.nz);
    int p = 0;
    for (int kk = 0; kk < gz; ++kk) {
        int k = kstart + kk;
        for (int j = L.ngy; j < L.ngy + ny; ++j) {
            for (int i = L.ngx; i < L.ngx + nx; ++i) {
                int id = F.I(i,j,k);
                F.Fvflux_momx[id] = buf[p++]; F.Fvflux_momy[id] = buf[p++]; F.Fvflux_momz[id] = buf[p++]; F.Fvflux_E[id] = buf[p++];
                F.Hvflux_momx[id] = buf[p++]; F.Hvflux_momy[id] = buf[p++]; F.Hvflux_momz[id] = buf[p++]; F.Hvflux_E[id] = buf[p++];
                F.Gvflux_momx[id] = buf[p++]; F.Gvflux_momy[id] = buf[p++]; F.Gvflux_momz[id] = buf[p++]; F.Gvflux_E[id] = buf[p++];
            }
        }
    }
    assert(p == (int)buf.size());
}

// High-level halo exchange routine (non-blocking) for gradient fields
inline void exchange_halos_viscous_flux(Field3D &F, CartDecomp &C, LocalDesc &L, HaloRequests &out_reqs) {
    int gx = L.ngx, gy = L.ngy, gz = L.ngz;
    int nx = L.nx, ny = L.ny, nz = L.nz;

    // viscous fluxes: 3 directions * 4 variables (rhou, rhov, rhow, E) = 12 variables
    const int nvars = 12;
    int pack_x_size = gx * ny * nz * nvars;
    int pack_y_size = gy * nx * nz * nvars;
    int pack_z_size = gz * nx * ny * nvars;

    std::vector<double> send_x_left(pack_x_size), send_x_right(pack_x_size);
    std::vector<double> recv_x_left(pack_x_size), recv_x_right(pack_x_size);
    std::vector<double> send_y_bot(pack_y_size), send_y_top(pack_y_size);
    std::vector<double> recv_y_bot(pack_y_size), recv_y_top(pack_y_size);
    std::vector<double> send_z_back(pack_z_size), send_z_front(pack_z_size);
    std::vector<double> recv_z_back(pack_z_size), recv_z_front(pack_z_size);

    pack_x_face_send_vis_flux(F, send_x_left, 1);
    pack_x_face_send_vis_flux(F, send_x_right, 0);
    pack_y_face_send_vis_flux(F, send_y_bot, 1);
    pack_y_face_send_vis_flux(F, send_y_top, 0);
    pack_z_face_send_vis_flux(F, send_z_back, 1);
    pack_z_face_send_vis_flux(F, send_z_front, 0);

    out_reqs.reqs.clear();
    out_reqs.stats.resize(6);
    out_reqs.reqs.resize(12);

    int tag = 200; // separate tag space from conserved
    // X-direction
    MPI_Irecv(recv_x_left.data(), pack_x_size, MPI_DOUBLE, L.nbr_xm, tag, C.cart_comm, &out_reqs.reqs[0]);
    MPI_Irecv(recv_x_right.data(), pack_x_size, MPI_DOUBLE, L.nbr_xp, tag+1, C.cart_comm, &out_reqs.reqs[1]);
    MPI_Isend(send_x_right.data(), pack_x_size, MPI_DOUBLE, L.nbr_xp, tag, C.cart_comm, &out_reqs.reqs[2]);
    MPI_Isend(send_x_left.data(), pack_x_size, MPI_DOUBLE, L.nbr_xm, tag+1, C.cart_comm, &out_reqs.reqs[3]);

    // Y-direction
    MPI_Irecv(recv_y_bot.data(), pack_y_size, MPI_DOUBLE, L.nbr_ym, tag+2, C.cart_comm, &out_reqs.reqs[4]);
    MPI_Irecv(recv_y_top.data(), pack_y_size, MPI_DOUBLE, L.nbr_yp, tag+3, C.cart_comm, &out_reqs.reqs[5]);
    MPI_Isend(send_y_top.data(), pack_y_size, MPI_DOUBLE, L.nbr_yp, tag+2, C.cart_comm, &out_reqs.reqs[6]);
    MPI_Isend(send_y_bot.data(), pack_y_size, MPI_DOUBLE, L.nbr_ym, tag+3, C.cart_comm, &out_reqs.reqs[7]);

    // Z-direction
    MPI_Irecv(recv_z_back.data(), pack_z_size, MPI_DOUBLE, L.nbr_zm, tag+4, C.cart_comm, &out_reqs.reqs[8]);
    MPI_Irecv(recv_z_front.data(), pack_z_size, MPI_DOUBLE, L.nbr_zp, tag+5, C.cart_comm, &out_reqs.reqs[9]);
    MPI_Isend(send_z_front.data(), pack_z_size, MPI_DOUBLE, L.nbr_zp, tag+4, C.cart_comm, &out_reqs.reqs[10]);
    MPI_Isend(send_z_back.data(), pack_z_size, MPI_DOUBLE, L.nbr_zm, tag+5, C.cart_comm, &out_reqs.reqs[11]);

    MPI_Waitall((int)out_reqs.reqs.size(), out_reqs.reqs.data(), MPI_STATUSES_IGNORE);

    unpack_x_face_recv_vis_flux(F, recv_x_left, 1);
    unpack_x_face_recv_vis_flux(F, recv_x_right, 0);
    unpack_y_face_recv_vis_flux(F, recv_y_bot, 1);
    unpack_y_face_recv_vis_flux(F, recv_y_top, 0);
    unpack_z_face_recv_vis_flux(F, recv_z_back, 1);
    unpack_z_face_recv_vis_flux(F, recv_z_front, 0);
}

// High-level halo exchange routine (non-blocking) for conserved variables
// This exchanges ghost layers in x, y, z directions. It assumes periodic or neighbor ranks set in LocalDesc.
inline void exchange_halos_physical(Field3D &F, CartDecomp &C, LocalDesc &L, HaloRequests &out_reqs) {
    // compute buffer sizes
    int gx = L.ngx, gy = L.ngy, gz = L.ngz;
    int nx = L.nx, ny = L.ny, nz = L.nz;

    int pack_x_size = gx * ny * nz * 5; // 5 conserved variables
    int pack_y_size = gy * nx * nz * 5;
    int pack_z_size = gz * nx * ny * 5;

    std::vector<double> send_x_left(pack_x_size), send_x_right(pack_x_size);
    std::vector<double> recv_x_left(pack_x_size), recv_x_right(pack_x_size);
    std::vector<double> send_y_bot(pack_y_size), send_y_top(pack_y_size);
    std::vector<double> recv_y_bot(pack_y_size), recv_y_top(pack_y_size);
    std::vector<double> send_z_back(pack_z_size), send_z_front(pack_z_size);
    std::vector<double> recv_z_back(pack_z_size), recv_z_front(pack_z_size);

    // pack data
    pack_x_face_send(F, send_x_left, 1);
    pack_x_face_send(F, send_x_right, 0);
    pack_y_face_send(F, send_y_bot, 1);
    pack_y_face_send(F, send_y_top, 0);
    pack_z_face_send(F, send_z_back, 1);
    pack_z_face_send(F, send_z_front, 0);

    // non-blocking receives then sends
    out_reqs.reqs.clear();
    out_reqs.stats.resize(6);
    out_reqs.reqs.resize(12);

    int tag = 100;
    // X-direction
    MPI_Irecv(recv_x_left.data(), pack_x_size, MPI_DOUBLE, L.nbr_xm, tag, C.cart_comm, &out_reqs.reqs[0]);
    MPI_Irecv(recv_x_right.data(), pack_x_size, MPI_DOUBLE, L.nbr_xp, tag+1, C.cart_comm, &out_reqs.reqs[1]);
    MPI_Isend(send_x_right.data(), pack_x_size, MPI_DOUBLE, L.nbr_xp, tag, C.cart_comm, &out_reqs.reqs[2]);
    MPI_Isend(send_x_left.data(), pack_x_size, MPI_DOUBLE, L.nbr_xm, tag+1, C.cart_comm, &out_reqs.reqs[3]);

    // Y-direction
    MPI_Irecv(recv_y_bot.data(), pack_y_size, MPI_DOUBLE, L.nbr_ym, tag+2, C.cart_comm, &out_reqs.reqs[4]);
    MPI_Irecv(recv_y_top.data(), pack_y_size, MPI_DOUBLE, L.nbr_yp, tag+3, C.cart_comm, &out_reqs.reqs[5]);
    MPI_Isend(send_y_top.data(), pack_y_size, MPI_DOUBLE, L.nbr_yp, tag+2, C.cart_comm, &out_reqs.reqs[6]);
    MPI_Isend(send_y_bot.data(), pack_y_size, MPI_DOUBLE, L.nbr_ym, tag+3, C.cart_comm, &out_reqs.reqs[7]);

    // Z-direction
    MPI_Irecv(recv_z_back.data(), pack_z_size, MPI_DOUBLE, L.nbr_zm, tag+4, C.cart_comm, &out_reqs.reqs[8]);
    MPI_Irecv(recv_z_front.data(), pack_z_size, MPI_DOUBLE, L.nbr_zp, tag+5, C.cart_comm, &out_reqs.reqs[9]);
    MPI_Isend(send_z_front.data(), pack_z_size, MPI_DOUBLE, L.nbr_zp, tag+4, C.cart_comm, &out_reqs.reqs[10]);
    MPI_Isend(send_z_back.data(), pack_z_size, MPI_DOUBLE, L.nbr_zm, tag+5, C.cart_comm, &out_reqs.reqs[11]);

    // Wait and unpack sequentially (could be overlapped with interior computation)
    MPI_Waitall((int)out_reqs.reqs.size(), out_reqs.reqs.data(), MPI_STATUSES_IGNORE);

    // Unpack
    unpack_x_face_recv(F, recv_x_left, 1);
    unpack_x_face_recv(F, recv_x_right, 0);
    unpack_y_face_recv(F, recv_y_bot, 1);
    unpack_y_face_recv(F, recv_y_top, 0);
    unpack_z_face_recv(F, recv_z_back, 1);
    unpack_z_face_recv(F, recv_z_front, 0);
}

// --------------------------- Utilities: initialize cart / local sizes ----------

// Build a Cartesian communicator and fill CartDecomp
inline void build_cart_decomp(CartDecomp &C) {
    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    C.size = world_size;
    C.rank = world_rank;

    // Ensure dims are zeroed so MPI_Dims_create will compute a valid decomposition
    C.dims[0] = C.dims[1] = C.dims[2] = 0;
    MPI_Dims_create(world_size, 3, C.dims);

    // Create Cartesian communicator; if it fails fall back to duplicating WORLD
    MPI_Cart_create(MPI_COMM_WORLD, 3, C.dims, C.periods, /*reorder=*/0, &C.cart_comm);
    if (C.cart_comm == MPI_COMM_NULL) {
        MPI_Comm_dup(MPI_COMM_WORLD, &C.cart_comm);
        C.coords[0] = C.coords[1] = C.coords[2] = 0;
    } else {
        MPI_Comm_rank(C.cart_comm, &C.rank);
        MPI_Comm_size(C.cart_comm, &C.size);
        MPI_Cart_coords(C.cart_comm, C.rank, 3, C.coords);
    }

    if (C.rank == 0) {
        std::cout << "MPI Cartesian Decomposition: periods = [" << C.periods[0] << ", " << C.periods[1] << ", " << C.periods[2] << "]" << std::endl;
        std::cout << " dim = [" << C.dims[0] << ", " << C.dims[1] << ", " << C.dims[2] << "]" << std::endl;
    }
}

// Compute local sizes and neighbor ranks and set into LocalDesc L
inline void compute_local_desc(const GridDesc &G, CartDecomp &C, LocalDesc &L, int ngx, int ngy, int ngz) {
    // split global grid into blocks; distribute remainder to lower coords
    int gx = C.dims[0], gy = C.dims[1], gz = C.dims[2];
    int px = C.coords[0], py = C.coords[1], pz = C.coords[2];

    // chunk 每个进程几个网格 and prefix 每个进程起始全局索引
    auto chunk = [](int N, int P, int coord){ int base = N / P; int rem = N % P; return base + (coord < rem ? 1 : 0); };
    auto prefix = [](int N, int P, int coord){ int base = N / P; int rem = N % P; return coord * base + std::min(coord, rem); };

    L.nx = chunk(G.global_nx, gx, px);
    L.ny = chunk(G.global_ny, gy, py);
    L.nz = chunk(G.global_nz, gz, pz);

    L.ox = prefix(G.global_nx, gx, px);
    L.oy = prefix(G.global_ny, gy, py);
    L.oz = prefix(G.global_nz, gz, pz);

    L.ngx = ngx; L.ngy = ngy; L.ngz = ngz;
    L.sx = L.nx + 2*L.ngx; L.sy = L.ny + 2*L.ngy; L.sz = L.nz + 2*L.ngz;

    // neighbors
    MPI_Cart_shift(C.cart_comm, 0, 1, &L.nbr_xm, &L.nbr_xp);
    MPI_Cart_shift(C.cart_comm, 1, 1, &L.nbr_ym, &L.nbr_yp);
    MPI_Cart_shift(C.cart_comm, 2, 1, &L.nbr_zm, &L.nbr_zp);
}

/*
 * Notes and next steps:
 *  - The pack/unpack routines pack only the 5 conserved variables. If you need to exchange more
 *    (primitive caches, viscous terms), you can pack them together into the same message.
 *  - For performance, consider using persistent MPI requests (MPI_Send_init / MPI_Recv_init)
 *    and reusing buffers.
 *  - To overlap communication and computation: post Irecv/Isend, compute interior points that do
 *    not depend on halo, then MPI_Wait for the halo and compute boundary stencils.
 *  - When moving to higher-order reconstructions, face buffers (flux_fx...) are helpful to store
 *    computed half-node fluxes. The face arrays allocated above satisfy sizes for that use.
 *  - I/O (VTK, HDF5) and checkpointing not implemented here; add parallel HDF5 for large runs.
 */

#endif // NS3D_FIELD_STRUCTURES_H
