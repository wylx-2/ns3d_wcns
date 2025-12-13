#include <iostream>
#include <vector>
#include <mpi.h>
#include <field_structures.h>
#include <ns3d_func.h>

// --------------------------- Simple demo main (initialization only) -----------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    // global info description
    CartDecomp C;
    SolverParams P;
    GridDesc G; 
    // initialize SolverParams
    read_solver_params_from_file("solver.in", P, G, C);
    build_cart_decomp(C);

    LocalDesc L; 
    compute_local_desc(G, C, L, P.ghost_layers, P.ghost_layers, P.ghost_layers);

    // check info
    if (C.rank == 0) {
        std::cout << "Global grid: " << G.global_nx << " x " << G.global_ny << " x " << G.global_nz << "\n";
        std::cout << "Domain: [" << G.x0 << ", " << G.x0 + G.Lx << "] x ["
                  << G.y0 << ", " << G.y0 + G.Ly << "] x ["
                  << G.z0 << ", " << G.z0 + G.Lz << "]\n";
        std::cout << "  dx, dy, dz: [" << G.dx << ", " << G.dy << ", " << G.dz << "]\n";
        std::cout << "MPI size: " << C.size << " dims: [" << C.dims[0] << ", " << C.dims[1] << ", " << C.dims[2] << "]\n";
        std::cout << "Solver Parameters:\n";
        std::cout << "  gamma: " << P.gamma << ", Pr: " << P.Pr << ", Ma: " << P.Ma << ", Re: " << P.Re << "\n";
        std::cout << "  Cv: " << P.Cv << ", Cp: " << P.Cp << ", Rgas: " << P.Rgas << "mu at T=1: " << P.get_mu(1.0) << "\n";
        std::cout << "  FVS type: ";
        switch (P.fvs_type) {
            case SolverParams::FVS_Type::StegerWarming: std::cout << "Steger-Warming\n"; break;
            case SolverParams::FVS_Type::LaxFriedrichs: std::cout << "Lax-Friedrichs\n"; break;
            case SolverParams::FVS_Type::Rusanov: std::cout << "Rusanov\n"; break;
            case SolverParams::FVS_Type::VanLeer: std::cout << "Van Leer\n"; break;
        }
        std::cout << "  CFL: " << P.cfl << " dt_fixed: " << P.dt_fixed << "\n";
        std::cout << "  if char_recon: " << (P.char_recon ? "true" : "false") << "\n";
        std::cout << "  Reconstruction: ";
        switch (P.recon) {
            case SolverParams::Reconstruction::WENO5: std::cout << "WENO5\n"; break;
            case SolverParams::Reconstruction::WCNS: std::cout << "WCNS\n"; break;
            case SolverParams::Reconstruction::LINEAR: std::cout << "LINEAR\n"; break;
            case SolverParams::Reconstruction::MDCD: std::cout << "MDCD\n"; break;
        }
        std::cout << "  Viscous scheme: ";
        switch (P.vis_scheme) {
            case SolverParams::ViscousScheme::C6th: std::cout << "6th-order central\n"; break;
            case SolverParams::ViscousScheme::C4th: std::cout << "4th-order central\n"; break;
        }
    }

    Field3D F; 
    F.allocate(L);
    // initialize_uniform_field(F, G, P);  // Initialize field
    // initialize_riemann_2d(F, G, P);
    // initialize_sod_shock_tube(F, G, P);
    // isotropic turbulence initialization
    // bar_urms_target = 1.0, k0 = 5.0, seed = 12345, rho0 = 1.0, p0 = 1.0
    init_isotropic_turbulence(F, G, C, P);
    // initialize_sine_x_field(F, G, P);

    apply_boundary(F, G, C, P); // apply boundary conditions and holo exchange
    F.primitiveToConserved(P); // update primitive variables (including ghosts)
    if (C.rank == 0)
        std::cout << "Initialization + halo exchange done\n";

    // output the initial field
    write_tecplot_field(F, G, C, P, 0.0);
    if(P.isotropic_analyse) isotropic_post_process(F, G, C, P, 0.0);

    // Main time-stepping loop
    time_advance(F, C, G, P);

    // output the final field
    write_tecplot_field(F, G, C, P, P.TotalTime);
    MPI_Finalize();
    return 0;
}