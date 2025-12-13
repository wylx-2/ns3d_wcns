#include "field_structures.h"
#include "ns3d_func.h"
#include <fstream>
#include <filesystem>

static const double PI = 3.14159265358979323846;

struct TurbStats {
    double kinetic_energy = 0.0;   // volume averaged 0.5(u^2+v^2+w^2)
    double dissipation = 0.0;      // ε
    double u_rms = 0.0;            // rms velocity
    double mean_mu = 0.0;        // mean dynamic viscosity
    double mean_rho = 0.0;       // mean density
    double taylor = 0.0;           // Taylor microscale λ
    double taylor_Li = 0.0;        // Taylor-scale by Li Xinliang
    double Re_lambda = 0.0;        // Reynolds number based on Taylor scale
    double Mach_t = 0.0;           // Turbulent Mach number
    double eta = 0.0;              // Kolmogorov scale
    double u_eta = 0.0;            // Kolmogorov velocity
    double tau_eta = 0.0;          // Kolmogorov time scale

    std::vector<double> k;         // wavenumbers
    std::vector<double> Ek;        // spectrum

    std::vector<double> R11;       // correlation longitudinal
    std::vector<double> R22;       // correlation transverse
    std::vector<double> r;         // physical lag distance
};

// -------------------------------------------------------------
// Compute R11, R22 using FFT-based correlation
// -------------------------------------------------------------
void compute_correlations_fftw(Field3D &F, GridDesc &G, CartDecomp &C,
                               std::vector<double> &R11,
                               std::vector<double> &R22,
                               std::vector<double> &r)
{
    fftw_mpi_init();
    MPI_Comm comm = C.cart_comm;
    int NX = G.global_nx, NY = G.global_ny, NZ = G.global_nz;

    ptrdiff_t local_n0, local_0_start;
    ptrdiff_t alloc_local = fftw_mpi_local_size_3d(NX,NY,NZ,comm,
                                                   &local_n0,&local_0_start);

    fftw_complex *Ux = fftw_alloc_complex(alloc_local);
    fftw_complex *Uy = fftw_alloc_complex(alloc_local);
    fftw_complex *Uz = fftw_alloc_complex(alloc_local);

    fftw_complex *Cx = fftw_alloc_complex(alloc_local);
    fftw_complex *Cy = fftw_alloc_complex(alloc_local);
    fftw_complex *Cz = fftw_alloc_complex(alloc_local);

    // Fill real-space velocity into FFT arrays
    {
        const LocalDesc &L = F.L;
        ptrdiff_t p = 0;
        for (ptrdiff_t i = local_0_start; i < local_0_start + local_n0; ++i)
        for (int j = 0; j < NY; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            int ii = int(i - L.ox);
            int jj = int(j - L.oy);
            int kk = int(k - L.oz);
            int id = F.I(ii+L.ngx, jj+L.ngy, kk+L.ngz);

            Ux[p][0] = F.u[id];  Ux[p][1] = 0.0;
            Uy[p][0] = F.v[id];  Uy[p][1] = 0.0;
            Uz[p][0] = F.w[id];  Uz[p][1] = 0.0;
            p++;
        }
    }

    // FFT(u)
    fftw_plan planUx = fftw_mpi_plan_dft_3d(NX,NY,NZ,Ux,Ux,comm,FFTW_FORWARD,FFTW_ESTIMATE);
    fftw_plan planUy = fftw_mpi_plan_dft_3d(NX,NY,NZ,Uy,Uy,comm,FFTW_FORWARD,FFTW_ESTIMATE);
    fftw_plan planUz = fftw_mpi_plan_dft_3d(NX,NY,NZ,Uz,Uz,comm,FFTW_FORWARD,FFTW_ESTIMATE);

    fftw_execute(planUx);
    fftw_execute(planUy);
    fftw_execute(planUz);

    // Rij(k) = U(k) * conj(U(k))
    ptrdiff_t p=0;
    for (ptrdiff_t i=0;i<local_n0;i++)
    for (int j=0;j<NY;j++)
    for (int k=0;k<NZ;k++)
    {
        double ar, ai;

        // x
        ar = Ux[p][0]; ai = Ux[p][1];
        Cx[p][0] = ar*ar + ai*ai;
        Cx[p][1] = 0.0;

        // y
        ar = Uy[p][0]; ai = Uy[p][1];
        Cy[p][0] = ar*ar + ai*ai;
        Cy[p][1] = 0.0;

        // z
        ar = Uz[p][0]; ai = Uz[p][1];
        Cz[p][0] = ar*ar + ai*ai;
        Cz[p][1] = 0.0;

        p++;
    }

    fftw_plan planCx = fftw_mpi_plan_dft_3d(NX,NY,NZ,Cx,Cx,comm,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_plan planCy = fftw_mpi_plan_dft_3d(NX,NY,NZ,Cy,Cy,comm,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_plan planCz = fftw_mpi_plan_dft_3d(NX,NY,NZ,Cz,Cz,comm,FFTW_BACKWARD,FFTW_ESTIMATE);

    fftw_execute(planCx);
    fftw_execute(planCy);
    fftw_execute(planCz);

    // Gather the 1D line r = displacement along x-axis
    int N = NX/2 + 1;
    R11.assign(N,0.0);
    R22.assign(N,0.0);

    // normalization: divide by variance and N³ (FFTW conventions)
    double local_var_u=0, local_var_v=0;
    {
        const LocalDesc &L = F.L;
        for (int kk=0; kk<L.nz; kk++)
        for (int jj=0; jj<L.ny; jj++)
        for (int ii=0; ii<L.nx; ii++)
        {
            int id = F.I(ii+L.ngx, jj+L.ngy, kk+L.ngz);
            local_var_u += F.u[id] * F.u[id];
            local_var_v += F.v[id] * F.v[id];
        }
    }
    double global_var_u=0, global_var_v=0;
    MPI_Allreduce(&local_var_u,&global_var_u,1,MPI_DOUBLE,MPI_SUM,comm);
    MPI_Allreduce(&local_var_v,&global_var_v,1,MPI_DOUBLE,MPI_SUM,comm);

    // Extract Cx(i,0,0), Cy(i,0,0), Cz(i,0,0)
    {
        const LocalDesc &L = F.L;
        for (int i=0;i<N;i++)
        {
            int gi = i; // global index
            // check if this rank owns it
            if (gi>=local_0_start && gi<local_0_start+local_n0)
            {
                ptrdiff_t li = gi - local_0_start;
                ptrdiff_t idx = (li*NY + 0)*NZ + 0;

                double Rxx = Cx[idx][0] / global_var_u;
                double Ryy = Cy[idx][0] / global_var_v;
                double Rzz = Cz[idx][0] / global_var_u; // use u variance for consistency

                double tmp1[3] = {Rxx,Ryy,Rzz};
                double tmp2[3]={0,0,0};

                MPI_Reduce(tmp1,tmp2,3,MPI_DOUBLE,MPI_SUM,0,comm);

                if (C.rank==0)
                {
                    R11[i] = tmp2[0];
                    R22[i] = (tmp2[1]+tmp2[2])*0.5;
                }
            }
            else
            {
                double tmp1[3]={0,0,0},tmp2[3]={0,0,0};
                MPI_Reduce(tmp1,tmp2,3,MPI_DOUBLE,MPI_SUM,0,comm);
                if (C.rank==0)
                {
                    R11[i]=tmp2[0];
                    R22[i]=(tmp2[1]+tmp2[2])*0.5;
                }
            }
        }
    }

    if (C.rank==0)
    {
        r.assign(N,0.0);
        double Lx=NX*G.dx;
        for(int i=0;i<N;i++)
            r[i] = double(i) * G.dx;
    }

    fftw_free(Ux); fftw_free(Uy); fftw_free(Uz);
    fftw_free(Cx); fftw_free(Cy); fftw_free(Cz);
    fftw_mpi_cleanup();
}

// -------------------------------------------------------------
// Main postprocess function
// -------------------------------------------------------------
void compute_turbulence_statistics(Field3D &F,
                                   const GridDesc &G,
                                   const SolverParams &P,
                                   const CartDecomp &C,
                                   const double current_time)
{
    TurbStats stats;

    // -------- 1. Compute spectrum (already done before) --------
    // compute_energy_spectrum(F, G, C, prefix + "_spectrum.dat");

    // Read spectrum back to stats
    if (C.rank==0) {
        std::string prefix;
        {
            std::stringstream ss;
            ss << "output/output_time_" << std::fixed << std::setprecision(5) << current_time;
            prefix = ss.str();
        }
        std::ifstream fin(prefix + "_spectrum.dat");
        double kk, Ek;
        while (fin >> kk >> Ek) {
            stats.k.push_back(kk);
            stats.Ek.push_back(Ek);
        }
        fin.close();
    }

    // -------- 2. Compute kinetic energy and RMS --------
    double local_energy=0, global_energy=0;
    double local_urms2=0, global_urms2=0;
    double local_sound_speed=0, global_sound_speed=0;
    double local_dudx2=0, global_dudx2=0;
    double local_mu=0, global_mu=0;
    double local_rho=0, global_rho=0;

    {
        const LocalDesc &L = F.L;
        compute_gradients_dudx(F, G);
        for (int k=0;k<L.nz;k++)
        for (int j=0;j<L.ny;j++)
        for (int i=0;i<L.nx;i++)
        {
            int id = F.I(i+L.ngx, j+L.ngy, k+L.ngz);
            double uu=F.u[id], vv=F.v[id], ww=F.w[id];
            local_energy += 0.5*(uu*uu + vv*vv + ww*ww);
            local_urms2 += (uu*uu + vv*vv + ww*ww);
            local_dudx2 += F.du_dx[id]*F.du_dx[id];
            local_sound_speed += std::sqrt(P.gamma * P.Rgas * F.T[id]);
            local_mu += P.get_mu(F.T[id]);
            local_rho += F.rho[id];
        }
    }

    MPI_Reduce(&local_energy,&global_energy,1,MPI_DOUBLE,MPI_SUM,0,C.cart_comm);
    MPI_Reduce(&local_urms2,&global_urms2,1,MPI_DOUBLE,MPI_SUM,0,C.cart_comm);
    MPI_Reduce(&local_dudx2,&global_dudx2,1,MPI_DOUBLE,MPI_SUM,0,C.cart_comm);
    MPI_Reduce(&local_sound_speed,&global_sound_speed,1,MPI_DOUBLE,MPI_SUM,0,C.cart_comm);
    MPI_Reduce(&local_mu,&global_mu,1,MPI_DOUBLE,MPI_SUM,0,C.cart_comm);
    MPI_Reduce(&local_rho,&global_rho,1,MPI_DOUBLE,MPI_SUM,0,C.cart_comm);

    double nu = 0.0;
    if (C.rank==0)
    {
        int N = G.global_nx * G.global_ny * G.global_nz;
        stats.mean_mu = global_mu / double(N);
        stats.mean_rho = global_rho / double(N);
        nu = stats.mean_mu / stats.mean_rho;

        stats.kinetic_energy = global_energy / double(N);
        stats.u_rms = std::sqrt(global_urms2 / double(3*N));
        stats.taylor_Li = stats.u_rms / std::sqrt(global_dudx2 / double(N));
        stats.Mach_t = std::sqrt(stats.kinetic_energy*2) / (global_sound_speed / double(N));
        stats.Re_lambda = stats.u_rms * stats.taylor_Li / nu;
        std::cout << "[Post] mean_mu = " << stats.mean_mu << ", mean_rho = " << stats.mean_rho
                  << ", u_rms = " << stats.u_rms
                  << ", dudx = " << global_dudx2 / double(N)
                  << ", Taylor_Li = " << stats.taylor_Li
                  << ", Mach_t = " << stats.Mach_t << "\n";
    }

    // -------- 3. Dissipation ε = 2 ν ∑ k² E(k) --------
    if (C.rank==0)
    {
        double eps=0;
        for (size_t i=0;i<stats.k.size();i++)
            eps += 2.0*nu * stats.k[i]*stats.k[i] * stats.Ek[i];

        stats.dissipation = eps;

        // -------- 4. Kolmogorov quantities --------
        stats.eta    = std::pow(nu*nu*nu / eps, 0.25);
        stats.u_eta  = std::pow(nu*eps,       0.25);
        stats.tau_eta= std::pow(nu/eps,       0.5);

        // -------- 5. Taylor microscale --------
        stats.taylor = std::sqrt(15.0 * nu * stats.u_rms*stats.u_rms / eps);
    }

    // -------- 7. Output statistics --------
    if (C.rank==0) {
        std::filesystem::path fpath("output/turbulence_stats.dat");
        bool exists = std::filesystem::exists(fpath);
        std::ofstream fout(fpath, std::ios::app);
        if (!fout) {
            std::cout << "[Error] Cannot open output/turbulence_stats.dat for writing.\n";
            return;
        }
        if (!exists) {
            fout << "current_time " << "Kinetic Energy " << "u_rms " << "Dissipation " << "Taylor " << "Kol_scale "
                 << "Kol_vel " << "Kol_time " << "Taylor_Li " << "Re_lambda " << "Mach_t\n";
        }
        fout << std::scientific << std::setprecision(8)
             << current_time << " "
             << stats.kinetic_energy << " "
             << stats.u_rms << " "
             << stats.dissipation << " "
             << stats.taylor << " "
             << stats.eta << " "
             << stats.u_eta << " "
             << stats.tau_eta << " "
             << stats.taylor_Li << " "
             << stats.Re_lambda << " "
             << stats.Mach_t << "\n";
    }

    /*
    // -------- 6. Compute correlations --------
    if (C.rank==0)
        std::cout << "[Post] Computing correlations...\n";
    compute_correlations_fftw(F,G,C, stats.R11, stats.R22, stats.r);
    */
}
