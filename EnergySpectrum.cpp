#include <mpi.h>
#include <cmath>
#include <vector>
#include <complex>
#include <iostream>
#include <fstream>
#include <fftw3.h>
#include "field_structures.h"

static const double PI = 3.14159265358979323846;

// ================================================================
// 将 Field3D 中本地物理区块的数据拷贝到线性数组
// ================================================================
void extract_local_velocity(const Field3D &F,
                            std::vector<double> &u,
                            std::vector<double> &v,
                            std::vector<double> &w)
{
    const LocalDesc &L = F.L;

    int Nloc = L.nx * L.ny * L.nz;
    u.resize(Nloc);
    v.resize(Nloc);
    w.resize(Nloc);

    int p = 0;

    for(int k = 0; k < L.nz; k++)
    for(int j = 0; j < L.ny; j++)
    for(int i = 0; i < L.nx; i++)
    {
        int id = F.I(i + L.ngx, j + L.ngy, k + L.ngz);

        u[p] = F.u[id];
        v[p] = F.v[id];
        w[p] = F.w[id];
        p++;
    }
}

// ================================================================
// 3D FFT using FFTW
// ================================================================
void fft3d_complex(int NX, int NY, int NZ,
                   fftw_complex *data,
                   int sign)
{
    fftw_plan plan = fftw_plan_dft_3d(
        NX, NY, NZ,
        data, data,
        sign, FFTW_ESTIMATE
    );

    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

// ================================================================
// Rank 0 进行能谱计算
// ================================================================
void compute_energy_spectrum_rank0(
        int NX, int NY, int NZ,
        const std::vector<double> &uall,
        const std::vector<double> &vall,
        const std::vector<double> &wall,
        const std::string &filename = "Energy-spectrum.dat")
{
    int N = NX*NY*NZ;

    // --- Make complex FFTW arrays ---
    fftw_complex *U = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *V = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *W = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    for (int id=0; id<N; id++) {
        U[id][0] = uall[id]; U[id][1] = 0;
        V[id][0] = vall[id]; V[id][1] = 0;
        W[id][0] = wall[id]; W[id][1] = 0;
    }

    // --- Forward FFT ---
    fft3d_complex(NX, NY, NZ, U, FFTW_FORWARD);
    fft3d_complex(NX, NY, NZ, V, FFTW_FORWARD);
    fft3d_complex(NX, NY, NZ, W, FFTW_FORWARD);

        // --- FFT归一化 ---
    double fft_norm = 1.0 / (NX * NY * NZ);
    for(int id = 0; id < N; id++) {
        U[id][0] *= fft_norm;
        U[id][1] *= fft_norm;
        V[id][0] *= fft_norm;
        V[id][1] *= fft_norm;
        W[id][0] *= fft_norm;
        W[id][1] *= fft_norm;
    }

    // --- Compute energy density |U|^2 + |V|^2 + |W|^2 ---
    std::vector<double> Er(N);
    for(int id=0; id<N; id++){
        double eu = U[id][0]*U[id][0] + U[id][1]*U[id][1];
        double ev = V[id][0]*V[id][0] + V[id][1]*V[id][1];
        double ew = W[id][0]*W[id][0] + W[id][1]*W[id][1];
        Er[id] = (eu + ev + ew) * 0.5;
    }

    // --- Shell-integrated energy spectrum ---
    int Kmax =  NX * 2;
    std::vector<double> Ek(Kmax+1, 0.0);
    std::vector<int> Nk(Kmax+1, 0);

    for(int k=0; k<NZ; k++)
    for(int j=0; j<NY; j++)
    for(int i=0; i<NX; i++)
    {
        int k1 = (i <= NX/2 ? i : i-NX);
        int k2 = (j <= NY/2 ? j : j-NY);
        int k3 = (k <= NZ/2 ? k : k-NZ);

        int kk = (int) std::sqrt(1.0*(k1*k1 + k2*k2 + k3*k3)) + 0.5;
        // int id = (i*NY + j)*NZ + k;
        int id = (k*NY + j)*NX + i;
        Ek[kk] += Er[id];
        Nk[kk] += 1;
    }

    // --- Output to file ---
    std::ofstream fout(filename);
    for(int k=0; k<=Kmax; k++)
    {
        fout << k << " " << Ek[k] << "\n";
    }
    fout.close();

    // --- Compute statistics (urms, integral length, Taylor scale) ---
    /*
    double Ek1=0, Ek2=0;
    for(int k=1; k<=Kmax; k++){
        Ek1 += Ek[k];
        Ek2 += Ek[k] / k;
    }

    double urms = std::sqrt(2.0*Ek1 / 3.0);
    double le   = 0.75 * PI * Ek2 / Ek1;
    double uxrms = 0.0;  // optional: compute ∂u/∂x RMS by FFT
    double tau = le / urms;

    std::cout << "urms = " << urms << "\n";
    std::cout << "Integral length scale le = " << le << "\n";
    std::cout << "Turnover time tau = " << tau << "\n";
    */

    fftw_free(U); fftw_free(V); fftw_free(W);
}

// ================================================================
// 主功能：在并行环境中计算 E(k)
// ================================================================
void compute_energy_spectrum(const Field3D &F,
                             const GridDesc &G,
                             const CartDecomp &C,
                             const std::string &filename = "Energy-spectrum.dat")
{
    const LocalDesc &L = F.L;

    int NX = G.global_nx;
    int NY = G.global_ny;
    int NZ = G.global_nz;

    // 1. 每个 rank 的本地物理区块
    std::vector<double> u_loc, v_loc, w_loc;
    extract_local_velocity(F, u_loc, v_loc, w_loc);

    int Nloc = L.nx * L.ny * L.nz;
    int N = NX * NY * NZ;

    // 2. Rank 0 收集所有数据
    std::vector<double> uall, vall, wall;
    if (C.rank == 0){
        uall.resize(N);
        vall.resize(N);
        wall.resize(N);
    }

    // 每个 rank 的数据在全局数组中的线性偏移（未直接使用）
    // int base = (L.ox*NY + L.oy)*NZ + L.oz; // unused, removed

    // Gather 方式：每个 rank 用 MPI_Gatherv 的 trick：Rank 0 手动 MPI_Recv
    if (C.rank == 0)
    {
        // Rank 0 首先放自己的数据
        for(int k=0; k<L.nz; k++)
        for(int j=0; j<L.ny; j++)
        for(int i=0; i<L.nx; i++)
        {
            int gx = L.ox + i;
            int gy = L.oy + j;
            int gz = L.oz + k;
            int gid = (gx*NY + gy)*NZ + gz;
            // local linear index in the same ordering as extract_local_velocity:
            // extract_local_velocity increments p in loops (k, j, i) so
            // p = (k*ny + j)*nx + i
            int lid = (k * L.ny + j) * L.nx + i;

            uall[gid] = u_loc[lid];
            vall[gid] = v_loc[lid];
            wall[gid] = w_loc[lid];
        }

        // 收集其他 rank
        for(int p=1; p<C.size; p++)
        {
            MPI_Status st;
            // 接收 LocalDesc
            LocalDesc Lp;
            MPI_Recv(&Lp, sizeof(LocalDesc), MPI_BYTE, p, 100, C.cart_comm, &st);

            int Np = Lp.nx * Lp.ny * Lp.nz;
            std::vector<double> ub(Np), vb(Np), wb(Np);

            MPI_Recv(ub.data(), Np, MPI_DOUBLE, p, 101, C.cart_comm, &st);
            MPI_Recv(vb.data(), Np, MPI_DOUBLE, p, 102, C.cart_comm, &st);
            MPI_Recv(wb.data(), Np, MPI_DOUBLE, p, 103, C.cart_comm, &st);

            // 放入全局数组
            int bp = (Lp.ox*NY + Lp.oy)*NZ + Lp.oz;

            int t = 0;
            for(int k=0; k<Lp.nz; k++)
            for(int j=0; j<Lp.ny; j++)
            for(int i=0; i<Lp.nx; i++)
            {
                int gx = Lp.ox + i;
                int gy = Lp.oy + j;
                int gz = Lp.oz + k;
                int gid = (gx*NY + gy)*NZ + gz;

                uall[gid] = ub[t];
                vall[gid] = vb[t];
                wall[gid] = wb[t];
                t++;
            }
        }
    }
    else
    {
        // 非 Rank 0 发送数据
        MPI_Send(&L, sizeof(LocalDesc), MPI_BYTE, 0, 100, C.cart_comm);
        MPI_Send(u_loc.data(), Nloc, MPI_DOUBLE, 0, 101, C.cart_comm);
        MPI_Send(v_loc.data(), Nloc, MPI_DOUBLE, 0, 102, C.cart_comm);
        MPI_Send(w_loc.data(), Nloc, MPI_DOUBLE, 0, 103, C.cart_comm);
    }

    MPI_Barrier(C.cart_comm);

    // 3. Rank 0 计算能谱
    if (C.rank == 0){
        compute_energy_spectrum_rank0(NX, NY, NZ, uall, vall, wall, filename);
        std::cout << "[Spectrum] Energy spectrum computed.\n";
    }
}
