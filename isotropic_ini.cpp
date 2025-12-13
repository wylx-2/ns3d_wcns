#include <mpi.h>
#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <iostream>
#include <fftw3.h>
#include "ns3d_func.h"
#include "field_structures.h"

static const double PI = 3.14159265358979323846;

// ------------------------------------------------------------
// 0~1 随机数
// ------------------------------------------------------------
double rnd01() {
    static thread_local std::mt19937_64 rng(123);
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

// ------------------------------------------------------------
// Rank 0 生成完整均匀各向同性湍流场
// ------------------------------------------------------------
void generate_full_turbulence(int NX, int NY, int NZ,
                              std::vector<double> &u,
                              std::vector<double> &v,
                              std::vector<double> &w)
{
    int N = NX * NY * NZ;

    std::vector<std::complex<double>> U(N), V(N), W(N);

    int k0 = 8;
    double A = 0.00013;
    double Ek0 = 3.0*A/64.0 * std::sqrt(2*PI) * std::pow(k0,5);
    double tao = std::sqrt(32.0/A) * std::pow(2*PI,0.25) * std::pow(k0,-3.5);

    std::cout << "[Turbulence] tau = " << tao << ", Ek0 = " << Ek0 << "\n";

    // Build Fourier velocity field
    for (int k=0; k<NZ; k++)
    for (int j=0; j<NY; j++)
    for (int i=0; i<NX; i++)
    {
        int id = (i*NY + j)*NZ + k;

        int k1 = (i <= NX/2 ? i : i-NX);
        int k2 = (j <= NY/2 ? j : j-NY);
        int k3 = (k <= NZ/2 ? k : k-NZ);

        int kk = k1*k1 + k2*k2 + k3*k3;

        if (kk == 0) {
            U[id]=0; V[id]=0; W[id]=0;
            continue;
        }

        double Ak = std::sqrt(2.0/3.0) *
                    std::sqrt(A*kk*std::exp(-2.0*kk/(double)(k0*k0)) / (4.0*PI));

        double a1 = 2*PI*rnd01();
        double a2 = 2*PI*rnd01();
        double a3 = 2*PI*rnd01();

        double v1r = Ak * std::sin(a1), v1i = Ak * std::cos(a1);
        double v2r = Ak * std::sin(a2), v2i = Ak * std::cos(a2);
        double v3r = Ak * std::sin(a3), v3i = Ak * std::cos(a3);

        double vkr = k1*v1r + k2*v2r + k3*v3r;
        double vki = k1*v1i + k2*v2i + k3*v3i;

        U[id] = std::complex<double>(k1*vkr/kk - v1r, 
                                     k1*vki/kk - v1i);
        V[id] = std::complex<double>(k2*vkr/kk - v2r,
                                     k2*vki/kk - v2i);
        W[id] = std::complex<double>(k3*vkr/kk - v3r,
                                     k3*vki/kk - v3i);
    }

    // Inverse transform
    fftw_plan pu = fftw_plan_dft_3d(NX,NY,NZ,
                                    reinterpret_cast<fftw_complex*>(U.data()),
                                    reinterpret_cast<fftw_complex*>(U.data()),
                                    FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_plan pv = fftw_plan_dft_3d(NX,NY,NZ,
                                    reinterpret_cast<fftw_complex*>(V.data()),
                                    reinterpret_cast<fftw_complex*>(V.data()),
                                    FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_plan pw = fftw_plan_dft_3d(NX,NY,NZ,
                                    reinterpret_cast<fftw_complex*>(W.data()),
                                    reinterpret_cast<fftw_complex*>(W.data()),
                                    FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(pu); fftw_execute(pv); fftw_execute(pw);
    fftw_destroy_plan(pu); fftw_destroy_plan(pv); fftw_destroy_plan(pw);

    double norm = 1.0 / (double)N;
    double Er = 0.0, Ei = 0.0;

    u.resize(N); v.resize(N); w.resize(N);

    for (int id=0; id<N; id++) {
        u[id] = U[id].real() ;
        v[id] = V[id].real() ;
        w[id] = W[id].real() ;
        Er += 0.5 * (u[id]*u[id] + v[id]*v[id] + w[id]*w[id]);
        Ei += 0.5 * (U[id].imag()*U[id].imag() + V[id].imag()*V[id].imag() + W[id].imag()*W[id].imag());
    }
    Er /= N;
    Ei /= N;

    std::cout << "[Turbulence] Initial energy: Er = " << Er << ", Ei = " << Ei << "\n";

    double scale = std::sqrt(Ek0 / Er);
    for (int id=0; id<N; id++) {
        u[id] *= scale;
        v[id] *= scale;
        w[id] *= scale;
    }

    std::cout << "[Turbulence] Energy normalized: scale=" << scale << "\n";


}

// ------------------------------------------------------------
// 主初始化函数：把全局湍流切片到各 rank
// ------------------------------------------------------------
void init_isotropic_turbulence(Field3D &F,
                               const GridDesc &G,
                               const CartDecomp &C,
                               const SolverParams &P)
{
    const LocalDesc &L = F.L;

    int NX = G.global_nx;
    int NY = G.global_ny;
    int NZ = G.global_nz;

    std::vector<double> Uall, Vall, Wall;

    // -------------------------------
    // Rank 0：生成完整湍流流场
    // -------------------------------
    if (C.rank == 0) {
        generate_full_turbulence(NX, NY, NZ, Uall, Vall, Wall);
        std::cout << "[Turbulence] Full field generated on rank 0\n";
    }

    // -------------------------------
    // 广播全局数据
    // -------------------------------
    if (C.rank != 0) {
        Uall.resize(NX*NY*NZ);
        Vall.resize(NX*NY*NZ);
        Wall.resize(NX*NY*NZ);
    }

    MPI_Bcast(Uall.data(), NX*NY*NZ, MPI_DOUBLE, 0, C.cart_comm);
    MPI_Bcast(Vall.data(), NX*NY*NZ, MPI_DOUBLE, 0, C.cart_comm);
    MPI_Bcast(Wall.data(), NX*NY*NZ, MPI_DOUBLE, 0, C.cart_comm);

    // -------------------------------
    // 提取本 rank 的物理区块
    // global index = (ox + i)
    // -------------------------------
    for (int k=0; k<L.nz; k++)
    for (int j=0; j<L.ny; j++)
    for (int i=0; i<L.nx; i++)
    {
        int gx = L.ox + i;
        int gy = L.oy + j;
        int gz = L.oz + k;

        int gid = (gx*NY + gy)*NZ + gz;

        int id = F.I(i + L.ngx,
                     j + L.ngy,
                     k + L.ngz);

        F.rho[id] = 1.0;
        F.T[id]   = 1.0;
        F.p[id]   = P.Rgas * F.rho[id] * F.T[id];
        F.u[id]   = Uall[gid];
        F.v[id]   = Vall[gid];
        F.w[id]   = Wall[gid];
    }

    std::cout << "[Turbulence] Rank " << C.rank << " field assigned.\n";
}
