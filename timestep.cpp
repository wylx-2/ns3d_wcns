#include "field_structures.h"
#include <mpi.h>
#include <cmath>

double compute_timestep(Field3D &F, const GridDesc &G, const SolverParams &P){
    double dt_local = 1e9;
    const int nx = F.L.nx, ny = F.L.ny, nz = F.L.nz;
    const int ngh = F.L.ngx; 
    const double dx = G.dx; // uniform grid assumed
    double dt_global; 

    for(int k=ngh;k<ngh+nz;++k)
    for(int j=ngh;j<ngh+ny;++j)
    for(int i=ngh;i<ngh+nx;++i){
        int id = F.I(i,j,k);
        double rho = F.rho[id]; double u = F.rhou[id]/rho;
        double p = (P.gamma-1.)*(F.E[id]-0.5*rho*u*u);
        double a = std::sqrt(P.gamma*p/rho);
        dt_local = std::min(dt_local, dx/(std::fabs(u)+a));
    }

    MPI_Allreduce(&dt_local,&dt_global,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
    return P.cfl*dt_global;
}