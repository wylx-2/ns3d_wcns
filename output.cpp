// output
#include "field_structures.h"
#include "ns3d_func.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <fftw3-mpi.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// tecplot 输出函数
// Write local (per-rank) field data to a Tecplot ASCII file.
// The file will contain a single structured ZONE with I=L.nx, J=L.ny, K=L.nz
// and point-packed data. Coordinates are cell centers computed from GridDesc.
void write_tecplot_field(const Field3D &F, const GridDesc &G, const CartDecomp &C, const SolverParams &P, double time)
{
	const LocalDesc &L = F.L;
	int rank = C.rank;

	// ensure output directory exists
	std::filesystem::path outdir("output");
	std::filesystem::path timedir = outdir / ("time_" + std::to_string(static_cast<int>(time * 100000)));
	std::error_code ec;
	std::filesystem::create_directories(timedir, ec);
	if (ec) {
		std::cerr << "Warning: could not create output directory 'output': " << ec.message() << "\n";
	}

	std::ostringstream ss;
	ss << timedir.string() << "/field" << "_rank" << rank << ".dat";
	std::ofstream ofs(ss.str());
	if (!ofs) {
		std::cerr << "Failed to open output file " << ss.str() << " for writing\n";
		return;
	}

	ofs << "TITLE = \"NS3D Field Rank " << rank << "  Time=" << std::scientific << std::setprecision(8) << time << "\"\n";
	ofs << "VARIABLES = \"X\" \"Y\" \"Z\" \"rho\" \"u\" \"v\" \"w\" \"E\" \"p\" \"T\"";
	ofs << "\n";

	// structured zone with local physical sizes
	// include time in zone title so merged files can see the time per-zone
	ofs << "ZONE T=\"rank_" << rank << " time=" << std::scientific << std::setprecision(8) << time << "\" I=" << L.nx << " J=" << L.ny << " K=" << L.nz
		<< " DATAPACKING=POINT\n";

	ofs << std::scientific << std::setprecision(8);

	// loop over physical cells in i (fast), j, k order
	for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
		for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
			for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
				int gid = F.I(i,j,k);
				int gi = L.ox + (i - L.ngx);
				int gj = L.oy + (j - L.ngy);
				int gk = L.oz + (k - L.ngz);
				double x = G.x0 + gi * G.dx;
				double y = G.y0 + gj * G.dy;
				double z = G.z0 + gk * G.dz;

				double rho = F.rho[gid];
				double rhou = F.rhou[gid];
				double rhov = F.rhov[gid];
				double rhow = F.rhow[gid];
				double E = F.E[gid];
				double u = F.u[gid];
				double v = F.v[gid];
				double w = F.w[gid];
				double p = F.p[gid];
				double T = F.T[gid];

				ofs << x << " " << y << " " << z << " " 
					<< rho << " " << u << " " << v << " " 
					<< w << " " << E << " " << p << " " << T;
				ofs << "\n";
			}
		}
	}

	ofs.close();
	std::cerr << "Wrote Tecplot file: " << ss.str() << " (rank " << rank << ", time=" << std::scientific << std::setprecision(8) << time << ")\n";
}


// Write residuals (per-equation L2 residuals and total energy) vs time step to a Tecplot-like ASCII table.
// Computes global L2 RMS of RHS for each conserved equation using RHS accessors in Field3D
// and writes a line: step Res_rho Res_rhou Res_rhov Res_rhow Res_E Etot
void write_residuals_tecplot(const Field3D &F, int step, const std::string &filename)
{
	const LocalDesc &L = F.L;
	// open file (overwrite if step==0, append otherwise)
	// ensure output directory exists
	std::filesystem::path outdir("output");
	std::error_code ec;
	std::filesystem::create_directories(outdir, ec);
	if (ec) {
		std::cerr << "Warning: could not create output directory 'output': " << ec.message() << "\n";
	}

	std::filesystem::path filepath = outdir / filename;
	std::ofstream ofs;
	if (step == 1) {
		ofs.open(filepath.string(), std::ofstream::out);
	} else {
		ofs.open(filepath.string(), std::ofstream::out | std::ofstream::app);
	}
	if (!ofs) {
		std::cerr << "Failed to open residuals file " << filepath.string() << "\n";
		return;
	}

	if (step == 1) {
		ofs << "TITLE = \"Residuals vs Step\"\n";
		ofs << "VARIABLES = \"Step\" \"Res_rho\" \"Res_rhou\" \"Res_rhov\" \"Res_rhow\" \"Res_E\" \"Etot\"\n";
	}

	ofs << std::scientific << std::setprecision(8);
	ofs << step << " " << F.global_res_rho << " " << F.global_res_rhou
		<< " " << F.global_res_rhov << " " << F.global_res_rhow << " " 
		<< F.global_res_E << " " << F.global_Etot << "\n";
	ofs.close();

}
