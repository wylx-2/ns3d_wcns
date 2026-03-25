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
#include <vector>
#include <fftw3-mpi.h>
#include <hdf5.h>
#include <H5FDmpio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// HDF5 输出函数
static bool write_hdf5_field(const Field3D &F, const GridDesc &G, const CartDecomp &C,
                             const std::filesystem::path &filepath, double time)
{
	const LocalDesc &L = F.L;
	const hsize_t global_dims[3] = {
		static_cast<hsize_t>(G.global_nz),
		static_cast<hsize_t>(G.global_ny),
		static_cast<hsize_t>(G.global_nx)
	};
	const hsize_t local_dims[3] = {
		static_cast<hsize_t>(L.nz),
		static_cast<hsize_t>(L.ny),
		static_cast<hsize_t>(L.nx)
	};
	const hsize_t start[3] = {
		static_cast<hsize_t>(L.oz),
		static_cast<hsize_t>(L.oy),
		static_cast<hsize_t>(L.ox)
	};
	const std::size_t npts = static_cast<std::size_t>(L.nx) * static_cast<std::size_t>(L.ny) * static_cast<std::size_t>(L.nz);

	std::vector<double> X(npts), Y(npts), Z(npts);
	std::vector<double> rho(npts), u(npts), v(npts), w(npts), E(npts), p(npts), T(npts);

	for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
		for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
			for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
				std::size_t lid = static_cast<std::size_t>(k - L.ngz) * static_cast<std::size_t>(L.ny) * static_cast<std::size_t>(L.nx)
					+ static_cast<std::size_t>(j - L.ngy) * static_cast<std::size_t>(L.nx)
					+ static_cast<std::size_t>(i - L.ngx);
				int gid = F.I(i, j, k);
				int gi = L.ox + (i - L.ngx);
				int gj = L.oy + (j - L.ngy);
				int gk = L.oz + (k - L.ngz);

				X[lid] = G.x0 + gi * G.dx;
				Y[lid] = G.y0 + gj * G.dy;
				Z[lid] = G.z0 + gk * G.dz;

				rho[lid] = F.rho[gid];
				u[lid] = F.u[gid];
				v[lid] = F.v[gid];
				w[lid] = F.w[gid];
				E[lid] = F.E[gid];
				p[lid] = F.p[gid];
				T[lid] = F.T[gid];
			}
		}
	}

#ifdef H5_HAVE_PARALLEL
	hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
	if (fapl < 0) {
		std::cerr << "Failed to create HDF5 file access property list\n";
		return false;
	}

	herr_t ierr = H5Pset_fapl_mpio(fapl, C.cart_comm, MPI_INFO_NULL);
	if (ierr < 0) {
		std::cerr << "Failed to set HDF5 MPI-IO file access property\n";
		H5Pclose(fapl);
		return false;
	}

	hid_t file = H5Fcreate(filepath.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
	H5Pclose(fapl);
	if (file < 0) {
		std::cerr << "Failed to create HDF5 file: " << filepath.string() << "\n";
		return false;
	}

	auto write_attr_i = [&](const char *name, int value) {
		hid_t aspace = H5Screate(H5S_SCALAR);
		hid_t attr = H5Acreate2(file, name, H5T_NATIVE_INT, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_NATIVE_INT, &value);
		H5Aclose(attr);
		H5Sclose(aspace);
	};

	auto write_attr_d = [&](const char *name, double value) {
		hid_t aspace = H5Screate(H5S_SCALAR);
		hid_t attr = H5Acreate2(file, name, H5T_NATIVE_DOUBLE, aspace, H5P_DEFAULT, H5P_DEFAULT);
		H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
		H5Aclose(attr);
		H5Sclose(aspace);
	};

	write_attr_i("global_nx", G.global_nx);
	write_attr_i("global_ny", G.global_ny);
	write_attr_i("global_nz", G.global_nz);
	write_attr_d("time", time);
	write_attr_d("dx", G.dx);
	write_attr_d("dy", G.dy);
	write_attr_d("dz", G.dz);

	hid_t filespace = H5Screate_simple(3, global_dims, nullptr);
	hid_t memspace = H5Screate_simple(3, local_dims, nullptr);
	ierr = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, local_dims, nullptr);
	if (ierr < 0) {
		std::cerr << "Failed to select hyperslab for HDF5 write\n";
		H5Sclose(memspace);
		H5Sclose(filespace);
		H5Fclose(file);
		return false;
	}

	hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
	if (xfer < 0) {
		std::cerr << "Failed to create HDF5 transfer property list\n";
		H5Sclose(memspace);
		H5Sclose(filespace);
		H5Fclose(file);
		return false;
	}
	ierr = H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
	if (ierr < 0) {
		std::cerr << "Failed to set collective MPI-IO transfer mode\n";
		H5Pclose(xfer);
		H5Sclose(memspace);
		H5Sclose(filespace);
		H5Fclose(file);
		return false;
	}

	auto write_dataset = [&](const char *name, const std::vector<double> &data) -> bool {
		hid_t dset = H5Dcreate2(file, name, H5T_IEEE_F64LE, filespace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		if (dset < 0) return false;
		herr_t ierr_local = H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, xfer, data.data());
		H5Dclose(dset);
		return ierr_local >= 0;
	};

	bool ok = true;
	ok = ok && write_dataset("X", X);
	ok = ok && write_dataset("Y", Y);
	ok = ok && write_dataset("Z", Z);
	ok = ok && write_dataset("rho", rho);
	ok = ok && write_dataset("u", u);
	ok = ok && write_dataset("v", v);
	ok = ok && write_dataset("w", w);
	ok = ok && write_dataset("E", E);
	ok = ok && write_dataset("p", p);
	ok = ok && write_dataset("T", T);

	H5Pclose(xfer);
	H5Sclose(memspace);
	H5Sclose(filespace);
	H5Fclose(file);

	if (!ok) {
		std::cerr << "Failed to write one or more datasets to HDF5 file: " << filepath.string() << "\n";
	}
	return ok;
#else
	if (C.rank == 0) {
		std::cerr << "Parallel HDF5 is required for single-file MPI output, but this build is serial HDF5\n";
	}
	return false;
#endif
}

void write_tecplot_field(const Field3D &F, const GridDesc &G, const CartDecomp &C, const SolverParams &P, double time)
{
	(void)P;

	// ensure output directory exists
	std::filesystem::path outdir("output");
	std::filesystem::path timedir = outdir / ("time_" + std::to_string(static_cast<int>(time * 100000)));
	std::error_code ec;
	std::filesystem::create_directories(timedir, ec);
	if (ec) {
		std::cerr << "Warning: could not create output directory 'output': " << ec.message() << "\n";
	}

	std::filesystem::path h5path = timedir / "field.h5";

	bool h5_ok = write_hdf5_field(F, G, C, h5path, time);

	if (h5_ok && C.rank == 0) {
		std::cerr << "Wrote HDF5 file: " << h5path.string() << " (time=" << std::scientific << std::setprecision(8) << time << ")\n";
	}
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
