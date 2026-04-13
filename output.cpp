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

				X[lid] = F.coord_x[gid];
				Y[lid] = F.coord_y[gid];
				Z[lid] = F.coord_z[gid];

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

void write_grid_metrics_tecplot_rank(const Field3D &F,
                                     const GridDesc &G,
                                     const CartDecomp &C,
                                     const std::string &filename_prefix)
{
	(void)G;
	const LocalDesc &L = F.L;

	std::filesystem::path outdir("output");
	std::error_code ec;
	std::filesystem::create_directories(outdir, ec);
	if (ec) {
		std::cerr << "Warning: could not create output directory 'output': " << ec.message() << "\n";
		return;
	}

	std::ostringstream oss;
	oss << filename_prefix << "_rank" << std::setw(4) << std::setfill('0') << C.rank << ".dat";
	std::filesystem::path filepath = outdir / oss.str();

	std::ofstream ofs(filepath.string(), std::ofstream::out);
	if (!ofs) {
		std::cerr << "Failed to open grid-metrics Tecplot file: " << filepath.string() << "\n";
		return;
	}

	ofs << "TITLE = \"Grid Metrics Per Rank\"\n";
	ofs << "VARIABLES = \"i\" \"j\" \"k\" \"gi\" \"gj\" \"gk\" "
	    << "\"x\" \"y\" \"z\" \"Ja\" "
	    << "\"xi_x\" \"xi_y\" \"xi_z\" "
	    << "\"eta_x\" \"eta_y\" \"eta_z\" "
	    << "\"zeta_x\" \"zeta_y\" \"zeta_z\"\n";
	ofs << "ZONE T=\"rank_" << C.rank << "\", I=" << L.nx << ", J=" << L.ny << ", K=" << L.nz
	    << ", DATAPACKING=POINT\n";

	ofs << std::scientific << std::setprecision(12);
	for (int k = L.ngz; k < L.nz + L.ngz; ++k) {
		for (int j = L.ngy; j < L.ny + L.ngy; ++j) {
			for (int i = L.ngx; i < L.nx + L.ngx; ++i) {
				const int li = i - L.ngx;
				const int lj = j - L.ngy;
				const int lk = k - L.ngz;
				const int gi = L.ox + li;
				const int gj = L.oy + lj;
				const int gk = L.oz + lk;
				const int id = F.I(i, j, k);

				ofs << li << " " << lj << " " << lk << " "
				    << gi << " " << gj << " " << gk << " "
				    << F.coord_x[id] << " " << F.coord_y[id] << " " << F.coord_z[id] << " "
				    << F.Ja[id] << " "
				    << F.xi_x[id] << " " << F.xi_y[id] << " " << F.xi_z[id] << " "
				   	<< F.eta_x[id] << " " << F.eta_y[id] << " " << F.eta_z[id] << " "
				    << F.zeta_x[id] << " " << F.zeta_y[id] << " " << F.zeta_z[id] << "\n";
			}
		}
	}

	ofs.close();
	std::cout << "Wrote grid-metrics Tecplot file: " << filepath.string() << "\n";
}

void write_grid_metric_derivatives_tecplot_rank(const Field3D &F,
                                                const GridDesc &G,
                                                const CartDecomp &C,
                                                const std::vector<double> &x_xi,
                                                const std::vector<double> &x_eta,
                                                const std::vector<double> &x_zeta,
                                                const std::vector<double> &y_xi,
                                                const std::vector<double> &y_eta,
                                                const std::vector<double> &y_zeta,
                                                const std::vector<double> &z_xi,
                                                const std::vector<double> &z_eta,
                                                const std::vector<double> &z_zeta,
                                                const std::string &filename_prefix)
{
	(void)G;
	const LocalDesc &L = F.L;
	const std::size_t tot = static_cast<std::size_t>(L.sx) * static_cast<std::size_t>(L.sy) * static_cast<std::size_t>(L.sz);
	if (x_xi.size() != tot || x_eta.size() != tot || x_zeta.size() != tot ||
	    y_xi.size() != tot || y_eta.size() != tot || y_zeta.size() != tot ||
	    z_xi.size() != tot || z_eta.size() != tot || z_zeta.size() != tot) {
		std::cerr << "write_grid_metric_derivatives_tecplot_rank: input size mismatch\n";
		return;
	}

	std::filesystem::path outdir("output");
	std::error_code ec;
	std::filesystem::create_directories(outdir, ec);
	if (ec) {
		std::cerr << "Warning: could not create output directory 'output': " << ec.message() << "\n";
		return;
	}

	std::ostringstream oss;
	oss << filename_prefix << "_rank" << std::setw(4) << std::setfill('0') << C.rank << ".dat";
	std::filesystem::path filepath = outdir / oss.str();

	std::ofstream ofs(filepath.string(), std::ofstream::out);
	if (!ofs) {
		std::cerr << "Failed to open raw-metric Tecplot file: " << filepath.string() << "\n";
		return;
	}

	ofs << "TITLE = \"Raw Metric Derivatives Per Rank\"\n";
	ofs << "VARIABLES = \"i\" \"j\" \"k\" \"gi\" \"gj\" \"gk\" "
	    << "\"x\" \"y\" \"z\" \"Ja\" "
	    << "\"x_xi\" \"x_eta\" \"x_zeta\" "
	    << "\"y_xi\" \"y_eta\" \"y_zeta\" "
	    << "\"z_xi\" \"z_eta\" \"z_zeta\"\n";
	ofs << "ZONE T=\"rank_" << C.rank << "\", I=" << L.nx << ", J=" << L.ny << ", K=" << L.nz
	    << ", DATAPACKING=POINT\n";

	ofs << std::scientific << std::setprecision(12);
	for (int k = L.ngz; k < L.nz + L.ngz; ++k) {
		for (int j = L.ngy; j < L.ny + L.ngy; ++j) {
			for (int i = L.ngx; i < L.nx + L.ngx; ++i) {
				const int li = i - L.ngx;
				const int lj = j - L.ngy;
				const int lk = k - L.ngz;
				const int gi = L.ox + li;
				const int gj = L.oy + lj;
				const int gk = L.oz + lk;
				const int id = F.I(i, j, k);

				ofs << li << " " << lj << " " << lk << " "
				    << gi << " " << gj << " " << gk << " "
				    << F.coord_x[id] << " " << F.coord_y[id] << " " << F.coord_z[id] << " "
				    << F.Ja[id] << " "
				    << x_xi[id] << " " << x_eta[id] << " " << x_zeta[id] << " "
				    << y_xi[id] << " " << y_eta[id] << " " << y_zeta[id] << " "
				    << z_xi[id] << " " << z_eta[id] << " " << z_zeta[id] << "\n";
			}
		}
	}

	ofs.close();
	std::cout << "Wrote raw metric-derivatives Tecplot file: " << filepath.string() << "\n";
}

void write_halfnode_z_tecplot_rank(const LocalDesc &L,
                                   const CartDecomp &C,
                                   const std::vector<double> &z_half,
                                   const std::string &filename_prefix)
{
	const std::size_t tot = static_cast<std::size_t>(L.sx) * static_cast<std::size_t>(L.sy) * static_cast<std::size_t>(L.sz);
	if (z_half.size() != tot) {
		std::cerr << "write_halfnode_z_tecplot_rank: input size mismatch\n";
		return;
	}

	std::filesystem::path outdir("output");
	std::error_code ec;
	std::filesystem::create_directories(outdir, ec);
	if (ec) {
		std::cerr << "Warning: could not create output directory 'output': " << ec.message() << "\n";
		return;
	}

	std::ostringstream oss;
	oss << filename_prefix << "_rank" << std::setw(4) << std::setfill('0') << C.rank << ".dat";
	std::filesystem::path filepath = outdir / oss.str();

	std::ofstream ofs(filepath.string(), std::ofstream::out);
	if (!ofs) {
		std::cerr << "Failed to open half-node z Tecplot file: " << filepath.string() << "\n";
		return;
	}

	ofs << "TITLE = \"Interpolated Half-node Z Coordinates Per Rank\"\n";
	ofs << "VARIABLES = \"i\" \"j\" \"k_face\" \"gi\" \"gj\" \"gk_face\" "
	    << "\"z_half\"\n";
	ofs << "ZONE T=\"rank_" << C.rank << "\", I=" << L.nx << ", J=" << L.ny << ", K=" << L.nz 
	    << ", DATAPACKING=POINT\n";

	ofs << std::scientific << std::setprecision(12);
	for (int kf = L.ngz; kf < L.ngz + L.nz; ++kf) {
		for (int j = L.ngy; j < L.ny + L.ngy; ++j) {
			for (int i = L.ngx; i < L.nx + L.ngx; ++i) {
				const int li = i - L.ngx;
				const int lj = j - L.ngy;
				const int lkf = kf - (L.ngz - 1);
				const int gi = L.ox + li;
				const int gj = L.oy + lj;
				const int gkf = L.oz + lkf;
				const int idn = idx3(i, j, kf, L); // use node index for x,y coords
				const int idh = idx_fy(i, j, kf, L);

				ofs << li << " " << lj << " " << lkf << " "
				    << gi << " " << gj << " " << gkf << " "
				    << z_half[idn] << "\n";
			}
		}
	}

	ofs.close();
	std::cout << "Wrote half-node z Tecplot file: " << filepath.string() << "\n";
}

void write_halfnode_invis_flux_tecplot_rank(const Field3D &F,
                                            const CartDecomp &C,
                                            char dir,
                                            unsigned long long call_id,
                                            const std::string &filename_prefix)
{
	const LocalDesc &L = F.L;
	std::filesystem::path outdir("output");
	std::error_code ec;
	std::filesystem::create_directories(outdir, ec);
	if (ec) {
		std::cerr << "Warning: could not create output directory 'output': " << ec.message() << "\n";
		return;
	}

	std::string dir_name;
	if (dir == 'x') dir_name = "xi";
	else if (dir == 'y') dir_name = "eta";
	else if (dir == 'z') dir_name = "zeta";
	else {
		std::cerr << "write_halfnode_invis_flux_tecplot_rank: unsupported direction '" << dir << "'\n";
		return;
	}

	std::ostringstream oss;
	oss << filename_prefix << "_" << dir_name << "_rank" << std::setw(4) << std::setfill('0') << C.rank
	    << "_call" << std::setw(6) << std::setfill('0') << call_id << ".dat";
	std::filesystem::path filepath = outdir / oss.str();

	std::ofstream ofs(filepath.string(), std::ofstream::out);
	if (!ofs) {
		std::cerr << "Failed to open half-node inviscid-flux file: " << filepath.string() << "\n";
		return;
	}

	ofs << "TITLE = \"Half-node Inviscid Flux (" << dir_name << ") Per Rank\"\n";
	if (dir == 'x') {
		ofs << "VARIABLES = \"i_face\" \"j\" \"k\" \"gi_face\" \"gj\" \"gk\" "
		    << "\"x_face\" \"y_face\" \"z_face\" "
		    << "\"Fx_mass\" \"Fx_momx\" \"Fx_momy\" \"Fx_momz\" \"Fx_E\"\n";
		ofs << "ZONE T=\"rank_" << C.rank << "\", I=" << (L.nx + 1) << ", J=" << L.ny << ", K=" << L.nz
		    << ", DATAPACKING=POINT\n";
		ofs << std::scientific << std::setprecision(12);
		for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
			for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
				for (int i_face = L.ngx - 1; i_face <= L.ngx + L.nx - 1; ++i_face) {
					const int li = i_face - (L.ngx - 1);
					const int gj = L.oy + (j - L.ngy);
					const int gk = L.oz + (k - L.ngz);
					const int idf = idx_fx(i_face, j, k, L);
					const int id0 = idx3(i_face, j, k, L);
					const int id1 = idx3(i_face + 1, j, k, L);
					const double x_face = 0.5 * (F.coord_x[id0] + F.coord_x[id1]);
					const double y_face = 0.5 * (F.coord_y[id0] + F.coord_y[id1]);
					const double z_face = 0.5 * (F.coord_z[id0] + F.coord_z[id1]);
					ofs << li << " " << (j - L.ngy) << " " << (k - L.ngz) << " "
					    << (L.ox + li) << " " << gj << " " << gk << " "
					    << x_face << " " << y_face << " " << z_face << " "
					    << F.flux_fx_mass[idf] << " " << F.flux_fx_momx[idf] << " "
					    << F.flux_fx_momy[idf] << " " << F.flux_fx_momz[idf] << " "
					    << F.flux_fx_E[idf] << "\n";
				}
			}
		}
	} else if (dir == 'y') {
		ofs << "VARIABLES = \"i\" \"j_face\" \"k\" \"gi\" \"gj_face\" \"gk\" "
		    << "\"x_face\" \"y_face\" \"z_face\" "
		    << "\"Fy_mass\" \"Fy_momx\" \"Fy_momy\" \"Fy_momz\" \"Fy_E\"\n";
		ofs << "ZONE T=\"rank_" << C.rank << "\", I=" << L.nx << ", J=" << (L.ny + 1) << ", K=" << L.nz
		    << ", DATAPACKING=POINT\n";
		ofs << std::scientific << std::setprecision(12);
		for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
			for (int j_face = L.ngy - 1; j_face <= L.ngy + L.ny - 1; ++j_face) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					const int lj = j_face - (L.ngy - 1);
					const int gi = L.ox + (i - L.ngx);
					const int gk = L.oz + (k - L.ngz);
					const int idf = idx_fy(i, j_face, k, L);
					const int id0 = idx3(i, j_face, k, L);
					const int id1 = idx3(i, j_face + 1, k, L);
					const double x_face = 0.5 * (F.coord_x[id0] + F.coord_x[id1]);
					const double y_face = 0.5 * (F.coord_y[id0] + F.coord_y[id1]);
					const double z_face = 0.5 * (F.coord_z[id0] + F.coord_z[id1]);
					ofs << (i - L.ngx) << " " << lj << " " << (k - L.ngz) << " "
					    << gi << " " << (L.oy + lj) << " " << gk << " "
					    << x_face << " " << y_face << " " << z_face << " "
					    << F.flux_fy_mass[idf] << " " << F.flux_fy_momx[idf] << " "
					    << F.flux_fy_momy[idf] << " " << F.flux_fy_momz[idf] << " "
					    << F.flux_fy_E[idf] << "\n";
				}
			}
		}
	} else {
		ofs << "VARIABLES = \"i\" \"j\" \"k_face\" \"gi\" \"gj\" \"gk_face\" "
		    << "\"x_face\" \"y_face\" \"z_face\" "
		    << "\"Fz_mass\" \"Fz_momx\" \"Fz_momy\" \"Fz_momz\" \"Fz_E\"\n";
		ofs << "ZONE T=\"rank_" << C.rank << "\", I=" << L.nx << ", J=" << L.ny << ", K=" << (L.nz + 1)
		    << ", DATAPACKING=POINT\n";
		ofs << std::scientific << std::setprecision(12);
		for (int k_face = L.ngz - 1; k_face <= L.ngz + L.nz - 1; ++k_face) {
			for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					const int lk = k_face - (L.ngz - 1);
					const int gi = L.ox + (i - L.ngx);
					const int gj = L.oy + (j - L.ngy);
					const int idf = idx_fz(i, j, k_face, L);
					const int id0 = idx3(i, j, k_face, L);
					const int id1 = idx3(i, j, k_face + 1, L);
					const double x_face = 0.5 * (F.coord_x[id0] + F.coord_x[id1]);
					const double y_face = 0.5 * (F.coord_y[id0] + F.coord_y[id1]);
					const double z_face = 0.5 * (F.coord_z[id0] + F.coord_z[id1]);
					ofs << (i - L.ngx) << " " << (j - L.ngy) << " " << lk << " "
					    << gi << " " << gj << " " << (L.oz + lk) << " "
					    << x_face << " " << y_face << " " << z_face << " "
					    << F.flux_fz_mass[idf] << " " << F.flux_fz_momx[idf] << " "
					    << F.flux_fz_momy[idf] << " " << F.flux_fz_momz[idf] << " "
					    << F.flux_fz_E[idf] << "\n";
				}
			}
		}
	}

	ofs.close();
	std::cout << "Wrote half-node inviscid flux file: " << filepath.string() << "\n";
}

void write_rhs_tecplot_rank(const Field3D &F,
                            const CartDecomp &C,
                            unsigned long long call_id,
                            const std::string &filename_prefix)
{
	const LocalDesc &L = F.L;
	std::filesystem::path outdir("output");
	std::error_code ec;
	std::filesystem::create_directories(outdir, ec);
	if (ec) {
		std::cerr << "Warning: could not create output directory 'output': " << ec.message() << "\n";
		return;
	}

	std::ostringstream oss;
	oss << filename_prefix << "_rank" << std::setw(4) << std::setfill('0') << C.rank
	    << "_call" << std::setw(6) << std::setfill('0') << call_id << ".dat";
	std::filesystem::path filepath = outdir / oss.str();

	std::ofstream ofs(filepath.string(), std::ofstream::out);
	if (!ofs) {
		std::cerr << "Failed to open RHS Tecplot file: " << filepath.string() << "\n";
		return;
	}

	ofs << "TITLE = \"RHS Values Per Rank\"\n";
	ofs << "VARIABLES = \"i\" \"j\" \"k\" \"gi\" \"gj\" \"gk\" "
	    << "\"x\" \"y\" \"z\" "
	    << "\"rhs_rho\" \"rhs_rhou\" \"rhs_rhov\" \"rhs_rhow\" \"rhs_E\"\n";
	ofs << "ZONE T=\"rank_" << C.rank << "\", I=" << L.nx << ", J=" << L.ny << ", K=" << L.nz
	    << ", DATAPACKING=POINT\n";

	ofs << std::scientific << std::setprecision(12);
	for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
		for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
			for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
				const int li = i - L.ngx;
				const int lj = j - L.ngy;
				const int lk = k - L.ngz;
				const int gi = L.ox + li;
				const int gj = L.oy + lj;
				const int gk = L.oz + lk;
				const int id = idx3(i, j, k, L);
				ofs << li << " " << lj << " " << lk << " "
				    << gi << " " << gj << " " << gk << " "
				    << F.coord_x[id] << " " << F.coord_y[id] << " " << F.coord_z[id] << " "
				    << F.rhs_rho[id] << " " << F.rhs_rhou[id] << " " << F.rhs_rhov[id] << " "
				    << F.rhs_rhow[id] << " " << F.rhs_E[id] << "\n";
			}
		}
	}

	ofs.close();
	std::cout << "Wrote RHS Tecplot file: " << filepath.string() << "\n";
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
		ofs << "VARIABLES = \"Step\" \"Res_rho\" \"Res_rhou\" \"Res_rhov\" \"Res_rhow\" \"Res_E\" \"Etot\" "
		    << "\"max_abs_u\" \"max_abs_v\" \"max_abs_w\" \"max_rho\" \"min_rho\" \"max_p\" \"min_p\" "
		    << "\"mean_u_x0\" \"mean_u_xpi\"\n";
	}

	ofs << std::scientific << std::setprecision(8);
	ofs << step << " " << F.global_res_rho << " " << F.global_res_rhou
		<< " " << F.global_res_rhov << " " << F.global_res_rhow << " " 
		<< F.global_res_E << " " << F.global_Etot << " "
		<< F.global_max_abs_u << " " << F.global_max_abs_v << " " << F.global_max_abs_w << " "
		<< F.global_max_rho << " " << F.global_min_rho << " "
		<< F.global_max_p << " " << F.global_min_p << " "
		<< F.mean_u_x0 << " " << F.mean_u_xpi << "\n";
	ofs.close();

}
