// 处理度量系数和雅可比相关的函数和数据结构
#include "field_structures.h"
#include "ns3d_func.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <filesystem>

static void mirror_fill_nonperiodic_ghost_layers(std::vector<double> &field,
                                                 const LocalDesc &L,
                                                 const CartDecomp &C,
                                                 const SolverParams &P)
{
	auto mirror_x = [&]() {
		if (L.nbr_xm == MPI_PROC_NULL && P.bc_xmin != SolverParams::BCType::Periodic) {
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
					for (int layer = 1; layer <= L.ngx; ++layer) {
						const int ig = L.ngx - layer;
						const int i0 = L.ngx;
						const int i1 = L.ngx + layer;
						field[idx3(ig, j, k, L)] = 2.0 * field[idx3(i0, j, k, L)] - field[idx3(i1, j, k, L)];
					}
				}
			}
		}

		if (L.nbr_xp == MPI_PROC_NULL && P.bc_xmax != SolverParams::BCType::Periodic) {
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
					for (int layer = 1; layer <= L.ngx; ++layer) {
						const int ig = L.ngx + L.nx - 1 + layer;
						const int i0 = L.ngx + L.nx - 1;
						const int i1 = L.ngx + L.nx - 1 - layer;
						field[idx3(ig, j, k, L)] = 2.0 * field[idx3(i0, j, k, L)] - field[idx3(i1, j, k, L)];
					}
				}
			}
		}
	};

	auto mirror_y = [&]() {
		if (L.nbr_ym == MPI_PROC_NULL && P.bc_ymin != SolverParams::BCType::Periodic) {
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int layer = 1; layer <= L.ngy; ++layer) {
						const int jg = L.ngy - layer;
						const int j0 = L.ngy;
						const int j1 = L.ngy + layer;
						field[idx3(i, jg, k, L)] = 2.0 * field[idx3(i, j0, k, L)] - field[idx3(i, j1, k, L)];
					}
				}
			}
		}

		if (L.nbr_yp == MPI_PROC_NULL && P.bc_ymax != SolverParams::BCType::Periodic) {
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int layer = 1; layer <= L.ngy; ++layer) {
						const int jg = L.ngy + L.ny - 1 + layer;
						const int j0 = L.ngy + L.ny - 1;
						const int j1 = L.ngy + L.ny - 1 - layer;
						field[idx3(i, jg, k, L)] = 2.0 * field[idx3(i, j0, k, L)] - field[idx3(i, j1, k, L)];
					}
				}
			}
		}
	};

	auto mirror_z = [&]() {
		if (L.nbr_zm == MPI_PROC_NULL && P.bc_zmin != SolverParams::BCType::Periodic) {
			for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int layer = 1; layer <= L.ngz; ++layer) {
						const int kg = L.ngz - layer;
						const int k0 = L.ngz;
						const int k1 = L.ngz + layer;
						field[idx3(i, j, kg, L)] = 2.0 * field[idx3(i, j, k0, L)] - field[idx3(i, j, k1, L)];
					}
				}
			}
		}

		if (L.nbr_zp == MPI_PROC_NULL && P.bc_zmax != SolverParams::BCType::Periodic) {
			for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int layer = 1; layer <= L.ngz; ++layer) {
						const int kg = L.ngz + L.nz - 1 + layer;
						const int k0 = L.ngz + L.nz - 1;
						const int k1 = L.ngz + L.nz - 1 - layer;
						field[idx3(i, j, kg, L)] = 2.0 * field[idx3(i, j, k0, L)] - field[idx3(i, j, k1, L)];
					}
				}
			}
		}
	};

	mirror_x();
	mirror_y();
	mirror_z();
}

static void copy_fill_nonperiodic_ghost_layers(std::vector<double> &field,
                                                 const LocalDesc &L,
                                                 const CartDecomp &C,
                                                 const SolverParams &P)
{
	auto copy_x = [&]() {
		if (L.nbr_xm == MPI_PROC_NULL && P.bc_xmin != SolverParams::BCType::Periodic) {
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
					for (int layer = 1; layer <= L.ngx; ++layer) {
						const int ig = L.ngx - layer;
						const int i1 = L.ngx + layer;
						field[idx3(ig, j, k, L)] = field[idx3(i1, j, k, L)];
					}
				}
			}
		}

		if (L.nbr_xp == MPI_PROC_NULL && P.bc_xmax != SolverParams::BCType::Periodic) {
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
					for (int layer = 1; layer <= L.ngx; ++layer) {
						const int ig = L.ngx + L.nx - 1 + layer;
						const int i1 = L.ngx + L.nx - 1 - layer;
						field[idx3(ig, j, k, L)] = field[idx3(i1, j, k, L)];
					}
				}
			}
		}
	};

	auto copy_y = [&]() {
		if (L.nbr_ym == MPI_PROC_NULL && P.bc_ymin != SolverParams::BCType::Periodic) {
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int layer = 1; layer <= L.ngy; ++layer) {
						const int jg = L.ngy - layer;
						const int j1 = L.ngy + layer;
						field[idx3(i, jg, k, L)] = field[idx3(i, j1, k, L)];
					}
				}
			}
		}

		if (L.nbr_yp == MPI_PROC_NULL && P.bc_ymax != SolverParams::BCType::Periodic) {
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int layer = 1; layer <= L.ngy; ++layer) {
						const int jg = L.ngy + L.ny - 1 + layer;
						const int j1 = L.ngy + L.ny - 1 - layer;
						field[idx3(i, jg, k, L)] = field[idx3(i, j1, k, L)];
					}
				}
			}
		}
	};

	auto copy_z = [&]() {
		if (L.nbr_zm == MPI_PROC_NULL && P.bc_zmin != SolverParams::BCType::Periodic) {
			for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int layer = 1; layer <= L.ngz; ++layer) {
						const int kg = L.ngz - layer;
						const int k1 = L.ngz + layer;
						field[idx3(i, j, kg, L)] = field[idx3(i, j, k1, L)];
					}
				}
			}
		}

		if (L.nbr_zp == MPI_PROC_NULL && P.bc_zmax != SolverParams::BCType::Periodic) {
			for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int layer = 1; layer <= L.ngz; ++layer) {
						const int kg = L.ngz + L.nz - 1 + layer;
						const int k1 = L.ngz + L.nz - 1 - layer;
						field[idx3(i, j, kg, L)] = field[idx3(i, j, k1, L)];
					}
				}
			}
		}
	};

	copy_x();
	copy_y();
	copy_z();
}

// 插值输入全场包括ghost的节点值，返回gh-1：gh+nx-1的半节点值，边界则返回gh层的半节点值
void interp_half_x(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L)
{
	for (int k = 0; k < L.sz; ++k) {
	for (int j = 0; j < L.sy; ++j) {
	for (int i = L.ngx - 1; i <= L.ngx + L.nx - 1; ++i) {
		face[idx_fx(i, j, k, L)] = (75.0 / 128.0) * (node[idx3(i, j, k, L)] + node[idx3(i + 1, j, k, L)])
				- (25.0 / 256.0) * (node[idx3(i - 1, j, k, L)] + node[idx3(i + 2, j, k, L)])
				+ ( 3.0 / 256.0) * (node[idx3(i - 2, j, k, L)] + node[idx3(i + 3, j, k, L)]);
	}}}
}

void interp_half_x_boundary(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L)
{
	if (L.nbr_xm != MPI_PROC_NULL && L.nbr_xp != MPI_PROC_NULL)  return;

	for (int k = 0; k < L.sz; ++k) {
	for (int j = 0; j < L.sy; ++j) {
		if (L.nbr_xm == MPI_PROC_NULL) {
			const int i0 = 0;
			face[idx_fx(i0, j, k, L)] = (1.0 / 128.0) * (315.0 * node[idx3(i0 + 1, j, k, L)] - 420.0 * node[idx3(i0 + 2, j, k, L)]
							+ 378.0 * node[idx3(i0 + 3, j, k, L)] - 180.0 * node[idx3(i0 + 4, j, k, L)]
							+ 35.0 * node[idx3(i0 + 5, j, k, L)]);
			face[idx_fx(i0 + 1, j, k, L)] = (1.0 / 128.0) * (35.0 * node[idx3(i0 + 1, j, k, L)] + 140.0 * node[idx3(i0 + 2, j, k, L)]
							- 70.0 * node[idx3(i0 + 3, j, k, L)] + 28.0 * node[idx3(i0 + 4, j, k, L)]
							- 5.0 * node[idx3(i0 + 5, j, k, L)]);
			face[idx_fx(i0 + 2, j, k, L)] = (1.0 / 128.0) * (- 5.0 * node[idx3(i0 + 1, j, k, L)] + 60.0 * node[idx3(i0 + 2, j, k, L)]
							+ 90.0 * node[idx3(i0 + 3, j, k, L)] - 20.0 * node[idx3(i0 + 4, j, k, L)]
							+ 3.0 * node[idx3(i0 + 5, j, k, L)]);
		}
		if (L.nbr_xp == MPI_PROC_NULL) {
			const int i1 = L.sx - 1;
			face[idx_fx(i1 - 1, j, k, L)] = (1.0 / 128.0) * (315.0 * node[idx3(i1 - 1, j, k, L)] - 420.0 * node[idx3(i1 - 2, j, k, L)]
							+ 378.0 * node[idx3(i1 - 3, j, k, L)] - 180.0 * node[idx3(i1 - 4, j, k, L)] 
							+ 35.0 * node[idx3(i1 - 5, j, k, L)]);
			face[idx_fx(i1 - 2, j, k, L)] = (1.0 / 128.0) * (35.0 * node[idx3(i1 - 1, j, k, L)] + 140.0 * node[idx3(i1 - 2, j, k, L)]
							- 70.0 * node[idx3(i1 - 3, j, k, L)] + 28.0 * node[idx3(i1 - 4, j, k, L)]
							- 5.0 * node[idx3(i1 - 5, j, k, L)]);
			face[idx_fx(i1 - 3, j, k, L)] = (1.0 / 128.0) * (- 5.0 * node[idx3(i1 - 1, j, k, L)] + 60.0 * node[idx3(i1 - 2, j, k, L)]
							+ 90.0 * node[idx3(i1 - 3, j, k, L)] - 20.0 * node[idx3(i1 - 4, j, k, L)]
							+ 3.0 * node[idx3(i1 - 5, j, k, L)]);
		}
	}}
}

void interp_half_y(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L)
{
	for (int k = 0; k < L.sz; ++k) {
	for (int i = 0; i < L.sx; ++i) {
	for (int j = L.ngy - 1; j <= L.ngy + L.ny - 1; ++j) {
		face[idx_fy(i, j, k, L)] = (75.0 / 128.0) * (node[idx3(i, j, k, L)] + node[idx3(i, j + 1, k, L)])
			- (25.0 / 256.0) * (node[idx3(i, j - 1, k, L)] + node[idx3(i, j + 2, k, L)])
			+ ( 3.0 / 256.0) * (node[idx3(i, j - 2, k, L)] + node[idx3(i, j + 3, k, L)]);
	}}}
}

void interp_half_y_boundary(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L)
{
	if (L.nbr_ym != MPI_PROC_NULL && L.nbr_yp != MPI_PROC_NULL)  return;

	for (int k = 0; k < L.sz; ++k) {
	for (int i = 0; i < L.sx; ++i) {
		if (L.nbr_ym == MPI_PROC_NULL) {
			const int j0 = 0;
			face[idx_fy(i, j0, k, L)] = (1.0 / 128.0) * (315.0 * node[idx3(i, j0 + 1, k, L)] - 420.0 * node[idx3(i, j0 + 2, k, L)] 
					+ 378.0 * node[idx3(i, j0 + 3, k, L)] - 180.0 * node[idx3(i,j0 + 4, k, L)] 
					+ 35.0 * node[idx3(i, j0 + 5, k, L)]);
			face[idx_fy(i, j0 + 1, k, L)] = (1.0 / 128.0) * (35.0 * node[idx3(i, j0 + 1, k, L)] + 140.0 * node[idx3(i, j0 + 2, k, L)] 
					- 70.0 * node[idx3(i, j0 + 3, k, L)] + 28.0 * node[idx3(i, j0 + 4, k, L)] 
					- 5.0 * node[idx3(i, j0 + 5, k, L)]);
			face[idx_fy(i, j0 + 2, k, L)] = (1.0 / 128.0) * (- 5.0 * node[idx3(i, j0 + 1, k, L)] + 60.0 * node[idx3(i, j0 + 2, k, L)]
					+ 90.0 * node[idx3(i, j0 + 3, k, L)] - 20.0 * node[idx3(i, j0 + 4, k, L)]
					+ 3.0 * node[idx3(i, j0 + 5, k, L)]);
		}
		if (L.nbr_yp == MPI_PROC_NULL) {
			const int j1 = L.sy - 1;
			face[idx_fy(i, j1 - 1, k, L)] = (1.0 / 128.0) * (315.0 * node[idx3(i, j1 - 1, k, L)] - 420.0 * node[idx3(i, j1 - 2, k, L)]
					+ 378.0 * node[idx3(i, j1 - 3, k, L)] - 180.0 * node[idx3(i, j1 - 4, k, L)] 
					+ 35.0 * node[idx3(i, j1 - 5, k, L)]);
			face[idx_fy(i, j1 - 2, k, L)] = (1.0 / 128.0) * (35.0 * node[idx3(i, j1 - 1, k, L)] + 140.0 * node[idx3(i, j1 - 2, k, L)]
					- 70.0 * node[idx3(i, j1 - 3, k, L)] + 28.0 * node[idx3(i, j1 - 4, k, L)]
					- 5.0 * node[idx3(i, j1 - 5, k, L)]);
			face[idx_fy(i, j1 - 3, k, L)] = (1.0 / 128.0) * (- 5.0 * node[idx3(i, j1 - 1, k, L)] + 60.0 * node[idx3(i, j1 - 2, k, L)]
					+ 90.0 * node[idx3(i, j1 - 3, k, L)] - 20.0 * node[idx3(i, j1 - 4, k, L)]
					+ 3.0 * node[idx3(i, j1 - 5, k, L)]);
		}
	}}
}

void interp_half_z(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L)
{
	for (int j = 0; j < L.sy; ++j) {
	for (int i = 0; i < L.sx; ++i) {
	for (int k = L.ngz - 1; k <= L.ngz + L.nz - 1; ++k) {
		face[idx_fz(i, j, k, L)] = (75.0 / 128.0) * (node[idx3(i, j, k, L)] + node[idx3(i, j, k + 1, L)])
			- (25.0 / 256.0) * (node[idx3(i, j, k - 1, L)] + node[idx3(i, j, k + 2, L)])
			+ ( 3.0 / 256.0) * (node[idx3(i, j, k - 2, L)] + node[idx3(i, j, k + 3, L)]);
	}}}
}

void interp_half_z_boundary(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L)
{
	if (L.nbr_zm != MPI_PROC_NULL && L.nbr_zp != MPI_PROC_NULL)  return;

	for (int j = 0; j < L.sy; ++j) {
	for (int i = 0; i < L.sx; ++i) {
		if (L.nbr_zm == MPI_PROC_NULL) {
			const int k0 = 0;
			face[idx_fz(i, j, k0, L)] = (1.0 / 128.0) * (315.0 * node[idx3(i, j, k0 + 1, L)] - 420.0 * node[idx3(i, j, k0 + 2, L)] 
				+ 378.0 * node[idx3(i, j, k0 + 3, L)] - 180.0 * node[idx3(i, j, k0 + 4, L)] 
				+ 35.0 * node[idx3(i, j, k0 + 5, L)]);
			face[idx_fz(i, j, k0 + 1, L)] = (1.0 / 128.0) * (35.0 * node[idx3(i, j, k0 + 1, L)] + 140.0 * node[idx3(i, j, k0 + 2, L)] 
				- 70.0 * node[idx3(i, j, k0 + 3, L)] + 28.0 * node[idx3(i, j, k0 + 4, L)] 
				- 5.0 * node[idx3(i, j, k0 + 5, L)]);
			face[idx_fz(i, j, k0 + 2, L)] = (1.0 / 128.0) * (- 5.0 * node[idx3(i, j, k0 + 1, L)] + 60.0 * node[idx3(i, j, k0 + 2, L)]
				+ 90.0 * node[idx3(i, j, k0 + 3, L)] - 20.0 * node[idx3(i, j, k0 + 4, L)]
				+ 3.0 * node[idx3(i, j, k0 + 5, L)]);
		}
		if (L.nbr_zp == MPI_PROC_NULL) {
			const int k1 = L.sz - 1;
			face[idx_fz(i, j, k1 - 1, L)] = (1.0 / 128.0) * (315.0 * node[idx3(i, j, k1 - 1, L)] - 420.0 * node[idx3(i, j, k1 - 2, L)]
				+ 378.0 * node[idx3(i, j, k1 - 3, L)] - 180.0 * node[idx3(i, j, k1 - 4, L)] 
				+ 35.0 * node[idx3(i, j, k1 - 5, L)]);
			face[idx_fz(i, j, k1 - 2, L)] = (1.0 / 128.0) * (35.0 * node[idx3(i, j, k1 - 1, L)] + 140.0 * node[idx3(i, j, k1 - 2, L)]
				- 70.0 * node[idx3(i, j, k1 - 3, L)] + 28.0 * node[idx3(i, j, k1 - 4, L)]
				- 5.0 * node[idx3(i, j, k1 - 5, L)]);
			face[idx_fz(i, j, k1 - 3, L)] = (1.0 / 128.0) * (- 5.0 * node[idx3(i, j, k1 - 1, L)] + 60.0 * node[idx3(i, j, k1 - 2, L)]
				+ 90.0 * node[idx3(i, j, k1 - 3, L)] - 20.0 * node[idx3(i, j, k1 - 4, L)]
				+ 3.0 * node[idx3(i, j, k1 - 5, L)]);
		}
	}}
}
void exchange_half_halo_x(std::vector<double> &a,
							const LocalDesc &L,
							const CartDecomp &C,
							int layers,
							int tag_base)
{
	// 半节点只交换次外层数据，最外层在各自插值中得到
	const int count = (layers-1) * L.sy * L.sz;
	std::vector<double> send_l(count), send_r(count), recv_l(count), recv_r(count);

	const int left_start  = L.ngx;
	const int right_start = L.ngx + L.nx - layers;

	int p = 0;
	for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
			for (int ii = 0; ii < layers - 1; ++ii) {
				send_l[p++] = a[idx_fx(left_start + ii, j, k, L)];
			}
		}
	}
	p = 0;
	for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
			for (int ii = 0; ii < layers - 1; ++ii) {
				send_r[p++] = a[idx_fx(right_start + ii, j, k, L)];
			}
		}
	}

	MPI_Request reqs[4];
	MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, L.nbr_xm, tag_base + 0, C.cart_comm, &reqs[0]);
	MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, L.nbr_xp, tag_base + 1, C.cart_comm, &reqs[1]);
	MPI_Isend(send_r.data(), count, MPI_DOUBLE, L.nbr_xp, tag_base + 0, C.cart_comm, &reqs[2]);
	MPI_Isend(send_l.data(), count, MPI_DOUBLE, L.nbr_xm, tag_base + 1, C.cart_comm, &reqs[3]);
	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

	if (L.nbr_xm != MPI_PROC_NULL) {
		p = 0;
		for (int k = 0; k < L.sz; ++k) {
			for (int j = 0; j < L.sy; ++j) {
				for (int ii = 0; ii < layers - 1; ++ii) {
					a[idx_fx(ii, j, k, L)] = recv_l[p++];
				}
			}
		}
	}

	if (L.nbr_xp != MPI_PROC_NULL) {
		p = 0;
		const int right_ghost_start = L.ngx + L.nx;
		for (int k = 0; k < L.sz; ++k) {
			for (int j = 0; j < L.sy; ++j) {
				for (int ii = 0; ii < layers - 1; ++ii) {
					a[idx_fx(right_ghost_start + ii, j, k, L)] = recv_r[p++];
				}
			}
		}
	}
}

void exchange_half_halo_y(std::vector<double> &a,
							const LocalDesc &L,
							const CartDecomp &C,
							int layers,
							int tag_base)
{
	const int count = (layers - 1) * L.sx * L.sz;
	std::vector<double> send_l(count), send_r(count), recv_l(count), recv_r(count);

	const int left_start  = L.ngy;
	const int right_start = L.ngy + L.ny - layers;

	int p = 0;
	for (int k = 0; k < L.sz; ++k) {
		for (int i = 0; i < L.sx; ++i) {
			for (int jj = 0; jj < layers - 1; ++jj) {
				send_l[p++] = a[idx_fy(i, left_start + jj, k, L)];
			}
		}
	}
	p = 0;
	for (int k = 0; k < L.sz; ++k) {
		for (int i = 0; i < L.sx; ++i) {
			for (int jj = 0; jj < layers - 1; ++jj) {
				send_r[p++] = a[idx_fy(i, right_start + jj, k, L)];
			}
		}
	}

	MPI_Request reqs[4];
	MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, L.nbr_ym, tag_base + 0, C.cart_comm, &reqs[0]);
	MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, L.nbr_yp, tag_base + 1, C.cart_comm, &reqs[1]);
	MPI_Isend(send_r.data(), count, MPI_DOUBLE, L.nbr_yp, tag_base + 0, C.cart_comm, &reqs[2]);
	MPI_Isend(send_l.data(), count, MPI_DOUBLE, L.nbr_ym, tag_base + 1, C.cart_comm, &reqs[3]);
	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

	if (L.nbr_ym != MPI_PROC_NULL) {
		p = 0;
		for (int k = 0; k < L.sz; ++k) {
			for (int i = 0; i < L.sx; ++i) {
				for (int jj = 0; jj < layers - 1; ++jj) {
					a[idx_fy(i, jj, k, L)] = recv_l[p++];
				}
			}
		}
	}

	if (L.nbr_yp != MPI_PROC_NULL) {
		p = 0;
		const int right_ghost_start = L.ngy + L.ny;
		for (int k = 0; k < L.sz; ++k) {
			for (int i = 0; i < L.sx; ++i) {
				for (int jj = 0; jj < layers - 1; ++jj) {
					a[idx_fy(i, right_ghost_start + jj, k, L)] = recv_r[p++];
				}
			}
		}
	}
}

void exchange_half_halo_z(std::vector<double> &a,
							const LocalDesc &L,
							const CartDecomp &C,
							int layers,
							int tag_base)
{
	const int count = (layers - 1) * L.sx * L.sy;
	std::vector<double> send_l(count), send_r(count), recv_l(count), recv_r(count);

	const int left_start  = L.ngz;
	const int right_start = L.ngz + L.nz - layers;

	int p = 0;
	for (int j = 0; j < L.sy; ++j) {
		for (int i = 0; i < L.sx; ++i) {
			for (int kk = 0; kk < layers - 1; ++kk) {
				send_l[p++] = a[idx_fz(i, j, left_start + kk, L)];
			}
		}
	}
	p = 0;
	for (int j = 0; j < L.sy; ++j) {
		for (int i = 0; i < L.sx; ++i) {
			for (int kk = 0; kk < layers - 1; ++kk) {
				send_r[p++] = a[idx_fz(i, j, right_start + kk, L)];
			}
		}
	}

	MPI_Request reqs[4];
	MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, L.nbr_zm, tag_base + 0, C.cart_comm, &reqs[0]);
	MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, L.nbr_zp, tag_base + 1, C.cart_comm, &reqs[1]);
	MPI_Isend(send_r.data(), count, MPI_DOUBLE, L.nbr_zp, tag_base + 0, C.cart_comm, &reqs[2]);
	MPI_Isend(send_l.data(), count, MPI_DOUBLE, L.nbr_zm, tag_base + 1, C.cart_comm, &reqs[3]);
	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

	if (L.nbr_zm != MPI_PROC_NULL) {
		p = 0;
		for (int j = 0; j < L.sy; ++j) {
			for (int i = 0; i < L.sx; ++i) {
				for (int kk = 0; kk < layers - 1; ++kk) {
					a[idx_fz(i, j, kk, L)] = recv_l[p++];
				}
			}
		}
	}

	if (L.nbr_zp != MPI_PROC_NULL) {
		p = 0;
		const int right_ghost_start = L.ngz + L.nz;
		for (int j = 0; j < L.sy; ++j) {
			for (int i = 0; i < L.sx; ++i) {
				for (int kk = 0; kk < layers - 1; ++kk) {
					a[idx_fz(i, j, right_ghost_start + kk, L)] = recv_r[p++];
				}
			}
		}
	}
}

void exchange_node_halo_x(std::vector<double> &a,
							const LocalDesc &L,
							const CartDecomp &C,
							int layers,
							int tag_base)
{
	// 节点交换ghost数据
	const int count = layers * L.sy * L.sz;
	std::vector<double> send_l(count), send_r(count), recv_l(count), recv_r(count);

	const int left_start  = L.ngx;
	const int right_start = L.ngx + L.nx - layers;

	int p = 0;
	for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
			for (int ii = 0; ii < layers; ++ii) {
				send_l[p++] = a[idx3(left_start + ii, j, k, L)];
			}
		}
	}
	p = 0;
	for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
			for (int ii = 0; ii < layers; ++ii) {
				send_r[p++] = a[idx3(right_start + ii, j, k, L)];
			}
		}
	}

	MPI_Request reqs[4];
	MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, L.nbr_xm, tag_base + 0, C.cart_comm, &reqs[0]);
	MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, L.nbr_xp, tag_base + 1, C.cart_comm, &reqs[1]);
	MPI_Isend(send_r.data(), count, MPI_DOUBLE, L.nbr_xp, tag_base + 0, C.cart_comm, &reqs[2]);
	MPI_Isend(send_l.data(), count, MPI_DOUBLE, L.nbr_xm, tag_base + 1, C.cart_comm, &reqs[3]);
	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

	p = 0;
	if(L.nbr_xm != MPI_PROC_NULL)
	{
		for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
		for (int ii = 0; ii < layers; ++ii) {
				a[idx3(ii, j, k, L)] = recv_l[p++];
		}}}
	}

	p = 0;
	if(L.nbr_xp != MPI_PROC_NULL)
	{
		const int right_ghost_start = L.ngx + L.nx;
		for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
		for (int ii = 0; ii < layers; ++ii) {
			a[idx3(right_ghost_start + ii, j, k, L)] = recv_r[p++];
		}}}
	}
}

void exchange_node_halo_y(std::vector<double> &a,
							const LocalDesc &L,
							const CartDecomp &C,
							int layers,
							int tag_base)
{
	const int count = layers * L.sx * L.sz;
	std::vector<double> send_l(count), send_r(count), recv_l(count), recv_r(count);

	const int left_start  = L.ngy;
	const int right_start = L.ngy + L.ny - layers;

	int p = 0;
	for (int k = 0; k < L.sz; ++k) {
		for (int i = 0; i < L.sx; ++i) {
			for (int jj = 0; jj < layers; ++jj) {
				send_l[p++] = a[idx3(i, left_start + jj, k, L)];
			}
		}
	}
	p = 0;
	for (int k = 0; k < L.sz; ++k) {
		for (int i = 0; i < L.sx; ++i) {
			for (int jj = 0; jj < layers; ++jj) {
				send_r[p++] = a[idx3(i, right_start + jj, k, L)];
			}
		}
	}

	MPI_Request reqs[4];
	MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, L.nbr_ym, tag_base + 0, C.cart_comm, &reqs[0]);
	MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, L.nbr_yp, tag_base + 1, C.cart_comm, &reqs[1]);
	MPI_Isend(send_r.data(), count, MPI_DOUBLE, L.nbr_yp, tag_base + 0, C.cart_comm, &reqs[2]);
	MPI_Isend(send_l.data(), count, MPI_DOUBLE, L.nbr_ym, tag_base + 1, C.cart_comm, &reqs[3]);
	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

	if(L.nbr_ym != MPI_PROC_NULL)
	{	
		p = 0;
		for (int k = 0; k < L.sz; ++k) {
		for (int i = 0; i < L.sx; ++i) {
		for (int jj = 0; jj < layers; ++jj) {
			a[idx3(i, jj, k, L)] = recv_l[p++];
		}}}
	}

	if(L.nbr_yp != MPI_PROC_NULL)
	{
		p = 0;
		const int right_ghost_start = L.ngy + L.ny;
		for (int k = 0; k < L.sz; ++k) {
		for (int i = 0; i < L.sx; ++i) {
		for (int jj = 0; jj < layers; ++jj) {
			a[idx3(i, right_ghost_start + jj, k, L)] = recv_r[p++];
		}}}
	}
}

void exchange_node_halo_z(std::vector<double> &a,
							const LocalDesc &L,
							const CartDecomp &C,
							int layers,
							int tag_base)
{
	const int count = layers * L.sx * L.sy;
	std::vector<double> send_l(count), send_r(count), recv_l(count), recv_r(count);

	const int left_start  = L.ngz;
	const int right_start = L.ngz + L.nz - layers;

	int p = 0;
	for (int j = 0; j < L.sy; ++j) {
		for (int i = 0; i < L.sx; ++i) {
			for (int kk = 0; kk < layers; ++kk) {
				send_l[p++] = a[idx3(i, j, left_start + kk, L)];
			}
		}
	}
	p = 0;
	for (int j = 0; j < L.sy; ++j) {
		for (int i = 0; i < L.sx; ++i) {
			for (int kk = 0; kk < layers; ++kk) {
				send_r[p++] = a[idx3(i, j, right_start + kk, L)];
			}
		}
	}

	MPI_Request reqs[4];
	MPI_Irecv(recv_l.data(), count, MPI_DOUBLE, L.nbr_zm, tag_base + 0, C.cart_comm, &reqs[0]);
	MPI_Irecv(recv_r.data(), count, MPI_DOUBLE, L.nbr_zp, tag_base + 1, C.cart_comm, &reqs[1]);
	MPI_Isend(send_r.data(), count, MPI_DOUBLE, L.nbr_zp, tag_base + 0, C.cart_comm, &reqs[2]);
	MPI_Isend(send_l.data(), count, MPI_DOUBLE, L.nbr_zm, tag_base + 1, C.cart_comm, &reqs[3]);
	MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

	if(L.nbr_zm != MPI_PROC_NULL)
	{
		p = 0;
		for (int j = 0; j < L.sy; ++j) {
		for (int i = 0; i < L.sx; ++i) {
		for (int kk = 0; kk < layers; ++kk) {
			a[idx3(i, j, kk, L)] = recv_l[p++];
		}}}
	}

	if(L.nbr_zp != MPI_PROC_NULL)
	{
	 	p = 0;
		const int right_ghost_start = L.ngz + L.nz;
		for (int j = 0; j < L.sy; ++j) {
		for (int i = 0; i < L.sx; ++i) {
		for (int kk = 0; kk < layers; ++kk) {
			a[idx3(i, j, right_ghost_start + kk, L)] = recv_r[p++];
		}}}
	}
}

// 输入tot=sx*sy*sz的phi，返回tot
// 计算dphi/dtheta并存入dphi_dtheta。theta指定求导方向，dxi/deta/dzeta。
// dtheta是对应方向的网格间距（物理距离）。返回false表示输入错误。
bool compute_dphi_dtheta(const std::vector<double> &phi,
						 const std::string &theta,
						 double dtheta,
						 const CartDecomp &C,
						 const LocalDesc &L,
						 std::vector<double> &dphi_dtheta)
{
	if (phi.size() != static_cast<std::size_t>(L.sx * L.sy * L.sz)) {
		std::cerr << "compute_dphi_dtheta: input size mismatch\n";
		return false;
	}
	if (dtheta == 0.0) {
		std::cerr << "compute_dphi_dtheta: dtheta must be non-zero\n";
		return false;
	}

	dphi_dtheta.assign(phi.size(), 0.0);
	const double inv_d = 1.0 / dtheta;

	const bool is_x = (theta == "xi" || theta == "x");
	const bool is_y = (theta == "eta" || theta == "y");
	const bool is_z = (theta == "zeta" || theta == "z");
	if (!is_x && !is_y && !is_z) {
		std::cerr << "compute_dphi_dtheta: unsupported theta='" << theta << "'\n";
		return false;
	}

	if (is_x) {
		std::vector<double> phi_half((L.sx-1) * L.sy * L.sz);
		interp_half_x(phi, phi_half, L);

		exchange_half_halo_x(phi_half, L, C, L.ngx, 1010);

		diff_x_half(phi_half, dphi_dtheta, inv_d, L);

		exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 1020);
		exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 1030);
		exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 1040);
		return true;
	}

	if (is_y) {
		std::vector<double> phi_half(L.sx * (L.sy-1) * L.sz);
		interp_half_y(phi, phi_half, L);

		exchange_half_halo_y(phi_half, L, C, L.ngy, 2010);

		diff_y_half(phi_half, dphi_dtheta, inv_d, L);

		exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 2020);
		exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 2030);
		exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 2040);
		return true;
	}

	std::vector<double> phi_half(L.sx * L.sy * (L.sz-1));
	interp_half_z(phi, phi_half, L);

	exchange_half_halo_z(phi_half, L, C, L.ngz, 3010);

	diff_z_half(phi_half, dphi_dtheta, inv_d, L);

	exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 3020);
	exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 3030);
	exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 3040);
	return true;
}

namespace {

inline double periodic_shift_component(const SolverParams &P, const std::string &theta, char coord_comp)
{
	if (theta == "xi" || theta == "x") {
		if (coord_comp == 'x') return P.periodic_xi_px;
		if (coord_comp == 'y') return P.periodic_xi_py;
		return P.periodic_xi_pz;
	}
	if (theta == "eta" || theta == "y") {
		if (coord_comp == 'x') return P.periodic_eta_px;
		if (coord_comp == 'y') return P.periodic_eta_py;
		return P.periodic_eta_pz;
	}
	if (coord_comp == 'x') return P.periodic_zeta_px;
	if (coord_comp == 'y') return P.periodic_zeta_py;
	return P.periodic_zeta_pz;
}

// 计算坐标分量对theta的导数，结果存入dphi_dtheta。theta指定求导方向，dxi/deta/dzeta。
// dtheta是对应方向的网格间距（物理距离）。返回false表示输入错误。
bool compute_dphi_dtheta_coord(const std::vector<double> &coord_component,
					   char coord_comp,
					   const std::string &theta,
					   double dtheta,
					   const CartDecomp &C,
					   const LocalDesc &L,
					   const SolverParams &P,
					   std::vector<double> &dphi_dtheta)
{
	const std::size_t tot = static_cast<std::size_t>(L.sx) * static_cast<std::size_t>(L.sy) * static_cast<std::size_t>(L.sz);
	if (coord_component.size() != tot) {
		std::cerr << "compute_dphi_dtheta_coord: input size mismatch\n";
		return false;
	}
	if (dtheta == 0.0) {
		std::cerr << "compute_dphi_dtheta_coord: dtheta must be non-zero\n";
		return false;
	}

	dphi_dtheta.assign(tot, 0.0);
	const double inv_d = 1.0 / dtheta;
	const bool is_x = (theta == "xi");
	const bool is_y = (theta == "eta");
	const bool is_z = (theta == "zeta");
	if (!is_x && !is_y && !is_z) {
		std::cerr << "compute_dphi_dtheta_coord: unsupported theta='" << theta << "'\n";
		return false;
	}

	const bool is_periodic_dir = (is_x && C.periods[0]) || (is_y && C.periods[1]) || (is_z && C.periods[2]);
	const double shift = is_periodic_dir ? periodic_shift_component(P, theta, coord_comp) : 0.0;

	if (is_x) {
		std::vector<double> coord_half((L.sx - 1) * L.sy * L.sz, 0.0);
		interp_half_x(coord_component, coord_half, L);
		exchange_half_halo_x(coord_half, L, C, L.ngx, 7010);
		if (C.coords[0] == 0) {
			for (int k = 0; k < L.sz; ++k) {
			for (int j = 0; j < L.sy; ++j) {
			for (int ii = 0; ii < L.ngx - 1; ++ii) {
					const int idh = idx_fx(ii, j, k, L);
					coord_half[idh] -= shift;
			}}}
	    }

		if (C.coords[0] == C.dims[0] - 1) {
			const int right_ghost_start = L.ngx + L.nx;
			for (int k = 0; k < L.sz ; ++k) {
			for (int j = 0; j < L.sy; ++j) {
			for (int ii = 0; ii < L.ngx - 1; ++ii) {
					const int idh = idx_fx(right_ghost_start + ii, j, k, L);
					coord_half[idh] += shift;
				}}}
		}
		interp_half_x_boundary(coord_component, coord_half, L);

		diff_x_half(coord_half, dphi_dtheta, inv_d, L);
		exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 7030);
		exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 7040);
		exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 7050);
		diff_x_half_boundary(coord_half, dphi_dtheta, inv_d, L);
		// copy_fill_nonperiodic_ghost_layers(dphi_dtheta, L, C, P);
		
		return true;
	}

	if (is_y) {
		std::vector<double> coord_half(L.sx * (L.sy - 1) * L.sz, 0.0);
		interp_half_y(coord_component, coord_half, L);
		exchange_half_halo_y(coord_half, L, C, L.ngy, 7110);
		if (C.coords[1] == 0) {
			for (int k = 0; k < L.sz ; ++k) {
			for (int i = 0; i < L.sx; ++i) {
			for (int jj = 0; jj < L.ngy - 1; ++jj) {
				const int idh = idx_fy(i, jj, k, L);
				coord_half[idh] -= shift;
			}}}
	    }
		if (C.coords[1] == C.dims[1] - 1) {
			const int right_ghost_start = L.ngy + L.ny;
			for (int k = 0; k < L.sz; ++k) {
			for (int i = 0; i < L.sx; ++i) {
			for (int jj = 0; jj < L.ngy - 1; ++jj) {
				const int idh = idx_fy(i, right_ghost_start + jj, k, L);
				coord_half[idh] += shift;
			}}}
		}
		interp_half_y_boundary(coord_component, coord_half, L);

		diff_y_half(coord_half, dphi_dtheta, inv_d, L);
		exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 7130);
		exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 7140);
		exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 7150);
		diff_y_half_boundary(coord_half, dphi_dtheta, inv_d, L);
		// copy_fill_nonperiodic_ghost_layers(dphi_dtheta, L, C, P);
		return true;
	}

	std::vector<double> coord_half(L.sx * L.sy * (L.sz - 1), 0.0);
	interp_half_z(coord_component, coord_half, L);
	exchange_half_halo_z(coord_half, L, C, L.ngz, 7210);
	if (C.coords[2] == 0) {
		for (int j = 0; j < L.sy; ++j) {
		for (int i = 0; i < L.sx; ++i) {
		for (int kk = 0; kk < L.ngz - 1; ++kk) {
			const int idh = idx_fz(i, j, kk, L);
			coord_half[idh] -= shift;
		}}}
	}
	if (C.coords[2] == C.dims[2] - 1) {
		const int right_ghost_start = L.ngz + L.nz;
		for (int j = 0; j < L.sy; ++j) {
		for (int i = 0; i < L.sx; ++i) {
		for (int kk = 0; kk < L.ngz - 1; ++kk) {
			const int idh = idx_fz(i, j, right_ghost_start + kk, L);
			coord_half[idh] += shift;
		}}}
	}
	interp_half_z_boundary(coord_component, coord_half, L);

	diff_z_half(coord_half, dphi_dtheta, inv_d, L);
	exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 7230);
	exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 7240);
	exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 7250);
	diff_z_half_boundary(coord_half, dphi_dtheta, inv_d, L);

	// copy_fill_nonperiodic_ghost_layers(dphi_dtheta, L, C, P);

	return true;
}

void apply_half_periodic_shift_x(std::vector<double> &prod_half,
								 const std::vector<double> &factor_half,
								 double shift,
								 const LocalDesc &L,
								 const CartDecomp &C)
{
	if (!C.periods[0] || shift == 0.0) return;

	if (C.coords[0] == 0) {
		for (int k = 0; k < L.sz ; ++k) {
			for (int j = 0; j < L.sy; ++j) {
				for (int ii = 0; ii < L.ngx - 1; ++ii) {
					const int idh = idx_fx(ii, j, k, L);
					prod_half[idh] -= factor_half[idh] * shift;
				}
			}
		}
	}

	if (C.coords[0] == C.dims[0] - 1) {
		const int right_ghost_start = L.ngx + L.nx;
		for (int k = 0; k < L.sz; ++k) {
			for (int j = 0; j < L.sy; ++j) {
				for (int ii = 0; ii < L.ngx - 1; ++ii) {
					const int idh = idx_fx(right_ghost_start + ii, j, k, L);
					prod_half[idh] += factor_half[idh] * shift;
				}
			}
		}
	}
}

void apply_half_periodic_shift_y(std::vector<double> &prod_half,
								 const std::vector<double> &factor_half,
								 double shift,
								 const LocalDesc &L,
								 const CartDecomp &C)
{
	if (!C.periods[1] || shift == 0.0) return;

	if (C.coords[1] == 0) {
		for (int k = 0; k < L.sz; ++k) {
			for (int i = 0; i < L.sx; ++i) {
				for (int jj = 0; jj < L.ngy - 1; ++jj) {
					const int idh = idx_fy(i, jj, k, L);
					prod_half[idh] -= factor_half[idh] * shift;
				}
			}
		}
	}

	if (C.coords[1] == C.dims[1] - 1) {
		const int right_ghost_start = L.ngy + L.ny;
		for (int k = 0; k < L.sz; ++k) {
			for (int i = 0; i < L.sx; ++i) {
				for (int jj = 0; jj < L.ngy - 1; ++jj) {
					const int idh = idx_fy(i, right_ghost_start + jj, k, L);
					prod_half[idh] += factor_half[idh] * shift;
				}
			}
		}
	}
}

void apply_half_periodic_shift_z(std::vector<double> &prod_half,
								 const std::vector<double> &factor_half,
								 double shift,
								 const LocalDesc &L,
								 const CartDecomp &C)
{
	if (!C.periods[2] || shift == 0.0) return;

	if (C.coords[2] == 0) {
		for (int j = 0; j < L.sy; ++j) {
			for (int i = 0; i < L.sx; ++i) {
				for (int kk = 0; kk < L.ngz - 1; ++kk) {
					const int idh = idx_fz(i, j, kk, L);
					prod_half[idh] -= factor_half[idh] * shift;
				}
			}
		}
	}

	if (C.coords[2] == C.dims[2] - 1) {
		const int right_ghost_start = L.ngz + L.nz;
		for (int j = 0; j < L.sy; ++j) {
			for (int i = 0; i < L.sx; ++i) {
				for (int kk = 0; kk < L.ngz - 1; ++kk) {
					const int idh = idx_fz(i, j, right_ghost_start + kk, L);
					prod_half[idh] += factor_half[idh] * shift;
				}
			}
		}
	}
}

bool compute_dphi_dtheta_coord_product(const std::vector<double> &periodic_factor,
									   const std::vector<double> &coord_component,
									   char coord_comp,
									   const std::string &theta,
									   double dtheta,
									   const CartDecomp &C,
									   const LocalDesc &L,
									   const SolverParams &P,
									   std::vector<double> &dphi_dtheta)
{
	const std::size_t tot = static_cast<std::size_t>(L.sx) * static_cast<std::size_t>(L.sy) * static_cast<std::size_t>(L.sz);
	if (periodic_factor.size() != tot || coord_component.size() != tot) {
		std::cerr << "compute_dphi_dtheta_coord_product: input size mismatch\n";
		return false;
	}
	if (dtheta == 0.0) {
		std::cerr << "compute_dphi_dtheta_coord_product: dtheta must be non-zero\n";
		return false;
	}

	std::vector<double> prod(tot, 0.0);

	for (std::size_t id = 0; id < tot; ++id) {
		prod[id] = periodic_factor[id] * coord_component[id];
	}
	mirror_fill_nonperiodic_ghost_layers(prod, L, C, P);

	dphi_dtheta.assign(tot, 0.0);
	const double inv_d = 1.0 / dtheta;
	const double shift = periodic_shift_component(P, theta, coord_comp);

	const bool is_x = (theta == "xi" || theta == "x");
	const bool is_y = (theta == "eta" || theta == "y");
	const bool is_z = (theta == "zeta" || theta == "z");
	if (!is_x && !is_y && !is_z) {
		std::cerr << "compute_dphi_dtheta_coord_product: unsupported theta='" << theta << "'\n";
		return false;
	}

	if (is_x) {
		std::vector<double> prod_half((L.sx - 1) * L.sy * L.sz, 0.0);
		std::vector<double> factor_half((L.sx - 1) * L.sy * L.sz, 0.0);
		interp_half_x(prod, prod_half, L);
		interp_half_x(periodic_factor, factor_half, L);
		exchange_half_halo_x(prod_half, L, C, L.ngx, 4010);
		exchange_half_halo_x(factor_half, L, C, L.ngx, 4020);
		apply_half_periodic_shift_x(prod_half, factor_half, shift, L, C);
		interp_half_x_boundary(prod, prod_half, L);

		diff_x_half(prod_half, dphi_dtheta, inv_d, L);
		exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 4030);
		exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 4040);
		exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 4050);
		diff_x_half_boundary(prod_half, dphi_dtheta, inv_d, L);
		// copy_fill_nonperiodic_ghost_layers(dphi_dtheta, L, C, P);
		return true;
	}

	if (is_y) {
		std::vector<double> prod_half(L.sx * (L.sy - 1) * L.sz, 0.0);
		std::vector<double> factor_half(L.sx * (L.sy - 1) * L.sz, 0.0);
		interp_half_y(prod, prod_half, L);
		interp_half_y(periodic_factor, factor_half, L);
		exchange_half_halo_y(prod_half, L, C, L.ngy, 5010);
		exchange_half_halo_y(factor_half, L, C, L.ngy, 5020);
		apply_half_periodic_shift_y(prod_half, factor_half, shift, L, C);
		interp_half_y_boundary(prod, prod_half, L);

		diff_y_half(prod_half, dphi_dtheta, inv_d, L);
		exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 5030);
		exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 5040);
		exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 5050);
		diff_y_half_boundary(prod_half, dphi_dtheta, inv_d, L);
		// copy_fill_nonperiodic_ghost_layers(dphi_dtheta, L, C, P);
		return true;
	}

	std::vector<double> prod_half(L.sx * L.sy * (L.sz - 1), 0.0);
	std::vector<double> factor_half(L.sx * L.sy * (L.sz - 1), 0.0);
	interp_half_z(prod, prod_half, L);
	interp_half_z(periodic_factor, factor_half, L);
	exchange_half_halo_z(prod_half, L, C, L.ngz, 6010);
	exchange_half_halo_z(factor_half, L, C, L.ngz, 6020);
	apply_half_periodic_shift_z(prod_half, factor_half, shift, L, C);
	interp_half_z_boundary(prod, prod_half, L);

	diff_z_half(prod_half, dphi_dtheta, inv_d, L);
	exchange_node_halo_x(dphi_dtheta, L, C, L.ngx, 6030);
	exchange_node_halo_y(dphi_dtheta, L, C, L.ngy, 6040);
	exchange_node_halo_z(dphi_dtheta, L, C, L.ngz, 6050);
	diff_z_half_boundary(prod_half, dphi_dtheta, inv_d, L);
	// copy_fill_nonperiodic_ghost_layers(dphi_dtheta, L, C, P);
	return true;
}

} // namespace


bool compute_metrics_and_jacobian(Field3D &F,
								  const GridDesc &G,
								  const CartDecomp &C,
								  const SolverParams &P)
{
	const LocalDesc &L = F.L;
	const std::size_t tot = static_cast<std::size_t>(L.sx) * static_cast<std::size_t>(L.sy) * static_cast<std::size_t>(L.sz);
	if (F.coord_x.size() != tot || F.coord_y.size() != tot || F.coord_z.size() != tot) {
		std::cerr << "compute_metrics_and_jacobian: coord array size mismatch\n";
		return false;
	}

	const double dxi   = G.dx;
	const double deta  = G.dy;
	const double dzeta = G.dz;

	std::vector<double> x_xi, x_eta, x_zeta;
	std::vector<double> y_xi, y_eta, y_zeta;
	std::vector<double> z_xi, z_eta, z_zeta;

	if (!compute_dphi_dtheta_coord(F.coord_x, 'x', "xi", dxi, C, L, P, x_xi)) return false;
	if (!compute_dphi_dtheta_coord(F.coord_x, 'x', "eta", deta, C, L, P, x_eta)) return false;
	if (!compute_dphi_dtheta_coord(F.coord_x, 'x', "zeta", dzeta, C, L, P, x_zeta)) return false;

	if (!compute_dphi_dtheta_coord(F.coord_y, 'y', "xi", dxi, C, L, P, y_xi)) return false;
	if (!compute_dphi_dtheta_coord(F.coord_y, 'y', "eta", deta, C, L, P, y_eta)) return false;
	if (!compute_dphi_dtheta_coord(F.coord_y, 'y', "zeta", dzeta, C, L, P, y_zeta)) return false;

	if (!compute_dphi_dtheta_coord(F.coord_z, 'z', "xi", dxi, C, L, P, z_xi)) return false;
	if (!compute_dphi_dtheta_coord(F.coord_z, 'z', "eta", deta, C, L, P, z_eta)) return false;
	if (!compute_dphi_dtheta_coord(F.coord_z, 'z', "zeta", dzeta, C, L, P, z_zeta)) return false;
	
	std::vector<double> d_p_yeta_z__zeta, d_p_yzeta_z__eta;
	std::vector<double> d_p_zeta_x__zeta, d_p_zzeta_x__eta;
	std::vector<double> d_p_xeta_y__zeta, d_p_xzeta_y__eta;

	std::vector<double> d_p_yzeta_z__xi, d_p_yxi_z__zeta;
	std::vector<double> d_p_zzeta_x__xi, d_p_zxi_x__zeta;
	std::vector<double> d_p_xzeta_y__xi, d_p_xxi_y__zeta;

	std::vector<double> d_p_yxi_z__eta, d_p_yeta_z__xi;
	std::vector<double> d_p_zxi_x__eta, d_p_zeta_x__xi;
	std::vector<double> d_p_xxi_y__eta, d_p_xeta_y__xi;

	if (!compute_dphi_dtheta_coord_product(y_eta, F.coord_z, 'z', "zeta", dzeta, C, L, P, d_p_yeta_z__zeta)) return false;
	if (!compute_dphi_dtheta_coord_product(y_zeta, F.coord_z, 'z', "eta",  deta,  C, L, P, d_p_yzeta_z__eta)) return false;
	if (!compute_dphi_dtheta_coord_product(z_eta, F.coord_x, 'x', "zeta", dzeta, C, L, P, d_p_zeta_x__zeta)) return false;
	if (!compute_dphi_dtheta_coord_product(z_zeta, F.coord_x, 'x', "eta", deta,  C, L, P, d_p_zzeta_x__eta)) return false;
	if (!compute_dphi_dtheta_coord_product(x_eta, F.coord_y, 'y', "zeta", dzeta, C, L, P, d_p_xeta_y__zeta)) return false;
	if (!compute_dphi_dtheta_coord_product(x_zeta, F.coord_y, 'y', "eta", deta,  C, L, P, d_p_xzeta_y__eta)) return false;

	if (!compute_dphi_dtheta_coord_product(y_zeta, F.coord_z, 'z', "xi",   dxi,   C, L, P, d_p_yzeta_z__xi)) return false;
	if (!compute_dphi_dtheta_coord_product(y_xi,   F.coord_z, 'z', "zeta", dzeta, C, L, P, d_p_yxi_z__zeta)) return false;
	if (!compute_dphi_dtheta_coord_product(z_zeta, F.coord_x, 'x', "xi",   dxi,   C, L, P, d_p_zzeta_x__xi)) return false;
	if (!compute_dphi_dtheta_coord_product(z_xi,   F.coord_x, 'x', "zeta", dzeta, C, L, P, d_p_zxi_x__zeta)) return false;
	if (!compute_dphi_dtheta_coord_product(x_zeta, F.coord_y, 'y', "xi",   dxi,   C, L, P, d_p_xzeta_y__xi)) return false;
	if (!compute_dphi_dtheta_coord_product(x_xi,   F.coord_y, 'y', "zeta", dzeta, C, L, P, d_p_xxi_y__zeta)) return false;

	if (!compute_dphi_dtheta_coord_product(y_xi,  F.coord_z, 'z', "eta", deta, C, L, P, d_p_yxi_z__eta)) return false;
	if (!compute_dphi_dtheta_coord_product(y_eta, F.coord_z, 'z', "xi",  dxi,  C, L, P, d_p_yeta_z__xi)) return false;
	if (!compute_dphi_dtheta_coord_product(z_xi,  F.coord_x, 'x', "eta", deta, C, L, P, d_p_zxi_x__eta)) return false;
	if (!compute_dphi_dtheta_coord_product(z_eta, F.coord_x, 'x', "xi",  dxi,  C, L, P, d_p_zeta_x__xi)) return false;
	if (!compute_dphi_dtheta_coord_product(x_xi,  F.coord_y, 'y', "eta", deta, C, L, P, d_p_xxi_y__eta)) return false;
	if (!compute_dphi_dtheta_coord_product(x_eta, F.coord_y, 'y', "xi",  dxi,  C, L, P, d_p_xeta_y__xi)) return false;

	for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
			for (int i = 0; i < L.sx; ++i) {
				const int id = idx3(i, j, k, L);
				F.xi_x[id]   = d_p_yeta_z__zeta[id] - d_p_yzeta_z__eta[id];
				F.xi_y[id]   = d_p_zeta_x__zeta[id] - d_p_zzeta_x__eta[id];
				F.xi_z[id]   = d_p_xeta_y__zeta[id] - d_p_xzeta_y__eta[id];

				F.eta_x[id]  = d_p_yzeta_z__xi[id]  - d_p_yxi_z__zeta[id];
				F.eta_y[id]  = d_p_zzeta_x__xi[id]  - d_p_zxi_x__zeta[id];
				F.eta_z[id]  = d_p_xzeta_y__xi[id]  - d_p_xxi_y__zeta[id];

				F.zeta_x[id] = d_p_yxi_z__eta[id]   - d_p_yeta_z__xi[id];
				F.zeta_y[id] = d_p_zxi_x__eta[id]   - d_p_zeta_x__xi[id];
				F.zeta_z[id] = d_p_xxi_y__eta[id]   - d_p_xeta_y__xi[id];

				F.Ja[id] = x_xi[id] * (y_eta[id] * z_zeta[id] - y_zeta[id] * z_eta[id])
						+ x_eta[id] * (y_zeta[id]  * z_xi[id] - y_xi[id] * z_zeta[id])
						+ x_zeta[id] * (y_xi[id] * z_eta[id]  - y_eta[id]  * z_xi[id]);
			}
		}
	}

	/*	
	// Debug dump: include ghost-layer values for coordinates, first derivatives and conservative metric terms.
	{
		std::error_code ec;
		std::filesystem::create_directories("output", ec);
		std::ostringstream oss;
		oss << "output/debug_metric_full_rank" << std::setw(4) << std::setfill('0') << C.rank << ".dat";
		std::ofstream ofs(oss.str(), std::ofstream::out);
		if (ofs) {
			ofs << "TITLE=\"Full Metric Debug (with ghost)\"\n";
			ofs << "VARIABLES="
			    << "\"i\" \"j\" \"k\" \"gi\" \"gj\" \"gk\" "
			    << "\"x\" \"y\" \"z\" \"Ja\" "
			    << "\"x_xi\" \"x_eta\" \"x_zeta\" "
			    << "\"y_xi\" \"y_eta\" \"y_zeta\" "
			    << "\"z_xi\" \"z_eta\" \"z_zeta\" "
			    << "\"d_p_yeta_z__zeta\" \"d_p_yzeta_z__eta\" "
			    << "\"d_p_zeta_x__zeta\" \"d_p_zzeta_x__eta\" "
			    << "\"d_p_xeta_y__zeta\" \"d_p_xzeta_y__eta\" "
			    << "\"d_p_yzeta_z__xi\" \"d_p_yxi_z__zeta\" "
			    << "\"d_p_zzeta_x__xi\" \"d_p_zxi_x__zeta\" "
			    << "\"d_p_xzeta_y__xi\" \"d_p_xxi_y__zeta\" "
			    << "\"d_p_yxi_z__eta\" \"d_p_yeta_z__xi\" "
			    << "\"d_p_zxi_x__eta\" \"d_p_zeta_x__xi\" "
			    << "\"d_p_xxi_y__eta\" \"d_p_xeta_y__xi\"\n";
			ofs << "ZONE T=\"rank_" << C.rank << "_with_ghost\", I=" << L.sx
			    << ", J=" << L.sy << ", K=" << L.sz << ", DATAPACKING=POINT\n";
			ofs << std::scientific << std::setprecision(12);

			for (int k = 0; k < L.sz; ++k) {
				for (int j = 0; j < L.sy; ++j) {
					for (int i = 0; i < L.sx; ++i) {
						const int id = idx3(i, j, k, L);
						const int gi = L.ox + (i - L.ngx);
						const int gj = L.oy + (j - L.ngy);
						const int gk = L.oz + (k - L.ngz);
						ofs << i << " " << j << " " << k << " "
						    << gi << " " << gj << " " << gk << " "
						    << F.coord_x[id] << " " << F.coord_y[id] << " " << F.coord_z[id] << " "
							<< F.Ja[id] << " "
						    << x_xi[id] << " " << x_eta[id] << " " << x_zeta[id] << " "
						    << y_xi[id] << " " << y_eta[id] << " " << y_zeta[id] << " "
						    << z_xi[id] << " " << z_eta[id] << " " << z_zeta[id] << " "
						    << d_p_yeta_z__zeta[id] << " " << d_p_yzeta_z__eta[id] << " "
						    << d_p_zeta_x__zeta[id] << " " << d_p_zzeta_x__eta[id] << " "
						    << d_p_xeta_y__zeta[id] << " " << d_p_xzeta_y__eta[id] << " "
						    << d_p_yzeta_z__xi[id] << " " << d_p_yxi_z__zeta[id] << " "
						    << d_p_zzeta_x__xi[id] << " " << d_p_zxi_x__zeta[id] << " "
						    << d_p_xzeta_y__xi[id] << " " << d_p_xxi_y__zeta[id] << " "
						    << d_p_yxi_z__eta[id] << " " << d_p_yeta_z__xi[id] << " "
						    << d_p_zxi_x__eta[id] << " " << d_p_zeta_x__xi[id] << " "
						    << d_p_xxi_y__eta[id] << " " << d_p_xeta_y__xi[id] << "\n";
					}
				}
			}
		}
	}

	// Debug testing

	for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
			for (int i = 0; i < L.sx; ++i) {
				const int id = idx3(i, j, k, L);
				//F.Ja[id] = G.Lx * G.Ly * G.Lz;
				//F.xi_x[id]   = 0.5 * F.Ja[id];
				F.xi_y[id]   = 0.0;
				F.xi_z[id]   = 0.0;
				F.eta_x[id]  = 0.0;
				//F.eta_y[id]  = 1.0 * F.Ja[id];
				F.eta_z[id]  = 0.0;
				F.zeta_x[id] = 0.0;
				F.zeta_y[id] = 0.0;
				//F.zeta_z[id] = 10.0 * F.Ja[id];
			}
		}
	}
	
	write_grid_metric_derivatives_tecplot_rank(F, G, C,
	                                          x_xi, x_eta, x_zeta,
	                                          y_xi, y_eta, y_zeta,
	                                          z_xi, z_eta, z_zeta,
	                                          "grid_metric_derivatives");
	*/
	interp_half_x(F.xi_x, F.xi_x_fx, L);
	interp_half_x(F.xi_y, F.xi_y_fx, L);
	interp_half_x(F.xi_z, F.xi_z_fx, L);
	exchange_half_halo_x(F.xi_x_fx, L, C, L.ngx, 8110);
	exchange_half_halo_x(F.xi_y_fx, L, C, L.ngx, 8120);
	exchange_half_halo_x(F.xi_z_fx, L, C, L.ngx, 8130);
	interp_half_x_boundary(F.xi_x, F.xi_x_fx, L);
	interp_half_x_boundary(F.xi_y, F.xi_y_fx, L);
	interp_half_x_boundary(F.xi_z, F.xi_z_fx, L);

	interp_half_y(F.eta_x, F.eta_x_fy, L);
	interp_half_y(F.eta_y, F.eta_y_fy, L);
	interp_half_y(F.eta_z, F.eta_z_fy, L);
	exchange_half_halo_y(F.eta_x_fy, L, C, L.ngy, 8210);
	exchange_half_halo_y(F.eta_y_fy, L, C, L.ngy, 8220);
	exchange_half_halo_y(F.eta_z_fy, L, C, L.ngy, 8230);
	interp_half_y_boundary(F.eta_x, F.eta_x_fy, L);
	interp_half_y_boundary(F.eta_y, F.eta_y_fy, L);
	interp_half_y_boundary(F.eta_z, F.eta_z_fy, L);

	interp_half_z(F.zeta_x, F.zeta_x_fz, L);
	interp_half_z(F.zeta_y, F.zeta_y_fz, L);
	interp_half_z(F.zeta_z, F.zeta_z_fz, L);
	exchange_half_halo_z(F.zeta_x_fz, L, C, L.ngz, 8310);
	exchange_half_halo_z(F.zeta_y_fz, L, C, L.ngz, 8320);
	exchange_half_halo_z(F.zeta_z_fz, L, C, L.ngz, 8330);
	interp_half_z_boundary(F.zeta_x, F.zeta_x_fz, L);
	interp_half_z_boundary(F.zeta_y, F.zeta_y_fz, L);
	interp_half_z_boundary(F.zeta_z, F.zeta_z_fz, L);

	// Debug output: half-node interpolated metrics
	/*
	{
		std::error_code ec;
		std::filesystem::create_directories("output", ec);
		std::ostringstream oss;
		oss << "output/debug_half_metrics_rank" << std::setw(4) << std::setfill('0') << C.rank << ".dat";
		std::ofstream ofs(oss.str(), std::ofstream::out);
		if (ofs) {
			ofs << "TITLE=\"Half-node Interpolated Metrics\"\n";
			ofs << "VARIABLES=\"dir\" \"i\" \"j\" \"k\" \"metric_x\" \"metric_y\" \"metric_z\"\n";
			ofs << std::scientific << std::setprecision(12);

			// x-face metrics: xi_*_fx, valid i-face range [ngx-1, ngx+nx-1]
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
					for (int i = L.ngx - 1; i <= L.ngx + L.nx - 1; ++i) {
						const int idf = idx_fx(i, j, k, L);
						ofs << 0 << " " << i << " " << j << " " << k << " "
						    << F.xi_x_fx[idf] << " " << F.xi_y_fx[idf] << " " << F.xi_z_fx[idf] << "\n";
					}
				}
			}

			// y-face metrics: eta_*_fy, valid j-face range [ngy-1, ngy+ny-1]
			for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int j = L.ngy - 1; j <= L.ngy + L.ny - 1; ++j) {
						const int idf = idx_fy(i, j, k, L);
						ofs << 1 << " " << i << " " << j << " " << k << " "
						    << F.eta_x_fy[idf] << " " << F.eta_y_fy[idf] << " " << F.eta_z_fy[idf] << "\n";
					}
				}
			}

			// z-face metrics: zeta_*_fz, valid k-face range [ngz-1, ngz+nz-1]
			for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
				for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
					for (int k = L.ngz - 1; k <= L.ngz + L.nz - 1; ++k) {
						const int idf = idx_fz(i, j, k, L);
						ofs << 2 << " " << i << " " << j << " " << k << " "
						    << F.zeta_x_fz[idf] << " " << F.zeta_y_fz[idf] << " " << F.zeta_z_fz[idf] << "\n";
					}
				}
			}
		}
	}*/

	
	// 返回节点处度量系数
	
	for (int k = 0; k < L.sz; ++k) {
		for (int j = 0; j < L.sy; ++j) {
			for (int i = 0; i < L.sx; ++i) {
				const int id = idx3(i, j, k, L);
				F.xi_x[id]   = F.xi_x[id] / F.Ja[id];
				F.xi_y[id]   = F.xi_y[id] / F.Ja[id];
				F.xi_z[id]   = F.xi_z[id] / F.Ja[id];
				F.eta_x[id]  = F.eta_x[id] / F.Ja[id];
				F.eta_y[id]  = F.eta_y[id] / F.Ja[id];
				F.eta_z[id]  = F.eta_z[id] / F.Ja[id];
				F.zeta_x[id] = F.zeta_x[id] / F.Ja[id];
				F.zeta_y[id] = F.zeta_y[id] / F.Ja[id];
				F.zeta_z[id] = F.zeta_z[id] / F.Ja[id];
			}
		}
	}

	return true;
}


