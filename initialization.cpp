#include "field_structures.h"
#include "ns3d_func.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <hdf5.h>
#include <H5FDmpio.h>

// 去掉字符串两端空格
static inline std::string trim(const std::string& s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

// 字符串转小写
static inline std::string lower(const std::string &s)
{
    std::string r=s;
    std::transform(r.begin(), r.end(), r.begin(), ::tolower);
    return r;
}

static inline bool parse_bool(const std::string &v)
{
    std::string lv = lower(v);
    return (lv == "true" || lv == "ture" || lv == "yes" || lv == "1" || lv == "on");
}

bool read_solver_params_from_file(
        const std::string &fname,
        SolverParams &P,
        GridDesc &G,
        CartDecomp &C)
{
    std::ifstream fin(fname);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open solver parameter file: " << fname << "\n";
        return false;
    }

    // ---- 默认值 ----
    P = SolverParams();  // 使用结构体默认构造
    // 网格默认值
    G.global_nx = 16; G.global_ny = 16; G.global_nz = 16;
    G.x0=G.y0=G.z0=0.0;

    std::string line;
    while (std::getline(fin, line))
    {
        line = trim(line);
        if (line.empty() || line[0]=='#') continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0,eq));
        std::string val = trim(line.substr(eq+1));

        std::string k = lower(key);
        std::string v = lower(val);

        // ---- 物理参数 ----
        if (k=="gamma") P.gamma = std::stod(val);
        else if (k=="pr") P.Pr = std::stod(val);
        else if (k=="ma") P.Ma = std::stod(val);
        else if (k=="re") P.Re = std::stod(val);

        // ---- 时间推进 ----
        else if (k=="cfl") P.cfl = std::stod(val);
        else if (k=="dt_fixed") P.dt_fixed = std::stod(val);

        // ---- 重构设置 ----
        else if (k=="fvs_type") {
            if (v=="stegerwarming") P.fvs_type = SolverParams::FVS_Type::StegerWarming;
            else if (v=="vanleer") P.fvs_type = SolverParams::FVS_Type::VanLeer;
            else if (v=="laxfriedrichs") P.fvs_type = SolverParams::FVS_Type::LaxFriedrichs;
        }
        else if (k=="interpolation") {
            if (v=="mdcd_hybrid") P.interpolation = SolverParams::Interpolation::MDCD_HYBRID;
            else if (v=="mdcd_linear") P.interpolation = SolverParams::Interpolation::MDCD_LINEAR;
            else if (v=="weno5") P.interpolation = SolverParams::Interpolation::WENO5;
            else if (v=="zero") P.interpolation = SolverParams::Interpolation::ZERO;
        }
        else if (k=="vis_scheme") {
            if (v=="c4") P.vis_scheme = SolverParams::ViscousScheme::C4th;
            else if (v=="c6") P.vis_scheme = SolverParams::ViscousScheme::C6th;
        }
        else if (k=="char_recon") {
            P.char_recon = (v=="yes" || v=="true");
        }
        else if (k=="riemann_solver") {
            if (v=="roe") P.riemann_solver = SolverParams::RiemannSolver::Roe;
            else if (v=="rusanov") P.riemann_solver = SolverParams::RiemannSolver::Rusanov;
            else if (v=="hllc") P.riemann_solver = SolverParams::RiemannSolver::HLLC;
            else if (v=="hll") P.riemann_solver = SolverParams::RiemannSolver::HLL;
            else if (v=="hllc_p") P.riemann_solver = SolverParams::RiemannSolver::HLLC_p;
            else if (v=="ausm") P.riemann_solver = SolverParams::RiemannSolver::AUSM;
        }
        
        else if (k=="mdcd_diss") P.mdcd_diss = std::stod(val);
        else if (k=="mdcd_disp") P.mdcd_disp = std::stod(val);

        // ---- 网格 ----
        else if (k=="global_nx") G.global_nx = std::stoi(val);
        else if (k=="global_ny") G.global_ny = std::stoi(val);
        else if (k=="global_nz") G.global_nz = std::stoi(val);

        else if (k=="x0") G.x0 = std::stod(val);
        else if (k=="y0") G.y0 = std::stod(val);
        else if (k=="z0") G.z0 = std::stod(val);

        else if (k=="lx") G.Lx = std::stod(val);
        else if (k=="ly") G.Ly = std::stod(val);
        else if (k=="lz") G.Lz = std::stod(val);

        // ---- simulation control (allow several common key names) ----
        else if (k=="max_steps") P.max_steps = std::stoi(val);
        else if (k=="monitor_stepfreq") P.monitor_Stepfreq = std::stoi(val);
        else if (k=="output_timefreq") P.output_Timefreq = std::stod(val);
        else if (k=="totaltime") P.TotalTime = std::stod(val);

        // ---- periodic coordinate shift vectors ----
        else if (k=="periodic_xi_px") P.periodic_xi_px = std::stod(val);
        else if (k=="periodic_eta_px") P.periodic_eta_px = std::stod(val);
        else if (k=="periodic_zeta_px") P.periodic_zeta_px = std::stod(val);
        else if (k=="periodic_xi_py") P.periodic_xi_py = std::stod(val);
        else if (k=="periodic_eta_py") P.periodic_eta_py = std::stod(val);
        else if (k=="periodic_zeta_py") P.periodic_zeta_py = std::stod(val);
        else if (k=="periodic_xi_pz") P.periodic_xi_pz = std::stod(val);
        else if (k=="periodic_eta_pz") P.periodic_eta_pz = std::stod(val);
        else if (k=="periodic_zeta_pz") P.periodic_zeta_pz = std::stod(val);

        // ---- initialization source ----
        else if (k=="restart") P.restart = parse_bool(val);
        else if (k=="restart_file") P.restart_file = val;
        else if (k=="use_grid_file") P.use_grid_file = parse_bool(val);
        else if (k=="grid_file") P.grid_file = val;

        // ---- 边界条件 ----
        auto parse_bc = [&](const std::string &v) {
            if (v=="inflow") return SolverParams::BCType::Inflow;
            if (v=="wall")     return SolverParams::BCType::Wall;
            if (v=="symmetry") return SolverParams::BCType::Symmetry;
            if (v=="outflow")  return SolverParams::BCType::Outflow;
            return SolverParams::BCType::Periodic;
        };

        if (k=="bc_xmin") P.bc_xmin = parse_bc(v);
        else if (k=="bc_xmax") P.bc_xmax = parse_bc(v);
        else if (k=="bc_ymin") P.bc_ymin = parse_bc(v);
        else if (k=="bc_ymax") P.bc_ymax = parse_bc(v);
        else if (k=="bc_zmin") P.bc_zmin = parse_bc(v);
        else if (k=="bc_zmax") P.bc_zmax = parse_bc(v);

        // ---- post-processing / monitor flags ----
        else if (k=="post_basicfield") P.post_basicfield = parse_bool(val);
        else if (k=="isotropicanalyse") P.isotropic_analyse = parse_bool(val);
        else if (k=="monitor_res") P.monitor_res = parse_bool(val);
        else if (k=="monitor_energy") P.monitor_energy = parse_bool(val);

        // ---- body-force source term ----
        else if (k=="use_body_force") P.use_body_force = parse_bool(val);
        else if (k=="body_force_x") P.body_force_x = std::stod(val);
        else if (k=="body_force_y") P.body_force_y = std::stod(val);
        else if (k=="body_force_z") P.body_force_z = std::stod(val);
    }

    fin.close();

    // -------------------------------
    // 后处理：设置周期性标志
    // -------------------------------
    if (P.bc_xmin==SolverParams::BCType::Periodic &&
        P.bc_xmax==SolverParams::BCType::Periodic)
        C.periods[0] = 1;
    if (P.bc_ymin==SolverParams::BCType::Periodic &&
        P.bc_ymax==SolverParams::BCType::Periodic)
        C.periods[1] = 1;
    if (P.bc_zmin==SolverParams::BCType::Periodic &&
        P.bc_zmax==SolverParams::BCType::Periodic)
        C.periods[2] = 1;
    const auto spacing_from_params = [](double Llen, int n, bool periodic) {
        const int denom = periodic ? n : (n - 1);
        return Llen / static_cast<double>(denom);
    };
    G.dx = spacing_from_params(G.Lx, G.global_nx, C.periods[0] != 0);
    G.dy = spacing_from_params(G.Ly, G.global_ny, C.periods[1] != 0);
    G.dz = spacing_from_params(G.Lz, G.global_nz, C.periods[2] != 0);

    // 根据重构格式设置ghost层数和stencil大小
    switch (P.interpolation) {
        case SolverParams::Interpolation::WENO5:
            P.ghost_layers = 3;
            P.stencil = 6;
            break;
        case SolverParams::Interpolation::ZERO:
            P.ghost_layers = 1;
            P.stencil = 2;
            break;
        case SolverParams::Interpolation::MDCD_LINEAR:
            P.ghost_layers = 3;
            P.stencil = 6;
            break;
        case SolverParams::Interpolation::MDCD_HYBRID:
            P.ghost_layers = 3;
            P.stencil = 6;
            break;
    }
    // 物理量
    P.Cv = 1.0/(P.gamma*(P.gamma-1.0)*P.Ma*P.Ma);
    P.Cp = P.Cv*P.gamma;
    P.Rgas = 1.0/(P.Ma*P.Ma*P.gamma);
    P.mu = 1.0 / P.Re;

    return true;
}

// 从 write_tecplot_field 生成的 Tecplot dat 文件读取并恢复物理量
bool initialize_from_tecplot(Field3D &F,
                             const GridDesc &G,
                             const CartDecomp &C,
                             const SolverParams &P,
                             const std::string &filename)
{
    // 假设输入为未分区的全域 Tecplot ASCII（单 ZONE，按 k-j-i 顺序点填充）。
    // 各 rank 逐行读取并仅保留落在本地物理区间的单元，避免额外通信。
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open Tecplot file: " << filename << "\n";
        return false;
    }

    // 跳过头部直到 ZONE 行
    std::string line;
    while (std::getline(fin, line)) {
        if (line.rfind("ZONE", 0) == 0) break;
    }

    const LocalDesc &L = F.L;
    const int nxg = G.global_nx;
    const int nyg = G.global_ny;
    const int nzg = G.global_nz;
    const long long total_cells = static_cast<long long>(nxg) * nyg * nzg;

    long long idx = 0;
    int gi = 0, gj = 0, gk = 0;
    while (idx < total_cells) {
        double x, y, z, rho, u, v, w, E, p, T;
        if (!(fin >> x >> y >> z >> rho >> u >> v >> w >> E >> p >> T)) {
            std::cerr << "Error: Tecplot data lines are fewer than expected when reading " << filename << "\n";
            return false;
        }

        // 将全局索引映射到本地物理区
        if (gi >= L.ox && gi < L.ox + L.nx &&
            gj >= L.oy && gj < L.oy + L.ny &&
            gk >= L.oz && gk < L.oz + L.nz) {
            int li = (gi - L.ox) + L.ngx;
            int lj = (gj - L.oy) + L.ngy;
            int lk = (gk - L.oz) + L.ngz;
            int id = F.I(li, lj, lk);

            F.rho[id]  = rho;
            F.rhou[id] = rho * u;
            F.rhov[id] = rho * v;
            F.rhow[id] = rho * w;
            F.E[id]    = E;

            F.u[id] = u;
            F.v[id] = v;
            F.w[id] = w;
            F.p[id] = p;
            F.T[id] = T;
        }

        // 推进全局计数器（x 最快）
        ++gi; ++idx;
        if (gi == nxg) { gi = 0; ++gj; }
        if (gj == nyg) { gj = 0; ++gk; }
    }

    return true;
}

// 从并行 HDF5 流场文件（write_tecplot_field 生成的 field.h5）恢复初场
bool initialize_from_hdf5(Field3D &F,
                          const GridDesc &G,
                          const CartDecomp &C,
                          const SolverParams &P,
                          const std::string &filename)
{
    (void)P;

#ifdef H5_HAVE_PARALLEL
    const LocalDesc &L = F.L;
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

    std::vector<double> rho(npts), u(npts), v(npts), w(npts), E(npts), p(npts), T(npts);

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

    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        std::cerr << "Failed to open HDF5 file: " << filename << "\n";
        return false;
    }

    hid_t memspace = H5Screate_simple(3, local_dims, nullptr);
    if (memspace < 0) {
        std::cerr << "Failed to create HDF5 memory dataspace\n";
        H5Fclose(file);
        return false;
    }

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    if (xfer < 0) {
        std::cerr << "Failed to create HDF5 transfer property list\n";
        H5Sclose(memspace);
        H5Fclose(file);
        return false;
    }

    ierr = H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    if (ierr < 0) {
        std::cerr << "Failed to set collective MPI-IO transfer mode for read\n";
        H5Pclose(xfer);
        H5Sclose(memspace);
        H5Fclose(file);
        return false;
    }

    auto read_dataset = [&](const char *name, std::vector<double> &data) -> bool {
        hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
        if (dset < 0) {
            std::cerr << "Failed to open dataset '" << name << "' in " << filename << "\n";
            return false;
        }

        hid_t filespace = H5Dget_space(dset);
        if (filespace < 0) {
            H5Dclose(dset);
            std::cerr << "Failed to get filespace for dataset '" << name << "'\n";
            return false;
        }

        herr_t ierr_local = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, local_dims, nullptr);
        if (ierr_local < 0) {
            H5Sclose(filespace);
            H5Dclose(dset);
            std::cerr << "Failed to select hyperslab for dataset '" << name << "'\n";
            return false;
        }

        ierr_local = H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, xfer, data.data());
        H5Sclose(filespace);
        H5Dclose(dset);
        if (ierr_local < 0) {
            std::cerr << "Failed to read dataset '" << name << "'\n";
            return false;
        }
        return true;
    };

    bool ok = true;
    ok = ok && read_dataset("rho", rho);
    ok = ok && read_dataset("u", u);
    ok = ok && read_dataset("v", v);
    ok = ok && read_dataset("w", w);
    ok = ok && read_dataset("E", E);
    ok = ok && read_dataset("p", p);
    ok = ok && read_dataset("T", T);

    H5Pclose(xfer);
    H5Sclose(memspace);
    H5Fclose(file);

    if (!ok) {
        return false;
    }

    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const std::size_t lid = static_cast<std::size_t>(k - L.ngz) * static_cast<std::size_t>(L.ny) * static_cast<std::size_t>(L.nx)
                    + static_cast<std::size_t>(j - L.ngy) * static_cast<std::size_t>(L.nx)
                    + static_cast<std::size_t>(i - L.ngx);
                const int id = F.I(i, j, k);

                F.rho[id] = rho[lid];
                F.rhou[id] = rho[lid] * u[lid];
                F.rhov[id] = rho[lid] * v[lid];
                F.rhow[id] = rho[lid] * w[lid];
                F.E[id] = E[lid];

                F.u[id] = u[lid];
                F.v[id] = v[lid];
                F.w[id] = w[lid];
                F.p[id] = p[lid];
                F.T[id] = T[lid];
            }
        }
    }

    return true;
#else
    if (C.rank == 0) {
        std::cerr << "Parallel HDF5 is required for initialize_from_hdf5, but this build is serial HDF5\n";
    }
    (void)F;
    (void)G;
    (void)filename;
    return false;
#endif
}

bool read_structured_grid_hdf5(const std::string &filename,
                               GridDesc &G,
                               const CartDecomp &C)
{
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "Failed to open structured grid file: " << filename << "\n";
        return false;
    }

    auto read_shape = [&](const char *name, hsize_t dims[3]) -> bool {
        hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
        if (dset < 0) {
            std::cerr << "Missing dataset '" << name << "' in " << filename << "\n";
            return false;
        }

        hid_t dspace = H5Dget_space(dset);
        if (dspace < 0) {
            H5Dclose(dset);
            std::cerr << "Failed to get dataspace for dataset '" << name << "'\n";
            return false;
        }

        const int ndims = H5Sget_simple_extent_ndims(dspace);
        if (ndims != 3) {
            H5Sclose(dspace);
            H5Dclose(dset);
            std::cerr << "Dataset '" << name << "' must be 3D ([nz, ny, nx]), got ndims=" << ndims << "\n";
            return false;
        }

        H5Sget_simple_extent_dims(dspace, dims, nullptr);
        H5Sclose(dspace);
        H5Dclose(dset);
        return true;
    };

    hsize_t dims_x[3] = {0, 0, 0};
    hsize_t dims_y[3] = {0, 0, 0};
    hsize_t dims_z[3] = {0, 0, 0};

    bool ok = true;
    ok = ok && read_shape("x", dims_x);
    ok = ok && read_shape("y", dims_y);
    ok = ok && read_shape("z", dims_z);

    H5Fclose(file);

    if (!ok) {
        return false;
    }

    const bool same_shape = (dims_x[0] == dims_y[0] && dims_x[1] == dims_y[1] && dims_x[2] == dims_y[2] &&
                             dims_x[0] == dims_z[0] && dims_x[1] == dims_z[1] && dims_x[2] == dims_z[2]);
    if (!same_shape) {
        std::cerr << "Dataset shapes of x/y/z are inconsistent in " << filename << "\n";
        return false;
    }

    G.global_nz = static_cast<int>(dims_x[0]);
    G.global_ny = static_cast<int>(dims_x[1]);
    G.global_nx = static_cast<int>(dims_x[2]);

    const auto spacing = [](int n, bool periodic) {
        const int denom = periodic ? n : (n - 1);
        return 1.0 / static_cast<double>(denom);
    };

    G.dx = spacing(G.global_nx, C.periods[0] != 0);
    G.dy = spacing(G.global_ny, C.periods[1] != 0);
    G.dz = spacing(G.global_nz, C.periods[2] != 0);

    return true;
}

static void extrapolate_nonperiodic_ghost_coords(Field3D &F,
                                                  const CartDecomp &C,
                                                  const SolverParams &P)
{
    // 简单镜像坐标
    const LocalDesc &L = F.L;

    auto extrapolate_axis_x = [&]() {
        if (L.nx < 2) return;

        if (L.nbr_xm == MPI_PROC_NULL && P.bc_xmin != SolverParams::BCType::Periodic) {
            for (int k = 0; k < L.sz; ++k) {
                for (int j = 0; j < L.sy; ++j) {
                    for (int layer = 1; layer <= L.ngx; ++layer) {
                        const int id0 = F.I(L.ngx, j, k);
                        const int id1 = F.I(L.ngx + layer, j, k);
                        const int idg = F.I(L.ngx - layer, j, k);
                        F.coord_x[idg] = 2.0 * F.coord_x[id0] -  F.coord_x[id1];
                        F.coord_y[idg] = 2.0 * F.coord_y[id0] -  F.coord_y[id1];
                        F.coord_z[idg] = 2.0 * F.coord_z[id0] -  F.coord_z[id1];
                    }
                }
            }
        }

        if (L.nbr_xp == MPI_PROC_NULL && P.bc_xmax != SolverParams::BCType::Periodic) {
            for (int k = 0; k < L.sz; ++k) {
                for (int j = 0; j < L.sy; ++j) {
                    for (int layer = 1; layer <= L.ngx; ++layer) {
                        const int id0 = F.I(L.ngx + L.nx - 1, j, k);
                        const int id1 = F.I(L.ngx + L.nx - 1 - layer, j, k);
                        const int idg = F.I(L.ngx + L.nx - 1 + layer, j, k);
                        F.coord_x[idg] = 2.0 * F.coord_x[id0] - F.coord_x[id1];
                        F.coord_y[idg] = 2.0 * F.coord_y[id0] - F.coord_y[id1];
                        F.coord_z[idg] = 2.0 * F.coord_z[id0] - F.coord_z[id1];
                    }
                }
            }
        }
    };

    auto extrapolate_axis_y = [&]() {
        if (L.ny < 2) return;

        if (L.nbr_ym == MPI_PROC_NULL && P.bc_ymin != SolverParams::BCType::Periodic) {
            for (int k = 0; k < L.sz; ++k) {
                for (int i = 0; i < L.sx; ++i) {
                    for (int layer = 1; layer <= L.ngy; ++layer) {
                        const int id0 = F.I(i, L.ngy, k);
                        const int id1 = F.I(i, L.ngy + layer, k);
                        const int idg = F.I(i, L.ngy - layer, k);
                        F.coord_x[idg] = 2.0 * F.coord_x[id0] - F.coord_x[id1];
                        F.coord_y[idg] = 2.0 * F.coord_y[id0] - F.coord_y[id1];
                        F.coord_z[idg] = 2.0 * F.coord_z[id0] - F.coord_z[id1];
                    }
                }
            }
        }

        if (L.nbr_yp == MPI_PROC_NULL && P.bc_ymax != SolverParams::BCType::Periodic) {
            for (int k = 0; k < L.sz; ++k) {
                for (int i = 0; i < L.sx; ++i) {
                    for (int layer = 1; layer <= L.ngy; ++layer) {
                        const int id0 = F.I(i, L.ngy + L.ny - 1, k);
                        const int id1 = F.I(i, L.ngy + L.ny - 1 - layer, k);
                        const int idg = F.I(i, L.ngy + L.ny - 1 + layer, k);
                        F.coord_x[idg] = 2.0 * F.coord_x[id0] - F.coord_x[id1];
                        F.coord_y[idg] = 2.0 * F.coord_y[id0] - F.coord_y[id1];
                        F.coord_z[idg] = 2.0 * F.coord_z[id0] - F.coord_z[id1];
                    }
                }
            }
        }
    };

    auto extrapolate_axis_z = [&]() {
        if (L.nz < 2) return;

        if (L.nbr_zm == MPI_PROC_NULL && P.bc_zmin != SolverParams::BCType::Periodic) {
            for (int j = 0; j < L.sy; ++j) {
                for (int i = 0; i < L.sx; ++i) {
                    for (int layer = 1; layer <= L.ngz; ++layer) {
                        const int id0 = F.I(i, j, L.ngz);
                        const int id1 = F.I(i, j, L.ngz + layer);
                        const int idg = F.I(i, j, L.ngz - layer);
                        F.coord_x[idg] = 2.0 * F.coord_x[id0] - F.coord_x[id1];
                        F.coord_y[idg] = 2.0 * F.coord_y[id0] - F.coord_y[id1];
                        F.coord_z[idg] = 2.0 * F.coord_z[id0] - F.coord_z[id1];
                    }
                }
            }
        }

        if (L.nbr_zp == MPI_PROC_NULL && P.bc_zmax != SolverParams::BCType::Periodic) {
            for (int j = 0; j < L.sy; ++j) {
                for (int i = 0; i < L.sx; ++i) {
                    for (int layer = 1; layer <= L.ngz; ++layer) {
                        const int id0 = F.I(i, j, L.ngz + L.nz - 1);
                        const int id1 = F.I(i, j, L.ngz + L.nz - 1 - layer);
                        const int idg = F.I(i, j, L.ngz + L.nz - 1 + layer);
                        F.coord_x[idg] = 2.0 * F.coord_x[id0] - F.coord_x[id1];
                        F.coord_y[idg] = 2.0 * F.coord_y[id0] - F.coord_y[id1];
                        F.coord_z[idg] = 2.0 * F.coord_z[id0] - F.coord_z[id1];
                    }
                }
            }
        }
    };

    extrapolate_axis_x();
    extrapolate_axis_y();
    extrapolate_axis_z();
}

bool read_structured_grid_hdf5_local(const std::string &filename,
                                     Field3D &F,
                                     const CartDecomp &C,
                                     const SolverParams &P)
{
#ifdef H5_HAVE_PARALLEL
    const LocalDesc &L = F.L;
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
    const std::size_t npts = static_cast<std::size_t>(L.nx)
                           * static_cast<std::size_t>(L.ny)
                           * static_cast<std::size_t>(L.nz);

    std::vector<double> x(npts), y(npts), z(npts);

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

    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (file < 0) {
        std::cerr << "Failed to open structured grid file: " << filename << "\n";
        return false;
    }

    hid_t memspace = H5Screate_simple(3, local_dims, nullptr);
    if (memspace < 0) {
        std::cerr << "Failed to create HDF5 memory dataspace for structured grid\n";
        H5Fclose(file);
        return false;
    }

    hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
    if (xfer < 0) {
        std::cerr << "Failed to create HDF5 transfer property list for structured grid\n";
        H5Sclose(memspace);
        H5Fclose(file);
        return false;
    }

    ierr = H5Pset_dxpl_mpio(xfer, H5FD_MPIO_COLLECTIVE);
    if (ierr < 0) {
        std::cerr << "Failed to set collective MPI-IO transfer mode for structured grid\n";
        H5Pclose(xfer);
        H5Sclose(memspace);
        H5Fclose(file);
        return false;
    }

    auto read_local_xyz = [&](const char *name, std::vector<double> &data) -> bool {
        hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
        if (dset < 0) {
            std::cerr << "Failed to open dataset '" << name << "' in " << filename << "\n";
            return false;
        }

        hid_t filespace = H5Dget_space(dset);
        if (filespace < 0) {
            H5Dclose(dset);
            std::cerr << "Failed to get filespace for dataset '" << name << "'\n";
            return false;
        }

        herr_t ierr_local = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, local_dims, nullptr);
        if (ierr_local < 0) {
            H5Sclose(filespace);
            H5Dclose(dset);
            std::cerr << "Failed to select hyperslab for dataset '" << name << "'\n";
            return false;
        }

        ierr_local = H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, xfer, data.data());
        H5Sclose(filespace);
        H5Dclose(dset);

        if (ierr_local < 0) {
            std::cerr << "Failed to read dataset '" << name << "'\n";
            return false;
        }
        return true;
    };

    bool ok = true;
    ok = ok && read_local_xyz("x", x);
    ok = ok && read_local_xyz("y", y);
    ok = ok && read_local_xyz("z", z);

    H5Pclose(xfer);
    H5Sclose(memspace);
    H5Fclose(file);

    if (!ok) {
        return false;
    }

    for (int k = L.ngz; k < L.ngz + L.nz; ++k) {
        for (int j = L.ngy; j < L.ngy + L.ny; ++j) {
            for (int i = L.ngx; i < L.ngx + L.nx; ++i) {
                const std::size_t lid = static_cast<std::size_t>(k - L.ngz) * static_cast<std::size_t>(L.ny) * static_cast<std::size_t>(L.nx)
                                      + static_cast<std::size_t>(j - L.ngy) * static_cast<std::size_t>(L.nx)
                                      + static_cast<std::size_t>(i - L.ngx);
                const int id = F.I(i, j, k);
                F.coord_x[id] = x[lid];
                F.coord_y[id] = y[lid];
                F.coord_z[id] = z[lid];
            }
        }
    }

    // 节点坐标ghost交换
    std::cout << "Exchanging node coordinates for ghost layers...\n";
    exchange_node_halo_x(F.coord_x, L, C, L.ngx, 820);
    exchange_node_halo_x(F.coord_y, L, C, L.ngx, 830);
    exchange_node_halo_x(F.coord_z, L, C, L.ngx, 840);

    exchange_node_halo_y(F.coord_x, L, C, L.ngy, 850);
    exchange_node_halo_y(F.coord_y, L, C, L.ngy, 860);
    exchange_node_halo_y(F.coord_z, L, C, L.ngy, 870);

    exchange_node_halo_z(F.coord_x, L, C, L.ngz, 880);
    exchange_node_halo_z(F.coord_y, L, C, L.ngz, 890);
    exchange_node_halo_z(F.coord_z, L, C, L.ngz, 900);

    // 非周期边界外侧 ghost 坐标采用简单镜像外推，保证坐标连续且不穿越边界
    extrapolate_nonperiodic_ghost_coords(F, C, P);

    // 对跨周期边界的ghost坐标附加平移，保证坐标连续
    if (C.periods[0]) {
        if (C.coords[0] == 0) {
            for (int k = 0; k < L.sz; ++k) {
                for (int j = 0; j < L.sy; ++j) {
                    for (int i = 0; i < L.ngx; ++i) {
                        int id = F.I(i, j, k);
                        F.coord_x[id] -= P.periodic_xi_px;
                        F.coord_y[id] -= P.periodic_xi_py;
                        F.coord_z[id] -= P.periodic_xi_pz;
                    }
                }
            }
        }
        if (C.coords[0] == C.dims[0] - 1) {
            for (int k = 0; k < L.sz; ++k) {
                for (int j = 0; j < L.sy; ++j) {
                    for (int i = L.ngx + L.nx; i < L.sx; ++i) {
                        int id = F.I(i, j, k);
                        F.coord_x[id] += P.periodic_xi_px;
                        F.coord_y[id] += P.periodic_xi_py;
                        F.coord_z[id] += P.periodic_xi_pz;
                    }
                }
            }
        }
    }

    if (C.periods[1]) {
        if (C.coords[1] == 0) {
            for (int k = 0; k < L.sz; ++k) {
                for (int j = 0; j < L.ngy; ++j) {
                    for (int i = 0; i < L.sx; ++i) {
                        int id = F.I(i, j, k);
                        F.coord_x[id] -= P.periodic_eta_px;
                        F.coord_y[id] -= P.periodic_eta_py;
                        F.coord_z[id] -= P.periodic_eta_pz;
                    }
                }
            }
        }
        if (C.coords[1] == C.dims[1] - 1) {
            for (int k = 0; k < L.sz; ++k) {
                for (int j = L.ngy + L.ny; j < L.sy; ++j) {
                    for (int i = 0; i < L.sx; ++i) {
                        int id = F.I(i, j, k);
                        F.coord_x[id] += P.periodic_eta_px;
                        F.coord_y[id] += P.periodic_eta_py;
                        F.coord_z[id] += P.periodic_eta_pz;
                    }
                }
            }
        }
    }

    if (C.periods[2]) {
        if (C.coords[2] == 0) {
            for (int k = 0; k < L.ngz; ++k) {
                for (int j = 0; j < L.sy; ++j) {
                    for (int i = 0; i < L.sx; ++i) {
                        int id = F.I(i, j, k);
                        F.coord_x[id] -= P.periodic_zeta_px;
                        F.coord_y[id] -= P.periodic_zeta_py;
                        F.coord_z[id] -= P.periodic_zeta_pz;
                    }
                }
            }
        }
        if (C.coords[2] == C.dims[2] - 1) {
            for (int k = L.ngz + L.nz; k < L.sz; ++k) {
                for (int j = 0; j < L.sy; ++j) {
                    for (int i = 0; i < L.sx; ++i) {
                        int id = F.I(i, j, k);
                        F.coord_x[id] += P.periodic_zeta_px;
                        F.coord_y[id] += P.periodic_zeta_py;
                        F.coord_z[id] += P.periodic_zeta_pz;
                    }
                }
            }
        }
    }

    return true;
#else
    if (C.rank == 0) {
        std::cerr << "Parallel HDF5 is required for read_structured_grid_hdf5_local, but this build is serial HDF5\n";
    }
    (void)F;
    (void)P;
    (void)filename;
    return false;
#endif
}

// 从 256^3 Tecplot 文件均匀抽样到 NX×NY×NZ 网格
bool initialize_from_tecplot_downsample(Field3D &F,
                                        const GridDesc &G,
                                        const CartDecomp &C,
                                        const SolverParams &P,
                                        const std::string &filename,
                                        int src_nx,
                                        int src_ny,
                                        int src_nz)
{
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open Tecplot file: " << filename << "\n";
        return false;
    }

    // 跳过头部直到 ZONE 行
    std::string line;
    while (std::getline(fin, line)) {
        if (line.rfind("ZONE", 0) == 0) break;
    }

    const LocalDesc &L = F.L;
    const long long total_src = 1LL * src_nx * src_ny * src_nz;

    // 预生成“源索引 -> 本地单元列表”的映射，按最近邻选取抽样点
    auto map_index = [](int g, int tgtN, int srcN) {
        if (tgtN <= 1) return 0;
        double pos = static_cast<double>(g) * static_cast<double>(srcN) /
                     static_cast<double>(tgtN);
        int idx = static_cast<int>(std::llround(pos));
        return std::clamp(idx, 0, srcN - 1);
    };

    std::unordered_map<long long, std::vector<int>> src_to_local;
    src_to_local.reserve(static_cast<size_t>(L.nx * L.ny * L.nz));

    for (int kk = 0; kk < L.nz; ++kk)
    for (int jj = 0; jj < L.ny; ++jj)
    for (int ii = 0; ii < L.nx; ++ii) {
        const int gi = L.ox + ii;
        const int gj = L.oy + jj;
        const int gk = L.oz + kk;

        const int si = map_index(gi, G.global_nx, src_nx);
        const int sj = map_index(gj, G.global_ny, src_ny);
        const int sk = map_index(gk, G.global_nz, src_nz);

        const long long sidx = (static_cast<long long>(sk) * src_ny + sj) * src_nx + si;

        const int id = F.I(ii + L.ngx, jj + L.ngy, kk + L.ngz);
        src_to_local[sidx].push_back(id);
    }

    long long idx = 0;
    size_t filled = 0;
    while (idx < total_src) {
        double x, y, z, rho, u, v, w, E, p, T;
        if (!(fin >> x >> y >> z >> rho >> u >> v >> w >> E >> p >> T)) {
            std::cerr << "Error: Tecplot data lines are fewer than expected when reading "
                      << filename << "\n";
            return false;
        }

        auto it = src_to_local.find(idx);
        if (it != src_to_local.end()) {
            for (int id : it->second) {
                F.rho[id]  = rho;
                F.rhou[id] = rho * u;
                F.rhov[id] = rho * v;
                F.rhow[id] = rho * w;
                F.E[id]    = E;

                F.u[id] = u;
                F.v[id] = v;
                F.w[id] = w;
                F.p[id] = p;
                F.T[id] = T;
            }
            filled += it->second.size();
        }

        ++idx;
    }

    if (filled < static_cast<size_t>(L.nx * L.ny * L.nz)) {
        std::cerr << "Warning: only filled " << filled << " / "
                  << (static_cast<size_t>(L.nx * L.ny * L.nz))
                  << " cells while sampling Tecplot file " << filename << "\n";
    }

    return true;
}

void initialize_riemann_2d(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    const LocalDesc &L = F.L;
    const double gamma = P.gamma;

    const double x_mid = 0.5;
    const double y_mid = 0.5;

    // ----- 遍历整个局部网格，包括所有 z 层 -----
    for (int k = 0; k < L.sz; ++k)
    {
        // z 不参与计算，只做复制
        for (int j = 0; j < L.sy; ++j)
        {

            for (int i = 0; i < L.sx; ++i)
            {
                double x = F.coord_x[F.I(i, j, k)];
                double y = F.coord_y[F.I(i, j, k)];

                double rho, u, v, w, p;
                w = 0.0;

                // ========= 四象限 Riemann ===============
                if (x >= x_mid && y >= y_mid) {          // 区域 I
                    //rho = 1.5;     u = 0.0;     v = 0.0;     p = 1.5;
                    //rho = 1.1;     u = 0.0;     v = 0.0;     p = 1.1;
                    rho = 1.0;     u = 0.75;     v = -0.5;     p = 1.0;
                }
                else if (x < x_mid && y >= y_mid) {      // 区域 II
                    // rho = 0.5323;  u = 1.206;   v = 0.0;     p = 0.3;
                    //rho = 0.5065;     u = 0.8939;     v = 0.0;     p = 0.35;
                    rho = 2.0;     u = 0.75;     v = 0.5;     p = 1.0;
                }
                else if (x < x_mid && y < y_mid) {       // 区域 III
                    //rho = 0.138;   u = 1.206;   v = 1.206;   p = 0.029;
                    //rho = 1.1;     u = 0.8939;     v = 0.8939;     p = 1.1;
                    rho = 1.0;     u = -0.75;     v = 0.5;     p = 1.0;
                }
                else {                                   // 区域 IV
                    //rho = 0.5323;  u = 0.0;     v = 1.206;   p = 0.3;
                    //rho = 0.5065;     u = 0.0;     v = 0.8939;     p = 0.35;
                    rho = 3.0;     u = -0.75;     v = -0.5;     p = 1.0;
                }

                // ========= 写入数据 =========
                int id = F.I(i,j,k);

                F.rho[id] = rho;
                F.u[id]   = u;
                F.v[id]   = v;
                F.w[id]   = w;
                F.p[id]   = p;
            }
        }
    }
}

void initialize_sod_shock_tube(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    // Sod shock tube along x direction
    LocalDesc &L = F.L;
    const double gamma = P.gamma;
    const double x_mid = 0.5 * G.global_nx * G.dx;

    for (int k=L.ngz; k<L.ngz+L.nz; ++k)
    for (int j=L.ngy; j<L.ngy+L.ny; ++j)
    for (int i=L.ngx; i<L.ngx+L.nx; ++i) {
        int id = F.I(i,j,k);
        double x = (L.ox + i - L.ngx + 0.5) * G.dx;
        double rho, u, v, w, p;
        v = 0.0; w = 0.0;
        if (x < x_mid) {
            rho = 1.0;
            u = 0.0;
            p = 1.0;
        } else {
            rho = 0.125;
            u = 0.0;
            p = 0.1;
        }
        F.rho[id] = rho;
        F.u[id] = u;
        F.v[id] = v;
        F.w[id] = w;
        F.p[id] = p;
    }
}

void initialize_uniform_field(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    // init some field (e.g., constant density + small velocity perturbation in interior)
    // initialize with consistent total energy so pressure is positive
    LocalDesc &L = F.L;
    const double p0 = 1.0; // reference pressure
    for (int k=L.ngz; k<L.ngz+L.nz; ++k)
    for (int j=L.ngy; j<L.ngy+L.ny; ++j)
    for (int i=L.ngx; i<L.ngx+L.nx; ++i) {
        int id = F.I(i,j,k);
        double rho = 1.0;
        double u = 1.0, v = 1.0, w = 1.0;
        F.rho[id] = rho;
        F.u[id] = u;
        F.v[id] = v;
        F.w[id] = w;
        F.p[id] = p0;
    }
}

void initialize_sine_x_field(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    // 一维沿x方向的正弦波分布，验证du_dx是否正确
    LocalDesc &L = F.L;
    const double p0 = 1.0; // reference pressure
    for (int k=L.ngz; k<L.ngz+L.nz; ++k)
    for (int j=L.ngy; j<L.ngy+L.ny; ++j)
    for (int i=L.ngx; i<L.ngx+L.nx; ++i) {
        int id = F.I(i,j,k);
        double rho = 1.0;
        double u = 0.0, v = 0.0, w = 0.0;
        F.rho[id] = rho;
        F.u[id] = std::sin(2.0 * M_PI * ( (L.ox + i - L.ngx + 0.5) * G.dx ) / G.Lx );
        F.v[id] = v;
        F.w[id] = w;
        F.p[id] = p0;
    }
}

void initialize_isentropic_vortex(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    LocalDesc &L = F.L;
    const double gamma = P.gamma;
    for (int k=L.ngz; k<L.ngz+L.nz; ++k)
    for (int j=L.ngy; j<L.ngy+L.ny; ++j)
    for (int i=L.ngx; i<L.ngx+L.nx; ++i) {
        int id = F.I(i,j,k);
        double x = F.coord_x[id];
        double y = F.coord_y[id];

        double x0 = 5;
        double y0 = 5;

        double r2 = 1.0 - ((x - x0)*(x - x0) + (y - y0)*(y - y0));
        double beta = 5.0; // vortex strength
        double u = 1.0 - beta / (2 * M_PI) * std::exp(0.5*r2) * (y - y0);
        double v = 1.0 + beta / (2 * M_PI) * std::exp(0.5*r2) * (x - x0);
        double w = 0.0;
        double T = 1.0 - (gamma - 1.0) * beta * beta / (8.0 * M_PI * M_PI * gamma) * std::exp(r2);

        F.rho[id] = std::pow(T, 1.0 / (gamma - 1.0));
        F.u[id] = u;
        F.v[id] = v;
        F.w[id] = w;
        F.p[id] = F.rho[id] * T;
    }
}

void initialize_Poiseuille_flow(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    LocalDesc &L = F.L;
    const double dpdx = P.body_force_x; // pressure gradient driving the flow
    const double mu = P.mu; // dynamic viscosity (arbitrary value for testing)

    for (int k=L.ngz; k<L.ngz+L.nz; ++k)
    for (int j=L.ngy; j<L.ngy+L.ny; ++j)
    for (int i=L.ngx; i<L.ngx+L.nx; ++i) {
        int id = F.I(i,j,k);
        double y = F.coord_y[id];

        double rho = 1.0;
        double u = (dpdx / (2.0 * mu)) * (1.0 - y) * y; // parabolic velocity profile
        double v = 0.0;
        double w = 0.0;
        double p = 1.0; //+ dpdx * (F.coord_x[id] - G.Lx/2); // linear pressure distribution

        F.rho[id] = rho;
        F.u[id] = u;
        F.v[id] = v;
        F.w[id] = w;
        F.p[id] = p;
    }


}

void initialize_spherical_riemann(Field3D &F, const GridDesc &G, const SolverParams &P)
{
    LocalDesc &L = F.L;
    const double gamma = P.gamma;
    const double x_mid = 0.0;
    const double y_mid = 0.0;
    const double z_mid = 0.4;

    for (int k=L.ngz; k<L.ngz+L.nz; ++k)
    for (int j=L.ngy; j<L.ngy+L.ny; ++j)
    for (int i=L.ngx; i<L.ngx+L.nx; ++i) {
        int id = F.I(i,j,k);
        double x = F.coord_x[id];
        double y = F.coord_y[id];
        double z = F.coord_z[id];
        double r2 = (x - x_mid)*(x - x_mid) + (y - y_mid)*(y - y_mid) + (z - z_mid)*(z - z_mid);

        double rho, u, v, w, p;
        if (r2 < 0.04) {
            p = 5.0;
        } else {
            p = 1.0;
        }
        F.rho[id] = 1.0;
        F.u[id] = 0.0;
        F.v[id] = 0.0;
        F.w[id] = 0.0;
        F.p[id] = p;
    }
}