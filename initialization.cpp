#include "field_structures.h"
#include "ns3d_func.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

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
    return (lv == "true" || lv == "yes" || lv == "1" || lv == "on");
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
        else if (k=="recon") {
            if (v=="mdcd") P.recon = SolverParams::Reconstruction::MDCD;
            else if (v=="weno5") P.recon = SolverParams::Reconstruction::WENO5;
            else if (v=="linear") P.recon = SolverParams::Reconstruction::LINEAR;
        }
        else if (k=="vis_scheme") {
            if (v=="c4") P.vis_scheme = SolverParams::ViscousScheme::C4th;
            else if (v=="c6") P.vis_scheme = SolverParams::ViscousScheme::C6th;
        }
        else if (k=="char_recon") {
            P.char_recon = (v=="yes" || v=="true");
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
    G.dx = G.Lx / (G.global_nx-1);
    G.dy = G.Ly / (G.global_ny-1);
    G.dz = G.Lz / (G.global_nz-1);

    // 根据重构格式设置ghost层数和stencil大小
    switch (P.recon) {
        case SolverParams::Reconstruction::WENO5:
            P.ghost_layers = 3;
            P.stencil = 6;
            break;
        case SolverParams::Reconstruction::LINEAR:
            P.ghost_layers = 2;
            P.stencil = 2;
            break;
        case SolverParams::Reconstruction::MDCD:
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
            double y = (L.oy + j - L.ngy + 0.5) * G.dy;

            for (int i = 0; i < L.sx; ++i)
            {
                double x = (L.ox + i - L.ngx + 0.5) * G.dx;

                double rho, u, v, w, p;
                w = 0.0;

                // ========= 四象限 Riemann ===============
                if (x >= x_mid && y >= y_mid) {          // 区域 I
                    rho = 1.5;     u = 0.0;     v = 0.0;     p = 1.5;
                }
                else if (x < x_mid && y >= y_mid) {      // 区域 II
                    rho = 0.5323;  u = 1.206;   v = 0.0;     p = 0.3;
                }
                else if (x < x_mid && y < y_mid) {       // 区域 III
                    rho = 0.138;   u = 1.206;   v = 1.206;   p = 0.029;
                }
                else {                                   // 区域 IV
                    rho = 0.5323;  u = 0.0;     v = 1.206;   p = 0.3;
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
        double u = 1.0, v = 1.0, w = 1.0;
        F.rho[id] = rho;
        F.u[id] = std::sin(2.0 * M_PI * ( (L.ox + i - L.ngx + 0.5) * G.dx ) / G.Lx );
        F.v[id] = v;
        F.w[id] = w;
        F.p[id] = p0;
    }
}