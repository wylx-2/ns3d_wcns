#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include "field_structures.h"
#include <mpi.h>
#include <iostream>
#include <string>
#include <complex>
#include <fftw3-mpi.h>
#include <iomanip>
#include <fstream>
#include <sstream>

// 读取solver.in文件初始化参数
bool read_solver_params_from_file(const std::string &fname, SolverParams &P, GridDesc &G, CartDecomp &C);

// 均匀场初始条件
void initialize_uniform_field(Field3D &F, const GridDesc &G, const SolverParams &P);

// 正弦波初始条件
void initialize_sine_x_field(Field3D &F, const GridDesc &G, const SolverParams &P);

// 2D Riemann 问题初始条件
void initialize_riemann_2d(Field3D &F, const GridDesc &G, const SolverParams &P);

// sod shock tube 初始条件
void initialize_sod_shock_tube(Field3D &F, const GridDesc &G, const SolverParams &P);

// 2D isentropic vortex 初始条件
void initialize_isentropic_vortex(Field3D &F, const GridDesc &G, const SolverParams &P);

// 球形Riemann问题初始条件
void initialize_spherical_riemann(Field3D &F, const GridDesc &G, const SolverParams &P);

// 平面Poiseuille流初始条件
void initialize_Poiseuille_flow(Field3D &F, const GridDesc &G, const SolverParams &P);

// 三维各向同均匀湍流初始条件
void generate_full_turbulence(int NX, int NY, int NZ,
                              std::vector<double> &u,
                              std::vector<double> &v,
                              std::vector<double> &w);
void init_isotropic_turbulence(Field3D &F, const GridDesc &G, const CartDecomp &C, const SolverParams &P);

// 槽道湍流初始条件
void initialize_channel_flow_turbulent(Field3D &F, const GridDesc &G, const SolverParams &P);

// 从 write_tecplot_field 的 dat 文件恢复初场
bool initialize_from_tecplot(Field3D &F,
                             const GridDesc &G,
                             const CartDecomp &C,
                             const SolverParams &P,
                             const std::string &filename);
// 从 write_tecplot_field 输出的 field.h5 并行读取并恢复初场
bool initialize_from_hdf5(Field3D &F,
                          const GridDesc &G,
                          const CartDecomp &C,
                          const SolverParams &P,
                          const std::string &filename);
// 读取结构网格 HDF5 文件中的尺寸信息（x/y/z shape: [nz, ny, nx]）
bool read_structured_grid_hdf5(const std::string &filename,
                               GridDesc &G,
                               const CartDecomp &C);
// 并行读取结构网格坐标并写入本 rank 的 F.coord_x/y/z（仅物理区）
bool read_structured_grid_hdf5_local(const std::string &filename,
                                     Field3D &F,
                                     const CartDecomp &C,
                                     const SolverParams &P);
// 从 256^3 Tecplot 文件均匀抽样到当前网格并初始化
bool initialize_from_tecplot_downsample(Field3D &F,
                                        const GridDesc &G,
                                        const CartDecomp &C,
                                        const SolverParams &P,
                                        const std::string &filename,
                                        int src_nx = 256,
                                        int src_ny = 256,
                                        int src_nz = 256);

// 边界条件处理函数
void apply_boundary(Field3D &F, GridDesc &G, CartDecomp &C, const SolverParams &P);

// 边界条件具体实现函数
void apply_wall_bc(Field3D &F, GridDesc &G, const LocalDesc &L, const SolverParams &P, int face_id);
void apply_symmetry_bc(Field3D &F, const LocalDesc &L, int face_id);
void apply_outflow_bc(Field3D &F, const LocalDesc &L, int face_id);
void apply_inflow_bc(Field3D &F, const LocalDesc &L, int face_id);

// 半节点通量边界处理
void apply_boundary_halfnode_flux(Field3D &F, const GridDesc &G, CartDecomp &C, const SolverParams &P);
void apply_outflow_bc_halfnode_flux(Field3D &F, const LocalDesc &L, int face_id);

void compute_invis_flux(Field3D &F, const SolverParams &P, const CartDecomp &C);
void compute_invis_flux_boundary(Field3D &F, const SolverParams &P);



void WCNS_Riemann_InviscidFlux(std::vector<double> &Fface,
                             const std::vector<std::vector<double>> &Ut,
                             const std::vector<std::vector<double>> &ut,
                             const SolverParams &P, double nx, double ny, double nz);
void Roe_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma);
void Rusanov_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma);
void HLLC_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma);
void HLLC_p_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma);
void HLL_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma);
void AUSM_Riemann_solver(std::vector<double> &Fface,
                 const std::vector<double> &UL, const std::vector<double> &UR,
                 double nx, double ny, double nz,
                 double gamma);
void compute_invis_dflux(Field3D &F, const SolverParams &P, const GridDesc &G);

// 插值
double interpolate_select(const std::vector<double> &vstencil, double flag, const SolverParams P);
// weno5 插值
double weno5_interpolate(const std::array<double,6> &stencil);
// zero 插值
double zero_interpolate(const std::array<double,2> &stencil);
// mdcd 线性插值
double mdcd_linear_interpolate(const std::array<double, 6> &stencil, SolverParams P);
// mdcd 混合插值        
double mdcd_hybrid_interpolate(const std::array<double, 6> &stencil, SolverParams P);


// 简单的线性重构（标量，2 点模板）
double linear_reconstruction(const std::array<double,2> &stencil);

// MDCD 重构（标量，6 点模板）
double mdcd_reconstruction(const std::array<double,6>& stencil, SolverParams P);

// WENO5 重构（标量，6 点模板）
double weno5_reconstruction(const std::array<double,6> &stencil);

// C6th 六阶中心差分重构（标量，6 点模板）
double c6th_reconstruction(const std::array<double,6> &stencil);

// C4th 四阶中心差分(标量，4 点模板)
double c4th_reconstruction(const std::array<double,4>& stencil);

// Runtime-sized reconstruction selector (accepts std::vector stencil)
double reconstruct_select(const std::vector<double> &stencil, double flag, const SolverParams P);

// RHS 计算占位符函数（用户需在此定义具体的通量差分或高阶算子）
void compute_rhs(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P, HaloRequests &out_reqs);

// 三阶 Runge-Kutta 时间推进
void runge_kutta_3(Field3D &F, CartDecomp &C, GridDesc &G,  SolverParams &P, HaloRequests &out_reqs, double dt);

// timestep calculation function
double compute_timestep(Field3D &F, const GridDesc &G, const SolverParams &P);

// 基于有限差分: 节点通量->重构到半节点->差分
void compute_flux_fd_x(Field3D &F, const GridDesc &G, const SolverParams &P);

// 计算无粘通量
void compute_flux(Field3D &F, const SolverParams &P);

// 计算数值无粘通量
void computeFVSFluxes(Field3D &F, const SolverParams &P);

// 重构无粘通量
void reconstructInviscidFlux(std::vector<double> &Fface,
                             const std::vector<std::vector<double>> &Ft,
                             const std::vector<std::vector<double>> &Ut,
                             const std::vector<std::vector<double>> &ut,
                             const SolverParams &P, int dim);

// 计算 Roe 平均态
void computeRoeAveragedState(double &rho_bar, double &rhou_bar, double &rhov_bar, double &rhow_bar,
                             double &h_bar, double &a_bar,
                             const double Ul[5], const double Ur[5],
                             double gamma);

// 计算特征分解
static void build_eigen_matrices(const double Ul[5], const double Ur[5],
                                 double nx, double ny, double nz,
                                 double gamma,
                                 double Lmat[5][5], double Rmat[5][5],
                                 double lambar[5]);

// 计算粘性通量
void compute_viscous_flux(Field3D &F, const CartDecomp &C, const GridDesc &G, const SolverParams &P);

// 计算粘性通量导数
void compute_vis_flux(Field3D &F, const GridDesc &G);

// Output full field in Tecplot ASCII format (per-rank file). Prefix will be used for filename: <prefix>_rank<id>.dat
// time: physical time to label the output (optional, default 0.0)
void write_tecplot_field(const Field3D &F, const GridDesc &G, const CartDecomp &C, const SolverParams &P, double time = 0.0);

// Write per-rank Tecplot file for local grid indices, physical coordinates,
// node Jacobian and metric coefficients.
void write_grid_metrics_tecplot_rank(const Field3D &F,
                                     const GridDesc &G,
                                     const CartDecomp &C,
                                     const std::string &filename_prefix = "grid_metrics");

// Write per-rank Tecplot file for raw metric derivatives, including x_xi/x_eta/.../z_zeta.
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
                                                const std::string &filename_prefix = "grid_metric_derivatives");

// Write per-rank Tecplot file for z-coordinate values interpolated onto z-half nodes.
void write_halfnode_z_tecplot_rank(const LocalDesc &L,
                                   const CartDecomp &C,
                                   const std::vector<double> &z_half,
                                   const std::string &filename_prefix = "grid_halfnode_z");

// Write per-rank Tecplot files for half-node inviscid fluxes in xi/eta/zeta directions.
void write_halfnode_invis_flux_tecplot_rank(const Field3D &F,
                                            const CartDecomp &C,
                                            char dir,
                                            unsigned long long call_id,
                                            const std::string &filename_prefix = "halfnode_invis_flux");

// Write per-rank Tecplot file for RHS values at nodes.
void write_rhs_tecplot_rank(const Field3D &F,
                            const CartDecomp &C,
                            unsigned long long call_id,
                            const std::string &filename_prefix = "rhs_values");

// Write residuals (per-equation L2 residuals and total energy) vs time step to a Tecplot-like ASCII table.
// The file will contain VARIABLES = "Step" "Res_rho" "Res_rhou" "Res_rhov" "Res_rhow" "Res_E" "Etot"
// If step==0 the file is overwritten and header is written; otherwise the line is appended.
void write_residuals_tecplot(const Field3D &F, int step, const std::string &filename = "residuals.dat");

// 计算能谱
void compute_energy_spectrum(const Field3D &F, const GridDesc &G, const CartDecomp &C,
                             const std::string &filename = "Energy-spectrum.dat");

// compute diagnostics: rms of RHS, total energy, residual
void compute_diagnostics(Field3D &F, const SolverParams &P, const GridDesc &G);
void compute_total_energy(Field3D &F, const GridDesc &G, const CartDecomp &C, const SolverParams &P);
void monitor_mean_u_sections(Field3D &F, const GridDesc &G, const CartDecomp &C, double current_time);

// main time advance loop with monitor & output
void time_advance(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P);

// post-processing for isotropic turbulence
void isotropic_post_process(Field3D &F, const GridDesc &G, const CartDecomp &C, const SolverParams &P, const double current_time);
void compute_turbulence_statistics(Field3D &F, const GridDesc &G, const SolverParams &P, const CartDecomp &C,
                                   const double current_time);
void compute_energy_spectrum_rank0(
        int NX, int NY, int NZ,
        const std::vector<double> &uall,
        const std::vector<double> &vall,
        const std::vector<double> &wall,
        const std::string &filename);
// compute dudx gradients用于湍流统计
void compute_gradients_dudx(Field3D &F, const GridDesc &G);

// 节点场到半节点面的方向插值（与度量系数构造一致）
void interp_half_x(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L);
void interp_half_y(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L);
void interp_half_z(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L);
// 边界近边界的半节点插值
void interp_half_x_boundary(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L);
void interp_half_y_boundary(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L);
void interp_half_z_boundary(const std::vector<double> &node, std::vector<double> &face, const LocalDesc &L);

// 交换半节点通量的 halo（仅物理区半节点）
void exchange_half_halo_x(std::vector<double> &a,const LocalDesc &L,const CartDecomp &C,int layers,int tag_base);
void exchange_half_halo_y(std::vector<double> &a,const LocalDesc &L,const CartDecomp &C,int layers,int tag_base);
void exchange_half_halo_z(std::vector<double> &a,const LocalDesc &L,const CartDecomp &C,int layers,int tag_base);
void exchange_node_halo_x(std::vector<double> &a,const LocalDesc &L,const CartDecomp &C,int layers,int tag_base);
void exchange_node_halo_y(std::vector<double> &a,const LocalDesc &L,const CartDecomp &C,int layers,int tag_base);
void exchange_node_halo_z(std::vector<double> &a,const LocalDesc &L,const CartDecomp &C,int layers,int tag_base);

// 差分
void diff_x_half(const std::vector<double> &flux_fx, std::vector<double> &rhs, double idx, const LocalDesc &L);
void diff_y_half(const std::vector<double> &flux_fy, std::vector<double> &rhs, double idy, const LocalDesc &L);
void diff_z_half(const std::vector<double> &flux_fz, std::vector<double> &rhs, double idz, const LocalDesc &L);
// 边界近边界的节点差分
void diff_x_half_boundary(const std::vector<double> &flux_fx, std::vector<double> &rhs, double idx, const LocalDesc &L);
void diff_y_half_boundary(const std::vector<double> &flux_fy, std::vector<double> &rhs, double idy, const LocalDesc &L);
void diff_z_half_boundary(const std::vector<double> &flux_fz, std::vector<double> &rhs, double idz, const LocalDesc &L);


// Compute directional derivative in computational space (xi/eta/zeta)
// for a scalar array with Field3D node layout.
bool compute_dphi_dtheta(const std::vector<double> &phi,
                         const std::string &theta,
                         double dtheta,
                         const CartDecomp &C,
                         const LocalDesc &L,
                         std::vector<double> &dphi_dtheta);

// Compute node metrics/Jacobian and directional half-node metric coefficients.
bool compute_metrics_and_jacobian(Field3D &F,
                                  const GridDesc &G,
                                  const CartDecomp &C,
                                  const SolverParams &P);