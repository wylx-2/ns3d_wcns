#include "ns3d_func.h"
#include "field_structures.h"

// -----------------------------------------------------------------------------
// 主时间推进循环 with monitor & output
// -----------------------------------------------------------------------------
void time_advance(Field3D &F, CartDecomp &C, GridDesc &G, SolverParams &P)
{
    double t_start = MPI_Wtime();
    double t_last = t_start;
    double current_time = 0.0;
    int max_steps = P.max_steps;
    int monitor_Stepfreq = P.monitor_Stepfreq;
    double output_Timefreq = P.output_Timefreq;
    double TotalTime = P.TotalTime;
    HaloRequests out_reqs;

    for (int step = 1; step <= max_steps; ++step)
    {
        double dt = P.dt_fixed;
        // record conserved variables at current step
        F.recordConservedTo0();

        const double eps = 1e-12;
        if (std::abs(TotalTime - current_time) <= eps) {
            break; // already at or beyond target time
        }

        // 计算时间步长
        if (P.dt_fixed < 0.0) dt = compute_timestep(F, G, P);
   
        bool if_output = false;
        if (current_time >= output_Timefreq - eps) {
            // already reached the scheduled output time; advance schedule first
            if_output = true;
            output_Timefreq += P.output_Timefreq;
        }
        if (current_time + dt > TotalTime) {
            dt = TotalTime - current_time; // adjust last step to hit TotalTime
        }
        if (current_time + dt > output_Timefreq) {
            dt = output_Timefreq - current_time; // adjust to hit output time
            output_Timefreq += P.output_Timefreq;
            if_output = true;
        }
        if (dt <= eps) {
            if (C.rank == 0) {
                std::cerr << "dt too small or zero; aborting to avoid stall at Time="
                          << current_time << "\n";
            }
            break;
        }
        current_time += dt;
        
        // 三阶Runge–Kutta推进
        runge_kutta_3(F, C, G, P, out_reqs, dt);
        if(C.rank == 0){
            std::cout << "Completed step " << step << " / " << max_steps << ", Time = " << current_time << " / " << TotalTime << "\n";
        }

        // 诊断与监控
        if (step % monitor_Stepfreq == 0 || step == 1) {
            if (P.monitor_res) {
                compute_diagnostics(F, P, G);
            }
            if (P.monitor_energy) {
                compute_total_energy(F, G, C, P);
            }

            double t_now = MPI_Wtime();
            double t_elapsed = t_now - t_start;
            double t_step = t_now - t_last;
            t_last = t_now;

            if (C.rank == 0) {
                std::cout << std::fixed << std::setprecision(6) << "[Step " << step << "] ";
                std::cout << "dt=" << dt << "  Time/step=" << t_step << "s  Elapsed=" << t_elapsed << "s";
                std::cout << "  Time=" << current_time << "/" << TotalTime;
                if (P.monitor_energy) {
                    std::cout << "  Ek_avg=" << F.global_Etot;
                }
                std::cout << "\n";
                if (P.monitor_res) {
                    std::stringstream ss;
                    ss << "output_residuals.dat";
                    write_residuals_tecplot(F, step, ss.str());
                }
            }
                if (P.isotropic_analyse) {
                    isotropic_post_process(F, G, C, P, current_time);
                }
        }

        // 输出流场文件
        if (if_output) {
            if(P.post_basicfield) {
                write_tecplot_field(F, G, C, P, current_time);
            }
            if (P.isotropic_analyse) {
                isotropic_post_process(F, G, C, P, current_time);
            }
        }

        if (current_time >= TotalTime) {
            break; // reached total simulation time
        }
    }

    if (C.rank == 0) {
        double total_time = MPI_Wtime() - t_start;
        std::cout << "Simulation complete. Total time: " << total_time << " s\n";
    }
}