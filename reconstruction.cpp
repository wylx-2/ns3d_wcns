#include "ns3d_func.h"
#include <array>
#include <vector>

// wrapper that selects recon method

// Public wrapper that accepts a runtime-sized stencil (std::vector)
// and forwards to the fixed-size implementations when possible.
// This implementation selects the reconstruction based on the
// `SolverParams::Reconstruction` enum and validates the stencil size
// against the expected template. If the stencil size does not match
// the expectation, a runtime error is thrown to catch misuses early.
double reconstruct_select(const std::vector<double> &vstencil, double flag, const SolverParams P)
{
    int n = (int)vstencil.size();
    SolverParams::Reconstruction r = P.recon;

    auto require_size = [&](int expected, const char *name) {
        if (n != expected) {
            throw std::runtime_error(std::string("reconstruct_select: reconstruction '") + name + " requires stencil size " + std::to_string(expected) + ", got " + std::to_string(n));
        }
    };

    // helper to optionally reverse according to sign(flag)
    if (r == SolverParams::Reconstruction::WENO5) {
        require_size(6, "WENO5");
        std::array<double,6> a5;
        for (int i = 0; i < 6; ++i) a5[i] = vstencil[i];
        if (flag < 0.0) std::reverse(a5.begin(), a5.end());
        return weno5_reconstruction(a5);
    }

    if (r == SolverParams::Reconstruction::LINEAR) {
        require_size(2, "LINEAR");
        std::array<double,2> a5;
        for (int i = 0; i < 2; ++i) a5[i] = vstencil[i];
        if (flag < 0.0) std::reverse(a5.begin(), a5.end());
        return linear_reconstruction(a5);
    }

    if (r == SolverParams::Reconstruction::MDCD) {
        // MDCD currently uses a 5-point template; map to linear/WENO5
        require_size(6, "MDCD");
        std::array<double,6> a5;
        for (int i = 0; i < 6; ++i) a5[i] = vstencil[i];
        if (flag < 0.0) std::reverse(a5.begin(), a5.end());
        // For now use the linear reconstructor as a placeholder for MDCD
        return mdcd_reconstruction(a5, P);
    }

    // fallback for unknown enum values or other cases: do a small
    // centered/biased average if the stencil is at least length 2.
    if (n >= 2) {
        int mid = n/2;
        if (flag >= 0.0) return 0.5 * (vstencil[mid] + vstencil[mid-1]);
        else return 0.5 * (vstencil[mid] + vstencil[mid+1]);
    }

    // last resort: return the single value or 0
    return vstencil.empty() ? 0.0 : vstencil[0];
}

// 简单的线性重构（标量，2点模板）
double linear_reconstruction(const std::array<double,2>& stencil) {
    return stencil[0];
}

// WENO5 重构（标量，6点模板）
double weno5_reconstruction(const std::array<double,6>& stencil) {
    double f0 = stencil[0];
    double f1 = stencil[1];
    double f2 = stencil[2];
    double f3 = stencil[3];
    double f4 = stencil[4];

    double eps = 1e-6;
    double alpha0 = 0.0, alpha1 = 0.0, alpha2 = 0.0;

    double beta0 = 13.0/12.0 * (f0 - 2*f1 + f2)*(f0 - 2*f1 + f2)
                 + 0.25 * (f0 - 4*f1 + 3*f2)*(f0 - 4*f1 + 3*f2);
    double beta1 = 13.0/12.0 * (f1 - 2*f2 + f3)*(f1 - 2*f2 + f3)
                 + 0.25 * (f1 - f3)*(f1 - f3);
    double beta2 = 13.0/12.0 * (f2 - 2*f3 + f4)*(f2 - 2*f3 + f4)
                 + 0.25 * (3*f2 - 4*f3 + f4)*(3*f2 - 4*f3 + f4);

    alpha0 = 0.1 / ((eps + beta0)*(eps + beta0));
    alpha1 = 0.6 / ((eps + beta1)*(eps + beta1));
    alpha2 = 0.3 / ((eps + beta2)*(eps + beta2));

    double asum = alpha0 + alpha1 + alpha2;
    double w0 = alpha0 / asum;
    double w1 = alpha1 / asum;
    double w2 = alpha2 / asum;

    double p0 = (2*f0 - 7*f1 + 11*f2) / 6.0;
    double p1 = (-f1 + 5*f2 + 2*f3) / 6.0;
    double p2 = (2*f2 + 5*f3 - f4) / 6.0;

    return w0 * p0 + w1 * p1 + w2 * p2;
}

double mdcd_reconstruction(const std::array<double,6>& stencil, SolverParams P) {
    double diss = P.mdcd_diss;
    double disp = P.mdcd_disp;
    return (1/2 * disp + 1 / 2 * diss) * stencil[0] +
           (-3/2 * disp - 5 / 2 * diss - 1 / 12) * stencil[1] +
           (7/12 + disp + 5 * diss) * stencil[2] +
           (7/12 + disp - 5 * diss ) * stencil[3] +
           (-3/2 * disp + 5 / 2 * diss - 1 / 12) * stencil[4] +
           (1/2 * disp - 1 / 2 * diss) * stencil[5];
}

// C6th 六阶中心差分
double c6th_reconstruction(const std::array<double,6>& stencil) {
    // 六阶中心差分重构公式
    return (1.0/60.0) * (stencil[0] - 8*stencil[1] + 37*stencil[2]
                        + 37*stencil[3] - 8*stencil[4] + stencil[5]);
}

// C4th 四阶中心差分
double c4th_reconstruction(const std::array<double,4>& stencil) {
    // 四阶中心差分重构公式
    return (1.0/12.0) * (-stencil[0] + 7*stencil[1] + 7*stencil[2] - stencil[3]);
} 
