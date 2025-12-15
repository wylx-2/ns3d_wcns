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
double interpolate_select(const std::vector<double> &vstencil, double flag, const SolverParams P)
{
    int n = (int)vstencil.size();
    SolverParams::Interpolation r = P.interpolation;

    auto require_size = [&](int expected, const char *name) {
        if (n != expected) {
            throw std::runtime_error(std::string("interpolate_select: interpolation '") + name + " requires stencil size " + std::to_string(expected) + ", got " + std::to_string(n));
        }
    };

    // helper to optionally reverse according to sign(flag)
    if (r == SolverParams::Interpolation::WENO5) {
        require_size(6, "WENO5");
        std::array<double,6> a5;
        for (int i = 0; i < 6; ++i) a5[i] = vstencil[i];
        if (flag < 0.0) std::reverse(a5.begin(), a5.end());
        return weno5_interpolate(a5);
    }

    if (r == SolverParams::Interpolation::ZERO) {
        require_size(2, "ZERO");
        std::array<double,2> a5;
        for (int i = 0; i < 2; ++i) a5[i] = vstencil[i];
        if (flag < 0.0) std::reverse(a5.begin(), a5.end());
        return zero_interpolate(a5);
    }

    if (r == SolverParams::Interpolation::MDCD) {
        // MDCD currently uses a 5-point template; map to linear/WENO5
        require_size(6, "MDCD");
        std::array<double,6> a5;
        for (int i = 0; i < 6; ++i) a5[i] = vstencil[i];
        if (flag < 0.0) std::reverse(a5.begin(), a5.end());
        // For now use the linear reconstructor as a placeholder for MDCD
        return mdcd_interpolate(a5, P);
    }

    // last resort: return the single value or 0
    return vstencil.empty() ? 0.0 : vstencil[0];
}

// 简单的零阶插值（标量，2点模板）
double zero_interpolate(const std::array<double,2>& stencil) {
    return stencil[0];
}

// WENO5 插值（标量，6点模板）
double weno5_interpolate(const std::array<double,6>& stencil){
    double f0 = stencil[0];
    double f1 = stencil[1];
    double f2 = stencil[2];
    double f3 = stencil[3];
    double f4 = stencil[4];

    double eps = 1e-6;
    double alpha0 = 0.0, alpha1 = 0.0, alpha2 = 0.0;

    double beta0 = (f0 - 2*f1 + f2)*(f0 - 2*f1 + f2)
                 + 0.25 * (f0 - 4*f1 + 3*f2)*(f0 - 4*f1 + 3*f2);
    double beta1 = (f1 - 2*f2 + f3)*(f1 - 2*f2 + f3)
                 + 0.25 * (f1 - f3)*(f1 - f3);
    double beta2 = (f2 - 2*f3 + f4)*(f2 - 2*f3 + f4)
                 + 0.25 * (3*f2 - 4*f3 + f4)*(3*f2 - 4*f3 + f4);

    alpha0 = (1.0/16.0) / ((eps + beta0)*(eps + beta0));
    alpha1 = (5.0/8.0) / ((eps + beta1)*(eps + beta1));
    alpha2 = (5.0/16.0) / ((eps + beta2)*(eps + beta2));

    double asum = alpha0 + alpha1 + alpha2;
    double w0 = alpha0 / asum;
    double w1 = alpha1 / asum;
    double w2 = alpha2 / asum;

    double p0 = (3*f0 - 10*f1 + 15*f2) / 8.0;
    double p1 = (-f1 + 6*f2 + 3*f3) / 8.0;
    double p2 = (3*f2 + 6*f3 - f4) / 8.0;

    return w0 * p0 + w1 * p1 + w2 * p2;
}

// MDCD 插值
double mdcd_interpolate(const std::array<double,6>& stencil, SolverParams P) {
    // MDCD interpolation implementation goes here
    std::cout << "MDCD interpolation not implemented." << std::endl;
    return stencil[2]; // placeholder
}