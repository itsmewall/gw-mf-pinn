#pragma once
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>
#include <random>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>

struct ClassicResult {
  const char* mode = "classic";
  double time_s = 0.0;
  std::vector<double> y;
  double scalar = 0.0;
};

struct VectorResult {
  const char* mode = "vector";
  size_t count = 0;
  double avg_residual = 0.0;
  double max_residual = 0.0;
  std::vector<double> best_x;
  double best_residual = 0.0;
  double time_s = 0.0;
};

struct SuperposeResult {
  const char* mode = "superpose";
  size_t N = 0;
  double prep_s = 0.0;
  double iters_s = 0.0;
  bool anneal = true;
  double weighted_cost = 0.0;
  std::vector<double> best_x;
  double best_residual = 0.0;
  struct Hit { std::vector<double> x; int hits; double residual; };
  std::vector<Hit> top_hits;
};

struct Problem {
  virtual ~Problem() = default;
  virtual ClassicResult classic_solve() = 0;
  virtual double residual(const std::vector<double>& x) = 0;
  virtual void sample_space(std::vector<std::vector<double>>& out) = 0;
};

namespace num {
  inline double simpson_adapt(const std::function<double(double)>& f,
                              double a, double b, double eps = 1e-9, int nmax = 20) {
    auto S = [&](double aa, double bb) {
      double c = 0.5*(aa+bb);
      double h = bb-aa;
      return (h/6.0)*(f(aa) + 4.0*f(c) + f(bb));
    };
    std::function<double(double,double,double,double,int)> rec =
      [&](double aa, double bb, double eps_, double whole, int depth)->double {
        double c = 0.5*(aa+bb);
        double L = S(aa, c), R = S(c, bb);
        if (depth <= 0) return L + R;
        if (std::fabs(L + R - whole) <= 15*eps_) return L + R + (L + R - whole)/15.0;
        return rec(aa, c, eps_/2.0, L, depth-1) + rec(c, bb, eps_/2.0, R, depth-1);
      };
    return rec(a, b, eps, S(a, b), nmax);
  }

  inline std::vector<double> rk4_step(
      const std::function<std::vector<double>(double, const std::vector<double>&)>& f,
      double t, const std::vector<double>& y, double h) {
    size_t n = y.size();
    std::vector<double> k1 = f(t, y);
    std::vector<double> tmp(n);

    for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + 0.5*h*k1[i];
    std::vector<double> k2 = f(t + 0.5*h, tmp);

    for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + 0.5*h*k2[i];
    std::vector<double> k3 = f(t + 0.5*h, tmp);

    for (size_t i = 0; i < n; ++i) tmp[i] = y[i] + h*k3[i];
    std::vector<double> k4 = f(t + h, tmp);

    std::vector<double> yn(n);
    for (size_t i = 0; i < n; ++i)
      yn[i] = y[i] + (h/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
    return yn;
  }
}

struct ClassicSolver { static ClassicResult solve(Problem& P); };
struct VectorScan   { static VectorResult run(Problem& P, size_t max_points = 0); };

struct SuperposeAnneal {
  static SuperposeResult run(Problem& P, int L = 24, double gamma = 6.0,
                             int shots = 50000, double p_shape = 0.6, bool anneal = true,
                             uint32_t seed = 42);
};
