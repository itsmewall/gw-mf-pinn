// src/main.cpp
#include "engine.hpp"
#include <iostream>
#include <cstdlib>
#include <complex>

// =========================
// Integral: f(x) = sin(x^3) em [a,b]
// =========================
struct Integral1D : public Problem {
  double a, b, eps;
  explicit Integral1D(double A, double B, double E=1e-9) : a(A), b(B), eps(E) {}
  ClassicResult classic_solve() override {
    auto f = [](double x){ return std::sin(x*x*x); };
    double val = num::simpson_adapt(f, a, b, eps, 30);
    ClassicResult R; R.scalar = val; return R;
  }
  double residual(const std::vector<double>& x) override {
    double mid = a + (b-a)*x[0];
    auto f = [](double t){ return std::sin(t*t*t); };
    double coarse = num::simpson_adapt(f, a, b, 1e-6, 8);
    double fine   = num::simpson_adapt(f, a, mid, 1e-6, 8)
                  + num::simpson_adapt(f, mid, b, 1e-6, 8);
    return std::fabs(fine - coarse);
  }
  void sample_space(std::vector<std::vector<double>>& out) override {
    const int N = 4096;
    out.resize(N);
    for (int i = 0; i < N; ++i) out[i] = { double(i)/double(N-1) };
  }
};

// =========================
// ODE: dy/dt = -y + sin(t) em [t0,t1], y(0) = y0
// =========================
struct ODE1D : public Problem {
  double t0, t1, y0;
  int steps;
  ODE1D(double T0, double T1, double Y0, int S) : t0(T0), t1(T1), y0(Y0), steps(S) {}
  ClassicResult classic_solve() override {
    double h = (t1 - t0)/double(steps);
    auto f = [](double t, const std::vector<double>& y){
      return std::vector<double>{ -y[0] + std::sin(t) };
    };
    double t = t0; std::vector<double> y{y0};
    for (int i = 0; i < steps; ++i) { y = num::rk4_step(f, t, y, h); t += h; }
    ClassicResult R; R.y = y; return R;
  }
  double residual(const std::vector<double>& x) override {
    int s = std::max(10, int(steps*(0.1 + 0.9*x[0])));
    double h1 = (t1 - t0)/double(s);
    auto f = [](double t, const std::vector<double>& y){
      return std::vector<double>{ -y[0] + std::sin(t) };
    };
    double t = t0; std::vector<double> y{y0};
    for (int i = 0; i < s; ++i) { y = num::rk4_step(f, t, y, h1); t += h1; }
    std::vector<double> y_coarse = y;

    int s2 = 2*s;
    double h2 = (t1 - t0)/double(s2);
    t = t0; y = {y0};
    for (int i = 0; i < s2; ++i) { y = num::rk4_step(f, t, y, h2); t += h2; }
    double err = std::fabs(y_coarse[0] - y[0]);
    return err;
  }
  void sample_space(std::vector<std::vector<double>>& out) override {
    const int N = 2048;
    out.resize(N);
    for (int i = 0; i < N; ++i) out[i] = { double(i)/double(N-1) };
  }
};

// =========================
// Schwarzschild físico 1D com mapeamento log e RMS local normalizado
// =========================
struct Schwarzschild1D : public Problem {
  double M, rmin, rmax;
  explicit Schwarzschild1D(double M_, double rmin_, double rmax_)
    : M(M_), rmin(rmin_), rmax(rmax_) {}
  ClassicResult classic_solve() override { ClassicResult R; R.scalar = 0.0; return R; }

  inline double map_x_to_r(double x) const {
    const double ratio = rmax / rmin;
    return rmin * std::pow(ratio, x);
  }
  inline double e2Phi(double r) const { return 1.0 - 2.0*M/r; }
  inline double Phi(double r)   const { return 0.5*std::log(e2Phi(r)); }
  inline double Lam(double r)   const { return -0.5*std::log(e2Phi(r)); }
  template <typename F>
  inline double d1(F&& Ffun, double r, double h) const {
    return (Ffun(r + h) - Ffun(r - h)) / (2.0*h);
  }
  template <typename F>
  inline double d2(F&& Ffun, double r, double h) const {
    return (Ffun(r + h) - 2.0*Ffun(r) + Ffun(r - h)) / (h*h);
  }
  double residual(const std::vector<double>& xvec) override {
    const double x = xvec[0];
    const double r = map_x_to_r(x);
    if (r <= 2.0*M) return 1e12;

    const int    K   = 5;
    const double h0  = std::max(1e-6, 5e-4*r);
    double acc = 0.0;
    int    cnt = 0;

    for (int k = -K; k <= K; ++k) {
      const double rk = r + k*h0;
      if (rk <= 2.0*M || rk >= rmax) continue;

      const double hk   = std::max(1e-6, 1e-3*rk);
      const double Phi1 = d1([&](double q){ return Phi(q); }, rk, hk);
      const double Lam1 = d1([&](double q){ return Lam(q); }, rk, hk);
      const double Phi2 = d2([&](double q){ return Phi(q); }, rk, hk);
      const double e2Laminv = std::exp(-2.0*Lam(rk));

      const double Gtt   = (1.0 - e2Laminv)/(rk*rk) + (2.0*Lam1*e2Laminv)/rk;
      const double Grr   = (1.0 - e2Laminv)/(rk*rk) - (2.0*Phi1*e2Laminv)/rk;
      const double Gthth = e2Laminv*(Phi2 + Phi1*Phi1 - Phi1*Lam1 + (Phi1 - Lam1)/rk);

      const double kappa = std::max(1e-18, M/(rk*rk*rk));
      const double resk  = std::sqrt(Gtt*Gtt + Grr*Grr + Gthth*Gthth) / kappa;

      acc += resk*resk;
      cnt += 1;
    }

    if (cnt == 0) return 1e12;
    double rms = std::sqrt(acc / double(cnt));
    const double edge = std::pow((r - rmin)/(rmax - rmin), 8.0);
    return rms * (1.0 + 0.1*edge);
  }
  void sample_space(std::vector<std::vector<double>>& out) override {
    const int N = 8192;
    out.resize(N);
    for (int i = 0; i < N; ++i) out[i] = { double(i)/double(N-1) };
  }
};

// =========================
// Navier–Stokes 2D periódico (Taylor–Green vortex)
// Domínio: [0, 2π] × [0, 2π], BCs periódicas
// ∂_t u + (u·∇)u = -∇p + ν ∇²u,  div u = 0
// IC: u0 = [ sin x cos y, -cos x sin y ], solução u(t)=u0 e^{-2νt}
// =========================
struct TaylorGreen2D : public Problem {
  int Nx, Ny;
  double nu;     // viscosidade
  double T;      // tempo final
  double dt;     // passo
  int jacobi_max; // iterações do solver de Poisson
  double jacobi_tol;

  TaylorGreen2D(int Nx_, int Ny_, double nu_, double T_, double dt_, int jacobi_max_=6000, double jacobi_tol_=1e-8)
    : Nx(Nx_), Ny(Ny_), nu(nu_), T(T_), dt(dt_), jacobi_max(jacobi_max_), jacobi_tol(jacobi_tol_) {}

  inline int id(int i, int j) const {
    i = (i % Nx + Nx) % Nx;
    j = (j % Ny + Ny) % Ny;
    return i + Nx*j;
  }

  inline void init_fields(std::vector<double>& u, std::vector<double>& v) const {
    u.assign(Nx*Ny, 0.0);
    v.assign(Nx*Ny, 0.0);
    for (int j = 0; j < Ny; ++j) {
      double y = 2.0*M_PI * (double)j / (double)Ny;
      for (int i = 0; i < Nx; ++i) {
        double x = 2.0*M_PI * (double)i / (double)Nx;
        u[id(i,j)] =  std::sin(x) * std::cos(y);
        v[id(i,j)] = -std::cos(x) * std::sin(y);
      }
    }
  }

  inline double dx(const std::vector<double>& a, int i, int j, double hx) const {
    return (a[id(i+1,j)] - a[id(i-1,j)]) / (2.0*hx);
  }
  inline double dy(const std::vector<double>& a, int i, int j, double hy) const {
    return (a[id(i,j+1)] - a[id(i,j-1)]) / (2.0*hy);
  }
  inline double lap(const std::vector<double>& a, int i, int j, double hx, double hy) const {
    return (a[id(i+1,j)] - 2.0*a[id(i,j)] + a[id(i-1,j)])/(hx*hx)
         + (a[id(i,j+1)] - 2.0*a[id(i,j)] + a[id(i,j-1)])/(hy*hy);
  }

  void step(std::vector<double>& u, std::vector<double>& v,
            std::vector<double>& p, double hx, double hy) const
  {
    std::vector<double> du(Nx*Ny), dv(Nx*Ny);
    std::vector<double> us(Nx*Ny), vs(Nx*Ny);
    std::vector<double> rhs(Nx*Ny);

    // 1) termos explícitos: advecção + difusão
    #pragma omp parallel for
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        int idx = id(i,j);
        double ux = dx(u,i,j,hx), uy = dy(u,i,j,hy);
        double vx = dx(v,i,j,hx), vy = dy(v,i,j,hy);
        double adv_u = u[idx]*ux + v[idx]*uy;
        double adv_v = u[idx]*vx + v[idx]*vy;
        du[idx] = -adv_u + nu * lap(u,i,j,hx,hy);
        dv[idx] = -adv_v + nu * lap(v,i,j,hx,hy);
      }
    }

    // u* intermediário
    #pragma omp parallel for
    for (int k = 0; k < Nx*Ny; ++k) {
      us[k] = u[k] + dt * du[k];
      vs[k] = v[k] + dt * dv[k];
    }

    // 2) RHS da pressão: (1/dt) div u*
    #pragma omp parallel for
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        rhs[id(i,j)] = (dx(us,i,j,hx) + dy(vs,i,j,hy)) / dt;
      }
    }

    // 3) resolve ∇²p = rhs por Jacobi periódico
    std::vector<double> p_new = p;
    double cx = 1.0/(hx*hx), cy = 1.0/(hy*hy);
    double c0 = 1.0/(2.0*(cx+cy));
    double res = 1.0;
    int it = 0;
    while (it < jacobi_max && res > jacobi_tol) {
      res = 0.0;
      #pragma omp parallel for reduction(+:res)
      for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
          double s = cx*(p[id(i+1,j)] + p[id(i-1,j)]) + cy*(p[id(i,j+1)] + p[id(i,j-1)]) - rhs[id(i,j)];
          double val = c0 * s;
          res += std::fabs(val - p[id(i,j)]);
          p_new[id(i,j)] = val;
        }
      }
      p.swap(p_new);
      res /= (Nx*Ny);
      ++it;
    }

    // 4) correção de velocidade
    #pragma omp parallel for
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        u[id(i,j)] = us[id(i,j)] - dt * dx(p,i,j,hx);
        v[id(i,j)] = vs[id(i,j)] - dt * dy(p,i,j,hy);
      }
    }
  }

  inline void exact(double t, int i, int j, double /*hx*/, double /*hy*/, double& ue, double& ve) const {
    double x = 2.0*M_PI * (double)i / (double)Nx;
    double y = 2.0*M_PI * (double)j / (double)Ny;
    double decay = std::exp(-2.0*nu*t);
    ue =  std::sin(x) * std::cos(y) * decay;
    ve = -std::cos(x) * std::sin(y) * decay;
  }

  void metrics(const std::vector<double>& u, const std::vector<double>& v, double t,
               double hx, double hy, double& l2, double& divavg) const
  {
    double se = 0.0, sd = 0.0;
    #pragma omp parallel for reduction(+:se,sd)
    for (int j = 0; j < Ny; ++j) {
      for (int i = 0; i < Nx; ++i) {
        double ue, ve; exact(t, i, j, hx, hy, ue, ve);
        double du = u[id(i,j)] - ue;
        double dv = v[id(i,j)] - ve;
        se += du*du + dv*dv;
        double div = dx(u,i,j,hx) + dy(v,i,j,hy);
        sd += std::fabs(div);
      }
    }
    l2 = std::sqrt(se / (Nx*Ny));
    divavg = sd / (Nx*Ny);
  }

  // ===== Implementações do contrato =====
  ClassicResult classic_solve() override {
    const double hx = 2.0*M_PI / (double)Nx;
    const double hy = 2.0*M_PI / (double)Ny;

    std::vector<double> u, v, p(Nx*Ny, 0.0);
    init_fields(u, v);

    int steps = (int)std::ceil(T/dt);
    double tcur = 0.0;
    for (int n = 0; n < steps; ++n) {
      step(u, v, p, hx, hy);
      tcur += dt;
    }

    double l2, divavg;
    metrics(u, v, tcur, hx, hy, l2, divavg);

    ClassicResult R;
    R.scalar = l2;         // erro L2 vs solução analítica
    R.y = {divavg};        // divergência média
    return R;
  }

  // residual: escolhe dt via x ∈ (0,1], dt = x * dt_max (CFL)
  double residual(const std::vector<double>& x) override {
    double hx = 2.0*M_PI / (double)Nx;
    double hy = 2.0*M_PI / (double)Ny;
    double umax0 = 1.0; // da IC
    double dt_cfl = 0.25 * std::min(hx, hy) / std::max(1e-6, umax0);
    double dt_vis = 0.25 * 0.5 * std::min(hx*hx, hy*hy) / std::max(1e-12, nu);
    double dt_max = std::min(dt_cfl, dt_vis);

    double xx = std::max(1e-3, std::min(1.0, x[0]));
    double dt_try = xx * dt_max;

    // simula até t=0.5 para avaliar erro vs analítico
    double t_end = std::min(0.5, T);
    int steps = std::max(1, (int)std::ceil(t_end/dt_try));

    std::vector<double> u, v, p(Nx*Ny, 0.0);
    init_fields(u, v);

    double tcur = 0.0;
    for (int n = 0; n < steps; ++n) {
      step(u, v, p, hx, hy);
      tcur += dt_try;
    }
    double l2, divavg;
    metrics(u, v, tcur, hx, hy, l2, divavg);

    return l2 + 0.1*divavg;
  }

  void sample_space(std::vector<std::vector<double>>& out) override {
    const int N = 2048;
    out.resize(N);
    for (int i = 0; i < N; ++i) out[i] = { (double)(i+1)/ (double)N }; // (0,1]
  }
};

// =========================
// Helpers
// =========================
static void print_vec(const std::vector<double>& v) {
  std::cout << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << v[i];
    if (i + 1 < v.size()) std::cout << ", ";
  }
  std::cout << "]";
}

// =========================
// MAIN
// Uso: ./calc_engine <task> <mode> [L] [gamma] [shots] [p_shape]
// task: integral | ode | bh | ns2d
// mode: classic | vector | superpose
// =========================
int main(int argc, char** argv) {
  std::string task = argc > 1 ? std::string(argv[1]) : "ns2d";
  std::string mode = argc > 2 ? std::string(argv[2]) : "classic";
  int    L       = argc > 3 ? std::atoi(argv[3]) : 32;
  double gamma   = argc > 4 ? std::atof(argv[4]) : 6.0;
  int    shots   = argc > 5 ? std::atoi(argv[5]) : 50000;
  double pshape  = argc > 6 ? std::atof(argv[6]) : 0.6;

  if (task == "integral") {
    Integral1D prob(0.0, 10.0, 1e-9);
    if (mode == "classic") {
      auto R = ClassicSolver::solve(prob);
      std::cout << "[classic integral] valor ~= " << R.scalar << " tempo=" << R.time_s << " s\n";
    } else if (mode == "vector") {
      auto R = VectorScan::run(prob);
      std::cout << "[vector integral] count=" << R.count
                << " best_res=" << R.best_residual
                << " tempo=" << R.time_s << " s\n";
    } else {
      auto R = SuperposeAnneal::run(prob, L, gamma, shots, pshape, true);
      std::cout << "[superpose integral] N=" << R.N
                << " weighted_cost=" << R.weighted_cost
                << " iters=" << R.iters_s << " s\n";
    }
  } else if (task == "ode") {
    ODE1D prob(0.0, 20.0, 1.0, 20000);
    if (mode == "classic") {
      auto R = ClassicSolver::solve(prob);
      std::cout << "[classic ode] y(T)~= "; print_vec(R.y);
      std::cout << " tempo=" << R.time_s << " s\n";
    } else if (mode == "vector") {
      auto R = VectorScan::run(prob);
      std::cout << "[vector ode] best_res=" << R.best_residual
                << " tempo=" << R.time_s << " s\n";
    } else {
      auto R = SuperposeAnneal::run(prob, L, gamma, shots, pshape, true);
      std::cout << "[superpose ode] N=" << R.N
                << " weighted_cost=" << R.weighted_cost
                << " prep=" << R.prep_s << " s iters=" << R.iters_s << " s\n";
    }
  } else if (task == "bh") {
    Schwarzschild1D prob(1.0, 2.2, 20.0);
    if (mode == "vector") {
      auto R = VectorScan::run(prob);
      std::cout << "[vector BH] best_x="; print_vec(R.best_x);
      std::cout << " best_res=" << R.best_residual
                << " tempo=" << R.time_s << " s\n";
    } else if (mode == "classic") {
      auto R = ClassicSolver::solve(prob);
      std::cout << "[classic BH] placeholder=" << R.scalar
                << " tempo=" << R.time_s << " s\n";
    } else {
      auto R = SuperposeAnneal::run(prob, L, gamma, shots, pshape, true);
      std::cout << "[superpose BH] N=" << R.N
                << " weighted_cost=" << R.weighted_cost
                << " prep=" << R.prep_s << " s iters=" << R.iters_s << " s\n";
      std::cout << "best_x="; print_vec(R.best_x);
      std::cout << " best_res=" << R.best_residual << "\n";
    }
  } else if (task == "ns2d") {
    // Params padrão: grade 256x256, nu=1e-3, T=1.0, dt=5e-4
    int Nx = 256, Ny = 256;
    double nu = 1e-3, T = 1.0, dt = 5e-4;
    if (argc > 7) { Nx = std::atoi(argv[7]); Ny = Nx; } // opcional: Nx via argv[7]
    if (argc > 8) nu = std::atof(argv[8]);              // argv[8]
    if (argc > 9) T  = std::atof(argv[9]);              // argv[9]
    if (argc > 10) dt = std::atof(argv[10]);            // argv[10]

    TaylorGreen2D prob(Nx, Ny, nu, T, dt, /*jacobi_max*/6000, /*jacobi_tol*/1e-8);

    if (mode == "classic") {
      auto R = ClassicSolver::solve(prob);
      std::cout << "[classic ns2d] L2_err=" << R.scalar
                << " div_avg=" << (R.y.empty()?0.0:R.y[0])
                << " tempo=" << R.time_s << " s\n";
    } else if (mode == "vector") {
      auto V = VectorScan::run(prob);
      std::cout << "[vector ns2d] best_x="; print_vec(V.best_x);
      std::cout << " best_res=" << V.best_residual
                << " tempo=" << V.time_s << " s\n";
    } else {
      auto S = SuperposeAnneal::run(prob, L, gamma, shots, pshape, true);
      std::cout << "[superpose ns2d] N=" << S.N
                << " weighted_cost=" << S.weighted_cost
                << " prep=" << S.prep_s << " s iters=" << S.iters_s << " s\n";
    }
  } else {
    std::cerr << "Tarefa desconhecida. Use: integral | ode | bh | ns2d\n";
    return 1;
  }
  return 0;
}
