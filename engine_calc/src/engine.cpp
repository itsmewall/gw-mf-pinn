#include "engine.hpp"
#ifdef _OPENMP
  #include <omp.h>
#endif

static double now_s() {
  using clk = std::chrono::high_resolution_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

ClassicResult ClassicSolver::solve(Problem& P) {
  double t0 = now_s();
  ClassicResult out = P.classic_solve();
  double t1 = now_s();
  out.time_s = t1 - t0;
  return out;
}

VectorResult VectorScan::run(Problem& P, size_t max_points) {
  std::vector<std::vector<double>> pts;
  P.sample_space(pts);
  if (max_points && pts.size() > max_points) pts.resize(max_points);

  double t0 = now_s();
  double best_r = std::numeric_limits<double>::infinity();
  std::vector<double> best_x;
  double acc = 0.0;
  double mx  = 0.0;

  #pragma omp parallel
  {
    double acc_local = 0.0;
    double mx_local  = 0.0;
    double best_r_local = std::numeric_limits<double>::infinity();
    std::vector<double> best_x_local;

    #pragma omp for nowait
    for (int i = 0; i < (int)pts.size(); ++i) {
      double r = P.residual(pts[i]);
      acc_local += r;
      if (r > mx_local) mx_local = r;
      if (r < best_r_local) {
        best_r_local = r;
        best_x_local = pts[i];
      }
    }

    #pragma omp critical
    {
      acc += acc_local;
      if (mx_local > mx) mx = mx_local;
      if (best_r_local < best_r) { best_r = best_r_local; best_x = best_x_local; }
    }
  }

  double t1 = now_s();
  VectorResult R;
  R.count = pts.size();
  R.avg_residual = pts.empty() ? 0.0 : acc / double(pts.size());
  R.max_residual = mx;
  R.best_x = best_x;
  R.best_residual = best_r;
  R.time_s = t1 - t0;
  return R;
}

SuperposeResult SuperposeAnneal::run(Problem& P, int L, double gamma, int shots,
                                     double p_shape, bool anneal, uint32_t seed) {
  std::vector<std::vector<double>> xs;
  P.sample_space(xs);
  const int N = (int)xs.size();
  if (N <= 0) throw std::runtime_error("sample_space vazio");

  double t0 = now_s();
  std::vector<double> rho(N);
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) rho[i] = P.residual(xs[i]);

  std::vector<int> order(N);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b){ return rho[a] < rho[b]; });

  std::vector<double> rho_n(N);
  for (int pos = 0; pos < N; ++pos) rho_n[order[pos]] = N > 1 ? double(pos)/double(N-1) : 0.0;
  double t1 = now_s();

  std::vector<double> ar(N, 1.0/std::sqrt((double)N));
  std::vector<double> ai(N, 0.0);

  std::vector<double> w(N);
  double wsum = 0.0;
  for (int i = 0; i < N; ++i) { w[i] = 1.0/(1.0 + rho_n[i]); wsum += w[i]; }
  for (int i = 0; i < N; ++i) w[i] /= wsum;

  auto oracle_phase = [&](double g){
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      double x = std::pow(rho_n[i], p_shape);
      double phi = -g * x;
      double c = std::cos(phi), s = std::sin(phi);
      double r0 = ar[i], im0 = ai[i];
      ar[i] = r0*c - im0*s;
      ai[i] = r0*s + im0*c;
    }
  };

  auto diffusion_simple = [&](){
    double mr = std::accumulate(ar.begin(), ar.end(), 0.0) / N;
    double mi = std::accumulate(ai.begin(), ai.end(), 0.0) / N;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      double r0 = ar[i], im0 = ai[i];
      ar[i] = 2.0*mr - r0;
      ai[i] = 2.0*mi - im0;
    }
  };

  auto diffusion_weighted = [&](){
    double den = 0.0, numr = 0.0, numi = 0.0;
    #pragma omp parallel for reduction(+:den,numr,numi)
    for (int i = 0; i < N; ++i) {
      den  += w[i]*w[i];
      numr += w[i]*ar[i];
      numi += w[i]*ai[i];
    }
    double pr = numr/den, pi = numi/den;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      double r0 = ar[i], im0 = ai[i];
      ar[i] = 2.0*w[i]*pr - r0;
      ai[i] = 2.0*w[i]*pi - im0;
    }
  };

  double t2 = now_s();
  if (anneal) {
    double g0 = gamma / double(std::max(1, L));
    for (int it = 0; it < L; ++it) {
      double g_it = g0 * double(it + 1);
      oracle_phase(g_it);
      diffusion_weighted();
    }
  } else {
    for (int it = 0; it < L; ++it) {
      oracle_phase(gamma);
      diffusion_simple();
    }
  }
  double t3 = now_s();

  std::vector<double> probs(N);
  double ps = 0.0;
  #pragma omp parallel for reduction(+:ps)
  for (int i = 0; i < N; ++i) {
    probs[i] = ar[i]*ar[i] + ai[i]*ai[i];
    ps += probs[i];
  }
  for (int i = 0; i < N; ++i) probs[i] /= ps;

  std::vector<double> cdf(N);
  std::partial_sum(probs.begin(), probs.end(), cdf.begin());
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  std::vector<int> counts(N, 0);
  for (int s = 0; s < shots; ++s) {
    double x = U(rng);
    int idx = int(std::lower_bound(cdf.begin(), cdf.end(), x) - cdf.begin());
    if (idx >= N) idx = N-1;
    counts[idx] += 1;
  }

  int best_idx = int(std::max_element(probs.begin(), probs.end()) - probs.begin());
  double weighted = 0.0;
  for (int i = 0; i < N; ++i) weighted += probs[i]*rho[i];

  std::vector<int> order_hits(N);
  std::iota(order_hits.begin(), order_hits.end(), 0);
  std::partial_sort(order_hits.begin(), order_hits.begin()+std::min(N,10),
                    order_hits.end(),
                    [&](int a, int b){ return counts[a] > counts[b]; });

  SuperposeResult R;
  R.N = N;
  R.prep_s = t1 - t0;
  R.iters_s = t3 - t2;
  R.anneal = anneal;
  R.weighted_cost = weighted;
  R.best_x = xs[best_idx];
  R.best_residual = rho[best_idx];
  for (int k = 0; k < std::min(N,10); ++k) {
    int idx = order_hits[k];
    SuperposeResult::Hit h;
    h.x = xs[idx];
    h.hits = counts[idx];
    h.residual = rho[idx];
    R.top_hits.push_back(std::move(h));
  }
  return R;
}
