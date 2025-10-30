#!/usr/bin/env python3
# cauchy_integral_puro.py
# Puro Python: sem NumPy, sem libs externas.

import math, time, argparse, random, cmath

PI = math.pi

# ===========================
# 1) Valor analítico via Cauchy
# ===========================
def analytic_cauchy():
    # Polos no semiplano superior: z = e^{iπ/4}, e^{i3π/4}
    i = 1j
    z1 = cmath.exp(i*PI/4.0)
    z2 = cmath.exp(i*3.0*PI/4.0)
    # Resíduo de 1/(z^4+1) em z0 é 1/(4 z0^3)
    res = 1.0/(4.0*(z1**3)) + 1.0/(4.0*(z2**3))
    integral = 2.0*PI*i*res
    # Deve dar π/√2
    return float(integral.real)

# ===========================
# 2) Quadratura truncada pura
#    x = tan t  em t ∈ [-T, T], T = arctan R
# ===========================
def f_real(x: float) -> float:
    return 1.0/(x*x*x*x + 1.0)

def g_t(t: float) -> float:
    x = math.tan(t)
    sec2 = 1.0/(math.cos(t)*math.cos(t))
    return f_real(x)*sec2

def simpson_adapt(f, a, b, eps=1e-10, nmax=30):
    def S(f, a, b):
        c = 0.5*(a+b); h = b-a
        return (h/6.0)*(f(a) + 4.0*f(c) + f(b))
    def rec(f, a, b, eps, whole, depth):
        c = 0.5*(a+b)
        L = S(f, a, c); R = S(f, c, b)
        if depth <= 0:
            return L + R
        if abs(L + R - whole) <= 15*eps:
            return L + R + (L + R - whole)/15.0
        return rec(f, a, c, eps/2.0, L, depth-1) + rec(f, c, b, eps/2.0, R, depth-1)
    return rec(f, a, b, eps, S(f, a, b), nmax)

def quad_trunc(R: float) -> float:
    # Integral simétrica em [-R, R] mapeada para t
    if R <= 0:
        return 0.0
    T = math.atan(R)
    return simpson_adapt(g_t, -T, T, eps=1e-10, nmax=30)

# ===========================
# 3) “Vector” e “Superpose” sem engine
# ===========================
def map_x_to_R(x: float) -> float:
    # x ∈ [0,1)  ->  R = tan(pi x / 2)
    x = max(0.0, min(0.999999, x))
    return math.tan(0.5*PI*x)

def grid_x(N: int):
    return [i/(N-1) for i in range(N)]

def vector_search(N: int):
    xs = grid_x(N)
    I_true = analytic_cauchy()
    t0 = time.perf_counter()
    best = (None, float("inf"), None)  # (R, err, approx)
    for x in xs:
        R = map_x_to_R(x)
        approx = quad_trunc(R)
        err = abs(approx - I_true)
        if err < best[1]:
            best = (R, err, approx)
    t1 = time.perf_counter()
    return {
        "R": best[0],
        "I_num": best[2],
        "err": best[1],
        "time_s": t1 - t0
    }

def superpose_anneal(N: int, L: int, gamma: float, shots: int, p_shape: float, seed: int = 42):
    # Estado sobre grade de x -> R; custo = |I_num(R) - I_true|
    xs = grid_x(N)
    I_true = analytic_cauchy()

    # 1) prepara custos e ranking
    t0 = time.perf_counter()
    costs = []
    for x in xs:
        R = map_x_to_R(x)
        approx = quad_trunc(R)
        costs.append(abs(approx - I_true))
    order = sorted(range(N), key=lambda i: costs[i])
    rank = [0]*N
    for pos, idx in enumerate(order):
        rank[idx] = pos
    # normaliza [0,1]
    rho_n = [rank[i]/float(max(1, N-1)) for i in range(N)]
    t1 = time.perf_counter()

    # 2) amplitudes complexas uniformes
    amp = [complex(1.0/math.sqrt(N), 0.0) for _ in range(N)]

    # 3) pesos para difusão ponderada por qualidade (menor rank => peso maior)
    w = [1.0/(1.0 + v) for v in rho_n]
    s = sum(w); w = [wi/s for wi in w]

    def oracle_phase(g):
        for i in range(N):
            x = rho_n[i]**p_shape
            phi = -g*x
            c = math.cos(phi); s_ = math.sin(phi)
            a = amp[i]
            amp[i] = complex(a.real*c - a.imag*s_, a.real*s_ + a.imag*c)

    def diffusion_weighted():
        den = 0.0; numr = 0.0; numi = 0.0
        for i in range(N):
            den  += w[i]*w[i]
            numr += w[i]*amp[i].real
            numi += w[i]*amp[i].imag
        pr = numr/den; pi = numi/den
        for i in range(N):
            ar, ai = amp[i].real, amp[i].imag
            amp[i] = complex(2.0*w[i]*pr - ar, 2.0*w[i]*pi - ai)

    # 4) iterações com annealing de fase
    t2 = time.perf_counter()
    g0 = gamma/float(max(1, L))
    for it in range(L):
        g_it = g0*(it+1)
        oracle_phase(g_it)
        diffusion_weighted()
    t3 = time.perf_counter()

    # 5) medição
    probs = [(a.real*a.real + a.imag*a.imag) for a in amp]
    ps = sum(probs); probs = [p/ps for p in probs]
    cdf = []
    acc = 0.0
    for p in probs:
        acc += p; cdf.append(acc)

    rng = random.Random(seed)
    def sample_one():
        x = rng.random()
        lo, hi = 0, N-1
        while lo < hi:
            mid = (lo+hi)//2
            if cdf[mid] < x: lo = mid+1
            else: hi = mid
        return lo

    counts = [0]*N
    for _ in range(shots):
        idx = sample_one()
        counts[idx] += 1

    best_idx = max(range(N), key=lambda i: probs[i])
    x_best = xs[best_idx]
    R_best = map_x_to_R(x_best)
    I_best = quad_trunc(R_best)
    err_best = abs(I_best - I_true)

    return {
        "N": N,
        "R": R_best,
        "I_num": I_best,
        "err": err_best,
        "prep_s": t1 - t0,
        "iters_s": t3 - t2,
        "best_x": x_best,
        "top_hits": sorted([(i, counts[i]) for i in range(N)], key=lambda kv: kv[1], reverse=True)[:10]
    }

# ===========================
# CLI
# ===========================
def main():
    ap = argparse.ArgumentParser(description="Integral ∫ dx/(x^4+1) por Cauchy e buscas puras, sem engine")
    ap.add_argument("mode", choices=["classic", "vector", "superpose"], help="método")
    ap.add_argument("--N", type=int, default=4096, help="tamanho da grade para vector/superpose")
    ap.add_argument("--L", type=int, default=32, help="iterações no superpose")
    ap.add_argument("--gamma", type=float, default=6.0, help="fase total no superpose")
    ap.add_argument("--shots", type=int, default=50000, help="amostras de medição no superpose")
    ap.add_argument("--p-shape", type=float, default=0.6, dest="p_shape", help="curvatura da fase vs rank")
    args = ap.parse_args()

    if args.mode == "classic":
        t0 = time.perf_counter()
        val = analytic_cauchy()
        t1 = time.perf_counter()
        print(f"[classic cauchy] I = ∫_(-∞)^(∞) dx/(x^4+1) = {val:.15f} tempo={t1-t0:.6f} s")
    elif args.mode == "vector":
        out = vector_search(args.N)
        print(f"[vector cauchy] melhor R≈{out['R']:.6f} I_num≈{out['I_num']:.15f} err≈{out['err']:.3e} tempo={out['time_s']:.6f} s")
    else:
        out = superpose_anneal(args.N, args.L, args.gamma, args.shots, args.p_shape)
        print(f"[superpose cauchy] N={out['N']} melhor R≈{out['R']:.6f} I_num≈{out['I_num']:.15f} err≈{out['err']:.3e} prep={out['prep_s']:.6f} s iters={out['iters_s']:.6f} s")

if __name__ == "__main__":
    main()
