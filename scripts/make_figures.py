"""
Figures for the seminar on token budgets for fleets of AI agents.

Model: N Engset agents (activation rate lam, deactivation rate nu = mu*c while tokens
remain), shared quota x in [0,B] refilled at rate p, net drift p - n c.
Throttle: on {x=0, n c > p} the fleet is held to p, so the aggregate completion rate is
mu*p = nu*(p/c) instead of n*nu.

Outputs (pgfplots-ready .tex files in ../figuras):
  sample_path.tex      n(t) and x(t) for one run
  dominant_rate.tex    exact dominant eigenvalue vs its square-root approximation
  throttle_scaling.tex P(throttled) vs B / (c sqrt(N rho (1-rho))) for several N
  decomposition.tex    throttled vs unthrottled: conditional law given x > 0

Run:  python3 make_figures.py
"""
import os
import numpy as np
from numba import njit

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figuras")


# ---------------------------------------------------------------------------
# Event-driven simulation of the PDMP
# ---------------------------------------------------------------------------
@njit(cache=True)
def simulate(N, lam, nu, c, p, B, T, throttle, nbins, seed, record):
    np.random.seed(seed)
    gam = lam + nu
    n = int(round(N * lam / gam))
    x = B / 2.0
    t = 0.0
    t_zero = 0.0      # time throttled (x=0, drift<0)
    t_full = 0.0      # time at x=B
    t_int = 0.0       # time in the interior
    s_n_int = 0.0     # integral of n over interior time
    hx = np.zeros(nbins)            # interior occupation of x
    hn = np.zeros(N + 1)            # interior occupation of n
    maxrec = 200000 if record else 1
    rt = np.zeros(maxrec)
    rn = np.zeros(maxrec)
    rx = np.zeros(maxrec)
    k = 0
    if record:
        rt[0] = t; rn[0] = n; rx[0] = x; k = 1
    eps = 1e-12
    while t < T:
        d = p - n * c
        at_zero = (x <= eps) and (d < 0)
        at_full = (x >= B - eps) and (d > 0)
        if at_zero:
            down = nu * p / c if throttle else n * nu
        else:
            down = n * nu
        up = (N - n) * lam
        R = up + down
        tau = np.random.exponential(1.0 / R)
        if at_zero or at_full:
            hit = np.inf
        elif d > 0:
            hit = (B - x) / d
        else:
            hit = x / (-d)
        if hit < tau:
            # move to the boundary, no jump yet
            dt = hit
            # interior occupation, piecewise linear in x
            t_int += dt
            s_n_int += n * dt
            hn[n] += dt
            m = 8
            for j in range(m):
                xm = x + d * dt * (j + 0.5) / m
                b = int(xm / B * nbins)
                if b >= nbins:
                    b = nbins - 1
                if b < 0:
                    b = 0
                hx[b] += dt / m
            x = B if d > 0 else 0.0
            t += dt
            if record and k < maxrec:
                rt[k] = t; rn[k] = n; rx[k] = x; k += 1
            continue
        dt = tau
        if at_zero:
            t_zero += dt
        elif at_full:
            t_full += dt
        else:
            t_int += dt
            s_n_int += n * dt
            hn[n] += dt
            m = 8
            for j in range(m):
                xm = x + d * dt * (j + 0.5) / m
                b = int(xm / B * nbins)
                if b >= nbins:
                    b = nbins - 1
                if b < 0:
                    b = 0
                hx[b] += dt / m
            x = x + d * dt
            if x < 0.0:
                x = 0.0
            if x > B:
                x = B
        t += dt
        if record and k < maxrec:
            rt[k] = t; rn[k] = n; rx[k] = x; k += 1
        if np.random.random() * R < up:
            n += 1
        else:
            n -= 1
        if record and k < maxrec:
            rt[k] = t; rn[k] = n; rx[k] = x; k += 1
    return t_zero, t_full, t_int, s_n_int, hx, hn, rt[:k], rn[:k], rx[:k]


def sigma(N, rho):
    return np.sqrt(N * rho * (1 - rho))


def p_of_beta(N, lam, nu, c, beta):
    rho = lam / (lam + nu)
    p = c * N * rho + beta * c * sigma(N, rho)
    if abs(p / c - round(p / c)) < 1e-6:     # keep p/c off the integers
        p += 1e-3 * c
    return p


def dominant_rate(N, lam, nu, c, p):
    gam = lam + nu
    rho = lam / gam
    return N * gam * (p - c * N * rho) / (p * (N * c - p))


# ---------------------------------------------------------------------------
# Figure 1: sample path
# ---------------------------------------------------------------------------
def fig_sample_path():
    # parameters chosen so that the quota spends most of its time in the interior
    # (long run: ~9% empty, ~15% full, ~76% interior)
    N, lam, nu, c = 20, 1.0, 3.0, 1.0
    p, B = 5.2, 4.0
    Tmax = 30.0
    *_, rt, rn, rx = simulate(N, lam, nu, c, p, B, Tmax, True, 10, 16, True)
    sel = rt <= Tmax
    rt, rn, rx = rt[sel], rn[sel], rx[sel]
    ncoords = " ".join(f"({a:.4f},{b:.0f})" for a, b in zip(rt, rn))
    xcoords = " ".join(f"({a:.4f},{b:.4f})" for a, b in zip(rt, rx))
    nstar = int(np.floor(p / c))
    body = rf"""%% Sample path: N={N}, lambda={lam}, nu={nu}, c={c}, p={p:.3f}, B={B}
\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=1 by 2, vertical sep=0.25cm, xlabels at=edge bottom, xticklabels at=edge bottom}},
  width=0.95\textwidth, height=3.3cm, xmin=0, xmax={Tmax:g}, grid=major, grid style={{gray!25}},
  tick label style={{font=\footnotesize}}, label style={{font=\footnotesize}}]
\nextgroupplot[ylabel={{$n(t)$}}, ymin=0, ymax=12, ytick={{0,{nstar},10}}, yticklabels={{0,$n^*$,10}}]
\addplot[const plot, azulcito, thick] coordinates {{{ncoords}}};
\addplot[dashed, rojito, domain=0:{Tmax:g}] {{{p/c:.4f}}};
\nextgroupplot[ylabel={{$x(t)$}}, xlabel={{time}}, ymin=-0.3, ymax={B+0.5:g}, ytick={{0,{B:g}}}, yticklabels={{0,$B$}}]
\addplot[verdecito!80!black, thick] coordinates {{{xcoords}}};
\end{{groupplot}}
\end{{tikzpicture}}
"""
    open(os.path.join(OUT, "sample_path.tex"), "w").write(body)
    print("sample path: p =", p, "n* =", nstar, "points", len(rt))


# ---------------------------------------------------------------------------
# Figure 2: dominant rate, exact vs square-root approximation
# ---------------------------------------------------------------------------
def fig_dominant_rate():
    lam, nu, c = 1.0, 3.0, 1.0
    gam = lam + nu
    rho = lam / gam
    Ns = np.unique(np.round(np.logspace(1, 4, 40)).astype(int))
    plots = []
    for beta, col in [(0.5, "verdecito"), (1.0, "azulcito"), (2.0, "rojito")]:
        pts = []
        for N in Ns:
            p = c * N * rho + beta * c * sigma(N, rho)
            if p >= N * c:
                continue
            r = dominant_rate(N, lam, nu, c, p)
            approx = gam * beta / (c * sigma(N, rho))
            pts.append(f"({N},{r/approx:.6f})")
        plots.append(rf"\addplot[{col}, very thick] coordinates {{{' '.join(pts)}}};" + "\n"
                     + rf"\addlegendentry{{$\beta={beta:g}$}}")
    body = rf"""%% exact dominant rate / square-root approximation, lambda={lam}, nu={nu}, c={c}
\begin{{tikzpicture}}
\begin{{semilogxaxis}}[width=\textwidth, height=0.62\textwidth, xmin=10, xmax=10000,
  ymin=0, ymax=1.1, grid=major, grid style={{gray!25}},
  xlabel={{fleet size $N$}}, ylabel={{$\theta^*/\theta^*_{{\sqrt N}}$}}, legend pos=south east,
  tick label style={{font=\footnotesize}}, label style={{font=\footnotesize}}]
{chr(10).join(plots)}
\addplot[dashed, gray, domain=10:10000] {{1}};
\end{{semilogxaxis}}
\end{{tikzpicture}}
"""
    open(os.path.join(OUT, "dominant_rate.tex"), "w").write(body)
    print("dominant rate figure done")


# ---------------------------------------------------------------------------
# Figure 3: throttling probability vs scaled bucket depth
# ---------------------------------------------------------------------------
def throttle_prob_numeric(N, lam, nu, c, p, B, h):
    """Stationary P(throttled) from the balance equations, with x discretised on a grid
    of step h (upwind: x moves one step at rate |d_n|/h). Throttled completions at mu*p."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    K = int(round(B / h))
    h = B / K
    nstar = int(np.floor(p / c))
    S = (N + 1) * (K + 1)
    idx = lambda n, k: n * (K + 1) + k
    rows, cols, vals = [], [], []
    for n in range(N + 1):
        d = p - n * c
        for k in range(K + 1):
            i = idx(n, k)
            if n < N:
                rows.append(i); cols.append(idx(n + 1, k)); vals.append((N - n) * lam)
            if n > 0:
                rate = nu * p / c if (k == 0 and n > nstar) else n * nu
                rows.append(i); cols.append(idx(n - 1, k)); vals.append(rate)
            if d > 0 and k < K:
                rows.append(i); cols.append(idx(n, k + 1)); vals.append(d / h)
            if d < 0 and k > 0:
                rows.append(i); cols.append(idx(n, k - 1)); vals.append(-d / h)
    Q = sp.csr_matrix((vals, (rows, cols)), shape=(S, S))
    Q = Q - sp.diags(np.asarray(Q.sum(axis=1)).ravel())
    A = Q.T.tolil()
    A[0, :] = np.ones(S)
    rhs = np.zeros(S); rhs[0] = 1.0
    pi = spla.spsolve(A.tocsc(), rhs)
    return sum(pi[idx(n, 0)] for n in range(nstar + 1, N + 1))


def fig_throttle_scaling():
    lam, nu, c, beta = 1.0, 3.0, 1.0, 1.0
    rho = lam / (lam + nu)
    bs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    plots = []
    rows = []
    for N, col, mark in [(10, "verdecito", "square*"), (40, "azulcito", "*"),
                         (160, "rojito", "triangle*"), (640, "violetita", "diamond*")]:
        p = p_of_beta(N, lam, nu, c, beta)
        s = c * sigma(N, rho)
        pts = []
        for b in bs:
            B = b * s
            # first-order grid: Richardson extrapolation from steps s/100 and s/200
            p1 = throttle_prob_numeric(N, lam, nu, c, p, B, s / 100)
            p2 = throttle_prob_numeric(N, lam, nu, c, p, B, s / 200)
            pz = 2 * p2 - p1
            pts.append(f"({b},{pz:.6g})")
            rows.append((N, b, pz))
        plots.append(rf"\addplot[{col}, thick, mark={mark}] coordinates {{{' '.join(pts)}}};" + "\n"
                     + rf"\addlegendentry{{$N={N}$}}")
    # check the grid solver against simulation at one point
    N = 40
    p = p_of_beta(N, lam, nu, c, beta)
    B = 1.0 * c * sigma(N, rho)
    tz, tf, ti, *_ = simulate(N, lam, nu, c, p, B, 2e5, True, 10, 99, False)
    s = c * sigma(N, rho)
    ext = 2 * throttle_prob_numeric(N, lam, nu, c, p, B, s / 200) - throttle_prob_numeric(N, lam, nu, c, p, B, s / 100)
    print("check N=40 b=1: grid (extrapolated)", ext, "sim", tz / (tz + tf + ti))
    # asymptotic slope: exp(-gamma*beta * b) in scaled units
    gam = lam + nu
    body = rf"""%% P(throttled) vs B/(c sqrt(N rho(1-rho))), beta={beta}, lambda={lam}, nu={nu}, c={c} (balance equations on a grid, extrapolated)
\begin{{tikzpicture}}
\begin{{semilogyaxis}}[width=\textwidth, height=0.62\textwidth, xmin=0, xmax=3.7,
  grid=major, grid style={{gray!25}},
  xlabel={{scaled depth $B/(c\sigma_N)$}}, ylabel={{$P(\text{{throttled}})$}}, legend style={{at={{(0.5,1.03)}}, anchor=south, legend columns=4, font=\scriptsize}},
  tick label style={{font=\footnotesize}}, label style={{font=\footnotesize}}]
{chr(10).join(plots)}
\end{{semilogyaxis}}
\end{{tikzpicture}}
"""
    open(os.path.join(OUT, "throttle_scaling.tex"), "w").write(body)
    for r in rows:
        print("throttle", r)


# ---------------------------------------------------------------------------
# Figure 4: decomposition test (throttled vs unthrottled)
# ---------------------------------------------------------------------------
def fmt(v):
    return f"{v:.3f}" if v >= 1e-3 else f"${v/10**np.floor(np.log10(v)):.1f}\\cdot 10^{{{int(np.floor(np.log10(v)))}}}$"


def fig_decomposition():
    N, lam, nu, c = 50, 1.0, 3.0, 1.0
    rho = lam / (lam + nu)
    s = c * sigma(N, rho)
    lines = []
    for beta in (-0.5, 0.0, 0.5, 1.0):
        p = p_of_beta(N, lam, nu, c, beta)
        B = 2.0 * s
        out = []
        for th in (True, False):
            tz, tf, ti, sni, hx, hn, *_ = simulate(N, lam, nu, c, p, B, 1.5e5, th, 40, 2026, False)
            w = tz + tf + ti
            pos = tf + ti
            out.append(dict(p0=tz / w, pF=tf / pos, m=sni / ti, hx=hx / hx.sum()))
        a, b = out
        tv = 0.5 * np.abs(a["hx"] - b["hx"]).sum()
        lines.append((beta, a["p0"], b["p0"], a["pF"], b["pF"], a["m"], b["m"], tv))
        print("decomp", lines[-1])
    rows = "\n".join(
        rf"$\phantom{{-}}{be:g}$ & {fmt(p0t)} & {fmt(p0f)} & {pFt:.4f} & {pFf:.4f} & {mt:.3f} & {mf:.3f} & {tv:.4f}\\"
        .replace(r"$\phantom{-}-", "$-")
        for be, p0t, p0f, pFt, pFf, mt, mf, tv in lines)
    body = rf"""%% decomposition test, N={N}, lambda={lam}, nu={nu}, c={c}, B=2 c sigma_N, simulation
\begin{{tabular}}{{c|cc|cc|cc|c}}
 & \multicolumn{{2}}{{c|}}{{$P(x=0)$}} & \multicolumn{{2}}{{c|}}{{$P(x=B\mid x>0)$}} & \multicolumn{{2}}{{c|}}{{$E[n\mid 0<x<B]$}} & TV of $x\mid$\\
$\beta$ & thr. & free & thr. & free & thr. & free & interior\\\hline
{rows}
\end{{tabular}}
"""
    open(os.path.join(OUT, "decomposition.tex"), "w").write(body)


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["path", "rate", "scaling", "decomp"]
    if "path" in todo: fig_sample_path()
    if "rate" in todo: fig_dominant_rate()
    if "scaling" in todo: fig_throttle_scaling()
    if "decomp" in todo: fig_decomposition()


# ---------------------------------------------------------------------------
# Figure 5: rate / quota trade-off at fixed throttling probability
# ---------------------------------------------------------------------------
def p_thr_extrap(N, lam, nu, c, p, B, s):
    p1 = throttle_prob_numeric(N, lam, nu, c, p, B, s / 100)
    p2 = throttle_prob_numeric(N, lam, nu, c, p, B, s / 200)
    return 2 * p2 - p1


def fig_tradeoff():
    lam, nu, c, N = 1.0, 3.0, 1.0, 160
    gam = lam + nu
    rho = lam / gam
    s = c * sigma(N, rho)
    betas = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    plots = []
    for eps, col, mark in [(1e-2, "verdecito", "square*"), (1e-3, "azulcito", "*"), (1e-4, "rojito", "triangle*")]:
        pts = []
        for beta in betas:
            p = p_of_beta(N, lam, nu, c, beta)
            # bisection on b in [0.02, 1.5 x rule]; b* = 0 if no quota is needed
            g = lambda bb: np.log(p_thr_extrap(N, lam, nu, c, p, bb * s, s)) - np.log(eps)
            lo, hi = 0.02, 1.5 * np.log(1 / eps) / (gam * beta)
            if g(lo) <= 0:
                b1 = 0.0
            else:
                for _ in range(14):
                    mid = 0.5 * (lo + hi)
                    if g(mid) > 0:
                        lo = mid
                    else:
                        hi = mid
                b1 = 0.5 * (lo + hi)
            pts.append(f"({beta},{b1:.4f})")
            print("tradeoff", eps, beta, round(b1, 4), "rule", round(np.log(1 / eps) / (gam * beta), 4))
        lab = {1e-2: "10^{-2}", 1e-3: "10^{-3}", 1e-4: "10^{-4}"}[eps]
        plots.append(rf"\addplot[{col}, thick, mark={mark}, only marks] coordinates {{{' '.join(pts)}}};" + "\n"
                     + rf"\addlegendentry{{$\varepsilon={lab}$}}" + "\n"
                     + rf"\addplot[{col}, dashed, domain=0.45:3.1, samples=60, forget plot] {{{np.log(1/eps)/gam:.5f}/x}};")
    body = rf"""%% quota needed vs refill margin at fixed P(throttled)=eps; N={N}, lambda={lam}, nu={nu}, c={c}
%% marks: balance equations on a grid (extrapolated); dashed: rule log(1/eps)/(gamma beta)
\begin{{tikzpicture}}
\begin{{axis}}[width=\textwidth, height=0.66\textwidth, xmin=0.3, xmax=3.2, ymin=0, ymax=5,
  grid=major, grid style={{gray!25}},
  xlabel={{refill margin $\beta$}}, ylabel={{quota $B^*/(c\sigma_N)$}}, legend pos=north east,
  tick label style={{font=\footnotesize}}, label style={{font=\footnotesize}}]
{chr(10).join(plots)}
\end{{axis}}
\end{{tikzpicture}}
"""
    open(os.path.join(OUT, "tradeoff.tex"), "w").write(body)


if __name__ == "__main__" and "tradeoff" in __import__("sys").argv:
    fig_tradeoff()
