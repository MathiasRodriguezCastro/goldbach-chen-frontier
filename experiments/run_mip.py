"""
Experimentos de optimización / selección (secciones 6-7 del .tex).

  A) Umbral de rescate de Chen: al restringir las representaciones a una banda
     balanceada p ∈ [N/2 - k, N/2], ¿a partir de qué semi-ancho k empieza a
     fallar el canal primo y los semiprimos (Chen) lo rescatan? (separable por N)

  B) Rama de mínima variación espacial (MIP, Gurobi): elegir una representación
     por N minimizando Σ|s_N - s_{N-2}|. Resulta geométricamente suave pero
     alterna de tipo: suavidad espacial ≠ suavidad de tipo.

Uso: python3 run_mip.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from mip import enumerate_window, solve_selection
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def kmin_prime_semi(T, N, kmax):
    """Distancia desde N/2 al primo p más cercano (hacia abajo) con N-p primo /
    semiprimo, restringido a p ∈ [N/2-kmax, N/2]. inf si no hay."""
    half = N // 2
    lo = max(2, half - kmax)
    p = T.primes
    p = p[(p >= lo) & (p <= half)]
    if p.size == 0:
        return np.inf, np.inf
    q = N - p
    d = half - p  # >=0
    dp = d[T.is_prime[q]]
    ds = d[T.is_semiprime[q]]
    kp = float(dp.min()) if dp.size else np.inf
    ks = float(ds.min()) if ds.size else np.inf
    return kp, ks


def rescue_experiment(T, scales, win=4000, kmax=800):
    """Para cada escala N0, fracción de N que requieren rescate vs semi-ancho k."""
    ks = np.unique(np.round(np.logspace(0, np.log10(kmax), 40)).astype(int))
    out = {}
    detail = {}
    for N0 in scales:
        Ns = range(N0, N0 + win + 1, 2)
        kp = np.empty(len(list(Ns)))
        ks_arr = np.empty_like(kp)
        for i, N in enumerate(range(N0, N0 + win + 1, 2)):
            kp[i], ks_arr[i] = kmin_prime_semi(T, N, kmax)
        fr = []
        nofeas = []
        for k in ks:
            need = np.mean((kp > k) & (ks_arr <= k))
            nf = np.mean((kp > k) & (ks_arr > k))
            fr.append(need); nofeas.append(nf)
        lab = f"N~{N0//1000}k"
        out[lab] = fr
        detail[lab] = {"k": ks.tolist(), "rescue_frac": fr, "nofeas_frac": nofeas}
    return ks, out, detail


def main():
    T = Tables(2_000_000)

    print("== A) Umbral de rescate de Chen ==")
    scales = [50_000, 200_000, 1_000_000]
    ks, fracs, detail = rescue_experiment(T, scales)
    p = plots.rescue_threshold(ks, fracs)
    print("   figura:", os.path.relpath(p))
    for lab, d in detail.items():
        # primer k donde el rescate baja de 1%
        arr = np.array(d["rescue_frac"]); kk = np.array(d["k"])
        below = kk[arr < 0.01]
        kcrit = int(below[0]) if below.size else None
        print(f"   {lab}: rescate<1% a partir de k≈{kcrit};  máx rescate={100*arr.max():.1f}%")

    print("== B) Rama de mínima variación espacial (MIP) ==")
    reps = enumerate_window(T, 100_000, 100_800)   # ~400 N pares, MIP modesto
    r = solve_selection(reps, beta=1.0, time_limit=120)
    Ns = sorted(r["sel"].keys())
    ss = [r["sel"][N][3] for N in Ns]
    ty = [r["sel"][N][2] for N in Ns]
    pth = plots.mip_branch(Ns, ss, ty)
    print(f"   N={r['n_N']}  TV(s)={r['total_variation']:.4f}  "
          f"semiprimos={r['n_semiprime']} ({100*r['n_semiprime']/r['n_N']:.1f}%)  "
          f"saltos_tipo={r['type_switches']}")
    print("   figura:", os.path.relpath(pth))

    # comparación: rama de SOLO primos (siempre existe, Goldbach) -> TV mayor pero 0 saltos
    reps_prime = {N: [x for x in reps[N] if x[2] == 0] for N in reps}
    rp = solve_selection(reps_prime, beta=1.0, time_limit=120)
    print(f"   [solo primos] TV(s)={rp['total_variation']:.4f}  semiprimos=0  saltos_tipo=0")

    with open(os.path.join(DATA, "mip_summary.json"), "w") as f:
        json.dump({
            "rescue": detail,
            "minTV_mixto": {"TV": r["total_variation"], "n_semi": r["n_semiprime"],
                            "switches": r["type_switches"], "nN": r["n_N"]},
            "minTV_solo_primos": {"TV": rp["total_variation"]},
        }, f, indent=2)


if __name__ == "__main__":
    main()
