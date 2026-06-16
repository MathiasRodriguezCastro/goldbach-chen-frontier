"""
Exponente de balance residual de Goldbach--Chen (tercer draft).

Calcula r_star(N) y lambda_Res(N) para todo N par hasta X, y responde las
preguntas empíricas del draft:
  Q1: crecimiento del máximo / percentiles de lambda_Res con X.
  Q2: distribución de r_star (multiplicador minimal); ¿spikes por divisibilidad?
  Q4: correlación de lambda_Res con la serie singular 𝔖(N) y omega(N).
  Q6: cobertura por diccionario de K multiplicadores pequeños.

Uso: python3 run_balance.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from heuristics import singular_series_goldbach
from balance import rstar_all, lambda_res_exact
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    X = 2_000_000
    T = Tables(X)
    N, rs = rstar_all(T)
    S = singular_series_goldbach(T.X, T.is_prime)
    _, lam = lambda_res_exact(T)
    ok = np.isfinite(lam) & (N >= 1000)

    # Q2: distribución de r_star + obstrucción por divisibilidad
    vals, cnts = np.unique(rs[rs >= 2], return_counts=True)
    rstar_dist = {int(v): int(c) for v, c in zip(vals, cnts)}
    div3 = (N % 3 == 0)
    div5 = (N % 5 == 0)

    # Q1: percentiles globales
    pcts = {q: float(np.percentile(lam[ok], q)) for q in (50, 90, 99, 99.9, 100)}

    # Q6: cobertura por diccionario de los K primos más chicos
    cover = {}
    cum = 0
    for v in sorted(rstar_dist):
        cum += rstar_dist[v]
        cover[v] = 100.0 * cum / len(N)

    # Q4: correlaciones
    m = ok & (N >= 20000)
    corr_S = float(np.corrcoef(np.log(S[N[m]]), lam[m])[0, 1])

    path = plots.balance_exponent(N, lam, rs, np.log(S))

    print(f"X={X}")
    print(f"  a_Res: mediana={1+pcts[50]:.4f}  p90={1+pcts[90]:.4f}  "
          f"p99={1+pcts[99]:.4f}  max={1+pcts[100]:.4f}")
    print(f"  r_star: " + "  ".join(f"{v}:{100*rstar_dist[v]/len(N):.1f}%"
                                    for v in sorted(rstar_dist) if rstar_dist[v] > 100))
    print(f"  obstrucción: r_star medio  3∤N={rs[~div3].mean():.3f} 3|N={rs[div3].mean():.3f} "
          f"| 5∤N={rs[~div5].mean():.3f} 5|N={rs[div5].mean():.3f}")
    print(f"  cobertura diccionario: {{2,3}}={cover.get(3,0):.2f}%  "
          f"{{..7}}={cover.get(7,0):.2f}%  {{..11}}={cover.get(11,0):.3f}%")
    print(f"  corr(lambda_Res, log 𝔖(N)) = {corr_S:+.3f}")
    print(f"figura: {os.path.relpath(path)}")

    # Q1 a varias escalas: ¿crece el máximo / percentiles?
    print("  --- Q1: percentiles de a_Res vs X ---")
    scan = {}
    for Xs in (100_000, 500_000, 2_000_000):
        Ts = T if Xs == X else Tables(Xs)
        Ns, rss = rstar_all(Ts)
        _, ls = lambda_res_exact(Ts)
        o = np.isfinite(ls) & (Ns >= 1000)
        row = {q: float(1 + np.percentile(ls[o], q)) for q in (50, 90, 99, 100)}
        scan[Xs] = row
        print(f"   X={Xs:>9}: mediana={row[50]:.4f} p90={row[90]:.4f} "
              f"p99={row[99]:.4f} max={row[100]:.4f}")

    json.dump({"X": X, "a_Res_pct": {k: 1 + v for k, v in pcts.items()},
               "rstar_dist": rstar_dist, "cover": cover, "corr_logS": corr_S,
               "rstar_3div": float(rs[div3].mean()), "rstar_3ndiv": float(rs[~div3].mean()),
               "scan_X": scan},
              open(os.path.join(DATA, "balance.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
