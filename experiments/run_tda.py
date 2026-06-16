"""
Persistencia topológica de valles (draft expandido §sec:tda).
¿Chen rellena los valles del cometa de Goldbach? Compara la persistencia de los
valles de R1/𝔖 y de R<=2/E<=2 (cada canal con su normalización de divisibilidad)
y la correlación de las anomalías. Genera figura 14.

Uso: python3 run_tda.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from counts import representation_counts
from heuristics import singular_series_goldbach, mertens_weight_W
from tda import valley_summary
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(1_000_000)
    N, R1, R2 = representation_counts(T)
    Rle2 = R1 + R2
    S = singular_series_goldbach(T.X, T.is_prime)[N]
    W = mertens_weight_W(N, T.is_prime)
    E1 = S
    Ele2 = S + np.sqrt(S) * W
    a1, pers1 = valley_summary(N, R1 / E1)
    a2, persc = valley_summary(N, Rle2 / Ele2)

    thrs = np.linspace(0.02, 0.30, 15)
    n_gold = [int(np.sum(pers1 > t)) for t in thrs]
    n_chen = [int(np.sum(persc > t)) for t in thrs]
    m = np.isfinite(a1) & np.isfinite(a2)
    corr = float(np.corrcoef(a1[m], a2[m])[0, 1])
    deep = np.argsort(np.nan_to_num(a1, nan=1.0))[:200]

    path = plots.tda(thrs, n_gold, n_chen, a1, a2, corr)

    print(f"X={T.X}")
    print(f"  valles prom>0.1: Goldbach={int(np.sum(pers1>0.1))}  Chen={int(np.sum(persc>0.1))}")
    print(f"  valles prom>0.2: Goldbach={int(np.sum(pers1>0.2))}  Chen={int(np.sum(persc>0.2))} "
          f"(Chen {100*np.sum(persc>0.2)/np.sum(pers1>0.2):.0f}% del de Goldbach)")
    print(f"  prominencia total: Goldbach={pers1.sum():.0f}  Chen={persc.sum():.0f}")
    print(f"  corr(anomalía Goldbach, Chen) = {corr:+.3f}")
    print(f"  200 N más frágiles: anomalía Goldbach={np.nanmean(a1[deep]):.3f}  "
          f"Chen={np.nanmean(a2[deep]):.3f}  -> Chen NO rellena (sigue bajo)")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"X": T.X, "corr_anomaly": corr,
               "valleys_gold": n_gold, "valleys_chen": n_chen, "thr": thrs.tolist(),
               "total_prom_gold": float(pers1.sum()), "total_prom_chen": float(persc.sum()),
               "fragile200_gold": float(np.nanmean(a1[deep])),
               "fragile200_chen": float(np.nanmean(a2[deep]))},
              open(os.path.join(DATA, "tda.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
