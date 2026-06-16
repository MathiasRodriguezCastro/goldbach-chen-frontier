"""
Driver principal: cribas, conteos, diagnósticos, heurística refinada y figuras.

Uso:
    python3 run_counts.py [X]          (default X = 2_000_000)

Guarda arrays en data/, figuras en figures/ y un resumen en data/summary.json.
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from counts import representation_counts, validate
from heuristics import (singular_series_goldbach, hl_prediction_R1,
                        refined_prediction_R2, mertens_weight_W,
                        measure_singular_exponent, TWIN_PRIME_C2)
import diagnostics as dg
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA, exist_ok=True)


def cv(x):
    x = x[np.isfinite(x)]
    return float(np.std(x) / np.mean(x))


def main(X):
    t0 = time.time()
    print(f"[1/5] Cribas y conteos hasta X={X} ...")
    T = Tables(X)
    N, R1, R2 = representation_counts(T)
    print(f"      {len(N)} enteros pares;  Goldbach R1>0 en todo el rango: {bool(np.all(R1>0))}")
    # validación FFT vs fuerza bruta
    rng = np.random.default_rng(0)
    ok = validate(T, list(rng.choice(N, size=300, replace=False)))
    print(f"      validación FFT vs fuerza bruta: {'OK' if ok else 'FALLA'}")

    print("[2/5] Serie singular y heurísticas ...")
    S = singular_series_goldbach(T.X, T.is_prime)
    W = mertens_weight_W(N, T.is_prime)
    predR1 = hl_prediction_R1(N, S)
    predR2 = refined_prediction_R2(N, S, T.is_prime)

    m = N >= 20000
    beta, r2reg = measure_singular_exponent(N, R1, R2, S, Nmin=20000)

    summary = {
        "X": int(X),
        "n_even": int(len(N)),
        "goldbach_holds": bool(np.all(R1 > 0)),
        "twin_prime_C2": TWIN_PRIME_C2,
        "R1_singular_exponent": 1.0,  # ref
        "R2_singular_exponent_beta": float(beta),
        "R2_singular_regression_R2coef": float(r2reg),
        "theta_mean": float(dg.theta(R1, R2)[m].mean()),
        "C_mean": float(dg.chen_surplus(R1, R2)[m].mean()),
        "cv_R1_naive": cv(R1[m] / (N[m] / np.log(N[m]) ** 2)),
        "cv_R1_singular": cv(R1[m] / predR1[m]),
        "cv_R2_tex": cv(R2[m] / (N[m] * np.log(np.log(N[m])) / np.log(N[m]) ** 2)),
        "cv_R2_refined_half": cv(R2[m] / predR2[m]),
        "median_R1_obs_over_pred": float(np.median(R1[m] / predR1[m])),
        "median_R2_obs_over_pred": float(np.median(R2[m] / predR2[m])),
    }
    for k in (1, 2, 3, 5, 10):
        summary[f"fragile_F{k}"] = int(len(dg.fragility_set(N, R1, R2, k)))

    print("[3/5] Geometría de factores B(q) ...")
    B, w = dg.balance_distribution(T, weighted=True)
    summary["B_mean_weighted"] = float(np.average(B, weights=w))
    summary["B_mass_below_0.1"] = float(w[B < 0.1].sum() / w.sum())
    summary["B_mass_above_0.4"] = float(w[B > 0.4].sum() / w.sum())

    print("[4/5] Guardando arrays y resumen ...")
    np.savez_compressed(os.path.join(DATA, "counts.npz"),
                        N=N, R1=R1, R2=R2, S=S[N], W=W)
    with open(os.path.join(DATA, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[5/5] Figuras ...")
    paths = []
    paths.append(plots.comets(N, R1, R2))
    paths.append(plots.singular_exponent(N, R1, R2, S))
    paths.append(plots.normalization_collapse(N, R1, S))
    paths.append(plots.theta_and_chen(N, R1, R2, W))
    paths.append(plots.balance_hist(B, w))
    for p in paths:
        print("      ", os.path.relpath(p))

    print(f"\nResumen ({time.time()-t0:.1f}s):")
    for kk, vv in summary.items():
        print(f"  {kk:32s} {vv}")
    return summary


if __name__ == "__main__":
    X = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    main(X)
