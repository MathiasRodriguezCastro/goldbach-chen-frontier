"""
Dinámica estadística de la frontera (draft expandido §sec:dynamics).
¿Es theta(N) predecible desde la aritmética local? Genera figura 15.

Uso: python3 run_dynamics.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from counts import representation_counts
from heuristics import singular_series_goldbach
from dynamics import binned_r2, group_r2
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(2_000_000)
    N, R1, R2 = representation_counts(T)
    S = singular_series_goldbach(T.X, T.is_prime)[N]
    theta = R1 / (R1 + R2)
    m = N >= 20000
    Nm, th, Sm = N[m], theta[m], S[m]

    r2_S, predS = binned_r2(th, np.log(Sm))
    res = th - predS
    # quitar la tendencia lenta (loglog) para medir la autocorrelación de corto rango
    res_dt = res - pd.Series(res).rolling(4001, center=True, min_periods=2000).mean().values
    res_dt = res_dt[np.isfinite(res_dt)]
    acf = [float(np.corrcoef(res_dt[:-k], res_dt[k:])[0, 1]) for k in range(1, 21)]

    dtheta = theta[1:] - theta[:-1]
    mm = N[1:] >= 20000
    dth = dtheta[mm]
    feat = np.c_[np.ones(len(dth)), np.log(S[:-1][mm]), np.log(S[1:][mm])]
    coef, *_ = np.linalg.lstsq(feat, dth, rcond=None)
    r2_d = 1 - np.var(dth - feat @ coef) / np.var(dth)

    # segmento para visualizar
    lo = np.searchsorted(Nm, 100000); hi = lo + 150
    path = plots.dynamics(Nm[lo:hi], th[lo:hi], predS[lo:hi],
                          list(range(1, 21)), acf, r2_S, r2_d)

    print(f"X={T.X}")
    print(f"  R^2(theta | 𝔖)           = {r2_S:.4f}")
    print(f"  R^2(Δtheta | salto 𝔖)    = {r2_d:.4f}")
    print(f"  ruido irreducible: std(res)={res.std():.4f} ({100*res.std()/th.std():.0f}% de la dispersión)")
    print(f"  autocorr lag-1 del residuo = {acf[0]:+.3f} (no es ruido blanco)")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"X": T.X, "r2_theta_S": r2_S, "r2_dtheta_jump": r2_d,
               "resid_std_frac": float(res.std() / th.std()), "acf": acf},
              open(os.path.join(DATA, "dynamics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
