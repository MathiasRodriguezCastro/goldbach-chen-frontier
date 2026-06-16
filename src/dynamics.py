"""
Dinámica estadística del estado de la frontera (draft expandido §sec:dynamics).

El estado X_N=(R1,R2,theta,H,𝔖,omega(N),N mod M) evoluciona con N. Pregunta: ¿la
cuota prima theta(N) y su incremento Δtheta_N son PREDECIBLES desde la aritmética
local de N (serie singular, divisibilidad), o hay ruido irreducible?

Como theta = 1/(1+R2/R1) y R2/R1 ~ 𝔖^{-1/2}W, theta es casi una función
determinista de 𝔖(N). Medimos:
  - R^2 de theta explicado por 𝔖(N) (predictor binned), y al añadir N mod 30.
  - el residuo: desviación (ruido irreducible) y autocorrelación (¿blanco?).
  - predictibilidad de Δtheta_N desde el salto de divisibilidad 𝔖(N)->𝔖(N+2).
"""
from __future__ import annotations
import numpy as np


def binned_r2(y, key, nbins=200):
    """R^2 de predecir y por la media de y en bins de `key`."""
    order = np.argsort(key)
    edges = np.linspace(key.min(), key.max(), nbins + 1)
    idx = np.clip(np.digitize(key, edges) - 1, 0, nbins - 1)
    pred = np.empty_like(y)
    for b in range(nbins):
        sel = idx == b
        if np.any(sel):
            pred[sel] = y[sel].mean()
    return 1.0 - np.var(y - pred) / np.var(y), pred


def group_r2(y, labels):
    """R^2 de predecir y por la media de y dentro de cada grupo (etiqueta discreta)."""
    pred = np.empty_like(y)
    for lab in np.unique(labels):
        sel = labels == lab
        pred[sel] = y[sel].mean()
    return 1.0 - np.var(y - pred) / np.var(y), pred


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from sieve import Tables
    from counts import representation_counts
    from heuristics import singular_series_goldbach

    T = Tables(2_000_000)
    N, R1, R2 = representation_counts(T)
    S = singular_series_goldbach(T.X, T.is_prime)[N]
    theta = R1 / (R1 + R2)
    m = N >= 20000
    Nm, th, Sm = N[m], theta[m], S[m]
    logS = np.log(Sm)

    r2_S, predS = binned_r2(th, logS)
    # añadir N mod 30 (divisibilidad fina) como grupo conjunto con bin de 𝔖
    res_S = th - predS
    mod30 = Nm % 30
    r2_mod_extra, _ = group_r2(res_S, mod30)  # cuánto explica mod30 del residuo
    # residuo final tras 𝔖 + mod30
    _, pred_mod = group_r2(res_S, mod30)
    res_final = res_S - pred_mod
    # autocorrelación lag-1 del residuo final (¿blanco?)
    ac1 = np.corrcoef(res_final[:-1], res_final[1:])[0, 1]

    print(f"X={T.X}")
    print(f"  Var(theta) total: std={th.std():.4f}")
    print(f"  R^2(theta | 𝔖(N))           = {r2_S:.4f}")
    print(f"  R^2 extra del residuo | Nmod30 = {r2_mod_extra:.4f}")
    print(f"  R^2(theta | 𝔖 + Nmod30)      = {1-np.var(res_final)/np.var(th):.4f}")
    print(f"  ruido irreducible: std(residuo) = {res_final.std():.4f} "
          f"({100*res_final.std()/th.std():.1f}% de la dispersión de theta)")
    print(f"  autocorr lag-1 del residuo = {ac1:+.4f}  ({'ruido ~blanco' if abs(ac1)<0.05 else 'queda estructura'})")

    # predictibilidad del incremento Δtheta_N desde el salto de divisibilidad
    dtheta = theta[1:] - theta[:-1]
    mm = N[1:] >= 20000
    dth = dtheta[mm]
    # salto de 𝔖: (logS(N), logS(N+2))
    feat = np.c_[np.log(S[:-1][mm]), np.log(S[1:][mm])]
    A = np.c_[np.ones(len(dth)), feat]
    coef, *_ = np.linalg.lstsq(A, dth, rcond=None)
    pred = A @ coef
    r2_d = 1 - np.var(dth - pred) / np.var(dth)
    print(f"  R^2(Δtheta | salto 𝔖(N)->𝔖(N+2)) = {r2_d:.4f}  "
          f"(la 'dinámica' la gobierna el cambio de divisibilidad)")
