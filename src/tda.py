"""
Persistencia topológica de los valles del cometa (draft expandido §sec:tda).

Homología persistente 0-dimensional del filtrado por sublevel de los conteos
normalizados. Los mínimos locales (valles, donde el canal es localmente débil)
nacen al subir el umbral y mueren al fusionarse con un vecino más profundo; la
persistencia death-birth es la PROMINENCIA del valle.

Pregunta del draft: ¿los valles de Goldbach (R1) desaparecen al pasar a Chen
(R<=2)? Si Chen "rellena" topológicamente los valles, su diagrama de persistencia
tiene menos puntos prominentes.

Para que los valles sean RELATIVOS (no dominados por la tendencia ~N/log^2N), se
normaliza cada conteo dividiéndolo por su mediana móvil (serie estacionaria ~1).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def detrend(R, window=2001):
    """R / mediana_móvil(R) -> serie estacionaria ~1 (valles = caídas <1)."""
    med = pd.Series(R.astype(float)).rolling(window, center=True, min_periods=window // 2).median()
    return (R / med.values)


def persistence_1d(f):
    """
    Persistencia 0-dim del filtrado sublevel de la secuencia f (mínimos locales).
    Devuelve array de (birth, death) de los valles finitos (el mínimo global es
    infinito y se omite). O(n α(n)).
    """
    n = len(f)
    order = np.argsort(f, kind="stable")
    parent = np.full(n, -1, dtype=np.int64)
    birth = np.zeros(n)
    added = np.zeros(n, dtype=bool)
    pairs = []

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for i in order:
        added[i] = True
        roots = []
        for j in (i - 1, i + 1):
            if 0 <= j < n and added[j]:
                roots.append(find(j))
        roots = list(dict.fromkeys(roots))  # únicos, preserva orden
        if not roots:
            parent[i] = i
            birth[i] = f[i]
        else:
            roots_sorted = sorted(roots, key=lambda r: birth[r])
            keep = roots_sorted[0]
            parent[i] = keep
            for r in roots_sorted[1:]:
                pairs.append((birth[r], f[i]))   # valle r muere en el saddle f[i]
                parent[r] = keep
    return np.array(pairs) if pairs else np.empty((0, 2))


def valley_summary(N, R, window=2001):
    """Persistencias de valles de la serie normalizada de R."""
    a = detrend(R, window)
    ok = np.isfinite(a)
    pairs = persistence_1d(a[ok])
    pers = (pairs[:, 1] - pairs[:, 0]) if len(pairs) else np.array([])
    return a, pers


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from sieve import Tables
    from counts import representation_counts

    from heuristics import singular_series_goldbach, mertens_weight_W
    T = Tables(1_000_000)
    N, R1, R2 = representation_counts(T)
    Rle2 = R1 + R2
    S = singular_series_goldbach(T.X, T.is_prime)[N]
    W = mertens_weight_W(N, T.is_prime)
    # Cada canal con SU normalización de divisibilidad: E1=S, E2=S^0.5 W, E<=2=E1+E2.
    E1 = S
    E2 = np.sqrt(S) * W
    Ele2 = E1 + E2
    a1, pers1 = valley_summary(N, R1 / E1)
    a2, persc = valley_summary(N, Rle2 / Ele2)
    print(f"X={T.X}  (cada canal normalizado por su serie singular)")
    for thr in (0.05, 0.1, 0.15, 0.2):
        n1 = int(np.sum(pers1 > thr)); nc = int(np.sum(persc > thr))
        print(f"  valles con prominencia > {thr:.2f}:  Goldbach={n1:5d}   Chen={nc:5d}   "
              f"(Chen/Goldbach={nc/max(n1,1):.2f})")
    print(f"  valle más profundo:  Goldbach={pers1.max():.3f}   Chen={persc.max():.3f}")
    print(f"  prominencia total:   Goldbach={pers1.sum():.1f}   Chen={persc.sum():.1f}")
    # ¿coinciden los valles? correlación de anomalías (normalizadas, mediana móvil)
    m = np.isfinite(a1) & np.isfinite(a2)
    print(f"  corr(anomalía Goldbach, anomalía Chen) = {np.corrcoef(a1[m], a2[m])[0,1]:+.3f}")
    # en los 200 valles más profundos de Goldbach, ¿qué hace Chen?
    deep = np.argsort(np.nan_to_num(a1, nan=1.0))[:200]
    print(f"  en los 200 N más frágiles (Goldbach): anomalía Goldbach media={np.nanmean(a1[deep]):.3f}  "
          f"Chen media={np.nanmean(a2[deep]):.3f}  (Chen~1 => rellena el valle)")
