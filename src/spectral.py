"""
Método del círculo EMPÍRICO: análisis espectral del cometa de Goldbach (ideas #3/#10).

(A) Espectro de Fourier de R_1 (destendenciado) sobre N par consecutivos: los picos
    aparecen en las frecuencias aritméticas a/q de primos chicos (periodo-N 6, 10, 30...)
    — los "arcos mayores" empíricos, con potencia que sigue la serie singular.
(B) Varianza de log R_1 explicada por la información modular local N mod m, acumulando
    primos: satura rápido (mod 6 ya capta el grueso) y converge al techo de 𝔖(N).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sieve import Tables
from counts import representation_counts
from heuristics import singular_series_goldbach


def detrended(R, win=2001):
    """R / mediana_móvil(R): aísla la oscilación aritmética de la tendencia N/log^2N."""
    tr = pd.Series(R.astype(float)).rolling(win, center=True, min_periods=win // 2).median().values
    a = R / tr
    return a - np.nanmean(a)


def power_spectrum(N, R1, Nlo, Nhi):
    """Espectro de potencia del cometa destendenciado en una ventana de N consecutivos."""
    m = (N >= Nlo) & (N < Nhi)
    a = np.nan_to_num(detrended(R1[m]))
    w = np.hanning(len(a))
    P = np.abs(np.fft.rfft(a * w)) ** 2
    freq = np.fft.rfftfreq(len(a))   # ciclos por paso (paso = 2 en N)
    return freq, P / P[1:].max()


def variance_by_modulus(N, R1, S, prime_sets, Nmin=20000, win=4001):
    """R^2 de la oscilación de log R1 (destendenciada) explicada por N mod prod(primes)."""
    m = N >= Nmin
    Nn, R1n, Sn = N[m], R1[m], S[m]
    # destendenciar log R1 (quita la tendencia lenta; queda 𝔖-oscilación + ruido HL)
    y = np.log(R1n.astype(float))
    y = y - pd.Series(y).rolling(win, center=True, min_periods=win // 2).mean().values
    ok = np.isfinite(y)
    Nn, Sn, y = Nn[ok], Sn[ok], y[ok]
    res = {}
    for B, ps in prime_sets:
        mod = int(np.prod(ps))
        cls = Nn % mod
        pred = np.zeros_like(y)
        for c in np.unique(cls):
            pred[cls == c] = y[cls == c].mean()
        res[B] = 1 - np.var(y - pred) / np.var(y)
    # techo: serie singular continua
    x = np.log(Sn)
    b, a0 = np.polyfit(x, y, 1)
    ceil = 1 - np.var(y - (a0 + b * x)) / np.var(y)
    return res, ceil


if __name__ == "__main__":
    T = Tables(2_000_000)
    N, R1, R2 = representation_counts(T)
    S = singular_series_goldbach(T.X, T.is_prime)[N]
    freq, P = power_spectrum(N, R1, 100000, 500000)
    # picos
    pk = np.argsort(-P)[:6]
    print("Picos espectrales del cometa (freq, periodo-N, potencia):")
    for k in sorted(pk, key=lambda i: -P[i]):
        if freq[k] > 0:
            print(f"  freq={freq[k]:.4f}  periodo-N={2/freq[k]:.1f}  potencia={P[k]:.3f}")
    psets = [(2, [2]), (3, [2, 3]), (5, [2, 3, 5]), (7, [2, 3, 5, 7]),
             (11, [2, 3, 5, 7, 11]), (13, [2, 3, 5, 7, 11, 13])]
    res, ceil = variance_by_modulus(N, R1, S, psets)
    print("\nVarianza de la oscilación de log R1 explicada por N mod (primos<=B):")
    for B, ps in psets:
        print(f"  primos<= {B:2d} (mod {int(np.prod(ps)):>5d}): R2={res[B]:.4f}  "
              f"({100*res[B]/ceil:.0f}% del techo 𝔖)")
    print(f"  techo (serie singular 𝔖 continua): R2={ceil:.4f}")
