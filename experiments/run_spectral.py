"""
Método del círculo empírico: análisis espectral del cometa de Goldbach (#3/#10).
Espectro de Fourier (arcos mayores en a/q) + varianza explicada por módulo (satura en
el techo de la serie singular). Genera figura 21.

Uso: python3 run_spectral.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from counts import representation_counts
from heuristics import singular_series_goldbach
from spectral import power_spectrum, variance_by_modulus
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(2_000_000)
    N, R1, R2 = representation_counts(T)
    S = singular_series_goldbach(T.X, T.is_prime)[N]

    freq, P = power_spectrum(N, R1, 100000, 500000)
    psets = [(2, [2]), (3, [2, 3]), (5, [2, 3, 5]), (7, [2, 3, 5, 7]),
             (11, [2, 3, 5, 7, 11]), (13, [2, 3, 5, 7, 11, 13])]
    res, ceil = variance_by_modulus(N, R1, S, psets)
    Bs = [b for b, _ in psets]; r2s = [res[b] for b in Bs]

    path = plots.spectral(freq, P, Bs, r2s, ceil)
    print("  espectro: pico dominante en freq=1/3 (mod 3, periodo-N=6); luego 1/5,2/5 (mod 5)")
    print("  varianza de la oscilación de log R1 explicada:")
    for b in Bs:
        print(f"    primos<= {b:2d}: {100*res[b]:.1f}%  ({100*res[b]/ceil:.0f}% del techo 𝔖)")
    print(f"    techo 𝔖: {100*ceil:.1f}%")
    print(f"  -> mod 6 ya capta {100*res[3]/ceil:.0f}% de lo explicable: el cometa es casi mod 6")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"B": Bs, "r2": r2s, "ceil": float(ceil)},
              open(os.path.join(DATA, "spectral.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
