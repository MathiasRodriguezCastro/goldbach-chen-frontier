"""
Reducción de primer momento (apéndice §A.5): verifica el Lema 3 y el Teorema 2.

  - Respuestas condicionales A_l=(l-1)/(l-2) (R1) y B_l=(l-1)(1-F_l)/(l-2+F_l) (R2),
    exactas vía equidistribución de primos en progresiones.
  - El exponente de primer momento beta2^(1) reproduce beta2 (log-regresión) con
    gap constante ~0.004 (Jensen) en todas las escalas.

Uso: python3 run_firstmoment.py
"""
import os
import sys
import json
import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from counts import representation_counts
from heuristics import singular_series_goldbach
from segmented import windowed_counts


def lemma3_table(T):
    """A_l, B_l medidos vs fórmulas (Lema 3), y F_l. Requiere FFT global (X<=~3e6)."""
    X = T.X
    N, R1, R2 = representation_counts(T)
    isp = T.is_prime.astype(float)
    primes = T.primes
    rows = []
    for l in (3, 5, 7, 11, 13, 17, 19):
        div = (N % l == 0)
        A = R1[div].mean() / R1[~div].mean()
        B = R2[div].mean() / R2[~div].mean()
        q = l * primes; q = q[q <= X]
        mask = np.zeros(X + 1); mask[q] = 1.0
        Farr = np.rint(fftconvolve(isp, mask)[N]).astype(np.int64)
        F = Farr.sum() / R2.sum()
        rows.append({"l": l, "A": float(A), "A_formula": (l - 1) / (l - 2),
                     "B": float(B), "B_formula": (l - 1) * (1 - F) / (l - 2 + F),
                     "F": float(F)})
    return rows


def fm_exponent(X):
    """beta2 (log-reg) y beta2^(1) (primer momento) en una ventana en X."""
    N, R1, R2, S = windowed_counts(X, 2_000_000)
    ok = R1 > 0
    N, R1, R2, S = N[ok], R1[ok], R2[ok], S[ok]
    beta_log = 1 + np.polyfit(np.log(S), np.log(R2 / R1), 1)[0]
    gl, dl, vl = [], [], []
    for l in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
        div = (N % l == 0)
        A = R1[div].mean() / R1[~div].mean()
        B = R2[div].mean() / R2[~div].mean()
        gl.append(np.log((l - 1) / (l - 2))); dl.append(np.log(B / A))
        vl.append((1. / l) * (1 - 1. / l))
    gl, dl, vl = map(np.array, (gl, dl, vl))
    beta_fm = 1 + np.sum(dl * gl * vl) / np.sum(gl ** 2 * vl)
    return float(beta_log), float(beta_fm)


def main():
    T = Tables(2_000_000)
    rows = lemma3_table(T)
    print("Lema 3 (X=2e6):  l    A_l    (l-1)/(l-2)    B_l    fórmula    F_l")
    for r in rows:
        print(f"            {r['l']:5d}  {r['A']:.4f}   {r['A_formula']:.4f}     "
              f"{r['B']:.4f}  {r['B_formula']:.4f}   {r['F']:.4f}")

    print("\nTeorema 2: beta2 (log-reg) vs beta2^(1) (primer momento)")
    scan = {}
    for X in (2_000_000, 20_000_000, 100_000_000):
        bl, bf = fm_exponent(X)
        scan[X] = {"beta_log": bl, "beta_fm": bf, "gap": bl - bf}
        print(f"   X={X:>11}: beta2={bl:.4f}  beta2^(1)={bf:.4f}  gap={bl-bf:+.4f}")

    json.dump({"lemma3": rows, "exponents": scan},
              open(os.path.join(os.path.dirname(__file__), "..", "data", "firstmoment.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
