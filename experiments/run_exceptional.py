"""
Cota de segundo momento / conjunto excepcional (apéndice §A.6).

Mide la concentración de R2/R1 alrededor de la predicción de canales W~(N):
  b(N) = (R2/R1) / W~(N),   CV(b) = sd(b)/mean(b).
Si CV(b) es pequeño y decrece, entonces R2/R1 = W~(N) para CASI TODO N (la densidad
del conjunto excepcional |R2/R1 - <b>W~| > eps·W~ es <= CV(b)^2/eps^2 por Chebyshev),
y la ley puntual beta2->1 vale c.t.p.

Uso: python3 run_exceptional.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from segmented import windowed_counts
from sieve import sieve_primes


def Wtilde(N, X, W=2_000_000):
    Nf = N.astype(float); logN = np.log(Nf); out = np.zeros(len(N))
    isp = sieve_primes(int(np.sqrt(X + W)) + 1); pr = np.nonzero(isp)[0]; pr = pr[pr >= 3]
    for r in pr:
        r = int(r); m = (N > r * r) & (N % r != 0)
        if m.any():
            out[m] += ((r - 1.) / (r - 2.)) * (1. / r) * (logN[m] / np.log(Nf[m] / r))
    return out


def main():
    rows = []
    for X in (20_000_000, 100_000_000):
        N, R1, R2, S = windowed_counts(X, 2_000_000)
        ok = R1 > 0
        N, R1, R2 = N[ok], R1[ok], R2[ok]
        Wt = Wtilde(N, X)
        b = (R2 / R1) / Wt
        bm = b.mean(); cv = b.std() / bm
        # conjunto excepcional DIRECTO: fracción con |b-<b>|/<b> > eps
        rel = np.abs(b - bm) / bm
        exc = {eps: float(np.mean(rel > eps)) for eps in (0.05, 0.10, 0.20)}
        cheb = {eps: float(min(1.0, cv ** 2 / eps ** 2)) for eps in (0.05, 0.10, 0.20)}
        rows.append({"X": X, "mean_b": float(bm), "CV": float(cv),
                     "poisson_floor": float(1 / np.sqrt(R2.mean())),
                     "exc_directo": exc, "exc_chebyshev": cheb})
        print(f"X={X}: CV(b)={cv:.4f}  (Poisson {1/np.sqrt(R2.mean()):.5f})")
        for eps in (0.05, 0.10, 0.20):
            print(f"   |desv|>{eps:.0%}: directo={100*exc[eps]:.3f}%   cota Chebyshev={100*cheb[eps]:.2f}%")

    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..", "data",
                                      "exceptional.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
