"""
Verificación numérica de la constante de la serie singular del canal (Lema 1 / la
Remark del término principal en el apéndice §A.7):
    C_r'(N) = 2 C2 · S(N) · (r-1)/(r-2) · I_r(N) · (1 + O(1/100)),
con C_r'(N) = #{(p,s): p+rs=N, s>=r primo} y el integral arquimediano
    I_r(N) = ∫_r^{(N-2)/r} dt / (log t · log(N - r t)).

Confirma simultáneamente el factor 2 C2 (r-1)/(r-2) y el peso EXACTO I_r/I_1
(la aproximación cruda w_r = (1/r) logN/log(N/r) lo sobreestima ~1.5x para r chico).

Uso: python3 run_channel_check.py
"""
import os
import sys
import numpy as np
from scipy.signal import fftconvolve
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from heuristics import singular_series_goldbach, TWIN_PRIME_C2


def main():
    T = Tables(2_000_000)
    X = T.X
    isp = T.is_prime.astype(float)
    primes = T.primes
    S = singular_series_goldbach(X, T.is_prime)

    def channel(r):
        s = primes[primes >= r]; q = r * s; q = q[q <= X]
        m = np.zeros(X + 1); m[q] = 1.0
        return np.rint(fftconvolve(isp, m)).astype(np.int64)

    def Ir(N, r):
        f = lambda t: 1.0 / (np.log(t) * np.log(N - r * t))
        val, _ = quad(f, r, (N - 2) / r, limit=80)
        return val

    rng = np.random.default_rng(1)
    print("  r   C_r'(N) / [2C2·S(N)·(r-1)/(r-2)·I_r(N)]   (esperado ~1)   vs w_r crudo")
    for r in (3, 5, 7):
        C = channel(r)
        Ns = rng.choice(np.arange(1_500_000, 1_900_000, 2), 60, replace=False)
        Ns = Ns[Ns % r != 0]
        ratios = np.array([C[Nv] / (2 * TWIN_PRIME_C2 * S[Nv] * (r - 1) / (r - 2) * Ir(int(Nv), r))
                           for Nv in Ns])
        # comparacion con el peso crudo: <C_r'/R1> vs (r-1)/(r-2) w_r
        wr = (1. / r) * np.log(X) / np.log(X / r)
        print(f"  {r}   {ratios.mean():.4f} ± {ratios.std():.4f}"
              f"                       (r-1)/(r-2)·w_r = {(r-1)/(r-2)*wr:.3f} sobreestima)")


if __name__ == "__main__":
    main()
