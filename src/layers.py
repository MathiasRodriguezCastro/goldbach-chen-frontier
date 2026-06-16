"""
Capas de complejidad aritmética (draft expandido, §3): generaliza R1, R2 a
  R_k(N) = #{p < N : p primo, Omega(N-p) = k},
es decir q = N - p tiene exactamente k factores primos (con multiplicidad).
R_1 = Goldbach, R_2 = semiprimo, R_3 = producto de tres primos, ...

Conexión con el hallazgo central: R_1 hereda la serie singular de Goldbach con
exponente 1, R_2 con ~1/2. ¿Cuál es el exponente beta_k de cada capa? Hipótesis
natural: beta_k decrece con k (cada factor de q puede absorber la divisibilidad
local de N, diluyendo el realce singular). Este módulo lo mide.

Todo por convolución FFT, igual que counts.py: R_k = 1_P * 1_{Omega==k}.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

from sieve import Tables


def layer_counts(T: Tables, Kmax: int = 6):
    """Devuelve (N_par, {k: R_k}) para k=1..Kmax sobre N par en [4..X]."""
    X = T.X
    a = T.is_prime.astype(np.float64)
    N = np.arange(4, X + 1, 2)
    out = {}
    for k in range(1, Kmax + 1):
        mask = (T.Omega == k).astype(np.float64)
        conv = fftconvolve(a, mask)
        out[k] = np.rint(conv[N]).astype(np.int64)
    return N, out


def composite_composite_count(T: Tables):
    """
    R_cc(N) = #{p<N : N-p = a·b con a,b>1 ambos COMPUESTOS} (draft §3).
    Un q admite factorización compuesto×compuesto sii q = a·b con a,b no primos.
    Condición: q no es primo, ni semiprimo, ni p^3 (=p·p^2, p^2 compuesto -> sí
    vale!), en general q vale si tiene una factorización en dos compuestos.
    Caracterización usada: q es "cc" sii Omega(q)>=4, O (Omega(q)>=2 y q tiene un
    divisor compuesto d con q/d compuesto). Aproximación robusta y barata:
    q es cc sii existe divisor d, 1<d<q, con d y q/d ambos compuestos.
    Aquí marcamos por criba: para cada par de compuestos (c1<=c2), c1*c2<=X.
    """
    X = T.X
    is_comp = np.zeros(X + 1, dtype=bool)
    n = np.arange(X + 1)
    is_comp[(T.Omega >= 2)] = True            # compuesto = Omega>=2 (no primo, no 1)
    is_comp[:4] = False                       # 0,1,2,3 no son compuestos
    comp_vals = np.nonzero(is_comp)[0]
    is_cc = np.zeros(X + 1, dtype=bool)
    sq = int(np.sqrt(X))
    small_comp = comp_vals[comp_vals <= sq]
    for c in small_comp:
        c = int(c)
        cc2 = comp_vals[(comp_vals >= c) & (comp_vals <= X // c)]
        is_cc[c * cc2] = True
    a = T.is_prime.astype(np.float64)
    N = np.arange(4, X + 1, 2)
    conv = fftconvolve(a, is_cc.astype(np.float64))
    return N, np.rint(conv[N]).astype(np.int64)


def singular_exponent_layer(N, Rk, R1, S, Nmin=20000):
    """beta_k model-light: pendiente de log(R_k/R_1) vs log S(N), +1."""
    m = (N >= Nmin) & (Rk > 0) & (R1 > 0)
    x = np.log(S[N[m]])
    y = np.log(Rk[m] / R1[m])
    b, a = np.polyfit(x, y, 1)
    r2 = 1.0 - np.var(y - (a + b * x)) / np.var(y)
    return b + 1.0, r2


if __name__ == "__main__":
    from heuristics import singular_series_goldbach
    T = Tables(2_000_000)
    N, R = layer_counts(T, Kmax=6)
    S = singular_series_goldbach(T.X, T.is_prime)
    m = N >= 20000
    tot = sum(R[k] for k in R)
    print(f"X={T.X}")
    print("  capa k   share(%)   beta_k     R^2     (beta_k vs 1/k)")
    for k in sorted(R):
        share = 100.0 * R[k][m].sum() / tot[m].sum()
        if k == 1:
            print(f"   k={k}    {share:6.2f}    1.000    (ref)    1/k={1/k:.3f}")
        else:
            bk, r2 = singular_exponent_layer(N, R[k], R[1], S)
            print(f"   k={k}    {share:6.2f}    {bk:.3f}    {r2:.3f}    1/k={1/k:.3f}")
