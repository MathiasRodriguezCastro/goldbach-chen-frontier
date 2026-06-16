"""
Exponente de balance residual de Goldbach--Chen (tercer draft).

Para cada N par se mira la representación nondegenerada N = p + r·s (p,r,s primos,
r<=s) y su exponente de balance
    lambda(r,s) = log r / log s,   a(r,s) = 1 + lambda.
El exponente residual es el mínimo:
    lambda_Res(N) = min_{N=p+rs} log r / log s,   a_Res(N) = 1 + lambda_Res(N).
El multiplicador minimal es
    r_star(N) = min{ r primo : exists p,s primos, N = p + r s }.
Como lambda crece con r (a s casi maximal ~ N/r), el minimizador usa el MENOR r
disponible: lambda_Res(N) ≈ log r_star / log(N/r_star).

Conexión con la serie singular (hallazgo de este repo): la misma divisibilidad de
N por primos pequeños que AGRANDA 𝔖(N) (realce de Goldbach) BLOQUEA los canales de
r pequeño: si ℓ|N y ℓ|r entonces p = N - rs ≡ 0 (mod ℓ) => p=ℓ. Por eso r_star(N)
crece con 𝔖(N): el balance residual es PEOR justo donde Goldbach es localmente rico.

r_star se calcula EXACTO para todo N por cobertura de canales vía FFT.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

from sieve import Tables


def rstar_all(T: Tables, rcap: int = 300):
    """
    r_star(N) para todo N par en [4..X], exacto, por cobertura de canales
    C_r(N) = #{(p,s): p+rs=N} > 0. Devuelve (N_par, r_star) con r_star=0 si
    ningún r<=rcap cubre (rarísimo).
    """
    X = T.X
    isp = T.is_prime.astype(np.float64)
    primes = T.primes
    r_star = np.zeros(X + 1, dtype=np.int32)
    covered = np.zeros(X + 1, dtype=bool)
    even = np.zeros(X + 1, dtype=bool)
    even[4::2] = True
    for r in primes[primes <= rcap]:
        r = int(r)
        s_primes = primes[primes >= r]
        idx = r * s_primes
        idx = idx[idx <= X]
        M = np.zeros(X + 1, dtype=np.float64)
        M[idx] = 1.0
        C = np.rint(fftconvolve(isp, M)[: X + 1])
        newly = (C > 0) & (~covered) & even
        r_star[newly] = r
        covered |= newly
        if covered[even].all():
            break
    N = np.arange(4, X + 1, 2)
    return N, r_star[N]


def balance_array(T: Tables):
    """balance(q)=log(spf q)/log(q/spf q) para q semiprimo; +inf en otro caso."""
    X = T.X
    bal = np.full(X + 1, np.inf, dtype=np.float64)
    semi = np.nonzero(T.is_semiprime)[0]
    r = T.spf[semi].astype(np.float64)
    s = (semi // T.spf[semi]).astype(np.float64)
    bal[semi] = np.log(r) / np.log(s)
    return bal


def lambda_res_exact(T: Tables, Pmax: int = 5000):
    """
    lambda_Res(N) EXACTO para todo N par: min sobre primos p<=Pmax de balance(N-p).
    El minimizador usa el rep de mayor s (menor p) en el canal de menor r, así que
    p<=Pmax lo captura salvo en N rarísimos (se devuelve +inf si ningún p<=Pmax
    da semiprimo, lo que indica que hay que subir Pmax). Devuelve (N, lambda).
    """
    X = T.X
    bal = balance_array(T)
    N = np.arange(4, X + 1, 2)
    lam = np.full(len(N), np.inf, dtype=np.float64)
    for p in T.primes[T.primes <= Pmax]:
        p = int(p)
        q = N - p
        cand = np.where(q >= 4, bal[np.clip(q, 0, X)], np.inf)
        np.minimum(lam, cand, out=lam)
    return N, lam


def residual_exponent_exact(T: Tables, Nval: int):
    """lambda_Res(N) exacto por enumeración (del blueprint del draft). Para validar."""
    primes = T.primes
    best = np.inf
    best_tr = None
    for r in primes:
        if r * r > Nval:
            break
        # s recorre primos >= r con r*s < Nval
        smax = (Nval - 2) // r
        s_cands = primes[(primes >= r) & (primes <= smax)]
        q = r * s_cands
        p = Nval - q
        good = T.is_prime[p]
        if np.any(good):
            s_ok = s_cands[good]
            lam = np.log(r) / np.log(s_ok.astype(float))
            j = int(np.argmin(lam))
            if lam[j] < best:
                best = float(lam[j]); best_tr = (int(r), int(s_ok[j]))
        # poda: si ya tenemos best, r mayores solo empeoran (lambda crece con r)
        if best < np.inf and np.log(r) / np.log(Nval / r) > best:
            break
    return best, best_tr


if __name__ == "__main__":
    from heuristics import singular_series_goldbach
    T = Tables(2_000_000)
    N, rs = rstar_all(T)
    S = singular_series_goldbach(T.X, T.is_prime)
    _, lam = lambda_res_exact(T)
    print(f"X={T.X}")
    # distribución de r_star
    vals, cnts = np.unique(rs, return_counts=True)
    print("  r_star  share%")
    for v, c in zip(vals[:8], cnts[:8]):
        print(f"   {v:4d}   {100*c/len(N):6.2f}")
    fin = np.isfinite(lam) & (N >= 1000)
    lf = lam[fin]
    print(f"  lambda_Res(N>=1000): mediana={np.median(lf):.4f}  media={np.mean(lf):.4f}  "
          f"max={np.max(lf):.4f}  a_Res mediana={1+np.median(lf):.4f}")
    print(f"  N>=1000 sin rep Chen residual (p<=Pmax): {int(np.sum(~np.isfinite(lam)&(N>=1000)))}")
    # conexión con la serie singular: r_star vs S(N) y vs 3|N
    div3 = (N % 3 == 0)
    print(f"  r_star medio | 3∤N: {rs[~div3].mean():.3f}   | 3|N: {rs[div3].mean():.3f}")
    # correlación lambda_Res con log S(N)
    m = (N >= 20000) & np.isfinite(lam)
    cc = np.corrcoef(np.log(S[N[m]]), lam[m])[0, 1]
    print(f"  corr(lambda_Res, log 𝔖(N)) = {cc:+.3f}  (positiva = balance peor donde Goldbach rico)")
    # validación del cálculo exacto (small-p) vs enumeración completa
    rng = np.random.default_rng(0)
    sample = rng.choice(N[N > 1000], 60, replace=False)
    err = []
    idx = {int(n): i for i, n in enumerate(N)}
    for Nv in sample:
        ex, _ = residual_exponent_exact(T, int(Nv))
        if np.isfinite(lam[idx[int(Nv)]]):
            err.append(abs(ex - lam[idx[int(Nv)]]))
    print(f"  |small-p vs enumeración completa| en {len(err)} N: "
          f"media={np.mean(err):.5f} max={np.max(err):.5f}")
