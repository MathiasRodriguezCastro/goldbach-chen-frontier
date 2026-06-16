"""
Goldbach en bases primoriales y geometría de acarreos (menú de "bases").

(A) Cobertura primorial. En base B_y = prod_{p<=y} p, todo primo grande vive en
    (Z/B_y)^x. Los residuos admisibles para N son
       A_y(N) = {a in (Z/B_y)^x : N-a in (Z/B_y)^x},
    y la cobertura local lambda_y(N)=|A_y(N)|/phi(B_y) factoriza por CRT como
       lambda_y(N) = prod_{3<=p<=y, p∤N} (p-2)/(p-1),
    de modo que lambda_y(N)/lambda_y(generico) = prod_{p|N,p<=y}(p-1)/(p-2) = 𝔖_y(N):
    ¡la cobertura primorial ES la serie singular truncada! La fracción del cometa que
    explica crece con y (= la saturación espectral: mod 6 ~85%, ... -> 99.9%).

(B) Geometría de acarreos (nuevo). En base b, #acarreos(p, N-p) = (S_b(p)+S_b(N-p)-S_b(N))/(b-1)
    con S_b la suma de dígitos. En base 2 esto es v_2(C(N,p)) (Kummer). Las representaciones
    de Goldbach tienen un EXCESO sistemático de acarreos sobre pares aleatorios: una firma
    digital, de origen aritmético (el último dígito del primo es coprimo con b). Se mide el
    exceso por base y la entropía de acarreo.
"""
from __future__ import annotations
import numpy as np

from sieve import Tables


def digit_sum(n, b):
    s = 0
    while n > 0:
        s += n % b; n //= b
    return s


def singular_truncated(N, y, is_prime):
    """lambda_y(N)/lambda_y(generico) = prod_{p|N, 3<=p<=y}(p-1)/(p-2) (cobertura relativa)."""
    odd_primes = np.nonzero(is_prime[:y + 1])[0]
    odd_primes = odd_primes[odd_primes > 2]
    val = 1.0
    for p in odd_primes:
        if N % p == 0:
            val *= (p - 1.0) / (p - 2.0)
    return val


def coverage_variance_explained(N, R1, T, ys=(3, 5, 7, 11, 13, 30)):
    """R^2 de log R1 explicado por la cobertura primorial 𝔖_y(N) (destendenciado)."""
    import pandas as pd
    m = N >= 20000
    Nn, R1n = N[m], R1[m]
    y = np.log(R1n.astype(float))
    y = y - pd.Series(y).rolling(4001, center=True, min_periods=2000).mean().values
    ok = np.isfinite(y); Nn, y = Nn[ok], y[ok]
    out = {}
    for Y in ys:
        Sy = np.array([singular_truncated(int(n), Y, T.is_prime) for n in Nn])
        x = np.log(Sy)
        b, a = np.polyfit(x, y, 1)
        out[Y] = 1 - np.var(y - (a + b * x)) / np.var(y)
    return out


def carry_excess(T, Nlo, Nhi, bases, stride=3, seed=0):
    """Exceso medio de acarreos (Goldbach - aleatorio) y entropía por base."""
    isp = T.is_prime; primes = T.primes
    rng = np.random.default_rng(seed)
    Ns = list(range(Nlo if Nlo % 2 == 0 else Nlo + 1, Nhi + 1, 2))[::stride]
    res = {}
    for b in bases:
        cg, cr, ent = [], [], []
        for N in Ns:
            pe = primes[primes <= N // 2]; pe = pe[isp[N - pe]]
            if len(pe) < 5:
                continue
            SN = digit_sum(int(N), b)
            c = np.array([(digit_sum(int(p), b) + digit_sum(int(N - p), b) - SN) / (b - 1) for p in pe])
            cg.append(c.mean())
            aa = rng.integers(2, N - 1, size=len(pe))
            cr.append(np.mean([(digit_sum(int(a), b) + digit_sum(int(N - a), b) - SN) / (b - 1) for a in aa]))
            v, ct = np.unique(c, return_counts=True); pmf = ct / ct.sum()
            ent.append(-(pmf * np.log(pmf)).sum())
        res[b] = {"gold": float(np.mean(cg)), "rand": float(np.mean(cr)),
                  "excess": float(np.mean(cg) - np.mean(cr)), "entropy": float(np.mean(ent))}
    return res


def carry_distribution(T, Nlo, Nhi, b, stride=1, seed=0):
    """Histograma agregado de #acarreos (Goldbach vs aleatorio) sobre la ventana, base b."""
    isp = T.is_prime; primes = T.primes
    rng = np.random.default_rng(seed)
    Ns = list(range(Nlo if Nlo % 2 == 0 else Nlo + 1, Nhi + 1, 2))[::stride]
    cg, cr = [], []
    for N in Ns:
        pe = primes[primes <= N // 2]; pe = pe[isp[N - pe]]
        if len(pe) < 5:
            continue
        SN = digit_sum(int(N), b)
        cg.extend([(digit_sum(int(p), b) + digit_sum(int(N - p), b) - SN) // (b - 1) for p in pe])
        aa = rng.integers(2, N - 1, size=len(pe))
        cr.extend([(digit_sum(int(a), b) + digit_sum(int(N - a), b) - SN) // (b - 1) for a in aa])
    mx = max(max(cg), max(cr)) + 1
    hg = np.bincount(cg, minlength=mx) / len(cg)
    hr = np.bincount(cr, minlength=mx) / len(cr)
    return np.arange(mx), hg, hr


if __name__ == "__main__":
    from counts import representation_counts
    T = Tables(2_000_000)
    N, R1, R2 = representation_counts(T)
    cov = coverage_variance_explained(N, R1, T)
    print("(A) Cobertura primorial 𝔖_y(N) explica de la oscilación de log R1:")
    for Y, r in cov.items():
        print(f"    y<= {Y:2d} (B_y primorial): R2={r:.4f}")
    car = carry_excess(T, 1_000_000, 1_006_000, [2, 3, 10])
    print("(B) Geometría de acarreos:")
    for b, r in car.items():
        print(f"    base {b:2d}: <carries> Goldbach={r['gold']:.3f} aleatorio={r['rand']:.3f} "
              f"EXCESO={r['excess']:+.3f}  H_acarreo={r['entropy']:.3f}")
