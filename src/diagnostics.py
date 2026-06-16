"""
Diagnósticos de la frontera Goldbach--Chen (sección 3 del .tex y extras).

Arrays vectorizados a partir de (N, R1, R2):
  theta(N) = R1/(R1+R2)            cuota de representaciones primo--primo
  C(N)     = R2/(R1+1)             excedente de Chen
  H(N)     = entropía binaria de theta
  fragility_set                    enteros k-frágiles
  A1, A2                           conteos normalizados (con/sin serie singular)

Enumeración explícita por N (usa spf) para representaciones individuales:
  representations(N)               lista de (p, q, tipo, a=factor menor de q)
  balance_distribution(...)        distribución de B(q)=log a/log q (ponderada)
  least_rescue(N)                  L1(N), L2(N) (menor q primo / semiprimo)
"""
from __future__ import annotations
import numpy as np

from sieve import Tables


# ----------------------------------------------------------------------
# Diagnósticos vectorizados (todo el rango)
# ----------------------------------------------------------------------
def theta(R1, R2):
    denom = R1 + R2
    out = np.where(denom > 0, R1 / np.maximum(denom, 1), 0.0)
    return out


def chen_surplus(R1, R2):
    return R2 / (R1 + 1.0)


def binary_entropy(th):
    th = np.clip(th, 1e-15, 1 - 1e-15)
    return -th * np.log(th) - (1 - th) * np.log(1 - th)


def fragility_set(N, R1, R2, k: int):
    """N par con R1<=k y R2>0 (Goldbach escaso pero Chen disponible)."""
    mask = (R1 <= k) & (R2 > 0)
    return N[mask]


# ----------------------------------------------------------------------
# Enumeración explícita por N
# ----------------------------------------------------------------------
def representations(T: Tables, Nval: int, semiprime_only: bool = False):
    """Devuelve array estructurado de representaciones N=p+q con p primo."""
    primes = T.primes
    primes = primes[primes < Nval]
    q = Nval - primes
    is_p = T.is_prime[q]
    is_s = T.is_semiprime[q]
    keep = is_s if semiprime_only else (is_p | is_s)
    p = primes[keep]
    qq = q[keep]
    typ = np.where(T.is_prime[qq], 1, 2)
    a = np.where(typ == 2, T.spf[qq], qq)  # factor menor de q (q mismo si es primo)
    return p, qq, typ, a


def least_rescue(T: Tables, Nval: int):
    """L1(N)=menor q primo, L2(N)=menor q semiprimo (o None)."""
    p, q, typ, a = representations(T, Nval)
    qp = q[typ == 1]
    qs = q[typ == 2]
    L1 = int(qp.min()) if qp.size else None
    L2 = int(qs.min()) if qs.size else None
    return L1, L2


def balance_distribution(T: Tables, weighted: bool = True):
    """
    Distribución global de B(q)=log a/log q sobre TODAS las representaciones
    de Chen con q semiprimo y p+q<=X par. Vectorizada: para cada semiprimo q,
    su peso es el nº de primos p de igual paridad con p<=X-q (= cuántas
    representaciones usan ese q). Si weighted=False, cada semiprimo cuenta 1.
    Devuelve (B_values, weights).
    """
    X = T.X
    # nº acumulado de primos impares <= m, y disponibilidad del primo 2.
    odd_prime = T.is_prime.copy()
    odd_prime[2] = False
    cum_oddprimes = np.cumsum(odd_prime)  # cum_oddprimes[m] = #primos impares <= m

    semis = np.nonzero(T.is_semiprime)[0]
    a = T.spf[semis].astype(np.float64)
    B = np.log(a) / np.log(semis.astype(np.float64))

    if not weighted:
        return B, np.ones_like(B)

    w = np.zeros(len(semis), dtype=np.float64)
    even_semi = (semis % 2) == 0           # q par semiprimo = 2·primo  -> empareja con p=2
    # q par: p debe ser par y primo => p=2, y 2<=X-q  => q<=X-2
    w[even_semi & (semis <= X - 2)] = 1.0
    # q impar: p impar primo, p<=X-q  => nº primos impares <= X-q
    odd_semi = ~even_semi
    lim = X - semis[odd_semi]
    lim = np.clip(lim, 0, X)
    w[odd_semi] = cum_oddprimes[lim]
    keep = w > 0
    return B[keep], w[keep]


if __name__ == "__main__":
    from counts import representation_counts
    from heuristics import singular_series_goldbach, hl_prediction_R1, refined_prediction_R2

    T = Tables(300000)
    N, R1, R2 = representation_counts(T)
    th = theta(R1, R2)
    C = chen_surplus(R1, R2)
    print(f"X={T.X}")
    print(f"  theta medio = {th[N>=1000].mean():.4f}  (cuota primo--primo)")
    print(f"  C(N) medio  = {C[N>=1000].mean():.4f}  (excedente de Chen)")
    for k in (1, 2, 3, 5):
        print(f"  |F_{k}| (k-frágiles) = {len(fragility_set(N,R1,R2,k))}")
    L1, L2 = least_rescue(T, 100000)
    print(f"  N=100000: L1={L1}  L2={L2}  (menor q primo / semiprimo)")
    B, w = balance_distribution(T, weighted=True)
    bmean = np.average(B, weights=w)
    print(f"  B(q) medio (ponderado) = {bmean:.4f}   (1/2=balanceado, →0 factor chico)")
    print(f"  masa con B(q)<0.1 = {w[B<0.1].sum()/w.sum():.3f};  con B(q)>0.4 = {w[B>0.4].sum()/w.sum():.3f}")
