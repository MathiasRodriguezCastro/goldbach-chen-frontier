"""
Formulación de transporte óptimo de la frontera Goldbach--Chen (draft expandido
§sec:transport).

Bajo la reflexión T(x)=1-x (que lleva p/N a q/N=1-p/N), se comparan:
  alpha   = distribución de p/N sobre los primos p<=N,
  beta_k  = distribución de q/N sobre los k-casi-primos q<=N (Omega(q)=k).
El defecto de reflexión de la capa k es
  Delta_k = W1( alpha, T# beta_k ),
y la versión de Chen mezcla las capas 1 y 2:
  Delta_<=2 = min_{lambda in [0,1]} W1( alpha, T#(lambda beta_1 + (1-lambda) beta_2) ).
Como lambda=1 reproduce Delta_1, siempre Delta_<=2 <= Delta_1; la pregunta del draft
es si la MEZCLA con semiprimos REDUCE el defecto ("colapso de transporte"), y si la
reducción viene de semiprimos con factor chico o balanceados.

W1 en [0,1] = integral |CDF_a - CDF_b|. La reflexión cumple CDF_{T#nu}(x)=1-CDF_nu(1-x).
"""
from __future__ import annotations
import numpy as np

from sieve import Tables

NB = 400
EDGES = np.linspace(0.0, 1.0, NB + 1)
MID = 0.5 * (EDGES[:-1] + EDGES[1:])
DX = 1.0 / NB


def _hist(x):
    h, _ = np.histogram(x, bins=EDGES, density=False)
    s = h.sum()
    return h / s if s > 0 else h.astype(float)


def _cdf(p):
    return np.cumsum(p)


def _reflect_cdf(p):
    """CDF de la medida reflejada T#nu, T(x)=1-x: CDF_{T#}(x)=1-CDF(1-x)."""
    # en la grilla de bins: reflejar el histograma y acumular
    return np.cumsum(p[::-1])


def w1(cdf_a, cdf_b):
    return float(np.sum(np.abs(cdf_a - cdf_b)) * DX)


def measures(T: Tables):
    """alpha=p/N (primos), beta1=q/N (primos, = alpha), beta2 (semiprimos) y
    los sub-perfiles de semiprimos con factor chico (B<0.2) vs balanceado (B>0.4)."""
    X = T.X
    primes = T.primes
    alpha = _hist(primes / X)
    semis = np.nonzero(T.is_semiprime)[0]
    a = T.spf[semis].astype(float); b = (semis // T.spf[semis]).astype(float)
    B = np.log(a) / np.log(semis.astype(float))
    beta2 = _hist(semis / X)
    beta2_small = _hist(semis[B < 0.2] / X)
    beta2_bal = _hist(semis[B > 0.4] / X)
    return alpha, beta2, beta2_small, beta2_bal


def transport_defects(T: Tables):
    alpha, beta2, b2s, b2b = measures(T)
    cdf_alpha = _cdf(alpha)
    # Delta_1: alpha vs reflexión de alpha (=beta1)
    d1 = w1(cdf_alpha, _reflect_cdf(alpha))
    # Delta_2: alpha vs reflexión de beta2
    d2 = w1(cdf_alpha, _reflect_cdf(beta2))
    # Delta_<=2: mezcla óptima lambda*beta1 + (1-lambda)*beta2
    ref1 = _reflect_cdf(alpha); ref2 = _reflect_cdf(beta2)
    lams = np.linspace(0, 1, 501)
    vals = [w1(cdf_alpha, lam * ref1 + (1 - lam) * ref2) for lam in lams]
    j = int(np.argmin(vals))
    d_le2 = vals[j]; lam_star = lams[j]
    # ¿de qué semiprimos viene la reducción? defecto con sub-perfiles
    d2_small = w1(cdf_alpha, _reflect_cdf(b2s))
    d2_bal = w1(cdf_alpha, _reflect_cdf(b2b))
    return {"Delta_1": d1, "Delta_2": d2, "Delta_le2": d_le2, "lambda_star": float(lam_star),
            "reduction_pct": 100 * (d1 - d_le2) / d1,
            "Delta_2_smallfactor": d2_small, "Delta_2_balanced": d2_bal}


if __name__ == "__main__":
    T = Tables(2_000_000)
    r = transport_defects(T)
    print(f"X={T.X}")
    for k, v in r.items():
        print(f"  {k:22s} {v:.5f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  -> colapso de transporte: Delta_<=2 {'<' if r['Delta_le2']<r['Delta_1'] else '>='} Delta_1 "
          f"({r['reduction_pct']:.1f}% de reducción, lambda*={r['lambda_star']:.2f})")
    print(f"  -> reducción dominada por semiprimos de factor "
          f"{'CHICO' if r['Delta_2_smallfactor']<r['Delta_2_balanced'] else 'BALANCEADO'} "
          f"(Δ2 factor-chico={r['Delta_2_smallfactor']:.4f} vs balanceado={r['Delta_2_balanced']:.4f})")
