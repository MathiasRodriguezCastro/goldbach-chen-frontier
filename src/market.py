"""
Goldbach como mercado de matching (ideas #5/#6 económicas).

Cada N par es un mercado que necesita dos insumos primos p+q=N. La liquidez es el
número de matches:
    L_G(N)=R_1(N)            (Goldbach: dos bienes primos),
    L_C(N)=R_1(N)+R_2(N)     (Chen: un bien puede ser semiprimo — sustituto compuesto),
    spread de Chen S_C(N)=R_2(N)/R_1(N)  (prima del sustituto).

Bajo un SHOCK DE OFERTA (cada primo desaparece con prob phi), un match de Goldbach
sobrevive con prob (1-phi)^2 (dos primos), pero uno de Chen con (1-phi)^3 (p y los dos
factores r,s del semiprimo). Resultado: el sustituto de Chen es liquidez de BUEN TIEMPO
— abundante en normalidad pero sin prima de resiliencia en crisis, porque el bien
compuesto requiere más sub-insumos y es más frágil cuando los primos escasean:
    S_C(phi) = S_C(0)·(1-phi)  ->  0.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq

from sieve import Tables
from counts import representation_counts


def liquidity(T: Tables):
    N, R1, R2 = representation_counts(T)
    return N, R1, R2


def surviving_liquidity(R1, R2, phi):
    """Liquidez esperada sobreviviente al shock phi."""
    a = (1 - phi) ** 2; b = (1 - phi) ** 3
    return R1 * a, R1 * a + R2 * b


def collapse_thresholds(R1, R2, sample_idx):
    """phi donde la liquidez esperada cae por debajo de 1 (Goldbach y Chen)."""
    phiG = 1 - 1 / np.sqrt(R1[sample_idx].astype(float))
    phiC = []
    for i in sample_idx:
        r1, r2 = float(R1[i]), float(R2[i])
        f = lambda a: r1 * a + r2 * a ** 1.5 - 1
        a = brentq(f, 1e-12, 1.0)
        phiC.append(1 - np.sqrt(a))
    return phiG, np.array(phiC)


if __name__ == "__main__":
    T = Tables(2_000_000)
    N, R1, R2 = liquidity(T)
    m = N >= 20000
    R1m, R2m = R1[m], R2[m]
    print(f"X={T.X}")
    print(f"  liquidez normal: L_G={R1m.mean():.0f}  L_C={(R1m+R2m).mean():.0f}  "
          f"spread S_C={np.mean(R2m/R1m):.2f}")
    for phi in (0.0, 0.5, 0.8, 0.9):
        LG, LC = surviving_liquidity(R1m, R2m, phi)
        sc = (LC - LG) / LG
        print(f"  phi={phi}: L_G={LG.mean():.0f}  L_C={LC.mean():.0f}  "
              f"spread S_C(phi)={sc.mean():.3f}  (= S_C(0)(1-phi)={np.mean(R2m/R1m)*(1-phi):.3f})")
    idx = np.nonzero(m)[0][np.random.default_rng(0).choice(int(m.sum()), 300, replace=False)]
    pG, pC = collapse_thresholds(R1, R2, idx)
    print(f"  umbral de colapso: phi_G={pG.mean():.4f}  phi_C={pC.mean():.4f}  "
          f"prima de resiliencia={ (pC-pG).mean():.4f} (~0: Chen no ayuda en crisis)")
