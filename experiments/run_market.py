"""
Goldbach como mercado de matching (ideas #5/#6). Liquidez, spread de Chen y
resiliencia a shocks de oferta. Hallazgo: el sustituto semiprimo de Chen es liquidez
de BUEN TIEMPO (3x más liquido en normalidad) pero SIN prima de resiliencia en crisis,
porque el bien compuesto necesita 3 primos (p,r,s) vs 2 -> S_C(phi)=S_C(0)(1-phi)->0.
Genera figura 19.

Uso: python3 run_market.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from market import liquidity, surviving_liquidity, collapse_thresholds
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(2_000_000)
    N, R1, R2 = liquidity(T)
    m = N >= 20000
    R1m, R2m = R1[m], R2[m]
    S0 = float(np.mean(R2m / R1m))

    phis = np.linspace(0, 0.999, 200)
    LG = np.array([surviving_liquidity(R1m, R2m, p)[0].mean() for p in phis])
    LC = np.array([surviving_liquidity(R1m, R2m, p)[1].mean() for p in phis])

    idx = np.nonzero(m)[0][np.random.default_rng(0).choice(int(m.sum()), 400, replace=False)]
    pG, pC = collapse_thresholds(R1, R2, idx)
    premium = float((pC - pG).mean())

    path = plots.market(phis, LG, LC, S0, premium)
    print(f"X={T.X}")
    print(f"  liquidez normal: L_G={R1m.mean():.0f}  L_C={(R1m+R2m).mean():.0f}  spread S_C={S0:.2f}")
    print(f"  umbral colapso: phi_G={pG.mean():.4f}  phi_C={pC.mean():.4f}")
    print(f"  prima de resiliencia (phi_C-phi_G) = {premium:.4f}  (~0: Chen es liquidez de buen tiempo)")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"X": T.X, "L_G": float(R1m.mean()), "L_C": float((R1m + R2m).mean()),
               "spread_S0": S0, "phi_G": float(pG.mean()), "phi_C": float(pC.mean()),
               "resilience_premium": premium},
              open(os.path.join(DATA, "market.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
