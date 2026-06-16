"""
Geometría discreta de la superficie de Chen p+rs=N (#14). Nube 2D de soluciones en
coordenadas log (u,v) y espectro de exponentes a=1+log r/log s. Las soluciones se
concentran cerca del borde de Goldbach (a->1). Genera figura 23.

Uso: python3 run_geometry.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from geometry import accumulate
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(2_000_000)
    H2, aedges, a_hist, npts = accumulate(T, 1_000_000, 1_005_000)
    amid = 0.5 * (aedges[:-1] + aedges[1:])
    amean = float(np.average(amid, weights=a_hist))
    cum = np.cumsum(a_hist) / a_hist.sum()
    amed = float(amid[np.searchsorted(cum, 0.5)])
    f_gold = float(a_hist[amid < 1.3].sum() / a_hist.sum())
    f_bal = float(a_hist[amid > 1.7].sum() / a_hist.sum())

    path = plots.geometry(H2, amid, a_hist, amed)
    print(f"  {npts} soluciones Chen sobre la superficie")
    print(f"  exponente a=1+log r/log s: media={amean:.3f}  mediana={amed:.3f}")
    print(f"  concentración Goldbach (a<1.3): {100*f_gold:.1f}%   balanceadas (a>1.7): {100*f_bal:.1f}%")
    print(f"  -> la nube se concentra cerca del borde r=1 (Goldbach); mediana muy < 1.9 (Li-Liu)")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"npts": int(npts), "a_mean": amean, "a_median": amed,
               "frac_a_lt_1.3": f_gold, "frac_a_gt_1.7": f_bal},
              open(os.path.join(DATA, "geometry.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
