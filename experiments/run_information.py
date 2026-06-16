"""
Contenido de información del cometa de Goldbach (#9). Presupuesto de varianza
(tendencia / serie singular / residuo) y blancura del residuo. Genera figura 22.

Uso: python3 run_information.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from counts import representation_counts
from information import budget, whiteness
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(2_000_000)
    N, R1, R2 = representation_counts(T)
    fr, resid, yt, mR1 = budget(N, R1)
    freq, P, ac1 = whiteness(resid)

    path = plots.information(fr, freq, P)
    print(f"X={T.X}")
    print(f"  presupuesto: tendencia {100*fr['trend']:.1f}%  "
          f"𝔖 {100*fr['singular']:.2f}%  irreducible {100*fr['irreducible']:.3f}%")
    print(f"  comprimible (tend+𝔖): {100*(fr['trend']+fr['singular']):.2f}%")
    print(f"  residuo: autocorr lag-1={ac1:+.4f} (blanco)  std={100*np.sqrt(np.var(resid)):.2f}% "
          f"(Poisson {100/np.sqrt(mR1):.2f}%)")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"X": T.X, "trend": float(fr["trend"]), "singular": float(fr["singular"]),
               "irreducible": float(fr["irreducible"]), "resid_ac1": ac1,
               "resid_std": float(np.sqrt(np.var(resid))), "poisson_floor": float(1/np.sqrt(mR1))},
              open(os.path.join(DATA, "information.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
