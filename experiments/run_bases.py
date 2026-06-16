"""
Goldbach en bases primoriales + geometría de acarreos (menú de "bases").

(A) Cobertura primorial 𝔖_y(N): la cobertura local en base B_y ES la serie singular
    truncada; explica el cometa cada vez mejor al crecer y (84.6% con B_6 -> 99.8%).
(B) Geometría de acarreos: las representaciones de Goldbach tienen un EXCESO sistemático
    de acarreos sobre pares aleatorios (firma digital, origen aritmético). En base 2 el
    nº de acarreos = v_2(C(N,p)) (Kummer). Genera figura 24.

Uso: python3 run_bases.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from counts import representation_counts
from bases import coverage_variance_explained, carry_excess, carry_distribution
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(2_000_000)
    N, R1, R2 = representation_counts(T)
    cov = coverage_variance_explained(N, R1, T)
    car = carry_excess(T, 1_000_000, 1_006_000, [2, 3, 4, 6, 8, 10, 12, 16])
    kcar, hg, hr = carry_distribution(T, 1_000_000, 1_004_000, 2)

    ebases = list(car.keys()); evals = [car[b]["excess"] for b in ebases]
    path = plots.bases(list(cov.keys()), list(cov.values()), kcar, hg, hr, ebases, evals)

    print("(A) cobertura primorial 𝔖_y explica de la oscilación de log R1:")
    for y, r in cov.items():
        print(f"    B_y (y<= {y:2d}): {100*r:.1f}%")
    print("(B) exceso de acarreos (Goldbach - aleatorio) por base:")
    for b in ebases:
        print(f"    base {b:2d}: {car[b]['excess']:+.3f}  (Goldbach {car[b]['gold']:.2f} vs azar {car[b]['rand']:.2f})")
    gm = float(np.average(kcar, weights=hg)); rm = float(np.average(kcar, weights=hr))
    print(f"  base 2: <acarreos> Goldbach={gm:.3f} vs aleatorio={rm:.3f}  (= v_2 C(N,p), Kummer)")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"coverage": {str(k): v for k, v in cov.items()},
               "carry_excess": {str(b): car[b] for b in ebases},
               "base2_gold_mean": gm, "base2_rand_mean": rm},
              open(os.path.join(DATA, "bases.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
