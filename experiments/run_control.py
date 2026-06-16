"""
Goldbach como problema de control (idea #11). ¿Qué política simple aclara el mercado
(halla un socio primo) más rápido? Una política CONSCIENTE DE RESIDUOS que usa N mod
(primos <= B) para evitar q=N-p divisible por primos chicos casi duplica el éxito y
reduce a la mitad el costo. Cuantifica el valor de la información aritmética local (la
serie singular) para el controlador. Genera figura 20.

Uso: python3 run_control.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from control import policies_window
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(2_000_000)
    Bs = (0, 3, 5, 7, 11, 13)
    _, out = policies_window(T, 1_000_000, 1_050_000, Bs=Bs)
    success = [out[B]["success"] for B in Bs]
    cost = [out[B]["cost_mean"] for B in Bs]

    path = plots.control(Bs, success, cost)
    print("  B (info)   éxito single-shot   costo secuencial medio")
    for B in Bs:
        tag = "naive" if B == 0 else f"coprimo<= {B}"
        print(f"   {B:>2} ({tag:11s})   {100*out[B]['success']:5.1f}%            {out[B]['cost_mean']:.2f}")
    print(f"  -> éxito {100*success[0]:.0f}%->{100*success[-1]:.0f}%, "
          f"costo {cost[0]:.2f}->{cost[-1]:.2f} ({100*(1-cost[-1]/cost[0]):.0f}% menos)")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"B": list(Bs), "success": success, "cost_mean": cost,
               "out": {str(B): out[B] for B in Bs}},
              open(os.path.join(DATA, "control.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
