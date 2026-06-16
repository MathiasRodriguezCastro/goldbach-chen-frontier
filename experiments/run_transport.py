"""
Transporte óptimo / defecto de reflexión (draft expandido §sec:transport).
Responde la pregunta de "colapso de transporte": ¿admitir semiprimos acerca la
nube reflejada a la distribución de primos? Genera figura 13 y escanea X.

Uso: python3 run_transport.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
import transport as tr
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    X = 2_000_000
    T = Tables(X)
    d = tr.transport_defects(T)
    alpha, beta2, b2s, b2b = tr.measures(T)
    refl_p = np.diff(np.concatenate([[0], tr._reflect_cdf(alpha)]))
    refl_s2 = np.diff(np.concatenate([[0], tr._reflect_cdf(beta2)]))
    path = plots.transport(tr.MID, alpha, refl_p, refl_s2, d)

    print(f"X={X}")
    print(f"  Delta_1={d['Delta_1']:.4f}  Delta_2={d['Delta_2']:.4f}  "
          f"Delta_<=2={d['Delta_le2']:.4f}  (lambda*={d['lambda_star']:.2f})")
    print(f"  colapso de transporte: {d['reduction_pct']:.1f}% de reducción "
          f"(reducción por factor {'CHICO' if d['Delta_2_smallfactor']<d['Delta_2_balanced'] else 'BALANCEADO'})")
    print(f"  figura: {os.path.relpath(path)}")

    # estabilidad en X
    print("  --- estabilidad: % de colapso vs X ---")
    scan = {}
    for Xs in (250_000, 1_000_000, 4_000_000):
        ds = tr.transport_defects(Tables(Xs))
        scan[Xs] = ds["reduction_pct"]
        print(f"   X={Xs:>9}: Delta_1={ds['Delta_1']:.4f} Delta_<=2={ds['Delta_le2']:.4f} "
              f"reducción={ds['reduction_pct']:.1f}% lambda*={ds['lambda_star']:.2f}")

    json.dump({"X": X, **d, "scan_reduction_pct": scan},
              open(os.path.join(DATA, "transport.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
