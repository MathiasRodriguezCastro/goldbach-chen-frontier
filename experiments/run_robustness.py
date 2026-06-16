"""
Robustez adversarial del grafo de Goldbach (ideas #2 grafo + #13 robustez).

Goldbach es HIPER-ROBUSTO a deleción aleatoria de primos (sobrevive ~97%) pero
CATASTRÓFICAMENTE FRÁGIL a un ataque modular: quitar una clase de residuo mod 3
(la mitad de los primos) destruye ~60% de los N, porque los primos sobrevivientes
(clase 2) solo suman a N≡1 mod 3 — una obstrucción LOCAL creada por el adversario.
Solo m=3,4 dan obstrucción (mod 5+ el sumset cubre todo). Genera figura 18.

Uso: python3 run_robustness.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from robustness import window_reps, random_damage, modular_damage, surviving_sumset
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def class1_fraction(T, m):
    """Fracción de primos ≡1 mod m (lo que elimina el ataque modular)."""
    p = T.primes
    return float(np.mean(p % m == 1))


def main():
    T = Tables(1_100_000)
    Ns, reps = window_reps(T, 1_000_000, 1_030_000)
    d = np.array([len(r) for r in reps])
    nprimes = len(T.primes)
    kappa_frac = float(d.min()) / nprimes

    phis = np.linspace(0, 0.999, 120)
    rand = random_damage(reps, phis)

    modpts = []
    results = {}
    for m, a, col, lab in [(3, 1, "#c0392b", "modular mod 3"),
                           (4, 1, "#e67e22", "modular mod 4"),
                           (5, 1, "#7f8c8d", "modular mod 5")]:
        frac, by_class, _ = modular_damage(T, Ns, reps, m, a)
        x = class1_fraction(T, m)
        units, sums, missed = surviving_sumset(m, a)
        modpts.append((x, frac, lab, col))
        results[f"mod{m}"] = {"removed_frac": x, "killed_frac": frac,
                              "surviving_units": units, "sumset": sums, "obstructed": missed}
        print(f"  modular clase 1 mod {m}: quita {100*x:.1f}% primos -> mata {100*frac:.1f}% N; "
              f"sobreviven {units}, sumset={sums}, obstruidas={missed}")

    _, by_class3, _ = modular_damage(T, Ns, reps, 3, 1)

    path = plots.robustness(phis, rand, modpts, kappa_frac, by_class3)
    print(f"  ALEATORIO: % N muertos @ phi=0.5/0.9/0.97/0.99 = "
          f"{[round(100*random_damage(reps,[p])[0],2) for p in (0.5,0.9,0.97,0.99)]}")
    print(f"  DIRIGIDO: kappa={int(d.min())} primos = {100*kappa_frac:.2f}% rompe Goldbach")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"window": [int(Ns[0]), int(Ns[-1])], "n_primes": nprimes,
               "kappa": int(d.min()), "kappa_frac": kappa_frac,
               "random_phi_0.97": float(random_damage(reps, [0.97])[0]),
               "modular": results, "byclass_mod3": by_class3},
              open(os.path.join(DATA, "robustness.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
