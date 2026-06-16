"""
Continuidad débil de las medidas de q/N (sección 4 del .tex: Question 1 y
Conjecture 3). La medida puntual mu_N es discontinua, pero ¿se estabiliza la
distribución de q/N al suavizar sobre ventanas N in [X, X+H]?

  - Panel A: mu_N individuales (ruidosas) vs densidad suavizada en ventana.
  - Panel B: densidad de q/N para q primo vs q semiprimo; ¿se acercan? ¿uniforme?
  - Panel C: distancia de variación total (TV) entre dos ventanas adyacentes vs H
             (si decrece -> convergencia débil).

Uso: python3 run_continuity.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from diagnostics import representations
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
NBINS = 50
BINS = np.linspace(0.0, 1.0, NBINS + 1)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def collect_xs(T, Nlo, Nhi, stride=2, offset=0):
    """q/N de las representaciones en la ventana, separadas por tipo.
    stride/offset permiten submuestrear N de la MISMA escala (para medir ruido)."""
    xp, xs = [], []
    start = (Nlo if Nlo % 2 == 0 else Nlo + 1) + 2 * offset
    for N in range(start, Nhi + 1, 2 * stride):
        p, q, typ, a = representations(T, N)
        x = q / N
        xp.append(x[typ == 1]); xs.append(x[typ == 2])
    return np.concatenate(xp), np.concatenate(xs)


def density(x):
    h, _ = np.histogram(x, bins=BINS, density=True)
    return h


def tv(d1, d2):
    """TV entre dos densidades (normalizadas a integral 1 sobre las celdas)."""
    p1 = d1 / d1.sum(); p2 = d2 / d2.sum()
    return 0.5 * np.abs(p1 - p2).sum()


def main():
    T = Tables(2_000_000)
    X0 = 1_000_000

    # --- Panel A/B: ventana de referencia ---
    H_ref = 4000  # nº de N pares
    xp, xs = collect_xs(T, X0, X0 + 2 * H_ref)
    dens_p = density(xp); dens_s = density(xs)
    tv_ps = tv(dens_p, dens_s)

    # medidas individuales (3 N aislados) -> ruidosas
    indiv = []
    for N in (X0 + 2, X0 + 1000, X0 + 5000):
        p, q, typ, a = representations(T, N)
        indiv.append((CENTERS, density((q[typ == 1]) / N)))

    # --- Panel C: estabilidad vs tamaño de ventana H ---
    # Dos submuestras de la MISMA escala (N alternos) -> TV = ruido muestral puro,
    # que debe decrecer ~1/sqrt(H) si la medida suavizada converge (conv. débil).
    Hs = [125, 250, 500, 1000, 2000, 4000]
    tv_stab = []
    for H in Hs:
        a_xp, _ = collect_xs(T, X0, X0 + 2 * H, stride=2, offset=0)  # N pares de la ventana
        b_xp, _ = collect_xs(T, X0, X0 + 2 * H, stride=2, offset=1)  # N impares (misma escala)
        tv_stab.append(tv(density(a_xp), density(b_xp)))

    path = plots.continuity(CENTERS, dens_p, dens_s, indiv, tv_ps, tv_stab, Hs)

    # cuán uniforme es la medida de q primo (TV contra la uniforme)
    tv_unif = tv(dens_p, np.ones_like(dens_p))
    # asimetría de cada tipo (media de q/N; 0.5 = simétrica)
    asym_p = float(np.mean(xp)); asym_s = float(np.mean(xs))

    print(f"figura: {os.path.relpath(path)}")
    print(f"  TV(primo, semiprimo)         = {tv_ps:.4f}")
    print(f"  TV(primo, uniforme)          = {tv_unif:.4f}   (q/N de q primo ~ uniforme)")
    print(f"  media q/N  primo={asym_p:.4f}  semiprimo={asym_s:.4f}  (0.5=simétrica)")
    print(f"  TV entre ventanas adyacentes vs H: {[round(t,4) for t in tv_stab]}")
    print(f"  -> {'DECRECE' if tv_stab[-1] < tv_stab[0] else 'no decrece'} con H "
          f"({tv_stab[0]:.4f} -> {tv_stab[-1]:.4f}): convergencia débil")

    json.dump({"tv_prime_semi": tv_ps, "tv_prime_uniform": tv_unif,
               "mean_qN_prime": asym_p, "mean_qN_semi": asym_s,
               "H": Hs, "tv_stability": tv_stab},
              open(os.path.join(DATA, "continuity.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
