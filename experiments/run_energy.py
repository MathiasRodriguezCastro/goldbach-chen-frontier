"""
Energía de estado fundamental aritmética y su termodinámica (idea favorita del
usuario, que subsume capas / función de partición / semianillos / grado del grafo).

  E(N) = min_{a+b=N}(Omega(a)+Omega(b)),  Goldbach <=> E(N)=2.
  Espectro n_E(N): n_2=R_1 (fundamental), n_3=2R_2 (primer excitado).
  Termodinámica Z_N(beta)=(f*f)(N): energía media U, calor específico C, beta*.

Hallazgo: beta* (temperatura de fusión del orden de Goldbach) está fuertemente
anti-correlacionado con la serie singular (donde S grande, más estados fundamentales,
funde a beta menor).

Uso: python3 run_energy.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from heuristics import singular_series_goldbach
from energy import thermodynamics, energy_spectrum, crossover_beta
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    T = Tables(1_000_000)
    betas = np.linspace(0.05, 6.0, 70)
    N, R1, th = thermodynamics(T, betas)
    bstar = crossover_beta(betas, th["C"])
    S = singular_series_goldbach(T.X, T.is_prime)[N]
    m = N >= 20000

    # espectro de un N representativo
    specE, specN = energy_spectrum(T, 500000, Emax=16)
    # curvas U, C promediadas sobre N (en m)
    Ubar = th["U"][:, m].mean(axis=1)
    Cbar = th["C"][:, m].mean(axis=1)

    path = plots.energy(specE[2:], specN[2:], betas, Ubar, Cbar,
                        bstar[m].mean(), np.log(S[m]), bstar[m])

    corr = float(np.corrcoef(np.log(S[m]), bstar[m])[0, 1])
    print(f"X={T.X}")
    print(f"  energía típica U(beta~0) = {Ubar[0]:.2f}   fundamental E(N)=2 (Goldbach)")
    print(f"  gap típico->fundamental = {Ubar[0]-2:.2f}  (~2loglogN-2 = {2*np.log(np.log(T.X))-2:.2f})")
    print(f"  beta* (fusión): media={bstar[m].mean():.3f}  rango=[{bstar[m].min():.2f},{bstar[m].max():.2f}]")
    print(f"  corr(beta*, log S(N)) = {corr:+.3f}   (S grande -> funde antes)")
    print(f"  degeneración fundamental media R_1 = {R1[m].mean():.0f}")
    print(f"  figura: {os.path.relpath(path)}")

    json.dump({"X": T.X, "U_typical": float(Ubar[0]), "beta_star_mean": float(bstar[m].mean()),
               "corr_bstar_logS": corr, "R1_mean": float(R1[m].mean()),
               "spectrum_N500000": {int(e): int(c) for e, c in zip(specE, specN)}},
              open(os.path.join(DATA, "energy.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
