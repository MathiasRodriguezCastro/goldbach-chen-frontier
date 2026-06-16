"""
Capas de complejidad aritmética (draft expandido §3): perfil rho_N(k) y, sobre
todo, el EXPONENTE SINGULAR beta_k de cada capa R_k (q con Omega(q)=k).

Hallazgo: beta_1=1, beta_2~1/2, y beta_k se vuelve NEGATIVO para k>=3: la serie
singular de Goldbach realza las capas bajas y SUPRIME las altas. Mecanismo: si
S(N) es grande (N divisible por primos chicos), q=N-p evita esos primos pequeños
y tiende a tener menos factores.

Uso: python3 run_layers.py [X]   (default 2_000_000)
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from heuristics import singular_series_goldbach
from layers import layer_counts, singular_exponent_layer, composite_composite_count
import plots

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main(X):
    T = Tables(X)
    Kmax = 7
    N, R = layer_counts(T, Kmax=Kmax)
    S = singular_series_goldbach(T.X, T.is_prime)
    m = N >= 20000
    tot = sum(R[k] for k in R)

    ks, shares, betas, r2s = [], [], [], []
    rows = []
    print(f"X={X}")
    print("  k   share%    beta_k    R^2")
    for k in sorted(R):
        share = 100.0 * R[k][m].sum() / tot[m].sum()
        if k == 1:
            bk, r2 = 1.0, 1.0
        else:
            bk, r2 = singular_exponent_layer(N, R[k], R[1], S)
        ks.append(k); shares.append(share); betas.append(bk); r2s.append(r2)
        rows.append({"k": k, "share_pct": share, "beta_k": bk, "R2": r2})
        print(f"  {k}   {share:6.2f}   {bk:+.3f}   {r2:.3f}")

    # Composite-composite rescue (draft §3)
    Ncc, Rcc = composite_composite_count(T)
    cc_pos = float(np.mean(Rcc[Ncc >= 20000] > 0))
    cc_mean = float(np.mean(Rcc[Ncc >= 20000]))
    print(f"  R_cc (q = compuesto×compuesto): media={cc_mean:.1f}, "
          f"fracción N con R_cc>0 = {cc_pos:.4f}")

    path = plots.layers(ks, shares, betas, r2s)
    print("figura:", os.path.relpath(path))

    json.dump({"X": X, "layers": rows, "Rcc_mean": cc_mean, "Rcc_pos_frac": cc_pos},
              open(os.path.join(DATA, "layers.json"), "w"), indent=2)


if __name__ == "__main__":
    X = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    main(X)
