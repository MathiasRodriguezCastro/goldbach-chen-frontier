"""
¿La deriva del exponente singular beta de R2 se detiene en 1/2 o sigue subiendo?

Mide beta LOCALMENTE en una ventana [X, X+W] (no acumulado), para X hasta ~3·10^8,
usando la convolución por bloques de src/segmented.py. beta local = pendiente de
log(R2/R1) vs log S(N) sobre los N de la ventana.

Uso: python3 run_scaling.py
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from segmented import windowed_counts
from heuristics import mertens_weight_W, TWIN_PRIME_C2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def beta_local(N, R1, R2, S):
    x = np.log(S)
    y = np.log(R2 / R1)
    b, a = np.polyfit(x, y, 1)
    r2 = 1.0 - np.var(y - (a + b * x)) / np.var(y)
    return b + 1.0, r2


def main():
    W = 2_000_000
    Xs = [int(v) for v in (6e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9)]  # múltiplos de W
    rows = []
    for X in Xs:
        t = time.time()
        N, R1, R2, S = windowed_counts(X, W)
        # algunos N pueden tener R1=0? Goldbach se cumple; por robustez filtramos.
        ok = R1 > 0
        N, R1, R2, S = N[ok], R1[ok], R2[ok], S[ok]
        b, r2 = beta_local(N, R1, R2, S)
        # CV del modelo de media potencia en la ventana
        Wt = mertens_weight_W(N, _is_prime_cache(X, W))
        base = N / np.log(N.astype(float)) ** 2
        model = 2 * TWIN_PRIME_C2 * S ** 0.5 * base * Wt
        cv = float(np.std(R2 / model) / np.mean(R2 / model))
        dt = time.time() - t
        rows.append({"X": X, "beta_local": float(b), "R2": float(r2),
                     "cv_halfpower": cv, "seconds": dt, "nN": int(len(N))})
        print(f"X={X:>11}: beta_local={b:.4f}  R^2={r2:.3f}  CV(S^.5)={cv:.4f}  "
              f"({dt:.0f}s, {len(N)} N)", flush=True)

    json.dump(rows, open(os.path.join(DATA, "scaling.json"), "w"), indent=2)

    Xa = np.array([r["X"] for r in rows], float)
    ba = np.array([r["beta_local"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogx(Xa, ba, "o-", ms=6, color="#c0392b", label=r"$\beta_{\rm local}(X)$")
    ax.axhline(0.5, color="k", ls="--", lw=1, label=r"$\beta=1/2$")
    ax.set_xlabel("$X$ (window centre)")
    ax.set_ylabel(r"local regression slope $\beta_2$")
    ax.set_title(r"Does $\beta_2$ of $R_2$ stabilize? (windows of $2\cdot10^6$)")
    ax.legend()
    fig.savefig(os.path.join(FIG, "09_beta_scaling.png"), dpi=300, bbox_inches="tight"); fig.savefig(os.path.join(FIG, "09_beta_scaling.pdf"), bbox_inches="tight")
    print("figure:", os.path.relpath(os.path.join(FIG, "09_beta_scaling.png")))


_cache = {}
def _is_prime_cache(X, W):
    """is_prime hasta sqrt(X+W) para mertens_weight_W (necesita primos pequeños)."""
    from sieve import sieve_primes
    key = int(np.sqrt(X + W)) + 1
    if key not in _cache:
        _cache[key] = sieve_primes(key)
    return _cache[key]


if __name__ == "__main__":
    main()
