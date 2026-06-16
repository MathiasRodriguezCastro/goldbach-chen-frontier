"""
Valida la ley derivada para el exponente singular de R2:
    beta_2(N) = 1 - c / <R2/R1>_N ,   con  <R2/R1> ~ log log N,
es decir el déficit 1-beta_2 decae como 1/<R2/R1> (≈1/loglogN), y beta_2 -> 1.

Deriva de R2/R1 = sum_r h_r(N) w_r(N) (la serie singular 𝔖 se cancela), cuya
correlación con log 𝔖 viene solo de los flips h_r en divisores pequeños de N
(perturbación O(1) sobre una suma ~loglogN). Test no circular: (1-beta_2)·<R2/R1>
debe ser ~constante. Genera figura 12.

Uso: python3 run_betalaw.py
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from segmented import windowed_counts
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    W = 2_000_000
    Xs = [int(v) for v in (6e6, 1e7, 2e7, 5e7, 1e8, 2e8)]
    betas, meanW = [], []
    for X in Xs:
        N, R1, R2, S = windowed_counts(X, W)
        ok = R1 > 0
        R1, R2, S = R1[ok], R2[ok], S[ok]
        b, a = np.polyfit(np.log(S), np.log(R2 / R1), 1)
        betas.append(b + 1.0)
        meanW.append(float(np.mean(R2 / R1)))
        print(f"X={X:>11}: beta2={b+1:.4f}  <R2/R1>={meanW[-1]:.4f}  "
              f"(1-beta2)*<R2/R1>={(1-(b+1))*meanW[-1]:.4f}", flush=True)

    betas = np.array(betas); meanW = np.array(meanW)
    inv = 1.0 / meanW
    # ajuste beta2 = 1 - c*inv  (recta que pasa por (0,1))
    c = -np.sum((betas - 1.0) * inv) / np.sum(inv * inv)
    rms = float(np.sqrt(np.mean((betas - (1 - c * inv)) ** 2)))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    xx = np.linspace(0, inv.max() * 1.05, 50)
    ax.plot(xx, 1 - c * xx, "-", color="#e67e22", lw=1.6,
            label=fr"$\beta_2=1-{c:.2f}/\langle R_2/R_1\rangle$  (rms {rms:.4f})")
    ax.scatter(inv, betas, s=55, color="#c0392b", zorder=3, label="medido (ventanas)")
    ax.scatter([0], [1], marker="*", s=180, color="k", zorder=3,
               label=r"límite $\beta_2\to1$ ($N\to\infty$)")
    ax.set_xlabel(r"$1/\langle R_2/R_1\rangle_N$  ($\sim 1/\log\log N$)")
    ax.set_ylabel(r"exponente singular local $\beta_2$")
    ax.set_title(r"Ley derivada: $\beta_2=1-c/\langle R_2/R_1\rangle\to1$")
    ax.legend(fontsize=9)
    path = os.path.join(FIG, "12_betalaw.png")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)

    json.dump({"X": Xs, "beta2": betas.tolist(), "meanW": meanW.tolist(),
               "c": float(c), "rms": rms,
               "product": ((1 - betas) * meanW).tolist()},
              open(os.path.join(DATA, "betalaw.json"), "w"), indent=2)
    print(f"  c={c:.4f}  rms={rms:.4f}  figura: {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
