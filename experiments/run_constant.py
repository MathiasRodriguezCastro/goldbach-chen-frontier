"""
La constante de tasa de la ley beta2 = 1 - c/<R2/R1> (apéndice §A.4).

Mide dos normalizaciones, kappa(X)=(1-beta2)loglogX y c(X)=(1-beta2)<R2/R1>,
a varias escalas y las extrapola en 1/loglogX. HALLAZGO: ambas DECRECEN y el límite
(~0.5-0.7, dependiente de convención) es corrección-dominado y NO determinable con
confianza hasta 1e9. Solo el ORDEN 1/loglogX es robusto. Genera figura 16.

OJO: lento (windowed_counts hasta 5e8 ~ 6 min).
Uso: python3 run_constant.py
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

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    Xs = [2_000_000, 20_000_000, 100_000_000, 500_000_000]
    kappa, c = [], []
    for X in Xs:
        N, R1, R2, S = windowed_counts(X, 2_000_000)
        ok = R1 > 0
        N, R1, R2, S = N[ok], R1[ok], R2[ok], S[ok]
        b2 = 1 + np.polyfit(np.log(S), np.log(R2 / R1), 1)[0]
        llX = np.log(np.log(X)); W0 = np.mean(R2 / R1)
        kappa.append((1 - b2) * llX); c.append((1 - b2) * W0)
        print(f"X={X:>11}: 1-beta2={1-b2:.4f}  kappa={kappa[-1]:.4f}  c={c[-1]:.4f}")

    inv = 1.0 / np.log(np.log(np.array(Xs, float)))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for y, lab, col in [(kappa, r"$\kappa=(1-\beta_2)\log\log X$", "#c0392b"),
                        (c, r"$c=(1-\beta_2)\langle R_2/R_1\rangle$", "#1f4fa0")]:
        a, b = np.polyfit(inv, y, 1)
        xs = np.linspace(0, inv.max() * 1.05, 50)
        ax.plot(xs, b + a * xs, "--", color=col, lw=1.3, label=lab + f"  (limit {b:.2f})")
        ax.scatter(inv, y, s=55, color=col, zorder=3)
        print(f"  {lab}: limit (1/loglogX->0) = {b:.3f}, slope {a:.2f}")
    ax.axvline(0, color="0.7", lw=0.8); ax.set_xlim(left=-0.01)
    ax.set_xlabel(r"$1/\log\log X$"); ax.set_ylabel("rate constant")
    ax.set_title(r"Rate constant: the $\sim2/\log\log X$ correction dominates; limit $\sim0.5$-$0.7$")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(FIG, "16_constante.png"), dpi=130, bbox_inches="tight")
    json.dump({"X": Xs, "kappa": kappa, "c": c},
              open(os.path.join(DATA, "constant.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
