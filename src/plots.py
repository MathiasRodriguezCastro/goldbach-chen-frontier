"""
Figuras de la frontera Goldbach--Chen. Cada función guarda un PNG en figures/.
Se usa un backend sin display (Agg).
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _subsample(N, *arrs, k=60000):
    if len(N) <= k:
        return (N,) + arrs
    idx = np.linspace(0, len(N) - 1, k).astype(int)
    return (N[idx],) + tuple(a[idx] for a in arrs)


def comets(N, R1, R2, name="01_comets.png"):
    Ns, R1s, R2s = _subsample(N, R1, R2)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(Ns, R1s, s=1, c="#1f4fa0", alpha=0.25, linewidths=0, label="$R_1$ (primo+primo)")
    ax[0].scatter(Ns, R2s, s=1, c="#c0392b", alpha=0.20, linewidths=0, label="$R_2$ (primo+semiprimo)")
    ax[0].set_xlabel("$N$"); ax[0].set_ylabel("nº de representaciones (ordenadas)")
    ax[0].set_title("Cometas de Goldbach y de semiprimos")
    lg = ax[0].legend(markerscale=8, framealpha=0.9)
    ax[1].scatter(Ns, R2s / (R1s + 1.0), s=1, c="#6c3483", alpha=0.25, linewidths=0)
    ax[1].set_xlabel("$N$"); ax[1].set_ylabel("$C(N)=R_2/(R_1+1)$")
    ax[1].set_title("Excedente de Chen $C(N)$")
    fig.tight_layout()
    return _save(fig, name)


def singular_exponent(N, R1, R2, S, name="02_singular_exponent.png", Nmin=20000):
    """Figura clave: log(R/base) vs log S(N) para R1 (pend≈1) y R2 (pend≈1/2)."""
    m = N >= Nmin
    Nf = N[m].astype(float); base = Nf / np.log(Nf) ** 2
    Sn = S[N[m]]
    x = np.log(Sn)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    refs = [1.0, 0.5]
    for k, (R, lab, col) in enumerate([(R1[m], "R_1", "#1f4fa0"), (R2[m], "R_2", "#c0392b")]):
        y = np.log(R / base)
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        x0, y0 = x.mean(), y.mean()
        ax[k].scatter(x, y, s=2, c=col, alpha=0.08, linewidths=0)
        ax[k].plot(xs, a + b * xs, "k-", lw=2, label=f"ajuste: $\\beta={b:.3f}$")
        ax[k].plot(xs, y0 + refs[k] * (xs - x0), "k--", lw=1.2,
                   label=fr"ref. $\beta={refs[k]:.1f}$")
        ax[k].set_xlabel(r"$\log \mathfrak{S}(N)$ (serie singular de Goldbach)")
        ax[k].set_ylabel(fr"$\log\,[{lab}/(N/\log^2 N)]$")
        ax[k].set_title(fr"${lab}$ vs serie singular")
        ax[k].legend(framealpha=0.9)
    fig.suptitle("Exponente singular: $R_1\\propto\\mathfrak{S}^{1}$, $R_2\\propto\\mathfrak{S}^{1/2}$", y=1.02)
    fig.tight_layout()
    return _save(fig, name)


def normalization_collapse(N, R1, S, name="03_normalizacion.png", Nmin=20000):
    """A1 sin vs con serie singular: muestra el colapso de las bandas."""
    m = N >= Nmin
    Nf = N[m].astype(float); base = Nf / np.log(Nf) ** 2
    Ns, A_raw, A_ss = _subsample(N[m], R1[m] / base, R1[m] / (S[N[m]] * base))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    ax[0].scatter(Ns, A_raw, s=1, c="#1f4fa0", alpha=0.15, linewidths=0)
    ax[0].set_title("$A_1=R_1/(N/\\log^2 N)$  (bandas por divisibilidad)")
    ax[0].set_xlabel("$N$"); ax[0].set_ylabel("$A_1$")
    ax[1].scatter(Ns, A_ss, s=1, c="#117a65", alpha=0.15, linewidths=0)
    ax[1].axhline(2 * 0.6601618, color="k", ls="--", lw=1, label="$2C_2$")
    ax[1].set_title("$R_1/(\\mathfrak{S}(N)\\,N/\\log^2 N)$  (colapsa a $2C_2$)")
    ax[1].set_xlabel("$N$"); ax[1].set_ylabel("normalizado"); ax[1].legend()
    fig.tight_layout()
    return _save(fig, name)


def theta_and_chen(N, R1, R2, W, name="04_theta_C_W.png", Nmin=20000):
    m = N >= Nmin
    Nf = N[m].astype(float)
    th = R1[m] / np.maximum(R1[m] + R2[m], 1)
    C = R2[m] / (R1[m] + 1.0)
    # binning para curvas medias
    nb = 200
    edges = np.linspace(Nf.min(), Nf.max(), nb + 1)
    idx = np.digitize(Nf, edges) - 1
    idx = np.clip(idx, 0, nb - 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    th_b = np.array([th[idx == i].mean() if np.any(idx == i) else np.nan for i in range(nb)])
    C_b = np.array([C[idx == i].mean() if np.any(idx == i) else np.nan for i in range(nb)])
    W_b = np.array([W[m][idx == i].mean() if np.any(idx == i) else np.nan for i in range(nb)])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    Ns, ths = _subsample(Nf, th, k=40000)
    ax[0].scatter(Ns, ths, s=1, c="#aed6f1", alpha=0.3, linewidths=0)
    ax[0].plot(cen, th_b, "b-", lw=2, label=r"media móvil")
    ax[0].set_title(r"Cuota primo--primo $\theta(N)=R_1/(R_1+R_2)$")
    ax[0].set_xlabel("$N$"); ax[0].set_ylabel(r"$\theta$"); ax[0].legend()
    ax[1].plot(cen, C_b, "-", c="#6c3483", lw=2, label=r"$C(N)$ (media)")
    ax[1].plot(cen, W_b, "--", c="#e67e22", lw=2, label=r"$W(N)$ (Mertens, predicho)")
    ax[1].set_title(r"$C(N)$ vs peso de Mertens $W(N)$")
    ax[1].set_xlabel("$N$"); ax[1].set_ylabel(r"valor"); ax[1].legend()
    fig.tight_layout()
    return _save(fig, name)


def balance_hist(B, w, name="05_balance.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(B, bins=80, weights=w, density=True, color="#16a085", alpha=0.85)
    ax.axvline(np.average(B, weights=w), color="k", ls="--", label=f"media={np.average(B,weights=w):.3f}")
    ax.set_xlabel(r"$B(q)=\log a/\log q$  (factor menor)")
    ax.set_ylabel("densidad (ponderada por representaciones)")
    ax.set_title("Geometría de factores de los semiprimos de Chen")
    ax.legend()
    return _save(fig, name)


def rescue_threshold(ks, fracs_by_scale, name="06_rescate.png"):
    """fracs_by_scale: dict 'etiqueta N' -> array de fracciones que necesitan rescate vs ks."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for lab, fr in fracs_by_scale.items():
        ax.plot(ks, 100 * np.array(fr), "-o", ms=3, label=lab)
    ax.set_xlabel("semi-ancho de banda $k$ (enteros bajo $N/2$)")
    ax.set_ylabel("% de $N$ que requieren rescate por semiprimo")
    ax.set_title("Umbral de rescate de Chen bajo restricción de balance")
    ax.set_xscale("log"); ax.legend()
    return _save(fig, name)


def continuity(centers, dens_p, dens_s, indiv, tv_ps, tv_stab, Hs, name="08_continuidad.png"):
    """
    Panel A: medidas μ_N individuales (ruidosas) vs densidad suavizada en ventana.
    Panel B: densidad de q/N para q primo vs q semiprimo + referencia uniforme.
    Panel C: distancia TV entre ventanas adyacentes vs tamaño de ventana H.
    """
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for x, d in indiv:
        ax[0].plot(x, d, color="0.75", lw=0.8, drawstyle="steps-mid")
    ax[0].plot(centers, dens_p, "b-", lw=2.2, label="suavizada en ventana")
    ax[0].plot([], [], color="0.75", lw=0.8, label=r"$\mu_N$ individual (ruidosa)")
    ax[0].set_title(r"Continuidad tras suavizar: $q/N$, $q$ primo")
    ax[0].set_xlabel("$q/N$"); ax[0].set_ylabel("densidad"); ax[0].legend(fontsize=8)

    ax[1].plot(centers, dens_p, "b-", lw=2, label="$q$ primo")
    ax[1].plot(centers, dens_s, "r-", lw=2, label="$q$ semiprimo")
    ax[1].axhline(1.0, color="k", ls="--", lw=1, label="uniforme")
    ax[1].set_title(fr"Tipos comparados (TV$={tv_ps:.3f}$)")
    ax[1].set_xlabel("$q/N$"); ax[1].set_ylabel("densidad"); ax[1].legend(fontsize=8)

    ax[2].loglog(Hs, tv_stab, "o-", color="#117a65")
    ax[2].set_title("Estabilidad: TV entre ventanas adyacentes")
    ax[2].set_xlabel("tamaño de ventana $H$ (nº de $N$)"); ax[2].set_ylabel("distancia TV")
    fig.tight_layout()
    return _save(fig, name)


def mip_branch(Ns, ss, types, name="07_rama_mip.png"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    Ns = np.array(Ns); ss = np.array(ss); types = np.array(types)
    ax.plot(Ns, ss, "-", c="0.6", lw=0.6, zorder=1)
    ax.scatter(Ns[types == 0], ss[types == 0], s=10, c="#1f4fa0", label="primo", zorder=2)
    ax.scatter(Ns[types == 1], ss[types == 1], s=10, c="#c0392b", label="semiprimo", zorder=2)
    ax.set_xlabel("$N$"); ax.set_ylabel("$s_N=q_N/N$")
    ax.set_title("Rama de mínima variación espacial: suave en $s_N$ pero alterna de tipo")
    ax.legend()
    return _save(fig, name)
