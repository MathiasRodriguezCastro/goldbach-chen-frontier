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


def balance_exponent(N, lam, rstar, logS, name="11_balance.png"):
    """Panel A: envolvente de percentiles de lambda_Res vs N. Panel B: histograma
    de r_star. Panel C: lambda_Res vs serie singular (correlación positiva)."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ok = np.isfinite(lam) & (N >= 1000)
    Nf = N[ok].astype(float); lamf = lam[ok]
    nb = 120
    edges = np.linspace(Nf.min(), Nf.max(), nb + 1)
    idx = np.clip(np.digitize(Nf, edges) - 1, 0, nb - 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    for q, lab, c in [(50, "mediana", "#1f4fa0"), (90, "p90", "#16a085"),
                      (99, "p99", "#e67e22"), (100, "máx", "#c0392b")]:
        curve = np.array([np.percentile(lamf[idx == i], q) if np.any(idx == i) else np.nan
                          for i in range(nb)])
        ax[0].plot(cen, curve, "-", color=c, lw=1.6, label=lab)
    ax[0].axhline(0.9, color="k", ls="--", lw=1, label="Li–Liu $\\lambda=0.9$")
    ax[0].set_xlabel("$N$"); ax[0].set_ylabel(r"$\lambda_{\rm Res}(N)=a_{\rm Res}-1$")
    ax[0].set_title("Envolvente del exponente de balance residual"); ax[0].legend(fontsize=8)

    vals, cnts = np.unique(rstar[rstar >= 2], return_counts=True)
    keep = vals <= 13
    ax[1].bar([str(v) for v in vals[keep]], 100 * cnts[keep] / len(N), color="#2e86c1")
    ax[1].set_xlabel(r"$r_\star(N)$ (multiplicador minimal)")
    ax[1].set_ylabel("share (%)"); ax[1].set_title("Multiplicador residual minimal")

    m = ok & (N >= 20000)
    xb = logS[N[m]]; yb = lam[m]
    nb2 = 40
    e2 = np.linspace(xb.min(), xb.max(), nb2 + 1)
    i2 = np.clip(np.digitize(xb, e2) - 1, 0, nb2 - 1)
    c2 = 0.5 * (e2[:-1] + e2[1:])
    mean_lam = np.array([yb[i2 == i].mean() if np.any(i2 == i) else np.nan for i in range(nb2)])
    ax[2].scatter(xb[::50], yb[::50], s=2, c="#aab7b8", alpha=0.3, linewidths=0)
    ax[2].plot(c2, mean_lam, "-", color="#c0392b", lw=2.2, label="media por bin")
    cc = np.corrcoef(xb, yb)[0, 1]
    ax[2].set_xlabel(r"$\log\mathfrak{S}(N)$"); ax[2].set_ylabel(r"$\lambda_{\rm Res}(N)$")
    ax[2].set_title(fr"Balance vs serie singular (corr$={cc:+.2f}$)"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def transport(mid, alpha, refl_p, refl_s2, defects, name="13_transporte.png"):
    """Panel A: densidad de primos vs reflejadas (primo/semiprimo). Panel B: defectos."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(mid, alpha, "-", color="#1f4fa0", lw=2, label=r"$\alpha$ (primos $p/N$)")
    ax[0].plot(mid, refl_p, "--", color="#16a085", lw=1.6, label=r"$T_\#\beta_1$ (primo reflejado)")
    ax[0].plot(mid, refl_s2, "-", color="#c0392b", lw=1.8, label=r"$T_\#\beta_2$ (semiprimo reflejado)")
    ax[0].set_xlabel("$x$"); ax[0].set_ylabel("densidad")
    ax[0].set_title(r"Bajo reflexión $T(x)=1-x$: $\beta_2$ matchea mejor a $\alpha$")
    ax[0].legend(fontsize=8)
    labels = [r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_{\leq2}$",
              r"$\Delta_2$ fact.chico", r"$\Delta_2$ balanc."]
    vals = [defects["Delta_1"], defects["Delta_2"], defects["Delta_le2"],
            defects["Delta_2_smallfactor"], defects["Delta_2_balanced"]]
    cols = ["#1f4fa0", "#c0392b", "#6c3483", "#e67e22", "#7f8c8d"]
    ax[1].bar(labels, vals, color=cols)
    ax[1].set_ylabel(r"defecto de reflexión $W_1$")
    ax[1].set_title(r"Colapso de transporte: $\Delta_{\leq2}$ es "
                    + f"{defects['reduction_pct']:.0f}% menor que " + r"$\Delta_1$")
    fig.tight_layout()
    return _save(fig, name)


def dynamics(seg_N, seg_theta, seg_pred, lags, acf, r2_theta, r2_dtheta, name="15_dinamica.png"):
    """Panel A: theta(N) vs predicción por 𝔖 en un segmento. Panel B: ACF del residuo."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(seg_N, seg_theta, "o-", ms=3, color="#1f4fa0", lw=0.8, label=r"$\theta(N)$ observado")
    ax[0].plot(seg_N, seg_pred, "-", color="#e67e22", lw=1.8, label=r"predicción por $\mathfrak{S}(N)$")
    ax[0].set_xlabel("$N$"); ax[0].set_ylabel(r"cuota prima $\theta(N)$")
    ax[0].set_title(fr"$\theta$ casi determinista en $\mathfrak{{S}}(N)$ ($R^2={r2_theta:.2f}$, $\Delta\theta$: {r2_dtheta:.2f})")
    ax[0].legend(fontsize=8)
    ax[1].stem(lags, acf, basefmt=" ")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("lag"); ax[1].set_ylabel("autocorrelación del residuo")
    ax[1].set_title("El residuo NO es ruido blanco (anomalía suave)")
    fig.tight_layout()
    return _save(fig, name)


def tda(thrs, n_gold, n_chen, a1, a2, corr, name="14_valles.png"):
    """Panel A: nº de valles vs prominencia (Goldbach vs Chen). Panel B: correlación
    de anomalías a1 vs a<=2 (los valles coinciden)."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy(thrs, n_gold, "o-", color="#1f4fa0", label="Goldbach $R_1$")
    ax[0].semilogy(thrs, n_chen, "s-", color="#c0392b", label=r"Chen $R_{\leq2}$")
    ax[0].set_xlabel("prominencia del valle (umbral)")
    ax[0].set_ylabel("nº de valles con prominencia mayor")
    ax[0].set_title("Persistencia de valles: Chen casi NO los rellena")
    ax[0].legend()
    m = np.isfinite(a1) & np.isfinite(a2)
    a1m, a2m = a1[m], a2[m]
    idx = np.linspace(0, len(a1m) - 1, 30000).astype(int)
    ax[1].scatter(a1m[idx], a2m[idx], s=2, c="#7f8c8d", alpha=0.25, linewidths=0)
    lim = [min(a1m.min(), a2m.min()), max(a1m.max(), a2m.max())]
    ax[1].plot(lim, lim, "k--", lw=1, label="diagonal")
    ax[1].set_xlabel("anomalía de Goldbach $R_1/\\mathfrak{S}$ (norm.)")
    ax[1].set_ylabel(r"anomalía de Chen $R_{\leq2}/E_{\leq2}$ (norm.)")
    ax[1].set_title(fr"Las anomalías COINCIDEN (corr$={corr:+.2f}$)")
    ax[1].legend()
    fig.tight_layout()
    return _save(fig, name)


def control(Bs, success, cost, name="20_control.png"):
    """Valor de la información aritmética: éxito y costo de aclaramiento vs presupuesto B."""
    fig, ax = plt.subplots(figsize=(7.8, 5))
    xs = [str(b) if b else "naive" for b in Bs]
    ax.plot(xs, 100 * np.array(success), "o-", color="#1f4fa0", lw=2, ms=7,
            label="éxito single-shot (%)")
    ax.set_ylabel("éxito single-shot (%)", color="#1f4fa0")
    ax.set_xlabel(r"presupuesto de información $B$ (controlador usa $N\,\mathrm{mod}\,\ell$, $\ell\leq B$)")
    ax2 = ax.twinx()
    ax2.plot(xs, cost, "s--", color="#c0392b", lw=2, ms=7, label="costo secuencial (intentos)")
    ax2.set_ylabel("intentos hasta aclarar (medio)", color="#c0392b")
    ax.set_title("Valor de la información aritmética para el controlador de Goldbach")
    fig.tight_layout()
    return _save(fig, name)


def information(fracs, freq, Pres, name="22_informacion.png"):
    """A: presupuesto de información (barra apilada). B: espectro del residuo (blanco)."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    labels = ["tendencia\n(N/log²N)", "serie singular\n$\\mathfrak{S}(N)$", "residuo\nirreducible"]
    vals = [100 * fracs["trend"], 100 * fracs["singular"], 100 * fracs["irreducible"]]
    cols = ["#aab7b8", "#1f4fa0", "#c0392b"]
    left = 0
    for v, lab, c in zip(vals, labels, cols):
        ax[0].barh([0], [v], left=left, color=c, height=0.5)
        ax[0].text(left + v / 2, 0, f"{lab}\n{v:.2f}%" if v < 5 else f"{lab}\n{v:.1f}%",
                   ha="center", va="center", fontsize=9, color="white" if v > 5 else "#c0392b")
        left += v
    ax[0].set_xlim(0, 100); ax[0].set_ylim(-1, 1); ax[0].set_yticks([])
    ax[0].set_xlabel("% de la varianza de $\\log R_1$")
    ax[0].set_title("El cometa es ~99.9% comprimible (local-aritmético)")
    ax[1].semilogy(freq, np.maximum(Pres, 1e-2), "-", color="#c0392b", lw=0.9)
    ax[1].axhline(1, color="0.4", ls="--", lw=1, label="blanco (potencia plana)")
    ax[1].set_xlabel("frecuencia"); ax[1].set_ylabel("potencia del residuo (norm.)")
    ax[1].set_title("El residuo es ruido BLANCO: sin estructura escondida")
    ax[1].set_ylim(1e-2, 1e2); ax[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def spectral(freq, P, Bs, r2s, ceil, name="21_espectral.png"):
    """A: espectro de potencia del cometa (arcos mayores). B: varianza explicada por módulo."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy(freq, np.maximum(P, 1e-4), "-", color="#1f4fa0", lw=1.0)
    for f, lab in [(1/3, "$1/3$ (mod 3, $T_N{=}6$)"), (1/5, "$1/5$ (mod 5)"),
                   (2/5, "$2/5$ (mod 5)")]:
        ax[0].axvline(f, color="#c0392b", ls="--", lw=1, alpha=0.7)
        ax[0].text(f, 1.3, lab, rotation=90, fontsize=7, color="#c0392b", va="bottom", ha="right")
    ax[0].set_xlabel("frecuencia (ciclos/paso en $N$ par)")
    ax[0].set_ylabel("potencia (normalizada)")
    ax[0].set_title("Espectro del cometa: arcos mayores en $a/q$ de primos chicos")
    ax[0].set_ylim(1e-4, 3)
    ax[1].plot([str(b) for b in Bs], 100 * np.array(r2s), "o-", color="#16a085", lw=2, ms=7,
               label="$N\\,\\mathrm{mod}\\,$(primos $\\leq B$)")
    ax[1].axhline(100 * ceil, color="#c0392b", ls="--", lw=1.2, label=f"techo $\\mathfrak{{S}}(N)$ ({100*ceil:.1f}%)")
    ax[1].set_xlabel("corte de primos $B$")
    ax[1].set_ylabel("% de la oscilación de $\\log R_1$ explicada")
    ax[1].set_title("El cometa es casi un fenómeno mod 6")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def geometry(H2, amid, a_hist, amed, name="23_geometria.png"):
    """A: densidad 2D de soluciones Chen en (u,v). B: espectro de exponentes a."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    Hm = np.ma.masked_where(H2.T <= 0, H2.T)
    im = ax[0].imshow(np.log10(Hm + 1), origin="lower", extent=[0, 1, 0, 1],
                      cmap="viridis", aspect="auto")
    ax[0].plot([0, 0.5], [0, 0.5], "w--", lw=1, label="$r=s$ (balanceado)")
    ax[0].plot([0, 0], [0, 1], "r-", lw=2, label="$r=1$ (frontera Goldbach)")
    ax[0].set_xlabel(r"$u=\log r/\log N$"); ax[0].set_ylabel(r"$v=\log s/\log N$")
    ax[0].set_title("Nube de soluciones Chen en la superficie $p+rs=N$")
    ax[0].legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax[0], label="$\\log_{10}$ densidad")
    ax[1].bar(amid, 100 * a_hist / a_hist.sum(), width=amid[1] - amid[0], color="#16a085", alpha=0.85)
    ax[1].axvline(amed, color="#1f4fa0", lw=2, label=f"mediana $a={amed:.2f}$")
    ax[1].axvline(1.9, color="#c0392b", ls="--", lw=1.5, label="Li–Liu $a=1.9$")
    ax[1].set_xlabel(r"exponente $a=1+\log r/\log s$")
    ax[1].set_ylabel("% de soluciones")
    ax[1].set_title("Espectro de exponentes: concentrado hacia Goldbach ($a\\to1$)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def bases(cov_y, cov_r2, kcar, hg, hr, excess_b, excess_v, name="24_bases.png"):
    """A: distribución de #acarreos Goldbach vs aleatorio (base 2). B: exceso vs base."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(kcar - 0.2, 100 * hr, width=0.4, color="#aab7b8", label="pares aleatorios")
    ax[0].bar(kcar + 0.2, 100 * hg, width=0.4, color="#c0392b", label="Goldbach")
    gm = np.average(kcar, weights=hg); rm = np.average(kcar, weights=hr)
    ax[0].axvline(gm, color="#c0392b", ls="--", lw=1); ax[0].axvline(rm, color="#7f8c8d", ls="--", lw=1)
    ax[0].set_xlabel("nº de acarreos en base 2  ($=v_2\\binom{N}{p}$, Kummer)")
    ax[0].set_ylabel("% de representaciones")
    ax[0].set_title(f"Firma digital: Goldbach acarrea más (+{gm-rm:.2f})")
    ax[0].legend(fontsize=8)
    ax[1].bar([str(b) for b in excess_b], excess_v, color="#1f4fa0", alpha=0.85)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("base $b$"); ax[1].set_ylabel("exceso de acarreos (Goldbach − aleatorio)")
    ax[1].set_title("El exceso de acarreos aparece en toda base")
    fig.tight_layout()
    return _save(fig, name)


def market(phis, LG, LC, S0, premium, name="19_mercado.png"):
    """A: liquidez sobreviviente G vs C. B: spread de Chen S_C(phi)=S_C(0)(1-phi)."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy(phis, LG, "-", color="#1f4fa0", lw=2.2, label="$L_G$ (Goldbach)")
    ax[0].semilogy(phis, LC, "-", color="#c0392b", lw=2.2, label="$L_C$ (Chen)")
    ax[0].axhline(1, color="0.6", ls=":", lw=1, label="umbral de iliquidez")
    ax[0].set_xlabel(r"shock de oferta $\phi$ (fracción de primos perdidos)")
    ax[0].set_ylabel("liquidez esperada sobreviviente")
    ax[0].set_title("Ambos mercados colapsan casi juntos ($\\phi\\approx0.99$)")
    ax[0].legend(fontsize=9)
    sc = (LC - LG) / LG
    ax[1].plot(phis, sc, "-", color="#6c3483", lw=2.2, label=r"$S_C(\phi)=(L_C-L_G)/L_G$")
    ax[1].plot(phis, S0 * (1 - phis), "--", color="#e67e22", lw=1.6, label=r"$S_C(0)\,(1-\phi)$")
    ax[1].set_xlabel(r"shock de oferta $\phi$")
    ax[1].set_ylabel(r"spread / prima del sustituto de Chen")
    ax[1].set_title(fr"El sustituto vale menos en crisis (prima resiliencia $\approx{premium:.3f}$)")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, name)


def robustness(phis, rand_curve, modpts, kappa_frac, byclass3, name="18_robustez.png"):
    """A: percolación aleatoria vs ataques modulares. B: obstrucción modular mod 3."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(phis, 100 * rand_curve, "-", color="#1f4fa0", lw=2.2, label="aleatorio")
    for (x, y, lab, col) in modpts:
        ax[0].scatter([x], [100 * y], s=90, color=col, zorder=3, label=lab)
    ax[0].axvline(kappa_frac, color="#16a085", ls="--", lw=1.2,
                  label=f"dirigido ($\\kappa$={100*kappa_frac:.1f}%)")
    ax[0].set_xlabel("fracción de primos eliminados")
    ax[0].set_ylabel("% de $N$ con Goldbach destruido")
    ax[0].set_title("Robusto al azar, frágil a la estructura")
    ax[0].legend(fontsize=8, loc="upper left")
    cls = list(byclass3.keys())
    ax[1].bar([f"$N\\equiv{c}$" for c in cls], [100 * byclass3[c] for c in cls],
              color=["#c0392b", "#16a085", "#c0392b"])
    ax[1].set_ylabel("% de $N$ destruidos (mod 3, quitar clase 1)")
    ax[1].set_title("Obstrucción modular: sobrevivientes (clase 2) solo suman a $N\\equiv1$")
    fig.tight_layout()
    return _save(fig, name)


def energy(specE, specN, betas, Ubar, Cbar, bstar_mean, logS, bstar, name="17_energia.png"):
    """A: espectro n_E(N). B: U(beta) y C(beta) (transición de fusión). C: beta* vs S."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
    ax[0].bar(specE, specN, color="#2e86c1", alpha=0.85)
    ax[0].bar([2], [specN[specE == 2][0]], color="#c0392b", label="fundamental $E=2$ (Goldbach)")
    ax[0].set_xlabel(r"energía $E=\Omega(a)+\Omega(b)$"); ax[0].set_ylabel("degeneración $n_E(N)$")
    ax[0].set_title("Espectro de energía aritmética de $N$"); ax[0].legend(fontsize=8)
    ax2 = ax[1].twinx()
    ax[1].plot(betas, Ubar, "-", color="#1f4fa0", lw=2, label=r"energía media $U(\beta)$")
    ax2.plot(betas, Cbar, "-", color="#c0392b", lw=1.8, label=r"calor específico $C(\beta)$")
    ax[1].axvline(bstar_mean, color="0.5", ls="--", lw=1)
    ax[1].axhline(2, color="0.7", ls=":", lw=1)
    ax[1].set_xlabel(r"$\beta$ (inversa de temperatura)")
    ax[1].set_ylabel(r"$U(\beta)$", color="#1f4fa0"); ax2.set_ylabel(r"$C(\beta)$", color="#c0392b")
    ax[1].set_title(fr"Fusión del orden de Goldbach ($\beta^*\approx{bstar_mean:.2f}$)")
    ax[1].legend(loc="upper right", fontsize=8)
    nb = 50
    e = np.linspace(logS.min(), logS.max(), nb + 1); idx = np.clip(np.digitize(logS, e) - 1, 0, nb - 1)
    cen = 0.5 * (e[:-1] + e[1:])
    mb = np.array([bstar[idx == i].mean() if np.any(idx == i) else np.nan for i in range(nb)])
    ax[2].scatter(logS[::40], bstar[::40], s=2, c="#aab7b8", alpha=0.3, linewidths=0)
    ax[2].plot(cen, mb, "-", color="#6c3483", lw=2.2, label="media por bin")
    cc = np.corrcoef(logS, bstar)[0, 1]
    ax[2].set_xlabel(r"$\log\mathfrak{S}(N)$"); ax[2].set_ylabel(r"$\beta^*(N)$")
    ax[2].set_title(fr"Temperatura de fusión vs serie singular (corr$={cc:+.2f}$)"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def layers(ks, shares, betas, betas_r2, name="10_capas.png"):
    """Panel A: perfil de capas rho(k). Panel B: exponente singular beta_k vs k."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(ks, shares, color="#2e86c1", alpha=0.85)
    ax[0].set_xlabel(r"capa $k=\Omega(q)$"); ax[0].set_ylabel("share (%)")
    ax[0].set_title(r"Perfil de complejidad aritmética $\rho(k)$")
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].plot(ks, betas, "o-", color="#c0392b", ms=7, label=r"$\beta_k$ medido")
    ax[1].plot(ks, [1.0 / k for k in ks], "s--", color="0.5", ms=5, label=r"$1/k$ (referencia)")
    ax[1].set_xlabel(r"capa $k=\Omega(q)$")
    ax[1].set_ylabel(r"exponente singular $\beta_k$")
    ax[1].set_title(r"$R_k\propto\mathfrak{S}(N)^{\beta_k}$: realce ($k\leq2$) vs supresión ($k\geq3$)")
    ax[1].legend()
    fig.tight_layout()
    return _save(fig, name)


def beta_scaling(Xs, betas, name="09_beta_scaling.png"):
    Xs = np.asarray(Xs, float); betas = np.asarray(betas, float)
    # ajuste loglog: beta = a + b·loglog X
    A = np.c_[np.ones_like(Xs), np.log(np.log(Xs))]
    c, *_ = np.linalg.lstsq(A, betas, rcond=None)
    xx = np.logspace(np.log10(Xs.min()), np.log10(4e18), 200)
    fit = c[0] + c[1] * np.log(np.log(xx))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(xx, fit, "-", color="#e67e22", lw=1.5,
                label=fr"ajuste $\beta_2={c[0]:.2f}+{c[1]:.2f}\,\log\log X$")
    ax.semilogx(Xs, betas, "o", ms=7, color="#c0392b", label=r"$\beta_2$ medido (ventana $2\cdot10^6$)")
    ax.axhline(0.5, color="0.4", ls="--", lw=1, label=r"$\beta=1/2$ (valor local en $X\sim10^6$)")
    ax.axhline(1.0, color="0.4", ls=":", lw=1, label=r"$\beta=1$ (herencia total)")
    ax.axvline(4e18, color="0.7", ls="-.", lw=1)
    ax.text(4e18, 0.52, "  Goldbach\n  verif.", fontsize=7, va="bottom")
    ax.set_xlabel("$X$ (centro de la ventana)")
    ax.set_ylabel(r"exponente singular local $\beta_2$")
    ax.set_title(r"$\beta_2(X)$ crece como $\log\log X$ — el ``$1/2$'' es solo local")
    ax.legend(fontsize=8, loc="upper left")
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
