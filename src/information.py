"""
Contenido de información del cometa de Goldbach (idea #9).

Descompone la varianza de log R_1(N) en: tendencia suave (N/log^2N), aritmética local
(serie singular 𝔖, captada por la clase N mod prod(primos chicos)), y residuo
irreducible. El residuo es la fluctuación genuina de Hardy--Littlewood: ~blanco, sin
estructura compresible. Conclusión informacional: el cometa es ~99.9% comprimible vía la
serie singular; su núcleo incompresible (~0.1%) es el azar irreducible de los pares de
primos. La firma: el espectro del residuo es PLANO (blanco), mientras el del cometa es
picudo (estructura aritmética).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sieve import Tables
from counts import representation_counts


def budget(N, R1, mod=30030, Nmin=20000, win=4001):
    """Devuelve (fracciones de varianza, residuo, índice válido)."""
    m = N >= Nmin
    Nn, R1n = N[m], R1[m]
    y = np.log(R1n.astype(float))
    trend = pd.Series(y).rolling(win, center=True, min_periods=win // 2).mean().values
    yt = y - trend
    ok = np.isfinite(yt)
    yt, Nok, trk = yt[ok], Nn[ok], trend[ok]
    cls = Nok % mod
    pred = np.zeros_like(yt)
    for c in np.unique(cls):
        pred[cls == c] = yt[cls == c].mean()
    resid = yt - pred
    Vt, VS, Vr = np.var(trk), np.var(pred), np.var(resid)
    tot = Vt + VS + Vr
    fracs = {"trend": Vt / tot, "singular": VS / tot, "irreducible": Vr / tot}
    return fracs, resid, yt, R1n.mean()


def whiteness(resid):
    """Espectro de potencia del residuo (debe ser plano = blanco)."""
    a = resid - resid.mean()
    P = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    freq = np.fft.rfftfreq(len(a))
    ac1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    return freq, P / P[1:].mean(), ac1


if __name__ == "__main__":
    T = Tables(2_000_000)
    N, R1, R2 = representation_counts(T)
    fr, resid, yt, mR1 = budget(N, R1)
    freq, P, ac1 = whiteness(resid)
    print("Presupuesto de información del cometa (Var log R1):")
    print(f"  tendencia suave:     {100*fr['trend']:.1f}%")
    print(f"  serie singular 𝔖:    {100*fr['singular']:.2f}%")
    print(f"  residuo irreducible: {100*fr['irreducible']:.3f}%")
    print(f"  -> comprimible (tend+𝔖): {100*(fr['trend']+fr['singular']):.2f}%")
    print(f"Residuo (azar HL): autocorr lag-1={ac1:+.4f} (blanco); "
          f"std={100*np.sqrt(np.var(resid)):.2f}% vs Poisson {100/np.sqrt(mR1):.2f}%")
    print(f"  espectro del residuo: plano? media potencia normalizada en [0.1,0.4] = "
          f"{P[(freq>0.1)&(freq<0.4)].mean():.2f} (≈1 = blanco)")
