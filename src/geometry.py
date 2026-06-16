"""
Geometría discreta de la superficie de Chen p+rs=N (idea #14).

Cada representación de Chen N=p+rs (q=N-p semiprimo, r=spf(q)<=s) es un punto sobre la
superficie. En coordenadas log normalizadas u=log r/log N, v=log s/log N, los puntos
viven en el triángulo {u<=v, u+v<=1}; el borde u=0 (r=1) es la frontera de Goldbach.
El exponente de Li--Liu de una solución es a=1+log r/log s. Pregunta: ¿las soluciones
se concentran cerca del borde de Goldbach (a->1)? Sí: la mediana de a es ~1.30, muy por
debajo del peor caso 1.9, y ~50% tienen a<1.3.

Acumula los histogramas (2D de (u,v) y 1D del exponente a) sobre una ventana sin guardar
los ~10^8 puntos.
"""
from __future__ import annotations
import numpy as np

from sieve import Tables


def accumulate(T: Tables, Nlo, Nhi, nb=160):
    X = T.X; isp = T.is_prime; spf = T.spf; primes = T.primes; Om = T.Omega
    H2 = np.zeros((nb, nb))                # densidad 2D en (u,v)
    a_hist = np.zeros(110)                 # exponente a en [1,2] en 110 bins
    aedges = np.linspace(1.0, 2.0, 111)
    npts = 0
    for N in range(Nlo if Nlo % 2 == 0 else Nlo + 1, Nhi + 1, 2):
        pe = primes[primes < N]; q = N - pe
        qs = q[Om[q] == 2]
        if qs.size == 0:
            continue
        r = spf[qs]; s = qs // r
        lN = np.log(N)
        u = np.log(r.astype(float)) / lN
        v = np.log(s.astype(float)) / lN
        a = 1.0 + np.log(r.astype(float)) / np.log(s.astype(float))
        H2 += np.histogram2d(u, v, bins=nb, range=[[0, 1], [0, 1]])[0]
        a_hist += np.histogram(a, bins=aedges)[0]
        npts += qs.size
    return H2, aedges, a_hist, npts


if __name__ == "__main__":
    T = Tables(2_000_000)
    H2, aedges, a_hist, npts = accumulate(T, 1_000_000, 1_005_000)
    amid = 0.5 * (aedges[:-1] + aedges[1:])
    amean = np.average(amid, weights=a_hist)
    cum = np.cumsum(a_hist) / a_hist.sum()
    amed = amid[np.searchsorted(cum, 0.5)]
    print(f"{npts} soluciones Chen")
    print(f"  exponente a=1+log r/log s: media={amean:.3f}  mediana={amed:.3f}")
    f_gold = a_hist[amid < 1.3].sum() / a_hist.sum()
    f_bal = a_hist[amid > 1.7].sum() / a_hist.sum()
    print(f"  concentración Goldbach (a<1.3): {100*f_gold:.1f}%   balanceadas (a>1.7): {100*f_bal:.1f}%")
