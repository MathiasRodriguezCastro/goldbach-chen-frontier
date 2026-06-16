"""
Energía de estado fundamental aritmética (la reformulación favorita del usuario, que
subsume capas, función de partición, semianillos y grado del grafo).

Para cada N par se define la energía de una descomposición a+b=N (a,b>=2):
    H(a,b) = Omega(a) + Omega(b),
y la energía de estado fundamental
    E(N) = min_{a+b=N} H(a,b).
Goldbach <=> E(N)=2 (ambos primos). El espectro de niveles
    n_E(N) = #{(a,b): a+b=N, a,b>=2, Omega(a)+Omega(b)=E}
cumple n_2 = R_1 (degeneración del fundamental = conteo de Goldbach), n_3 = 2 R_2
(primer excitado = primo+semiprimo, Chen). La termodinámica:
    Z_N(beta) = sum_{a+b=N} e^{-beta H} = (f_beta * f_beta)(N),  f_beta(n)=e^{-beta Omega(n)},
una sola FFT por beta da Z para TODO N. De ahí energía media U, entropía S, energía
libre F y calor específico C; el pico de C(beta) marca la "temperatura de fusión"
beta*(N) del orden de Goldbach.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

from sieve import Tables


def _weights(T: Tables, beta: float):
    """f(n)=e^{-beta Omega(n)} (n>=2), g=Omega·f, h=Omega^2·f."""
    Om = T.Omega.astype(np.float64)
    valid = np.arange(len(Om)) >= 2
    f = np.where(valid, np.exp(-beta * Om), 0.0)
    g = np.where(valid, Om * f, 0.0)
    h = np.where(valid, Om * Om * f, 0.0)
    return f, g, h


def thermodynamics(T: Tables, betas):
    """
    Para cada beta y N par, devuelve U (energía media), C (calor específico),
    F (energía libre), S (entropía), y la ocupación del fundamental p0.
    Devuelve dict de arrays indexados [i_beta, i_N].
    """
    X = T.X
    N = np.arange(4, X + 1, 2)
    U = np.empty((len(betas), len(N)))
    C = np.empty_like(U)
    F = np.empty_like(U)
    p0 = np.empty_like(U)
    R1 = np.rint(fftconvolve((T.Omega == 1).astype(float),
                             (T.Omega == 1).astype(float))[N]).astype(np.int64)
    for i, b in enumerate(betas):
        f, g, h = _weights(T, b)
        Z = fftconvolve(f, f)[N]
        Z1 = 2.0 * fftconvolve(g, f)[N]                       # sum H e^{-bH}
        Z2 = 2.0 * fftconvolve(h, f)[N] + 2.0 * fftconvolve(g, g)[N]  # sum H^2 e^{-bH}
        U[i] = Z1 / Z
        C[i] = b * b * (Z2 / Z - (Z1 / Z) ** 2)              # beta^2 Var(H)
        F[i] = -np.log(Z) / b
        p0[i] = R1 * np.exp(-2.0 * b) / Z                    # ocupación del fundamental E=2
    return N, R1, dict(U=U, C=C, F=F, p0=p0)


def energy_spectrum(T: Tables, Nval: int, Emax: int = 20):
    """Espectro n_E(Nval) por enumeración directa (un N)."""
    Om = T.Omega
    a = np.arange(2, Nval - 1)
    b = Nval - a
    E = Om[a] + Om[b]
    hist = np.bincount(E, minlength=Emax + 1)[:Emax + 1]
    return np.arange(Emax + 1), hist


def crossover_beta(betas, C):
    """beta*(N) = argmax_beta C(beta,N) (temperatura de fusión del orden de Goldbach)."""
    idx = np.argmax(C, axis=0)
    return np.asarray(betas)[idx]


if __name__ == "__main__":
    from heuristics import singular_series_goldbach
    T = Tables(500_000)
    betas = np.linspace(0.05, 6, 60)
    N, R1, th = thermodynamics(T, betas)
    bstar = crossover_beta(betas, th["C"])
    m = N >= 20000
    print(f"X={T.X}")
    print(f"  energía típica U(beta~0) media = {th['U'][0][m].mean():.2f}  "
          f"(~2loglogN={2*np.log(np.log(T.X)):.2f})")
    print(f"  energía libre F(beta=6) media = {th['F'][-1][m].mean():.3f}  (-> E(N)=2)")
    print(f"  beta* (pico de calor específico): media={bstar[m].mean():.3f} "
          f"rango=[{bstar[m].min():.2f},{bstar[m].max():.2f}]")
    # ¿beta* correlaciona con la serie singular?
    S = singular_series_goldbach(T.X, T.is_prime)[N]
    cc = np.corrcoef(np.log(S[m]), bstar[m])[0, 1]
    print(f"  corr(beta*, log S(N)) = {cc:+.3f}")
    E, hist = energy_spectrum(T, 200000)
    print(f"  espectro de N=200000: n_E para E=2..8 = {hist[2:9]}  (n_2=R_1={R1[N==200000][0]})")
