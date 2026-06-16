"""
Conteos de representación R1, R2, R<=2 para todos los enteros hasta X
vía convolución aditiva (FFT).

  R1(N) = sum_{p<N} 1_P(p) 1_P(N-p)         = (1_P  * 1_P )(N)
  R2(N) = sum_{p<N} 1_P(p) 1_{S2}(N-p)      = (1_P  * 1_S2)(N)

donde * es la convolución aditiva discreta (autocorrelación de soporte sobre N+).
Esto coincide exactamente con las definiciones del .tex (conteos ORDENADOS por
el primer sumando p; para N=2p el término p=N/2 se cuenta una sola vez).

La FFT introduce error de redondeo O(eps · n · max); para los rangos de este
proyecto los conteos son enteros pequeños y `np.rint` los recupera exactamente.
`brute_counts` valida un sub-rango por fuerza bruta.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

from sieve import Tables


def representation_counts(T: Tables):
    """Devuelve (N_even, R1, R2) como arrays alineados sobre N par en [4..X]."""
    X = T.X
    a = T.is_prime.astype(np.float64)
    s2 = T.is_semiprime.astype(np.float64)

    # Convoluciones aditivas: índice N de la salida = i+j.
    conv_pp = fftconvolve(a, a)          # longitud 2X+1
    conv_ps = fftconvolve(a, s2)

    R1_all = np.rint(conv_pp[: X + 1]).astype(np.int64)
    R2_all = np.rint(conv_ps[: X + 1]).astype(np.int64)

    N = np.arange(4, X + 1, 2)
    return N, R1_all[N], R2_all[N]


def brute_counts(T: Tables, Nval: int):
    """R1(N), R2(N) por fuerza bruta (validación)."""
    primes = T.primes
    primes = primes[primes < Nval]
    q = Nval - primes
    r1 = int(np.count_nonzero(T.is_prime[q]))
    r2 = int(np.count_nonzero(T.is_semiprime[q]))
    return r1, r2


def validate(T: Tables, sample_Ns) -> bool:
    """Compara la versión FFT contra fuerza bruta en una lista de N."""
    N, R1, R2 = representation_counts(T)
    idx = {int(n): i for i, n in enumerate(N)}
    ok = True
    for Nv in sample_Ns:
        if Nv not in idx:
            continue
        b1, b2 = brute_counts(T, Nv)
        f1, f2 = int(R1[idx[Nv]]), int(R2[idx[Nv]])
        if (b1, b2) != (f1, f2):
            print(f"  MISMATCH N={Nv}: FFT=({f1},{f2}) brute=({b1},{b2})")
            ok = False
    return ok


if __name__ == "__main__":
    T = Tables(20000)
    N, R1, R2 = representation_counts(T)
    # Goldbach: R1(N)>0 para todo N par > 2 en el rango.
    assert np.all(R1 > 0), "¡Goldbach falla en el rango! (bug)"
    # Validación FFT vs fuerza bruta.
    import numpy.random as npr
    npr.seed(0)
    sample = list(npr.choice(N, size=200, replace=False)) + [4, 6, 100, 2 * 4999]
    assert validate(T, sample), "FFT != fuerza bruta"
    print(f"counts.py OK — X={T.X}, #N={len(N)}")
    print(f"  R1(100)={R1[(N==100)][0]}  R2(100)={R2[(N==100)][0]}")
    print(f"  max R1={R1.max()} en N={int(N[R1.argmax()])}")
    print(f"  promedio R2/R1 = {np.mean(R2/np.maximum(R1,1)):.3f}")
