"""
Conteos R1, R2 en una VENTANA [X, X+W] para X grande (hasta ~3·10^8), sin FFT de
longitud X. Idea: las cribas is_prime / is_semiprime sí caben en RAM hasta ~10^9
(arrays de bytes), pero la convolución completa no. Como solo necesitamos
R1(N), R2(N) para N en una ventana cerca de X, basta una **convolución por
bloques**:

  (a * b)[n] = sum_{I,J bloques} (a_I * b_J)  colocado en offset (I+J)·W,

y para la salida en el bloque M = X/W solo contribuyen los pares de bloques con
I+J ∈ {M-1, M}. Son ~M convoluciones FFT de longitud 2W: O(X log W) en tiempo,
O(W) de memoria para las FFT. Esto permite medir el exponente singular β a escalas
inalcanzables para la FFT global.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve

from sieve import sieve_primes


def _windowed_conv(a: np.ndarray, b: np.ndarray, X: int, W: int) -> np.ndarray:
    """Devuelve (a*b)[X : X+W] (longitud W). Requiere X múltiplo de W."""
    assert X % W == 0, "X debe ser múltiplo de W"
    L = len(a)
    nblocks = (L + W - 1) // W
    M = X // W
    out = np.zeros(2 * W)  # un par de bloques convoluciona a soporte de 2W
    for I in range(nblocks):
        for J in (M - 1 - I, M - I):
            if J < I or J >= nblocks:   # J>=I evita duplicar (sumamos 2x si I!=J)
                continue
            # Convertir a float SOLO el bloque (16 MB), no el array completo (GBs).
            aI = a[I * W:(I + 1) * W].astype(np.float64)
            bJ = b[J * W:(J + 1) * W].astype(np.float64)
            if aI.size == 0 or bJ.size == 0:
                continue
            c = fftconvolve(aI, bJ)        # longitud len(aI)+len(bJ)-1
            base = (I + J) * W - X         # offset dentro de [X, X+2W)
            seg, off = _place(c, base, 2 * W)
            if I == J:
                if seg is not None:
                    out[off:off + len(seg)] += seg          # par I=J: una vez
            else:
                # a*b NO es simétrica: sumamos aI*bJ y aJ*bI por separado
                if seg is not None:
                    out[off:off + len(seg)] += seg          # aI*bJ
                aJ = a[J * W:(J + 1) * W].astype(np.float64)
                bI = b[I * W:(I + 1) * W].astype(np.float64)
                c2 = fftconvolve(aJ, bI)                     # aJ*bI (mismo offset)
                seg2, off2 = _place(c2, base, 2 * W)
                if seg2 is not None:
                    out[off2:off2 + len(seg2)] += seg2
    return out[:W]


def _place(c, base, length):
    """Recorta c colocado en offset `base` al rango [0, length)."""
    lo = base
    hi = base + len(c)
    a0 = max(0, lo); b0 = min(length, hi)
    if a0 >= b0:
        return None, 0
    return c[a0 - lo:b0 - lo], a0


def semiprime_mask(L: int, is_prime: np.ndarray, primes: np.ndarray) -> np.ndarray:
    """
    Máscara de semiprimos en [0..L] por enumeración p·q (p<=q primos). Solo recorre
    primos hasta sqrt(L) (~2000 para L=3·10^8): mucho más rápido que cribar Omega.
    """
    is_semi = np.zeros(L + 1, dtype=bool)
    sq = int(np.sqrt(L))
    small = primes[primes <= sq]
    for p in small:
        p = int(p)
        qs = primes[(primes >= p) & (primes <= L // p)]
        is_semi[p * qs] = True
    return is_semi


def windowed_counts(X: int, W: int):
    """
    R1, R2 para N par en [X, X+W). Devuelve (N, R1, R2, S) con S = serie singular
    de Goldbach en la ventana. Requiere X múltiplo de W.
    """
    L = X + W
    is_prime = sieve_primes(L)
    primes = np.nonzero(is_prime)[0]
    s2 = semiprime_mask(L, is_prime, primes)   # bool; se convierte por bloque

    R1w = np.rint(_windowed_conv(is_prime, is_prime, X, W)).astype(np.int64)
    R2w = np.rint(_windowed_conv(is_prime, s2, X, W)).astype(np.int64)

    # serie singular de Goldbach en la ventana (criba windowed)
    S = _windowed_singular_series(X, W, is_prime)

    n = np.arange(X, X + W)
    even = (n % 2) == 0
    N = n[even]
    return N, R1w[even], R2w[even], S[even]


def _windowed_singular_series(X: int, W: int, is_prime: np.ndarray) -> np.ndarray:
    """
    S(N)=prod_{p|N,p>2}(p-1)/(p-2) EXACTA para n en [X, X+W).
    Aplica los primos pequeños (p<=sqrt) con la criba barata y recupera el ÚNICO
    cofactor primo grande (p>sqrt) factorizando el residuo por bloques.
    """
    S = np.ones(W, dtype=np.float64)
    n = np.arange(X, X + W, dtype=np.int64)
    res = n.copy()
    # quitar el factor 2 (la serie singular solo usa primos impares)
    while np.any(res % 2 == 0):
        m = res % 2 == 0
        res[m] //= 2

    rmax = int(np.sqrt(X + W)) + 1
    odd_primes = np.nonzero(is_prime[:rmax + 1])[0]
    odd_primes = odd_primes[odd_primes > 2]
    for p in odd_primes:
        p = int(p)
        start = (-X) % p                 # primer múltiplo de p en [X, X+W)
        if start >= W:
            continue
        S[start::p] *= (p - 1.0) / (p - 2.0)
        # quitar TODAS las potencias de p del residuo (solo en los múltiplos)
        sub = res[start::p]
        while True:
            mm = sub % p == 0
            if not mm.any():
                break
            sub[mm] //= p
        res[start::p] = sub
    # lo que queda >1 es un único primo grande p>sqrt: aporta (p-1)/(p-2)
    big = res > 1
    S[big] *= (res[big] - 1.0) / (res[big] - 2.0)
    return S


if __name__ == "__main__":
    # Validación: la versión por bloques debe COINCIDIR con la FFT global.
    from counts import representation_counts
    from sieve import Tables
    from heuristics import singular_series_goldbach

    Xv, Wv = 180000, 20000
    Nw, R1w, R2w, Sw = windowed_counts(Xv, Wv)

    T = Tables(Xv + Wv)
    Ng, R1g, R2g = representation_counts(T)
    Sg = singular_series_goldbach(T.X, T.is_prime)
    idx = {int(n): i for i, n in enumerate(Ng)}
    ok = True
    for i, n in enumerate(Nw):
        j = idx[int(n)]
        if (R1w[i], R2w[i]) != (int(R1g[j]), int(R2g[j])):
            print(f"  MISMATCH N={n}: win=({R1w[i]},{R2w[i]}) global=({R1g[j]},{R2g[j]})")
            ok = False
            if i > 5:
                break
    # serie singular
    smax = np.max(np.abs(Sw - Sg[Nw]))
    print(f"segmented.py: conteos {'OK' if ok else 'FALLA'}; "
          f"max|ΔS|={smax:.2e}; #N={len(Nw)}")
