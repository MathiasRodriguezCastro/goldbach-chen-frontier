"""
Heurísticas de Hardy--Littlewood y una heurística REFINADA para R2.

------------------------------------------------------------------------
R1 (Goldbach, Hardy--Littlewood):
    R1(N) ~ 2 C2 · S(N) · N / log^2 N,
con C2 = constante de primos gemelos y la serie singular de Goldbach
    S(N) = prod_{p | N, p>2} (p-1)/(p-2).

------------------------------------------------------------------------
R2 (prima + semiprimo) — heurística refinada (contribución de este repo):

Escribimos el semiprimo q = N - p = r·s con r<=s primos impares, y sumamos
sobre el factor pequeño r:
    R2(N) = sum_{r primo} #{ s primo : N - r s primo },
es decir, para cada r el sumando es un conteo "tipo Goldbach" de la forma
lineal (s, N - r s). Su serie singular de dos formas factoriza como

    S2(r,N) = 2 C2 · S(N) · (r-1)/(r-2)      (para r ∤ N, r>2),

¡la MISMA serie singular de Goldbach S(N)! Por lo tanto

    R2(N) ~ 2 C2 · S(N) · (N / log^2 N) · W(N),
    W(N) := sum_{r primo, 3<=r<=sqrt(N)} (r-1)/(r-2) · (1/r) · logN/log(N/r).

Dos consecuencias falsables:
  (P1)  El factor log log N del .tex ES la suma de Mertens W(N) (Mertens:
        sum_{r<=sqrt N} 1/r ~ log log N).
  (P2)  Como R1 ~ 2 C2 S(N) N/log^2 N, el ratio de Chen se simplifica:
            C(N) = R2(N)/(R1(N)+1) ~ W(N),
        ¡independiente de la aritmética de N! S(N) se CANCELA. Predice que el
        ratio de Chen es casi determinista (una curva de Mertens suave),
        aunque R1 y R2 por separado fluctúen fuertemente con los divisores de N.
"""
from __future__ import annotations
import numpy as np

# Constante de primos gemelos C2 = prod_{p>2} (1 - 1/(p-1)^2).
TWIN_PRIME_C2 = 0.6601618158468695739278121


def singular_series_goldbach(X: int, is_prime: np.ndarray) -> np.ndarray:
    """S(N) = prod_{p|N, p>2} (p-1)/(p-2) para N en [0..X] (criba multiplicativa)."""
    X = int(X)
    S = np.ones(X + 1, dtype=np.float64)
    odd_primes = np.nonzero(is_prime)[0]
    odd_primes = odd_primes[odd_primes > 2]
    for p in odd_primes:
        S[p::p] *= (p - 1.0) / (p - 2.0)
    return S


def hl_prediction_R1(N: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Predicción HL para R1 sobre los N dados (S = serie singular completa)."""
    Nf = N.astype(np.float64)
    return 2.0 * TWIN_PRIME_C2 * S[N] * Nf / np.log(Nf) ** 2


def mertens_weight_W(N: np.ndarray, is_prime: np.ndarray) -> np.ndarray:
    """
    W(N) = sum_{r primo, 3<=r<=sqrt(N)} (r-1)/(r-2) · (1/r) · logN/log(N/r).

    Acumulación vectorizada sobre primos pequeños r (cada r toca los N con N>r^2).
    """
    Nf = N.astype(np.float64)
    logN = np.log(Nf)
    W = np.zeros_like(Nf)
    Xmax = int(N.max())
    rmax = int(np.sqrt(Xmax)) + 1
    small_primes = np.nonzero(is_prime[: rmax + 1])[0]
    small_primes = small_primes[small_primes >= 3]
    for r in small_primes:
        r = int(r)
        mask = N > r * r  # r <= sqrt(N)
        if not np.any(mask):
            continue
        nn = Nf[mask]
        contrib = ((r - 1.0) / (r - 2.0)) * (1.0 / r) * (logN[mask] / np.log(nn / r))
        W[mask] += contrib
    return W


# Exponente singular EMPÍRICO de R2 (hallazgo de este repo): R2 hereda ~la RAÍZ
# de la serie singular de Goldbach, no la serie completa. Deriva lentamente con X
# (≈0.45 en 5·10^5 → ≈0.53 en 8·10^6); 1/2 es la mejor descripción efectiva.
R2_SINGULAR_EXPONENT = 0.5


def refined_prediction_R2(N, S, is_prime, beta: float = R2_SINGULAR_EXPONENT):
    """R2(N) ~ 2 C2 · S(N)^beta · (N/log^2 N) · W(N), con beta≈1/2 (no 1)."""
    Nf = N.astype(np.float64)
    W = mertens_weight_W(N, is_prime)
    return 2.0 * TWIN_PRIME_C2 * S[N] ** beta * (Nf / np.log(Nf) ** 2) * W


def measure_singular_exponent(N, R1, R2, S, Nmin: int = 5000):
    """
    Mide el exponente singular de R2 de forma model-light: usa R1 como medidor de
    S(N) y regresa log(R2/R1) contra log S(N). Pendiente = beta - 1.
    Devuelve (beta, R2_de_la_regresion).
    """
    m = N >= Nmin
    Sn = S[N[m]]
    x = np.log(Sn)
    y = np.log(R2[m] / R1[m])
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    r2 = 1.0 - np.var(y - yhat) / np.var(y)
    return b + 1.0, r2


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from sieve import Tables
    from counts import representation_counts

    T = Tables(200000)
    N, R1, R2 = representation_counts(T)
    S = singular_series_goldbach(T.X, T.is_prime)

    # Trabajamos lejos de los bordes (W(N)=0 para N<=9; primos chicos dominan).
    m = N >= 1000
    Nm, R1m, R2m = N[m], R1[m], R2[m]
    predR1 = hl_prediction_R1(Nm, S)
    predR2 = refined_prediction_R2(Nm, S, T.is_prime)
    W = mertens_weight_W(Nm, T.is_prime)

    cR1 = np.median(R1m / predR1)
    cR2 = np.median(R2m / predR2)
    print(f"X={T.X}  (N>=1000)")
    print(f"  R1: razón mediana obs/pred = {cR1:.4f}")
    print(f"  R2: razón mediana obs/pred = {cR2:.4f}")

    def cv(x):
        x = x[np.isfinite(x)]
        return np.std(x) / np.mean(x)

    print("  --- dispersión relativa (CV) de la razón observado/modelo (menor=mejor) ---")
    print(f"  R1 / (N/log^2N)            CV={cv(R1m/(Nm/np.log(Nm)**2)):.3f}")
    print(f"  R1 / HL(2C2·S·N/log^2N)    CV={cv(R1m/predR1):.3f}   <- serie singular")
    print(f"  R2 / (N·loglogN/log^2N)    CV={cv(R2m/(Nm*np.log(np.log(Nm))/np.log(Nm)**2)):.3f}   <- heurística .tex")
    print(f"  R2 / refinado(2C2·S·N·W)   CV={cv(R2m/predR2):.3f}   <- modelo refinado")

    # ---- Predicción P2: el ratio de Chen C(N) ~ W(N), INDEPENDIENTE de S(N) ----
    C = R2m / (R1m + 1.0)
    ratio = C / W                       # debería concentrarse cerca de una constante
    print("  --- P2: ratio de Chen C(N)=R2/(R1+1) vs W(N) (Mertens) ---")
    print(f"  mediana C(N)/W(N) = {np.median(ratio):.4f}   CV(C/W) = {cv(ratio):.3f}")
    print(f"  (compárese: CV de C(N) crudo = {cv(C):.3f};  CV de R2 crudo = {cv(R2m):.3f})")
