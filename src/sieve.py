"""
Cribas aritméticas vectorizadas para el proyecto Goldbach--Chen.

Provee, para un tope X:
  - is_prime[n]      : máscara booleana de primalidad           (Eratóstenes)
  - Omega[n]         : Omega(n) = #factores primos con multiplicidad
  - spf[n]           : smallest prime factor (0 para n<2)
  - máscaras derivadas: semiprimos (Omega==2), P2 = primos ∪ semiprimos

La criba de Omega usa el truco de potencias de primo: para cada potencia p^k<=X
se suma 1 a todos los múltiplos de p^k; entonces Omega[n] = sum_{p,k} [p^k | n]
= sum_p v_p(n), que es exactamente Omega con multiplicidad.

Todo es O(X log log X) en tiempo y O(X) en memoria.
"""
from __future__ import annotations
import numpy as np


def sieve_primes(X: int) -> np.ndarray:
    """Máscara booleana is_prime[0..X]."""
    X = int(X)
    is_prime = np.ones(X + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(X**0.5) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    return is_prime


def omega_sieve(X: int, is_prime: np.ndarray | None = None) -> np.ndarray:
    """Omega(n) (con multiplicidad) para n en [0..X] vía potencias de primo."""
    X = int(X)
    if is_prime is None:
        is_prime = sieve_primes(X)
    Omega = np.zeros(X + 1, dtype=np.int16)
    primes = np.nonzero(is_prime)[0]
    for p in primes:
        pk = int(p)
        while pk <= X:
            Omega[pk::pk] += 1
            pk *= int(p)
    return Omega


def spf_sieve(X: int) -> np.ndarray:
    """Smallest prime factor de cada n en [0..X] (spf[0]=spf[1]=0)."""
    X = int(X)
    spf = np.zeros(X + 1, dtype=np.int64)
    for i in range(2, X + 1):
        if spf[i] == 0:  # i es primo
            spf[i :: i] = np.where(spf[i :: i] == 0, i, spf[i :: i])
    return spf


class Tables:
    """Contenedor perezoso de cribas reutilizables para un tope X dado."""

    def __init__(self, X: int):
        self.X = int(X)
        self.is_prime = sieve_primes(self.X)
        self.Omega = omega_sieve(self.X, self.is_prime)
        self.is_semiprime = self.Omega == 2
        self.is_P2 = (self.Omega >= 1) & (self.Omega <= 2)  # primos ∪ semiprimos
        self._spf = None

    @property
    def spf(self) -> np.ndarray:
        if self._spf is None:
            self._spf = spf_sieve(self.X)
        return self._spf

    @property
    def primes(self) -> np.ndarray:
        return np.nonzero(self.is_prime)[0]


if __name__ == "__main__":
    # Validación rápida contra valores conocidos.
    T = Tables(100)
    assert T.is_prime[97] and not T.is_prime[91]
    assert T.Omega[12] == 3 and T.Omega[97] == 1
    assert T.is_semiprime[6] and T.is_semiprime[9] and not T.is_semiprime[8]
    assert T.is_P2[7] and T.is_P2[6] and not T.is_P2[8]  # 8 = 2^3 (Omega=3)
    assert T.spf[15] == 3 and T.spf[49] == 7 and T.spf[97] == 97
    print("sieve.py: todas las aserciones OK")
