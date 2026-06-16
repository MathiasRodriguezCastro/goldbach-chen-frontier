"""
Robustez adversarial del grafo de Goldbach (ideas #2 grafo + #13 robustez).

Grafo bipartito N <-> p con arista si N-p es primo; grado d_N = #representaciones.
Un adversario elimina primos; N "sobrevive" si conserva alguna representación con
ambos primos vivos. Tres regímenes:

  - Aleatorio: cada primo muere con prob phi. P(N falla) = (1-(1-phi)^2)^{d_N}.
    Goldbach es HIPER-ROBUSTO (sobrevive hasta phi~0.97 por la enorme redundancia).
  - Dirigido: para matar el N más débil basta quitar 1 primo por cada par => d_min
    primos (corte de vértices de un emparejamiento). kappa = d_min.
  - Modular: quitar los primos de una clase a mod m. Los primos sobrevivientes viven
    en (Z/m)^x \ {a}; su SUMSET puede no cubrir todas las clases pares => obstrucción
    LOCAL creada por el ataque. Para m=3, quitar la clase 1 (la mitad de los primos)
    mata 2/3 de los N (los sobrevivientes, clase 2, solo suman a N≡1).
"""
from __future__ import annotations
import numpy as np

from sieve import Tables


def window_reps(T: Tables, Nlo: int, Nhi: int):
    """Devuelve (Ns, reps) con reps[i] = primos p<=N/2 tal que N-p es primo."""
    isp = T.is_prime
    primes = T.primes
    Ns = np.arange(Nlo if Nlo % 2 == 0 else Nlo + 1, Nhi + 1, 2)
    reps = []
    for N in Ns:
        pe = primes[primes <= N // 2]
        pe = pe[isp[N - pe]]
        reps.append(pe)
    return Ns, reps


def random_damage(reps, phis):
    """Fracción esperada de N fallados vs phi (deleción aleatoria)."""
    d = np.array([len(r) for r in reps], dtype=float)
    out = []
    for phi in phis:
        pfail = (1 - (1 - phi) ** 2) ** d
        out.append(float(pfail.mean()))
    return np.array(out)


def modular_damage(T, Ns, reps, m, a):
    """Quitar primos ≡a mod m: fracción de N totalmente muertos y supervivencia por
    clase de N mod m."""
    killed = np.zeros(len(Ns), dtype=bool)
    surv_frac = np.zeros(len(Ns))
    for i, N in enumerate(Ns):
        pe = reps[i]; q = N - pe
        alive = (pe % m != a) & (q % m != a)
        surv_frac[i] = alive.mean() if len(pe) else 0.0
        killed[i] = not alive.any()
    by_class = {c: float(np.mean(killed[Ns % m == c])) for c in range(m)}
    return float(killed.mean()), by_class, surv_frac


def surviving_sumset(m, a):
    """Clases pares de N que el sumset de (Z/m)^x\\{a} NO cubre (obstrucción)."""
    units = [u for u in range(m) if np.gcd(u, m) == 1 and u != a]
    sums = set((u + v) % m for u in units for v in units)
    even_classes = [c for c in range(m)]  # N par; clase mod m
    missed = [c for c in even_classes if c not in sums]
    return units, sorted(sums), missed


if __name__ == "__main__":
    T = Tables(1_100_000)
    Ns, reps = window_reps(T, 1_000_000, 1_030_000)
    d = np.array([len(r) for r in reps])
    nprimes = len(T.primes)
    print(f"Ventana: {len(Ns)} N pares; d_min={d.min()}, d_max={d.max()}, #primos={nprimes}")
    kappa = int(d.min())
    print(f"  DIRIGIDO: kappa={kappa} primos ({100*kappa/nprimes:.2f}%) rompe Goldbach")
    phis = np.array([0.5, 0.9, 0.95, 0.97, 0.99])
    rd = random_damage(reps, phis)
    print(f"  ALEATORIO: frac N fallados @ phi={list(phis)} = {np.round(rd,4)}")
    for m, a in [(3, 1), (4, 1), (5, 1)]:
        frac, by_class, _ = modular_damage(T, Ns, reps, m, a)
        units, sums, missed = surviving_sumset(m, a)
        print(f"  MODULAR (clase {a} mod {m}): mata {100*frac:.1f}% de N; "
              f"sobreviven clases {units}, sumset={sums}, N-clases obstruidas={missed}")
