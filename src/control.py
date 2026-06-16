"""
Goldbach como problema de control (idea #11).

Trayectoria N_t = N_0 + 2t. En cada N el controlador elige un primo p (acción) y el
sistema responde q=N-p; éxito si q es primo (mercado aclarado). Goldbach = existe una
política que siempre acierta. Pregunta computable: ¿qué política simple aclara el
mercado más rápido / más confiable?

Política CONSCIENTE DE RESIDUOS: usa N mod (primos <= B) para elegir p tal que q=N-p
sea coprimo con esos primos (evita desperdiciar intentos en q divisibles por 3,5,...).
Cuantifica el valor de la información aritmética local (la serie singular) para el
controlador. Dos métricas:
  - single-shot: tasa de éxito de elegir UN p (N-p primo);
  - secuencial: nº de intentos hasta hallar un socio primo (costo de aclaramiento).
"""
from __future__ import annotations
import numpy as np

from sieve import Tables


def _small_primes(B):
    return [p for p in (3, 5, 7, 11, 13) if p <= B]


def policies_window(T: Tables, Nlo, Nhi, Bs=(0, 3, 5, 7), stride=5):
    """Para una ventana de N par, tasa de éxito single-shot y costo secuencial medio
    de la política consciente-de-residuos con presupuesto de info B (B=0 = naive)."""
    isp = T.is_prime
    primes = T.primes
    Ns = np.arange(Nlo if Nlo % 2 == 0 else Nlo + 1, Nhi + 1, 2)[::stride]
    sp = primes[primes <= Nhi // 2 + 1]
    out = {}
    for B in Bs:
        small = _small_primes(B)
        succ = 0; cost = []
        for N in Ns:
            cand = sp[sp <= N // 2]
            q = N - cand
            # "bueno" = q coprimo con los primos <= B
            good_mask = np.ones(len(cand), dtype=bool)
            for ell in small:
                good_mask &= (q % ell != 0)
            # single-shot: mayor p (cerca N/2) entre los buenos
            gi = np.nonzero(good_mask)[0]
            psel = cand[gi[-1]] if len(gi) else cand[-1]
            succ += bool(isp[N - psel])
            # secuencial: probar buenos (de mayor a menor p) y luego el resto
            order = np.concatenate([gi[::-1], np.nonzero(~good_mask)[0][::-1]])
            qo = N - cand[order]
            hit = np.argmax(isp[qo]) if isp[qo].any() else len(qo) - 1
            cost.append(hit + 1)
        out[B] = {"success": succ / len(Ns), "cost_mean": float(np.mean(cost)),
                  "cost_p90": float(np.percentile(cost, 90))}
    return Ns, out


if __name__ == "__main__":
    T = Tables(2_000_000)
    Ns, out = policies_window(T, 1_000_000, 1_050_000)
    print(f"Ventana [1e6,1.05e6], {len(Ns)} N (submuestra)")
    print("  B (info)   éxito single-shot   costo secuencial medio   p90")
    for B, r in out.items():
        tag = "naive" if B == 0 else f"coprimo<= {B}"
        print(f"   {B:>2} ({tag:11s})   {100*r['success']:5.1f}%            "
              f"{r['cost_mean']:.2f}                {r['cost_p90']:.0f}")
    b0, bm = out[0], out[max(out)]
    print(f"  -> con info hasta {max(out)}: éxito {100*b0['success']:.0f}%->{100*bm['success']:.0f}%, "
          f"costo {b0['cost_mean']:.2f}->{bm['cost_mean']:.2f} ({100*(1-bm['cost_mean']/b0['cost_mean']):.0f}% menos)")
