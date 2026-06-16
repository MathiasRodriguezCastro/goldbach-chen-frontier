"""
Formulaciones MIP de selección sobre la frontera Goldbach--Chen (secciones 6-7
del .tex), resueltas con Gurobi.

Idea: elegir UNA representación N=p+q por cada N par de una ventana, minimizando
combinaciones de:
  - conmutaciones de tipo   v_N = |u_N - u_{N-2}|       (oscilación primo<->semiprimo)
  - variación espacial      r_N = |s_N - s_{N-2}|       (suavidad de q_N/N)
  - desbalance              |s_N - 1/2|
y/o el nº total de semiprimos usados (formulación de mínimo-semiprimo).

Como Goldbach se cumple empíricamente (R1>0), minimizar v_N a secas da 0 (elegir
siempre primo). Las preguntas con contenido aparecen al RESTRINGIR el conjunto
admisible (p.ej. una banda balanceada p∈[N/2-ΔN, N/2]) — entonces puede ser
necesario un semiprimo: el "rescate de Chen" se vuelve forzado y medible.
"""
from __future__ import annotations
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from sieve import Tables


def enumerate_window(T: Tables, Nlo: int, Nhi: int, band: float | None = None):
    """
    Para cada N par en [Nlo, Nhi], lista de representaciones admisibles
    (p, q, tipo, s=q/N). Si band=δ (0<δ<=0.5), solo se admiten p con
    p/N ∈ [1/2-δ, 1/2] (representaciones balanceadas, q>=p).
    Devuelve dict N -> list[(p, q, tipo, s)].
    """
    reps = {}
    primes = T.primes
    for N in range(Nlo if Nlo % 2 == 0 else Nlo + 1, Nhi + 1, 2):
        pe = primes[primes < N]
        if band is not None:
            lo = (0.5 - band) * N
            pe = pe[(pe >= lo) & (pe <= N / 2)]
        q = N - pe
        is_p = T.is_prime[q]
        is_s = T.is_semiprime[q]
        keep = is_p | is_s
        p = pe[keep]; qq = q[keep]
        typ = np.where(T.is_prime[qq], 0, 1)  # 0=primo, 1=semiprimo
        lst = [(int(p[i]), int(qq[i]), int(typ[i]), qq[i] / N) for i in range(len(p))]
        reps[N] = lst
    return reps


def solve_selection(reps, alpha=0.0, beta=0.0, gamma=0.0, lam_semiprime=0.0,
                    time_limit=60, verbose=False):
    """
    Resuelve el MIP de selección. Pesos:
      alpha  -> Σ conmutaciones de tipo v_N
      beta   -> Σ variación espacial r_N
      gamma  -> Σ |s_N - 1/2|
      lam_semiprime -> Σ u_N (nº de semiprimos usados)
    Devuelve dict con N, p, q, tipo (u), s seleccionados y métricas.
    """
    Ns = sorted(reps.keys())
    feasible = [N for N in Ns if len(reps[N]) > 0]
    infeasible = [N for N in Ns if len(reps[N]) == 0]

    m = gp.Model("seleccion_goldbach_chen")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit

    z, u, s = {}, {}, {}
    for N in feasible:
        zz = []
        for i, (p, q, t, sv) in enumerate(reps[N]):
            zz.append(m.addVar(vtype=GRB.BINARY, name=f"z_{N}_{i}"))
        z[N] = zz
        m.addConstr(gp.quicksum(zz) == 1, name=f"cover_{N}")
        u[N] = m.addVar(vtype=GRB.BINARY, name=f"u_{N}")
        m.addConstr(u[N] == gp.quicksum(reps[N][i][2] * zz[i] for i in range(len(zz))))
        s[N] = m.addVar(lb=0.0, ub=1.0, name=f"s_{N}")
        m.addConstr(s[N] == gp.quicksum(reps[N][i][3] * zz[i] for i in range(len(zz))))

    obj = gp.LinExpr()
    if lam_semiprime:
        obj += lam_semiprime * gp.quicksum(u[N] for N in feasible)
    if gamma:
        for N in feasible:
            d = m.addVar(lb=0.0)
            m.addConstr(d >= s[N] - 0.5); m.addConstr(d >= 0.5 - s[N])
            obj += gamma * d
    # acoplamientos entre N consecutivos (N, N-2 ambos factibles)
    for N in feasible:
        if (N - 2) in u:  # consecutivo factible
            if alpha:
                v = m.addVar(lb=0.0)
                m.addConstr(v >= u[N] - u[N - 2]); m.addConstr(v >= u[N - 2] - u[N])
                obj += alpha * v
            if beta:
                r = m.addVar(lb=0.0)
                m.addConstr(r >= s[N] - s[N - 2]); m.addConstr(r >= s[N - 2] - s[N])
                obj += beta * r
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    sel = {}
    for N in feasible:
        i = int(np.argmax([zz.X for zz in z[N]]))
        p, q, t, sv = reps[N][i]
        sel[N] = (p, q, t, sv)
    n_semi = sum(1 for N in feasible if sel[N][2] == 1)
    switches = sum(1 for N in feasible if (N - 2) in sel and sel[N][2] != sel[N - 2][2])
    tv = sum(abs(sel[N][3] - sel[N - 2][3]) for N in feasible if (N - 2) in sel)
    return {
        "sel": sel, "feasible": feasible, "infeasible": infeasible,
        "obj": m.ObjVal, "status": m.Status,
        "n_semiprime": n_semi, "type_switches": switches, "total_variation": tv,
        "n_N": len(feasible),
    }


if __name__ == "__main__":
    T = Tables(60000)
    print("== Experimento A: ventana sin restricción, min variación espacial ==")
    reps = enumerate_window(T, 10000, 12000)
    rA = solve_selection(reps, beta=1.0)
    print(f"  N={rA['n_N']}  TV(s)={rA['total_variation']:.3f}  semiprimos={rA['n_semiprime']}"
          f"  switches={rA['type_switches']}")

    print("== Experimento B: banda balanceada δ=0.05, min semiprimos ==")
    repsB = enumerate_window(T, 10000, 12000, band=0.05)
    rB = solve_selection(repsB, lam_semiprime=1.0)
    print(f"  N factibles={rB['n_N']}  N SIN rep balanceada={len(rB['infeasible'])}")
    print(f"  semiprimos forzados (mínimo)={rB['n_semiprime']}  switches={rB['type_switches']}")
