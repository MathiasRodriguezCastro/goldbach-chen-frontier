"""
Auditoría del estimador beta_2 (responde R5 y S2 del peer review).

R5: define UN estimador (regresión OLS de log(R2/R1) sobre log S(N) sobre [Nmin,X])
    y reporta su error estándar; explica la diferencia entre el valor ACUMULADO
    (sobre [2,X], el de la Tabla 1) y el LOCAL en ventana (el del apéndice).
S2: cuantifica la contribución del átomo 3|N a la pendiente medida, recomputando
    el exponente con la subpoblación 3∤N (átomo removido).

Uso: python3 run_estimator_audit.py [X]   (def X=2_000_000)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sieve import Tables
from counts import representation_counts
from heuristics import singular_series_goldbach


def ols_slope_se(x, y):
    """Pendiente OLS, intercepto, R^2 y error estándar de la pendiente."""
    n = x.size
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    b = np.sum((x - xm) * (y - ym)) / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    s2 = np.sum(resid ** 2) / (n - 2)
    se_b = np.sqrt(s2 / sxx)
    r2 = 1.0 - np.var(resid) / np.var(y)
    return b, a, r2, se_b, n


def beta2(N, R1, R2, S, mask):
    x = np.log(S[N[mask]])
    y = np.log(R2[mask] / R1[mask])
    b, a, r2, se, n = ols_slope_se(x, y)
    return b + 1.0, se, r2, n  # slope = beta2 - 1


def main():
    X = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    T = Tables(X)
    N, R1, R2 = representation_counts(T)
    S = singular_series_goldbach(T.X, T.is_prime)
    Nmin = 5000
    base = (N >= Nmin) & (R1 > 0) & (R2 > 0)

    print(f"=== Auditoría del estimador beta_2  (X={X:,}, N>={Nmin}) ===\n")

    # --- R5: UN estimador acumulado sobre [Nmin, X] con error estándar ---
    b_all, se_all, r2_all, n_all = beta2(N, R1, R2, S, base)
    print("R5  Estimador canónico: OLS de log(R2/R1) ~ log S(N), acumulado sobre [Nmin,X]")
    print(f"    beta_2 = {b_all:.4f} +/- {se_all:.4f} (SE)   R^2={r2_all:.3f}   n={n_all:,}")
    print(f"    -> el '0.497' de la Tabla 1 es ESटE estimador (acumulado).")

    # comparar con la mediana de pendientes locales en ventanas de 2e5
    win = 200_000
    edges = list(range(Nmin, X - win, win))
    locs = []
    for lo in edges:
        m = base & (N >= lo) & (N < lo + win)
        if m.sum() < 2000:
            continue
        bw, _, _, _ = beta2(N, R1, R2, S, m)
        locs.append(bw)
    locs = np.array(locs)
    print(f"    pendiente LOCAL (ventanas de {win:,}): mediana={np.median(locs):.4f}, "
          f"media={locs.mean():.4f}, ultima={locs[-1]:.4f}  (n_ventanas={locs.size})")
    print(f"    -> el '0.531' del apéndice es la ventana LOCAL alta en X (no la acumulada).\n")

    # --- S2: contribución del átomo 3|N ---
    m3 = base & (N % 3 != 0)        # átomo removido
    m3c = base & (N % 3 == 0)
    b_no3, se_no3, r2_no3, n_no3 = beta2(N, R1, R2, S, m3)
    print("S2  Contribución del canal r=3 (el átomo 3|N) a la pendiente:")
    print(f"    beta_2 (todos)      = {b_all:.4f}   deficit 1-beta_2 = {1-b_all:.4f}")
    print(f"    beta_2 (3 no divide N) = {b_no3:.4f} +/- {se_no3:.4f}   deficit = {1-b_no3:.4f}   n={n_no3:,}")
    share = 1.0 - (1 - b_no3) / (1 - b_all)
    print(f"    -> removiendo el átomo 3|N el deficit cae {100*share:.0f}%: "
          f"el canal r=3 explica ~{100*share:.0f}% de (1-beta_2).")

    # chequeo del mecanismo: 3|N duplica S y bloquea r=3
    logS3 = np.log(S[N[m3c]]).mean()
    logS_no3 = np.log(S[N[m3]]).mean()
    ratio3 = np.median(R2[m3c] / R1[m3c])
    ratio_no3 = np.median(R2[m3] / R1[m3])
    print(f"    mecanismo: <log S | 3|N>={logS3:.3f} vs <log S | 3∤N>={logS_no3:.3f} (Δ≈log2={np.log(2):.3f});")
    print(f"               mediana R2/R1 | 3|N = {ratio3:.3f} < | 3∤N = {ratio_no3:.3f} (canal r=3 bloqueado).")


if __name__ == "__main__":
    main()
