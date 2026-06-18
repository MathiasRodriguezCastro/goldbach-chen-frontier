"""
Two robustness checks for the singular-exponent drift beta_2(X).

(A) Cramer/Poisson NULL CONTROL. Replace the actual prime and semiprime indicators by
    independent Poisson counts with the singular-series-matched local means
        E[R_1(N)] = 2 C2 S(N) N/log^2 N,    E[R_2(N)] = E[R_1(N)] * W_f(N),
    and run the SAME beta_2 regression on the surrogate. If the surrogate recovers the
    observed drift, the drift is a consequence of the local blocking structure (W_f), not of
    higher-order arithmetic correlations among the actual primes. S(N) and W_f(N) depend only on
    the small prime divisors of N, so this needs no sieve to X (seconds, not minutes).

(B) MODEL COMPARISON for the drift. Over the computed range loglog X in [2.75, 3.03] the data
    cannot, by themselves, separate the law beta_2 = 1 - c/loglog X (limit 1) from a finite
    limit beta_2 = beta_inf - c/loglog X (beta_inf < 1) or a different power
    beta_2 = 1 - c/(loglog X)^p. We fit all of them and report the fits honestly.

Uso: python3 run_nullcontrol.py
"""
import os
import sys
import json
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from segmented import _windowed_singular_series
from heuristics import TWIN_PRIME_C2


def blocked_Wf(N, isp):
    """Blocked channel sum W_f(N) = sum_{3<=r<=sqrt N, r NOT | N} (r-1)/(r-2) (1/r) logN/log(N/r).

    The r|N exclusion is the mechanism: blocking those channels is what couples W_f to S(N)."""
    Nf = N.astype(np.float64)
    logN = np.log(Nf)
    W = np.zeros_like(Nf)
    rmax = int(np.sqrt(int(N.max()))) + 1
    small = np.nonzero(isp[:rmax + 1])[0]
    small = small[small >= 3]
    for r in small:
        r = int(r)
        mask = (N > r * r) & (N % r != 0)        # r <= sqrt(N) AND r does not divide N
        if not mask.any():
            continue
        nn = Nf[mask]
        W[mask] += ((r - 1.0) / (r - 2.0)) * (1.0 / r) * (logN[mask] / np.log(nn / r))
    return W

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
W = 2_000_000
XS = [int(v) for v in (6e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9)]
NDRAW = 32
SEED = 20260619


def small_is_prime(M):
    s = np.ones(M + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(M ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def beta2_slope(S, R1, R2):
    ok = (R1 > 0) & (R2 > 0)
    x = np.log(S[ok])
    y = np.log(R2[ok] / R1[ok])
    return np.polyfit(x, y, 1)[0] + 1.0


def main():
    rng = np.random.default_rng(SEED)
    isp = small_is_prime(int((XS[-1] + W) ** 0.5) + 2)

    # ---- (A) Cramer/Poisson null control --------------------------------------------------
    null = []
    for X in XS:
        S_full = _windowed_singular_series(X, W, isp)
        n = np.arange(X, X + W)
        ev = n % 2 == 0
        N, S = n[ev].astype(np.int64), S_full[ev]
        Wf = blocked_Wf(N, isp)
        Nf = N.astype(np.float64)
        lam1 = 2.0 * TWIN_PRIME_C2 * S * Nf / np.log(Nf) ** 2
        lam2 = lam1 * Wf
        betas = np.empty(NDRAW)
        for k in range(NDRAW):
            R1 = rng.poisson(lam1)
            R2 = rng.poisson(lam2)
            betas[k] = beta2_slope(S, R1, R2)
        null.append({"X": X, "beta2_cramer": float(betas.mean()),
                     "sd": float(betas.std()), "meanWf": float(Wf.mean())})
        print(f"X={X:>11}: beta2_Cramer={betas.mean():.4f} +/- {betas.std():.4f}  "
              f"<Wf>={Wf.mean():.4f}", flush=True)

    # actual beta_2 from the block-bootstrap run
    boot = json.load(open(os.path.join(DATA, "bootstrap.json")))
    actual = {p["X"]: p["beta2"] for p in boot["per_X"]}
    for row in null:
        row["beta2_actual"] = actual.get(row["X"])

    # ---- (B) model comparison on the actual beta_2 ----------------------------------------
    X = np.array([p["X"] for p in boot["per_X"]], float)
    b = np.array([p["beta2"] for p in boot["per_X"]])
    L = np.log(np.log(X + W / 2.0))

    def rms(pred):
        return float(np.sqrt(np.mean((b - pred) ** 2)))

    # M1: beta = 1 - c/L  (limit 1, 1 param)
    c1 = float(np.sum((1.0 - b) * (1.0 / L)) / np.sum((1.0 / L) ** 2))
    m1 = {"form": "1 - c/loglogX", "c": c1, "rms": rms(1 - c1 / L), "limit": 1.0}
    # M2: beta = binf - c/L  (free limit, 2 params)  -> linear in (1, 1/L)
    A = np.vstack([np.ones_like(L), 1.0 / L]).T
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    binf, negc = coef
    resid = b - A @ coef
    dof = len(b) - 2
    cov = (resid @ resid / dof) * np.linalg.inv(A.T @ A)
    se_binf = float(np.sqrt(cov[0, 0]))
    m2 = {"form": "binf - c/loglogX", "binf": float(binf), "se_binf": se_binf,
          "c": float(-negc), "rms": rms(A @ coef)}
    # M3: beta = 1 - c/L^p  (free power, 2 params)
    try:
        popt, pcov = curve_fit(lambda L, c, p: 1 - c / L ** p, L, b,
                               p0=[c1, 1.0], maxfev=10000)
        m3 = {"form": "1 - c/(loglogX)^p", "c": float(popt[0]), "p": float(popt[1]),
              "se_p": float(np.sqrt(pcov[1, 1])), "rms": rms(1 - popt[0] / L ** popt[1])}
    except Exception as e:
        m3 = {"form": "1 - c/(loglogX)^p", "error": str(e)}
    # linear (eq. beta-loglog): beta = a + b*L
    bl, al = np.polyfit(L, b, 1)
    mlin = {"form": "a + b*loglogX", "a": float(al), "b": float(bl), "rms": rms(al + bl * L)}

    out = {"W": W, "seed": SEED, "loglogX_range": [float(L.min()), float(L.max())],
           "null_control": null,
           "models": {"M1_limit1": m1, "M2_freelimit": m2, "M3_power": m3, "linear": mlin}}
    json.dump(out, open(os.path.join(DATA, "nullcontrol.json"), "w"), indent=2)

    print("\n=== (A) Cramer null control: actual vs surrogate beta_2 ===")
    for r in null:
        print(f"  X={r['X']:>11}: actual={r['beta2_actual']:.4f}  "
              f"Cramer={r['beta2_cramer']:.4f} (sd {r['sd']:.4f})")
    print(f"\n=== (B) model comparison (loglogX in "
          f"[{L.min():.2f},{L.max():.2f}], 8 windows) ===")
    print(f"  M1  beta=1-c/L           : c={m1['c']:.3f}            rms={m1['rms']:.4f}")
    print(f"  M2  beta=binf-c/L        : binf={m2['binf']:.3f}+/-{m2['se_binf']:.3f}  "
          f"rms={m2['rms']:.4f}")
    if "p" in m3:
        print(f"  M3  beta=1-c/L^p         : p={m3['p']:.3f}+/-{m3['se_p']:.3f}      "
              f"rms={m3['rms']:.4f}")
    print(f"  lin beta=a+b*L           : a={mlin['a']:.3f} b={mlin['b']:.3f}  "
          f"rms={mlin['rms']:.4f}")
    print(f"  saved: {os.path.relpath(os.path.join(DATA, 'nullcontrol.json'))}")


if __name__ == "__main__":
    main()
