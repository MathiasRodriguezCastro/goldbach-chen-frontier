"""
Block-bootstrap confidence intervals for the singular exponent beta_2(X), the
non-circular product (1-beta_2)*<R2/R1>, and the loglog-drift fit (eq. beta-loglog).

Consecutive even N are arithmetically dependent (they share small-prime structure),
so an i.i.d. bootstrap would understate the error. We use a MOVING-BLOCK bootstrap
over the per-N sequence within each window [X, X+W]: blocks of L consecutive even
integers are resampled with replacement (wrap-around), preserving short-range
arithmetic dependence. For each resample we recompute

    beta_2 = 1 + slope of  log(R2/R1)  on  log S(N),
    product = (1 - beta_2) * mean(R2/R1),

and, jointly across windows, refit  beta_2(X) = a + b*loglog X  to test that the
upward drift slope b is significant (b>0). Reports 95% percentile CIs.

Also extends the non-circular validation grid to X=10^9 (run_betalaw stopped at 2e8).

Uso: python3 run_bootstrap.py
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from segmented import windowed_counts

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

W = 2_000_000          # window length
L = 1_000              # block length (even-N index units; spans ~2000 integers)
B = 1_000              # bootstrap resamples
XS = [int(v) for v in (6e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8, 1e9)]
SEED = 20260617


def beta2_product(x, y, ratio):
    """beta_2 (=1+slope of y on x) and the non-circular product.

    Slope via cov/var (much faster than polyfit under heavy bootstrap)."""
    mx, my = x.mean(), y.mean()
    b = (np.dot(x, y) / x.size - mx * my) / (np.dot(x, x) / x.size - mx * mx)
    beta2 = b + 1.0
    return beta2, (1.0 - beta2) * ratio.mean()


def block_idx(n, L, rng):
    """Moving-block bootstrap index vector of length n (wrap-around blocks)."""
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=nb)
    idx = (starts[:, None] + np.arange(L)[None, :]).ravel() % n
    return idx[:n]


def ci(samples, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def main():
    rng = np.random.default_rng(SEED)
    cols = {}   # X -> (x, y, ratio)
    point = []  # per-X point estimates
    for X in XS:
        t = time.time()
        N, R1, R2, S = windowed_counts(X, W)
        ok = R1 > 0
        R1, R2, S = R1[ok], R2[ok], S[ok]
        x, y, ratio = np.log(S), np.log(R2 / R1), R2 / R1
        cols[X] = (x, y, ratio)
        beta2, prod = beta2_product(x, y, ratio)
        point.append({"X": X, "beta2": float(beta2), "meanW": float(ratio.mean()),
                      "product": float(prod), "nN": int(len(x))})
        print(f"X={X:>11}: beta2={beta2:.4f}  <R2/R1>={ratio.mean():.4f}  "
              f"product={prod:.4f}  ({time.time()-t:.0f}s)", flush=True)

    # ---- moving-block bootstrap -------------------------------------------------
    loglogX = np.array([np.log(np.log(X + W / 2)) for X in XS])
    beta_bs = {X: np.empty(B) for X in XS}
    prod_bs = {X: np.empty(B) for X in XS}
    slope_bs = np.empty(B)      # loglog-drift slope b
    intsc_bs = np.empty(B)      # loglog-drift intercept a
    t = time.time()
    for k in range(B):
        bvec = np.empty(len(XS))
        for j, X in enumerate(XS):
            x, y, ratio = cols[X]
            idx = block_idx(len(x), L, rng)
            beta2, prod = beta2_product(x[idx], y[idx], ratio[idx])
            beta_bs[X][k] = beta2
            prod_bs[X][k] = prod
            bvec[j] = beta2
        b, a = np.polyfit(loglogX, bvec, 1)
        slope_bs[k] = b
        intsc_bs[k] = a
    print(f"  bootstrap B={B}, L={L}: {time.time()-t:.0f}s", flush=True)

    # loglog-drift point fit on the point estimates
    b0, a0 = np.polyfit(loglogX, [p["beta2"] for p in point], 1)
    slo_lo, slo_hi = ci(slope_bs)
    int_lo, int_hi = ci(intsc_bs)

    out = {"W": W, "L": L, "B": B, "seed": SEED, "per_X": []}
    for p in point:
        X = p["X"]
        bl, bh = ci(beta_bs[X])
        pl, ph = ci(prod_bs[X])
        p.update({"beta2_lo": bl, "beta2_hi": bh,
                  "product_lo": pl, "product_hi": ph})
        out["per_X"].append(p)
    out["loglog_fit"] = {"intercept": float(a0), "slope": float(b0),
                         "slope_lo": slo_lo, "slope_hi": slo_hi,
                         "intercept_lo": int_lo, "intercept_hi": int_hi}
    json.dump(out, open(os.path.join(DATA, "bootstrap.json"), "w"), indent=2)

    print("\n=== beta_2(X) with 95% block-bootstrap CI ===")
    for p in out["per_X"]:
        print(f"  X={p['X']:>11}: beta2={p['beta2']:.3f} "
              f"[{p['beta2_lo']:.3f}, {p['beta2_hi']:.3f}]   "
              f"product={p['product']:.3f} [{p['product_lo']:.3f}, {p['product_hi']:.3f}]")
    print(f"\nloglog-drift fit  beta2 = {a0:.3f} + {b0:.3f} * loglog X")
    print(f"  slope b in [{slo_lo:.3f}, {slo_hi:.3f}]  (b>0 => significant upward drift)")
    print(f"  saved: {os.path.relpath(os.path.join(DATA, 'bootstrap.json'))}")


if __name__ == "__main__":
    main()
