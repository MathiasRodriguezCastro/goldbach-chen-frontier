#!/usr/bin/env python3
"""
Regenerate every figure and derived table of the paper from scratch.

Usage:
    python3 run_all.py [--fast] [--with-gurobi]

  --fast        skip the slow drivers (scaling to 1e9, betalaw, constant, bases);
                regenerates the rest in a few minutes.
  --with-gurobi also run the optional selection-MIP driver (needs a Gurobi license).

Each row of DRIVERS maps a paper figure/table to the script that produces it
(see also the "Figure -> driver" table in the README). Self-tests live in each
src/<module>.py and run with `python3 src/<module>.py`.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

# (driver, args, produces, slow?, needs_gurobi?)
DRIVERS = [
    ("run_counts.py",          ["2000000"], "Figs 1-5, summary.json",            False, False),
    ("run_estimator_audit.py", [],          "estimator/atom audit (sec 3)",      False, False),
    ("run_layers.py",          [],          "Fig 10, layers.json",               False, False),
    ("run_balance.py",         [],          "Fig 11, balance.json",              False, False),
    ("run_continuity.py",      [],          "Fig 8, continuity.json",            False, False),
    ("run_transport.py",       [],          "Fig 13, transport.json",            False, False),
    ("run_tda.py",             [],          "Fig 14, tda.json",                  False, False),
    ("run_dynamics.py",        [],          "Fig 15, dynamics.json",             False, False),
    ("run_firstmoment.py",     [],          "first-moment responses (App A)",    False, False),
    ("run_exceptional.py",     [],          "concentration table (App A)",       False, False),
    ("run_spectral.py",        [],          "Fig 21, spectral.json",             False, False),
    ("run_bases.py",           [],          "Fig 24, bases.json",                True,  False),
    ("run_scaling.py",         [],          "Fig 9, scaling.json (sieve to 1e9)",True,  False),
    ("run_bootstrap.py",       [],          "block-bootstrap CIs, bootstrap.json",True,  False),
    ("run_betalaw.py",         [],          "Fig 12, betalaw.json",              True,  False),
    ("run_constant.py",        [],          "Fig 16, constant",                  True,  False),
    # supplement figures
    ("run_energy.py",          [],          "Fig 17 (supplement)",               False, False),
    ("run_robustness.py",      [],          "Fig 18 (supplement)",               False, False),
    ("run_market.py",          [],          "Fig 19 (supplement)",               False, False),
    ("run_control.py",         [],          "Fig 20 (supplement)",               False, False),
    ("run_information.py",     [],          "Fig 22 (supplement)",               False, False),
    ("run_geometry.py",        [],          "Fig 23 (supplement)",               False, False),
    # optional, needs a Gurobi license
    ("run_mip.py",             [],          "Figs 6-7 (selection MIP)",          False, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip slow drivers")
    ap.add_argument("--with-gurobi", action="store_true", help="also run the MIP driver")
    args = ap.parse_args()

    ok, failed, skipped = [], [], []
    t0 = time.time()
    for driver, dargs, produces, slow, needs_gurobi in DRIVERS:
        if slow and args.fast:
            skipped.append((driver, "slow"))
            continue
        if needs_gurobi and not args.with_gurobi:
            skipped.append((driver, "needs --with-gurobi"))
            continue
        print(f">>> {driver:24s} -> {produces}")
        t = time.time()
        r = subprocess.run([sys.executable, str(EXP / driver), *dargs])
        dt = time.time() - t
        if r.returncode == 0:
            ok.append((driver, dt))
            print(f"    OK ({dt:.0f}s)")
        else:
            failed.append((driver, r.returncode))
            print(f"    FAIL (exit {r.returncode})")

    print("\n=== summary ===")
    print(f"  ok:      {len(ok)}  ({time.time()-t0:.0f}s total)")
    print(f"  failed:  {len(failed)}  {[d for d, _ in failed]}")
    print(f"  skipped: {len(skipped)}  {[d for d, _ in skipped]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
