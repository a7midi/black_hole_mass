\
#!/usr/bin/env python
"""
20_test_periodicity_M2.py

Scan lattice spacing Δ for periodicity in M^2 phases and apply Rayleigh/Kuiper tests.
Bootstrap significance and optional selection-aware nulls via mock catalogs.

Outputs: plots/scan_power_vs_delta.png and CSV with scan results.
"""
from __future__ import annotations
import os, sys, argparse, json
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import DERIVED, PLOTS, compute_phase_power, auto_delta_grid, rng

def load_samples(dataset: str) -> pd.DataFrame:
    fp = os.path.join(DERIVED, "M2_samples.parquet")
    df = pd.read_parquet(fp)
    if dataset == "all":
        return df
    else:
        return df[df["dataset"].isin([dataset])].copy()

def scan_periodicity(M2: np.ndarray, deltas: np.ndarray) -> pd.DataFrame:
    rows = []
    for d in deltas:
        stats = compute_phase_power(M2, d)
        rows.append({"delta": d, **stats})
    return pd.DataFrame(rows)

def bootstrap_null(M2: np.ndarray, deltas: np.ndarray, B: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Nonparametric bootstrap by resampling with replacement from M2 (preserves selection-induced shape).
    For each bootstrap sample we compute max Rayleigh R across Δ grid; same for Kuiper V.
    """
    R = rng(seed)
    rows = []
    n = len(M2)
    for b in tqdm(range(B), desc="bootstrap"):
        idx = R.integers(0, n, size=n)
        M2b = M2[idx]
        res = scan_periodicity(M2b, deltas)
        rows.append({"max_R": res["R"].max(), "max_V": res["V"].max()})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["GW","XRB","EHT","all"], default="all")
    ap.add_argument("--B", type=int, default=400, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-Aeff", action="store_true", help="use A_eff instead of M2 when spins available")
    ap.add_argument("--n-delta", type=int, default=160)
    args = ap.parse_args()

    df = load_samples(args.dataset)
    x = df["A_eff"].values if args.use_Aeff else df["M2"].values
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    deltas = auto_delta_grid(x, n=args.n_delta)

    scan = scan_periodicity(x, deltas)
    scan.to_csv(os.path.join(PLOTS, f"scan_periodicity_{args.dataset}.csv"), index=False)

    # Bootstrap null for multiple-testing corrected significance
    boot = bootstrap_null(x, deltas, B=args.B, seed=args.seed)
    boot.to_csv(os.path.join(PLOTS, f"bootstrap_null_{args.dataset}.csv"), index=False)

    # Compute empirical p-values for the observed max statistics
    obs_max_R = scan["R"].max()
    obs_max_V = scan["V"].max()
    p_R = (boot["max_R"] >= obs_max_R).mean()
    p_V = (boot["max_V"] >= obs_max_V).mean()

    # Plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.semilogx(scan["delta"], scan["R"])
    plt.xlabel(r"$\Delta$ in $M^2$ units")
    plt.ylabel("Rayleigh resultant R")
    plt.title(f"Periodicity scan in $M^2$ ({args.dataset}); maxR={obs_max_R:.3f}, p≈{p_R:.3g}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"scan_power_vs_delta_{args.dataset}.png"), dpi=180)

    plt.figure()
    plt.semilogx(scan["delta"], scan["V"])
    plt.xlabel(r"$\Delta$ in $M^2$ units")
    plt.ylabel("Kuiper V")
    plt.title(f"Kuiper scan in $M^2$ ({args.dataset}); maxV={obs_max_V:.3f}, p≈{p_V:.3g}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"scan_power_vs_delta_kuiper_{args.dataset}.png"), dpi=180)

    print(json.dumps({
        "dataset": args.dataset,
        "use_Aeff": bool(args.use_Aeff),
        "obs_max_R": float(obs_max_R),
        "obs_max_V": float(obs_max_V),
        "p_R_boot": float(p_R),
        "p_V_boot": float(p_V),
        "best_delta_by_R": float(scan.loc[scan["R"].idxmax(), "delta"]),
        "best_delta_by_V": float(scan.loc[scan["V"].idxmax(), "delta"]),
    }, indent=2))

if __name__ == "__main__":
    main()
