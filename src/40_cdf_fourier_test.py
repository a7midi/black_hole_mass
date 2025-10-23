\
#!/usr/bin/env python
"""
40_cdf_fourier_test.py

Empirical CDF staircase residuals for M^2 and a simple Fourier spectrum of the residuals.
Reports the dominant frequency and the implied Δ ≈ 1/f_peak in the rank domain, then
maps it back to the M^2 domain using the observed dynamic range.
"""
from __future__ import annotations
import os, argparse, numpy as np, pandas as pd, json
import matplotlib.pyplot as plt

from common import DERIVED, PLOTS, fourier_power_on_ecdf_residuals

def load_x(dataset: str, use_Aeff: bool):
    df = pd.read_parquet(os.path.join(DERIVED, "M2_samples.parquet"))
    if dataset != "all":
        df = df[df["dataset"].isin([dataset])]
    x = df["A_eff"].values if use_Aeff else df["M2"].values
    return x[np.isfinite(x)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["GW","XRB","EHT","all"], default="all")
    ap.add_argument("--use-Aeff", action="store_true")
    args = ap.parse_args()

    x = load_x(args.dataset, args.use_Aeff)
    freqs, power = fourier_power_on_ecdf_residuals(x)
    # ignore the DC bin (freq 0)
    freqs, power = freqs[1:], power[1:]
    k = np.argmax(power)
    f_peak = float(freqs[k])
    # Map heuristic: in rank domain, one cycle ≈ one period across [0,1], so Δ_rank ≈ 1/f_peak.
    # Convert to M^2 using the observed range (units-free mapping).
    M2_lo, M2_hi = float(np.min(x)), float(np.max(x))
    Delta_guess = (M2_hi - M2_lo) / max(f_peak, 1e-6)

    # Save figure
    plt.figure()
    plt.plot(freqs, power)
    plt.xlabel("Frequency (rank domain)")
    plt.ylabel("Power")
    plt.title(f"ECDF residual spectrum ({args.dataset}); f_peak={f_peak:.3g} -> Δ≈{Delta_guess:.3g}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"ecdf_fourier_{args.dataset}.png"), dpi=180)

    with open(os.path.join(PLOTS, f"ecdf_fourier_{args.dataset}.json"), "w") as f:
        json.dump({"f_peak_rank": f_peak, "Delta_guess_M2": Delta_guess}, f, indent=2)

    print(json.dumps({"f_peak_rank": f_peak, "Delta_guess_M2": Delta_guess}, indent=2))

if __name__ == "__main__":
    main()
