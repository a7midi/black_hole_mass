#!/usr/bin/env python
"""
30_bayes_mixture_lattice.py  —  lightweight Bayesian test for lattice periodicity in M^2

Model idea (robust + fast on laptops):
  • Work with phases φ = 2π · frac(M2 / Δ).
  • If M2 lives near a lattice nΔ with Gaussian fuzz σ ≪ Δ, the φ’s cluster near 0.
  • We model φ with a mixture: Uniform(0,2π) with weight (1−f) plus VonMises(μ=0, κ) with weight f.
  • Priors: log Δ ~ Uniform[log Dmin, log Dmax],  f ~ Beta(1,1),  κ ~ HalfNormal(σ=5).
  • Derived: σ/Δ ≈ 1 / (2π √κ)  (large-κ approximation connecting von Mises to wrapped normal).

Why this design?
  - It avoids building a huge Gaussian comb in M^2 directly (saves RAM and avoids float32/64 clashes).
  - It is still faithful to the key prediction: periodic alignment of M^2 modulo Δ.
  - It nests H0 (no lattice) at f=0, so we can report a Savage–Dickey style Bayes factor via a small-ε interval.

CLI:
  python src/30_bayes_mixture_lattice.py --dataset GW --draws 1200 --tune 800 --max-N 40000 --chains 2 --cores 1
"""
from __future__ import annotations
import os, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyMC imports (float64 everywhere to avoid dtype crashes)
import pymc as pm
import pytensor.tensor as pt
# Force float64 in the backend if available
try:
    import pytensor
    pytensor.config.floatX = "float64"
except Exception:
    try:
        import aesara
        aesara.config.floatX = "float64"
    except Exception:
        pass

# -------------------------------
# Utilities
# -------------------------------
def load_M2(dataset: str, samples_path: Path = Path("data/derived/M2_samples.parquet"),
            use_Aeff: bool = False, max_N: int | None = None, seed: int = 42) -> np.ndarray:
    """Return a 1D numpy array of M^2 (or A_eff proxy) for the selected dataset."""
    df = pd.read_parquet(samples_path)
    if dataset != "all":
        df = df.query("dataset == @dataset").copy()
        if df.empty:
            raise SystemExit(f"No rows for dataset={dataset} in {samples_path}")

    col = "Aeff" if use_Aeff and "Aeff" in df.columns else "M2"
    x = df[col].to_numpy(dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if max_N is not None and len(x) > max_N:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=max_N, replace=False)
    return x

def auto_delta_bounds(x: np.ndarray) -> tuple[float, float]:
    """Loose but safe bounds for Δ given the support of x = M^2 (in Msun^2)."""
    q1, q99 = np.quantile(x, [0.01, 0.99])
    Dmin = max(q1/30.0, 1e-3)     # let Δ be smaller than the lower scale, but not crazy small
    Dmax = q99/2.0                # Δ cannot exceed half the upper support (else few bins exist)
    if Dmax <= Dmin:
        Dmax = Dmin * 5.0
    return float(Dmin), float(Dmax)

def phases_from_M2(x: np.ndarray, Delta):
    """φ = 2π · frac(x / Δ) computed symbolically with PyMC math."""
    return 2.0 * np.pi * ( (x/Delta) - pm.math.floor(x/Delta) )

def posterior_interval_BF10(samples_f: np.ndarray, eps: float = 0.01) -> float:
    """
    Savage–Dickey (interval) approximation for BF10 (lattice vs no-lattice).
    Prior: f ~ Beta(1,1) so P(f<eps) = eps.
    Posterior: p_post = fraction of posterior draws with f<eps.
    Then BF10 ≈ eps / p_post  (smaller posterior mass near 0 means stronger evidence for lattice).
    """
    p_post = max(1e-9, np.mean(samples_f < eps))
    return eps / p_post

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=str, default="data/derived/M2_samples.parquet")
    ap.add_argument("--dataset", choices=["GW","XRB","EHT","all"], default="all")
    ap.add_argument("--use-Aeff", action="store_true", help="use A_eff proxy if present")
    ap.add_argument("--max-N", type=int, default=40000, help="downsample to at most this many points")
    ap.add_argument("--draws", type=int, default=1200)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=1)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="results/bayes")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---------- data ----------
    x = load_M2(args.dataset, Path(args.samples), use_Aeff=args.use_Aeff, max_N=args.max_N, seed=args.seed)
    if len(x) < 100:
        raise SystemExit(f"Too few samples after filtering/downsampling: N={len(x)}")
    Dmin, Dmax = auto_delta_bounds(x)

    # Cast to float64 constant for PyMC graph
    x_const = pm.math.constant(x.astype(np.float64))

    # ---------- model ----------
    with pm.Model() as model:
        # Priors
        logD = pm.Uniform("logDelta", lower=np.log(Dmin), upper=np.log(Dmax))
        Delta = pm.Deterministic("Delta", pm.math.exp(logD))

        # κ >= 0 (concentration); use a weakly-informative half-normal
        kappa = pm.HalfNormal("kappa", sigma=5.0)

        # lattice weight f in (0,1)
        f_latt = pm.Beta("f_latt", alpha=1.0, beta=1.0)

        # Derived: approx σ/Δ from κ (large-κ approximation for von Mises ~ wrapped normal)
        sigma_over_Delta = pm.Deterministic("sigma_over_Delta", 1.0/(2.0*np.pi*pm.math.sqrt(kappa + 1e-12)))

        # Phases φ = 2π frac(x/Δ)
        phi = phases_from_M2(x_const, Delta)

        # Von Mises term
        vm = pm.VonMises.dist(mu=0.0, kappa=kappa)
        log_vm = pm.logp(vm, phi)

        # Uniform term on [0,2π)
        logU = -np.log(2.0*np.pi)

        # Log-likelihood for the mixture on the circle
        log_mix = pm.math.logaddexp(pt.log1p(-f_latt) + logU,
                                  pm.math.log(f_latt) + log_vm)
        pm.Potential("likelihood", pm.math.sum(log_mix))

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.seed,
            init="jitter+adapt_diag",
            progressbar=True,
        )

    # ---------- summaries ----------
    post = idata.posterior
    Delta_samps = np.asarray(post["Delta"]).reshape(-1)
    theta_samps = np.asarray(post["sigma_over_Delta"]).reshape(-1)
    f_samps     = np.asarray(post["f_latt"]).reshape(-1)

    BF10 = posterior_interval_BF10(f_samps, eps=0.01)

    # ---------- save ----------
    import json
    summary = {
        "dataset": args.dataset,
        "N_used": int(len(x)),
        "Delta_median": float(np.median(Delta_samps)),
        "Delta_16": float(np.quantile(Delta_samps, 0.16)),
        "Delta_84": float(np.quantile(Delta_samps, 0.84)),
        "sigma_over_Delta_median": float(np.median(theta_samps)),
        "sigma_over_Delta_16": float(np.quantile(theta_samps, 0.16)),
        "sigma_over_Delta_84": float(np.quantile(theta_samps, 0.84)),
        "f_latt_median": float(np.median(f_samps)),
        "f_latt_16": float(np.quantile(f_samps, 0.16)),
        "f_latt_84": float(np.quantile(f_samps, 0.84)),
        "BF10_eps0.01": float(BF10),
        "Dmin": Dmin, "Dmax": Dmax,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---------- plots ----------
    try:
        import arviz as az
    except Exception:
        az = None

    # Trace / posterior plots
    if az is not None:
        az.to_netcdf(idata, str(outdir / "posterior.nc"))
        az.plot_posterior(idata, var_names=["Delta", "sigma_over_Delta", "f_latt"], ref_val=[None, None, 0.0])
        plt.tight_layout()
        plt.savefig(outdir / "posterior_marginals.png", dpi=180)
        plt.close()

    # Simple corner-like joint for Δ and f
    try:
        import corner
        arr = np.vstack([Delta_samps, theta_samps, f_samps]).T
        corner.corner(arr, labels=[r"$\Delta$ ($M_\odot^2$)", r"$\sigma/\Delta$", r"$f_{\rm latt}$"])
        plt.savefig(outdir / "corner.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
