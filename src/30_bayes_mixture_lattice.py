\
#!/usr/bin/env python
"""
30_bayes_mixture_lattice.py

Bayesian comparison: smooth baseline vs. lattice mixture for the distribution of M^2.
Implements a *single* mixture model containing both a smooth component and a lattice component
with mixing fraction f_latt. The null (no lattice) corresponds to f_latt=0.

We report posterior for Δ, σ/Δ, f_latt and a **Savage–Dickey Bayes factor** for H1 (f>0) vs H0 (f=0).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import pytensor.tensor as pt

from common import DERIVED, PLOTS, auto_delta_grid, rng

def load_M2(dataset: str, use_Aeff: bool = False) -> np.ndarray:
    df = pd.read_parquet(os.path.join(DERIVED, "M2_samples.parquet"))
    if dataset != "all":
        df = df[df["dataset"].isin([dataset])]
    x = df["A_eff"].values if use_Aeff else df["M2"].values
    x = x[np.isfinite(x)]
    return x

def broken_powerlaw_baseline(x: np.ndarray):
    """
    Return a simple fixed-shape baseline logpdf up to an additive constant, for reweighting inside PyMC.
    We avoid introducing free scales: the break is at the median of x; slopes are set by the empirical
    tails via Hill-type estimates.
    """
    xn = x / np.median(x)
    left_slope = 1.0 + np.clip((np.log(np.median(xn)) - np.log(np.percentile(xn, 25))) / (np.log(np.percentile(xn, 25))), -5, 5)
    right_slope = 1.0 + np.clip((np.log(np.percentile(xn, 75)) - np.log(np.median(xn))) / (np.log(np.percentile(xn, 75))), -5, 5)
    # clamp for stability
    left_slope = float(np.nan_to_num(left_slope, nan=2.0))
    right_slope = float(np.nan_to_num(right_slope, nan=2.0))
    xm = np.median(x)
    def logpdf_fn(z):
        z = pt.clip(z, 1e-12, np.max(x)*10)
        logz = pt.log(z / xm)
        # piecewise slopes
        logp = pt.switch(pt.lt(z, xm), -left_slope * pt.abs(logz), -right_slope * pt.abs(logz))
        return logp - pm.math.logsumexp(logp)  # stabilized
    return logpdf_fn

def lattice_logpdf(z, Delta, sigma, xmin, xmax):
    """
    Compute logpdf for the lattice-mixture (sum_n N(z; nΔ, σ^2)) with n covering [xmin, xmax].
    Uniform w_n in the covered range.
    """
    n_min = pt.cast(pt.floor(xmin / Delta) - 2, "int32")
    n_max = pt.cast(pt.ceil(xmax / Delta) + 2, "int32")
    n = pt.arange(n_min, n_max + 1)
    centers = Delta * pt.cast(n, Delta.dtype)
    # Gaussian kernels
    logk = -0.5 * ((z[:, None] - centers[None, :]) / sigma)**2 - pt.log(sigma) - 0.5 * pt.log(2*np.pi)
    # Uniform mixture over n
    logpdf = pm.math.logsumexp(logk, axis=1) - pt.log(n.shape[0])
    return logpdf

def savage_dickey_BF(prior_a: float, prior_b: float, posterior_samples_f: np.ndarray, at: float = 0.0, eps: float = 1e-3):
    """
    Savage-Dickey density ratio for nested hypothesis H0: f=0 vs H1: f>0
    with Beta(prior_a, prior_b) prior on f in [0,1].
    We estimate posterior density near 0 using a small window [0, eps].
    """
    from scipy.stats import beta as beta_dist
    prior_density_at_0 = beta_dist.pdf(at + eps/2, prior_a, prior_b)  # approximate at boundary
    post = np.asarray(posterior_samples_f)
    post_density_near_0 = np.mean((post >= at) & (post <= at + eps)) / eps
    BF10 = prior_density_at_0 / max(post_density_near_0, 1e-12)
    return float(BF10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["GW","XRB","EHT","all"], default="all")
    ap.add_argument("--use-Aeff", action="store_true")
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    x = load_M2(args.dataset, use_Aeff=args.use_Aeff)
    x = x[np.isfinite(x)]
    xmin, xmax = float(np.min(x)), float(np.max(x))

    # Prior ranges based only on data extents
    Dgrid = auto_delta_grid(x, n=100)
    Dmin, Dmax = float(np.min(Dgrid)), float(np.max(Dgrid))

    with pm.Model() as model:
        # Lattice parameters
        log_Delta = pm.Uniform("log_Delta", lower=np.log(Dmin), upper=np.log(Dmax))
        Delta = pm.Deterministic("Delta", pt.exp(log_Delta))
        theta = pm.Beta("theta", alpha=2.0, beta=2.0)     # σ/Δ in (0,1)
        sigma = pm.Deterministic("sigma", pt.exp(Delta) * theta)

        # Lattice fraction
        a0, b0 = 1.0, 1.0  # Uniform prior on [0,1]
        f_latt = pm.Beta("f_latt", alpha=a0, beta=b0)

        # Smooth baseline (fixed shape, normalized by construction as log-weights)
        base_logpdf_fn = broken_powerlaw_baseline(x)

        z = pm.MutableData("z", x.astype("float64"))
        logpdf_latt = lattice_logpdf(z, Delta=pt.exp(Delta), sigma=sigma, xmin=xmin, xmax=xmax)
        logpdf_base = base_logpdf_fn(z)

        # Mixture log-likelihood
        logp = pm.logaddexp(pt.log(f_latt) + logpdf_latt, pt.log1p(-f_latt) + logpdf_base)
        pm.Potential("lik", logp.sum())

        idata = pm.sample(draws=args.draws, tune=args.tune, target_accept=0.9, chains=4, random_seed=args.seed, compute_convergence_checks=True)

    # Summaries
    az.summary(idata, var_names=["Delta","theta","sigma","f_latt"]).to_csv(os.path.join(PLOTS, f"bayes_summary_{args.dataset}.csv"))
    post = az.extract(idata, var_names=["Delta","theta","sigma","f_latt"]).to_dataframe()
    post["Delta"] = np.exp(post["Delta"])

    # Savage–Dickey Bayes factor for f_latt
    BF10 = savage_dickey_BF(1.0, 1.0, post["f_latt"].values, at=0.0, eps=1e-3)
    with open(os.path.join(PLOTS, f"bayes_BF_{args.dataset}.json"), "w") as f:
        import json
        json.dump({"BF10_f_latt_gt_0_vs_eq_0": BF10}, f, indent=2)

    # Simple plots
    import matplotlib.pyplot as plt

    # Posterior of Delta and sigma/Delta
    plt.figure()
    plt.hist(post["Delta"], bins=60, density=True)
    plt.xlabel(r"$\Delta$ in $M^2$ units")
    plt.ylabel("Posterior density")
    plt.title(f"Posterior of lattice spacing Δ ({args.dataset})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"posterior_Delta_{args.dataset}.png"), dpi=180)

    plt.figure()
    plt.hist(post["theta"], bins=60, density=True)
    plt.xlabel(r"$\sigma/\Delta$")
    plt.ylabel("Posterior density")
    plt.title(f"Posterior of jitter ratio σ/Δ ({args.dataset})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f"posterior_sigma_over_Delta_{args.dataset}.png"), dpi=180)

    # Joint posterior
    try:
        import corner
        corner.corner(post[["Delta","theta","f_latt"]].values, labels=[r"$\Delta$", r"$\sigma/\Delta$", r"$f_{\rm latt}$"])
        plt.savefig(os.path.join(PLOTS, f"corner_Delta_theta_flatt_{args.dataset}.png"), dpi=180)
    except Exception:
        pass

    print(f"[done] Savage–Dickey BF10 (lattice vs no lattice) = {BF10:.3g}")

if __name__ == "__main__":
    main()
