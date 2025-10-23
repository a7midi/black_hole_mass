\
#!/usr/bin/env python
"""
10_prepare_samples.py

Read cached GW event posteriors, BlackCAT XRB masses, and optional EHT SMBH masses.
Produce a unified M^2 sample table with consistent units (Msun) and optional area proxy A_eff.

Saves: data/derived/M2_samples.parquet
"""
from __future__ import annotations
import os, sys, glob, argparse, yaml
import numpy as np
import pandas as pd

from common import RAW_GW, RAW_XRB, RAW_EHT, DERIVED, ensure_parquet_engine, effective_area_proxy, rng

def load_gw_samples(n_samples: int = 4000, seed: int = 0) -> pd.DataFrame:
    """
    Load thinned per-event parquet files with component masses (source frame).
    Return long-form samples with one row per component per draw.
    """
    files = sorted(glob.glob(os.path.join(RAW_GW, "*.parquet")))
    rows = []
    R = rng(seed)
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if not {"mass_1_source","mass_2_source"}.issubset(df.columns):
            continue
        # Thin uniformly at random to at most n_samples per label
        if "label" in df.columns:
            for lbl, grp in df.groupby("label"):
                g = grp.sample(min(n_samples, len(grp)), random_state=R.integers(0, 2**32-1))
                # Component 1
                rows.append(pd.DataFrame({
                    "dataset": "GW",
                    "source": os.path.basename(fp).replace(".parquet",""),
                    "component": "m1",
                    "mass_Msun": g["mass_1_source"].values,
                    "spin": g["a_1"].values if "a_1" in g.columns else np.full(len(g), np.nan),
                }))
                # Component 2
                rows.append(pd.DataFrame({
                    "dataset": "GW",
                    "source": os.path.basename(fp).replace(".parquet",""),
                    "component": "m2",
                    "mass_Msun": g["mass_2_source"].values,
                    "spin": g["a_2"].values if "a_2" in g.columns else np.full(len(g), np.nan),
                }))
        else:
            g = df.sample(min(n_samples, len(df)), random_state=R.integers(0, 2**32-1))
            rows.append(pd.DataFrame({
                "dataset": "GW",
                "source": os.path.basename(fp).replace(".parquet",""),
                "component": "m1",
                "mass_Msun": g["mass_1_source"].values,
                "spin": g["a_1"].values if "a_1" in g.columns else np.full(len(g), np.nan),
            }))
            rows.append(pd.DataFrame({
                "dataset": "GW",
                "source": os.path.basename(fp).replace(".parquet",""),
                "component": "m2",
                "mass_Msun": g["mass_2_source"].values,
                "spin": g["a_2"].values if "a_2" in g.columns else np.full(len(g), np.nan),
            }))
    if not rows:
        return pd.DataFrame(columns=["dataset","source","component","mass_Msun","spin"])
    out = pd.concat(rows, ignore_index=True)
    return out

def load_xrb_samples(n_per: int = 4000, seed: int = 1) -> pd.DataFrame:
    """
    Read BlackCAT table and draw samples per source using a conservative uncertainty model:
    - If +/- errors present, draw from a split-normal.
    - Else, assume 10% relative Gaussian.
    """
    import scipy.stats as st
    R = rng(seed)
    fp_parq = os.path.join(RAW_XRB, "blackcat_bh_masses.parquet")
    fp_csv = os.path.join(RAW_XRB, "blackcat_bh_masses.csv")
    if os.path.exists(fp_parq):
        df = pd.read_parquet(fp_parq)
    elif os.path.exists(fp_csv):
        df = pd.read_csv(fp_csv)
    else:
        return pd.DataFrame(columns=["dataset","source","component","mass_Msun","spin"])

    rows = []
    for _, row in df.iterrows():
        name = str(row["name"])
        m = float(row["mass_Msun"])
        lo = float(row.get("mass_err_lo", np.nan))
        hi = float(row.get("mass_err_hi", np.nan))
        if np.isnan(lo) or np.isnan(hi) or lo <= 0 or hi <= 0:
            sigma = 0.1 * m
            samples = R.normal(m, sigma, size=n_per)
        else:
            # Split normal: left sigma=lo, right sigma=hi
            left = st.truncnorm.rvs(a=(0 - m)/lo, b=(np.inf - m)/lo, loc=m, scale=lo, size=n_per//2, random_state=R)
            right = st.truncnorm.rvs(a=(0 - m)/hi, b=(np.inf - m)/hi, loc=m, scale=hi, size=n_per - len(left), random_state=R)
            samples = np.concatenate([left, right])
        samples = samples[samples > 0]
        rows.append(pd.DataFrame({
            "dataset": "XRB",
            "source": name,
            "component": "single",
            "mass_Msun": samples,
            "spin": np.full(len(samples), np.nan),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["dataset","source","component","mass_Msun","spin"])

def load_eht_samples(n_per: int = 6000, seed: int = 2) -> pd.DataFrame:
    """
    Load EHT SMBH masses from YAML and draw Gaussian samples.
    """
    R = rng(seed)
    fp = os.path.join(RAW_EHT, "masses.yaml")
    if not os.path.exists(fp):
        return pd.DataFrame(columns=["dataset","source","component","mass_Msun","spin"])
    import yaml
    est = yaml.safe_load(open(fp))
    rows = []
    for name, rec in est.items():
        m = float(rec["mass_Msun"])
        s = float(rec.get("sigma_Msun", 0.05*m))
        samples = R.normal(m, s, size=n_per)
        rows.append(pd.DataFrame({
            "dataset": "EHT",
            "source": name,
            "component": "single",
            "mass_Msun": samples[samples > 0],
            "spin": np.full(np.sum(samples > 0), np.nan)
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["dataset","source","component","mass_Msun","spin"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=4000, help="per-event or per-source sampling cap")
    ap.add_argument("--include-eht", action="store_true", help="include SMBHs if YAML present")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    parquet_engine = ensure_parquet_engine()

    gw = load_gw_samples(n_samples=args.n_samples, seed=args.seed)
    xrb = load_xrb_samples(n_per=args.n_samples, seed=args.seed + 1)
    eht = load_eht_samples(n_per=args.n_samples, seed=args.seed + 2) if args.include_eht else \
          (pd.DataFrame(columns=gw.columns))
    df = pd.concat([gw, xrb, eht], ignore_index=True)
    if not len(df):
        print("[warn] no samples assembled"); return

    df["M2"] = df["mass_Msun"]**2
    # Area proxy with spins if known
    df["A_eff"] = effective_area_proxy(df["mass_Msun"].values, df["spin"].values if "spin" in df.columns else None)

    out = os.path.join(DERIVED, "M2_samples.parquet")
    df.to_parquet(out, index=False, engine=parquet_engine)
    print(f"[done] wrote {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
