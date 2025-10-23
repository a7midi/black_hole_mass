#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare unified M^2 samples for GW, XRB (BlackCAT), and EHT.

Outputs: data/derived/M2_samples.parquet with columns:
    dataset ∈ {"GW","XRB","EHT"}
    source  : provenance string (e.g. "GW190412_053044:mass_1_source", "M87*")
    M       : mass in solar masses
    M2      : M**2 in solar-mass squared

Default paths follow the project layout; override with CLI flags.

No arbitrary constants are introduced; we only propagate reported uncertainties.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# ------------------------ helpers ------------------------

def _rsplit_normal(mu: float, sig_lo: float, sig_hi: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw from a two-piece normal centered at mu with left sigma=sig_lo and right sigma=sig_hi.
    Sampling is exact via a mixture of left/right half-normals with weights proportional to sigmas.
    """
    if sig_lo <= 0 or sig_hi <= 0:
        # fall back: symmetric if anything is non-positive
        s = max(sig_lo, sig_hi, 1e-12)
        return rng.normal(mu, s, size)

    p_left = sig_lo / (sig_lo + sig_hi)
    u = rng.random(size)
    z = np.abs(rng.standard_normal(size))
    out = np.empty(size)
    left = u < p_left
    out[left]  = mu - z[left]  * sig_lo
    out[~left] = mu + z[~left] * sig_hi
    return out


def _read_any_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV/TSV reader (auto-detects delimiter; respects '#' comments).
    """
    return pd.read_csv(path, sep=None, engine="python", comment="#")


# ------------------------ GW ------------------------

def stack_gw(gw_dir: Path, thin: int = 1) -> pd.DataFrame:
    """
    Read all *.parquet files in gw_dir and extract component source-frame masses.
    Returns a dataframe with columns: dataset, source, M, M2.
    """
    rows = []
    pq_files = sorted(gw_dir.glob("*.parquet"))
    for pq in pq_files:
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            print(f"[skip] {pq.name}: {e}")
            continue

        # Prefer standard LVK names
        preferred = [c for c in ("mass_1_source", "mass_2_source") if c in df.columns]
        # Otherwise, accept any mass_*_source columns
        if not preferred:
            preferred = [c for c in df.columns if c.startswith("mass_") and c.endswith("_source")]

        for col in preferred:
            m = pd.to_numeric(df[col], errors="coerce")
            m = m[np.isfinite(m)]
            if thin > 1:
                m = m.iloc[::thin]
            if len(m) == 0:
                continue
            src = f"{pq.stem}:{col}"
            tmp = pd.DataFrame({
                "dataset": "GW",
                "source":  src,
                "M":       m.values,
            })
            tmp["M2"] = tmp["M"]**2
            rows.append(tmp)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        print(f"[ok] GW: stacked {len(out):,} rows from {gw_dir}")
        return out
    else:
        print(f"[warn] GW: no parquet files in {gw_dir}")
        return pd.DataFrame(columns=["dataset","source","M","M2"])


# ------------------------ XRB (BlackCAT Table A.4) ------------------------

def sample_xrb(blackcat_csv: Path, samples_per_obj: int = 1000, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Draw samples from BlackCAT Table A.4 masses.
    Uses a split normal with (mu=M1, sigma_lo=e_M1, sigma_hi=E_M1).
    Rows lacking M1 or both errors are skipped (no ad-hoc uncertainties).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = _read_any_csv(blackcat_csv)
    # Normalize columns
    cols = {c.strip(): c for c in df.columns}
    need = ["Name","M1"]
    for c in need:
        if c not in cols:
            raise RuntimeError(f"BlackCAT file missing required column '{c}'. Found: {list(df.columns)}")

    # Optional asymmetric errors
    c_hi = "E_M1" if "E_M1" in cols else None
    c_lo = "e_M1" if "e_M1" in cols else None

    keep = df[[cols["Name"], cols["M1"]]].copy()
    keep.columns = ["name","mu"]
    if c_hi: keep["errp"] = pd.to_numeric(df[cols[c_hi]], errors="coerce")
    else:    keep["errp"] = np.nan
    if c_lo: keep["errm"] = pd.to_numeric(df[cols[c_lo]], errors="coerce")
    else:    keep["errm"] = np.nan

    keep["mu"]   = pd.to_numeric(keep["mu"], errors="coerce")
    keep = keep[np.isfinite(keep["mu"])]

    rows = []
    for _, r in keep.iterrows():
        name = str(r["name"]).strip()
        mu   = float(r["mu"])
        errp = float(r["errp"]) if np.isfinite(r["errp"]) else np.nan
        errm = float(r["errm"]) if np.isfinite(r["errm"]) else np.nan

        if not np.isfinite(errp) and not np.isfinite(errm):
            # Skip if no uncertainty reported (parameter-free policy: avoid inventing widths)
            continue

        if not np.isfinite(errp): errp = errm
        if not np.isfinite(errm): errm = errp

        mdraw = _rsplit_normal(mu, errm, errp, samples_per_obj, rng)
        tmp = pd.DataFrame({
            "dataset": "XRB",
            "source":  name,
            "M":       mdraw,
        })
        tmp["M2"] = tmp["M"]**2
        rows.append(tmp)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        print(f"[ok] XRB: sampled {len(out):,} rows from {blackcat_csv}")
        return out
    else:
        print(f"[warn] XRB: nothing sampled (check that M1 and at least one of E_M1/e_M1 are present)")
        return pd.DataFrame(columns=["dataset","source","M","M2"])


# ------------------------ EHT (SMBH masses) ------------------------

def sample_eht(eht_csv: Path, samples_per_obj: int = 10_000, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Sample EHT SMBH masses. Accepts either:
        - columns 'mu' and 'sigma'  (both in solar masses), or
        - columns 'mass_msun' and 'sigma_msun'
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = _read_any_csv(eht_csv)
    df.columns = [c.strip() for c in df.columns]

    # Flexible column mapping
    col_map = {}
    if "mu" in df.columns:
        col_map["mu"] = "mu"
    elif "mass_msun" in df.columns:
        col_map["mu"] = "mass_msun"

    if "sigma" in df.columns:
        col_map["sigma"] = "sigma"
    elif "sigma_msun" in df.columns:
        col_map["sigma"] = "sigma_msun"

    if "name" not in df.columns:
        raise RuntimeError(f"EHT file missing 'name' column. Found columns: {list(df.columns)}")

    if "mu" not in col_map:
        raise RuntimeError(f"EHT file missing a mass column ('mu' or 'mass_msun'). Found columns: {list(df.columns)}")
    if "sigma" not in col_map:
        raise RuntimeError(f"EHT file missing an uncertainty column ('sigma' or 'sigma_msun'). Found columns: {list(df.columns)}")

    sub = df[["name", col_map["mu"], col_map["sigma"]]].rename(columns={col_map["mu"]: "mu", col_map["sigma"]: "sigma"})
    sub["mu"]    = pd.to_numeric(sub["mu"], errors="coerce")
    sub["sigma"] = pd.to_numeric(sub["sigma"], errors="coerce")
    sub = sub[np.isfinite(sub["mu"]) & np.isfinite(sub["sigma"]) & (sub["sigma"] > 0)]

    rows = []
    for _, r in sub.iterrows():
        name  = str(r["name"]).strip()
        mu    = float(r["mu"])
        sigma = float(r["sigma"])
        mdraw = rng.normal(mu, sigma, samples_per_obj)
        tmp = pd.DataFrame({
            "dataset": "EHT",
            "source":  name,
            "M":       mdraw,
        })
        tmp["M2"] = tmp["M"]**2
        rows.append(tmp)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        print(f"[ok] EHT: sampled {len(out):,} rows from {eht_csv}")
        return out
    else:
        print(f"[warn] EHT: nothing sampled (check columns and positive sigmas)")
        return pd.DataFrame(columns=["dataset","source","M","M2"])


# ------------------------ main ------------------------

def main():
    p = argparse.ArgumentParser(description="Prepare unified M^2 samples for GW + XRB + EHT")
    p.add_argument("--gw-dir", default="data/raw/gw", type=Path)
    p.add_argument("--xrb-csv", default="data/external/xrb/blackcat_tablea4.csv", type=Path)
    p.add_argument("--eht-csv", default="data/external/eht/eht_masses.csv", type=Path)
    p.add_argument("--out", default="data/derived/M2_samples.parquet", type=Path)

    p.add_argument("--thin-gw", type=int, default=1, help="Keep 1 out of N GW posterior rows")
    p.add_argument("--xrb-samples", type=int, default=1000, help="Samples per XRB object")
    p.add_argument("--eht-samples", type=int, default=10_000, help="Samples per EHT object")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    gw  = stack_gw(args.gw_dir, thin=args.thin_gw)
    xrb = sample_xrb(args.xrb_csv, samples_per_obj=args.xrb_samples, rng=rng)
    eht = sample_eht(args.eht_csv, samples_per_obj=args.eht_samples, rng=rng)

    all_frames = [df for df in [gw, xrb, eht] if len(df)]
    if not all_frames:
        raise SystemExit("[error] No data assembled. Check your inputs.")

    all_ = pd.concat(all_frames, ignore_index=True)
    # Drop any non-finite / non-positive masses defensively
    all_ = all_[np.isfinite(all_["M"]) & (all_["M"] > 0)]
    all_["M2"] = all_["M"]**2

    # Save
    try:
        all_.to_parquet(args.out, index=False)
    except Exception:
        # fallback for environments without pyarrow
        all_.to_parquet(args.out, index=False, engine="fastparquet")
    print(f"[ok] wrote {args.out} with {len(all_):,} rows")
    print(all_["dataset"].value_counts())


if __name__ == "__main__":
    main()
