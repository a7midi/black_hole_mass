\
#!/usr/bin/env python
"""
02_fetch_blackcat_xrb.py

Fetch stellar-mass BH measurements from the BlackCAT catalog and cache a CSV/Parquet.
We use pandas.read_html to scrape the main table if reachable; otherwise, the script
uses a curated fallback subset if available in data/raw/xrb/blackcat_fallback.csv.

Uncertainties are propagated by sampling later in 10_prepare_samples.py.
"""
from __future__ import annotations
import os, sys, argparse, io
import numpy as np
import pandas as pd
import requests
from common import RAW_XRB, ensure_parquet_engine

BLACKCAT_URLS = [
    # Known public mirrors/snapshots may change; we try a couple.
    "https://www.astro.puc.cl/~blackcat/",
    "https://www.astro.puc.cl/~blackcat/table.html",
]

def try_fetch() -> pd.DataFrame | None:
    for url in BLACKCAT_URLS:
        try:
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            # Parse any tables on the page
            tables = pd.read_html(r.text)
            # Heuristic: pick the largest table
            tbl = max(tables, key=lambda t: t.shape[0]*t.shape[1])
            return tbl
        except Exception as e:
            continue
    return None

def coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Try to identify BH name and mass columns
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}
    name_col = None
    mass_col = None
    mass_lo = None
    mass_hi = None

    # Try typical column names
    for k, c in cols.items():
        if any(x in k for x in ["source", "name", "system"]):
            name_col = c
        if "mass" in k and ("bh" in k or "m_bh" in k or "mbh" in k or k == "mass"):
            mass_col = c
        if any(x in k for x in ["-err", "minus", "lower"]):
            mass_lo = c
        if any(x in k for x in ["+err", "plus", "upper"]):
            mass_hi = c

    if name_col is None:
        name_col = df.columns[0]
    if mass_col is None:
        # fall back: find any column with numeric and reasonable scale (2-100)
        for c in df.columns:
            try:
                x = pd.to_numeric(df[c], errors="coerce")
                if x.dropna().between(2, 100).mean() > 0.3:
                    mass_col = c; break
            except Exception:
                pass
    # Build a simplified table
    out = pd.DataFrame({
        "name": df[name_col].astype(str),
        "mass_Msun": pd.to_numeric(df[mass_col], errors="coerce")
    })
    if mass_lo and mass_hi:
        out["mass_err_lo"] = pd.to_numeric(df[mass_lo], errors="coerce")
        out["mass_err_hi"] = pd.to_numeric(df[mass_hi], errors="coerce")
    else:
        out["mass_err_lo"] = np.nan
        out["mass_err_hi"] = np.nan

    out = out.dropna(subset=["mass_Msun"]).reset_index(drop=True)
    # Rough filtering of likely BHs
    out = out[out["mass_Msun"] > 2.5].copy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fallback", default=os.path.join(RAW_XRB, "blackcat_fallback.csv"))
    args = ap.parse_args()

    os.makedirs(RAW_XRB, exist_ok=True)
    parquet_engine = ensure_parquet_engine()

    df = try_fetch()
    if df is None:
        if os.path.exists(args.fallback):
            print("[info] using fallback CSV:", args.fallback)
            df = pd.read_csv(args.fallback)
        else:
            # minimal curated fallback
            data = [
                # name, mass, lo, hi
                ("Cyg X-1", 21.2, 2.2, 2.2),
                ("LMC X-1", 10.9, 1.4, 1.4),
                ("GX 339-4", 9.0, 1.5, 1.5),
                ("A0620-00", 6.6, 0.25, 0.25),
                ("GRO J1655-40", 6.3, 0.3, 0.3),
                ("V404 Cyg", 9.0, 0.6, 0.6),
                ("XTE J1118+480", 7.6, 0.7, 0.7),
                ("M33 X-7", 15.65, 1.45, 1.45),
            ]
            df = pd.DataFrame(data, columns=["name","mass_Msun","mass_err_lo","mass_err_hi"])
            print("[warn] BlackCAT fetch failed; wrote minimal fallback table.")

    else:
        df = coerce_columns(df)

    out_csv = os.path.join(RAW_XRB, "blackcat_bh_masses.csv")
    out_parq = os.path.join(RAW_XRB, "blackcat_bh_masses.parquet")
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parq, index=False, engine=parquet_engine)
    print(f"[done] saved {len(df)} rows to {out_csv} and {out_parq}")

if __name__ == "__main__":
    main()
