\
#!/usr/bin/env python
"""
01_fetch_gw_posteriors.py

Download LVK GW posterior samples (GWTC-3/4) from GWOSC Event API and cache locally.
Extract source-frame component masses (m1_source, m2_source) and spins if present.
Saves one parquet per event with thin posterior samples for downstream processing.

Usage:
  python src/01_fetch_gw_posteriors.py --catalogs GWTC-3-confident GWTC-4-confident --max-events 999 --thin 5
"""
from __future__ import annotations
import os, sys, argparse, re, time, json
from typing import Dict, List, Optional
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd

from pesummary.io import read as ps_read

from common import RAW_GW, ensure_parquet_engine

GWOSC_API_BASE = "https://www.gw-openscience.org/eventapi/json"

def list_events(catalog: str) -> Dict:
    url = f"{GWOSC_API_BASE}/{catalog}/"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def pick_posterior_files(meta: Dict) -> List[str]:
    """Return a prioritized list of (likely) posterior sample URLs for an event."""
    urls = []
    try:
        files = meta.get("files", {})
        for name, rec in files.items():
            desc = rec.get("description", "").lower()
            if "posterior" in desc or "samples" in desc or "pesummary" in desc:
                for u in rec.get("links", []):
                    if u.endswith(".h5") or u.endswith(".hdf5"):
                        urls.append(u)
    except Exception:
        pass
    # De-duplicate, preserve order
    seen = set()
    kept = []
    for u in urls:
        if u not in seen:
            seen.add(u); kept.append(u)
    return kept

def try_download(url: str, out_path: str) -> bool:
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        return False

def extract_masses(h5_path: str, thin: int = 10, nmax: int = 10000) -> Optional[pd.DataFrame]:
    """
    Use pesummary to read posterior samples. Returns a DataFrame with m1_source, m2_source,
    chi1/chi2 (if available), event label.
    """
    try:
        data = ps_read(h5_path)
    except Exception as e:
        print(f"[warn] pesummary failed to read {h5_path}: {e}", file=sys.stderr)
        return None

    # pesummary returns a SamplesDict-like interface; pick the first label
    labels = list(data.labels) if hasattr(data, "labels") else []
    if not labels:
        try:
            # fallback for older structures
            df = data.samples_dict
            labels = list(df.keys())
        except Exception:
            labels = []

    df_all = []
    for lbl in labels or []:
        try:
            samp = data.samples(lab=lbl)
        except Exception:
            try:
                samp = data.samples_dict[lbl]
            except Exception:
                continue
        pdf = samp.to_pandas() if hasattr(samp, "to_pandas") else pd.DataFrame(samp)

        # Try common column names
        cols = pdf.columns
        m1 = None; m2 = None
        for c in ["mass_1_source", "m1_source", "source_mass_1", "mass_1"]:
            if c in cols:
                m1 = pdf[c].values; break
        for c in ["mass_2_source", "m2_source", "source_mass_2", "mass_2"]:
            if c in cols:
                m2 = pdf[c].values; break
        if m1 is None or m2 is None:
            continue

        # Spins (dimensionless) if present (effective aligned or individual)
        a1 = None; a2 = None
        for c in ["a_1", "chi_1", "spin_1z", "chi1", "a1"]:
            if c in cols:
                a1 = pdf[c].values; break
        for c in ["a_2", "chi_2", "spin_2z", "chi2", "a2"]:
            if c in cols:
                a2 = pdf[c].values; break

        # Thin to reduce size
        arrs = [m1, m2]
        L = min(len(x) for x in arrs if x is not None)
        step = max(int(thin), 1)
        idx = np.arange(0, min(L, nmax), step, dtype=int)
        out = {
            "mass_1_source": np.asarray(m1)[idx],
            "mass_2_source": np.asarray(m2)[idx],
            "label": np.array([str(lbl)]*len(idx))
        }
        if a1 is not None:
            out["a_1"] = np.asarray(a1)[idx]
        if a2 is not None:
            out["a_2"] = np.asarray(a2)[idx]
        df_all.append(pd.DataFrame(out))

    if not df_all:
        return None

    df = pd.concat(df_all, ignore_index=True)
    # Event name heuristic from filename
    ev = os.path.basename(h5_path).split("_")[0]
    df["event"] = ev
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogs", nargs="+", default=["GWTC-3-confident"])
    ap.add_argument("--max-events", type=int, default=9999)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(RAW_GW, exist_ok=True)
    parquet_engine = ensure_parquet_engine()

    n_saved = 0
    for cat in args.catalogs:
        meta = list_events(cat)
        events = meta.get("events", {})
        for ev_name, ev_meta in tqdm(list(events.items())[:args.max_events], desc=f"{cat}"):
            # Skip repeats
            out_parquet = os.path.join(RAW_GW, f"{ev_name}.parquet")
            if (not args.force) and os.path.exists(out_parquet):
                continue

            urls = pick_posterior_files(ev_meta)
            success = False
            for i, url in enumerate(urls[:4]):  # try a few
                tmp = os.path.join(RAW_GW, f"{ev_name}.{i}.h5")
                if not os.path.exists(tmp):
                    ok = try_download(url, tmp)
                    if not ok:
                        continue
                df = extract_masses(tmp, thin=args.thin)
                if df is not None and len(df):
                    df.to_parquet(out_parquet, index=False, engine=parquet_engine)
                    n_saved += 1
                    success = True
                    break
            if not success:
                # record failure marker
                open(os.path.join(RAW_GW, f"{ev_name}.failed"), "w").write("no posterior parsed")

    print(f"[done] saved {n_saved} event parquet files in {RAW_GW}")

if __name__ == "__main__":
    main()
