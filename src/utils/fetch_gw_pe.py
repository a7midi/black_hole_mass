# src/utils/fetch_gw_pe.py
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pesummary.gw.fetch import fetch_open_samples

def fetch_one(event: str, outdir: Path) -> Path | None:
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        data = fetch_open_samples(event, read_file=True, unpack=True)  # handle GWTC-1/2/3
        labels = list(getattr(data, "labels", data.samples_dict.keys()))
        for label in labels:
            sd = data.samples_dict[label]
            keys = set(sd.keys())
            if {"mass_1_source", "mass_2_source"} <= keys:
                df = pd.DataFrame({k: np.asarray(v) for k, v in sd.items()})
                break
            if {"mass_1", "mass_2", "redshift"} <= keys:
                m1 = np.asarray(sd["mass_1"]) / (1.0 + np.asarray(sd["redshift"]))
                m2 = np.asarray(sd["mass_2"]) / (1.0 + np.asarray(sd["redshift"]))
                df = pd.DataFrame({"mass_1_source": m1, "mass_2_source": m2})
                break
        else:
            raise RuntimeError(f"{event}: no usable mass columns in analyses {labels}")

        # Save a compact parquet (requires pyarrow/fastparquet)
        out = outdir / f"{event}.parquet"
        df.to_parquet(out, index=False)
        return out
    except Exception as e:
        (outdir / f"{event}.failed").write_text(str(e))
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", nargs="+", required=True)
    p.add_argument("--outdir", default="data/raw/gw")
    args = p.parse_args()

    outdir = Path(args.outdir)
    for ev in args.events:
        out = fetch_one(ev, outdir)
        if out:
            print(f"[ok] {ev} -> {out}")
        else:
            print(f"[fail] {ev}")

if __name__ == "__main__":
    sys.exit(main())
