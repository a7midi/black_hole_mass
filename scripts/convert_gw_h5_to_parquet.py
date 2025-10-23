# save as: scripts/convert_gw_h5_to_parquet.py
import os, glob, re, pandas as pd
from pesummary.io.read import read

IN_DIRS  = ["data/external/gwpe/GWTC-2.1", "data/external/gwpe/GWTC-3"]
OUT_DIR  = "data/raw/gw"
os.makedirs(OUT_DIR, exist_ok=True)

def event_stem(path):
    # Pull canonical event id from filename, e.g.
    # .../IGWN-GWTC2p1-v2-GW190412_053044_PEDataRelease_cosmo.h5 -> GW190412_053044
    m = re.search(r"(GW\d{6}(?:_\d{6})?)", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]

def choose_label(samples_dict):
    # Prefer "C01:Mixed" if present; else the largest chain
    keys = list(samples_dict.keys())
    for k in keys:
        if "C01" in k or "Mixed" in k:
            return k
    # fallback: most samples
    return max(keys, key=lambda k: len(samples_dict[k]))

def process_one(path):
    ev = event_stem(path)
    try:
        res = read(path)
        lbl = choose_label(res.samples_dict)
        df  = res.samples_dict[lbl].to_pandas()
        # Keep only source-frame masses/spins if present; leave others for later stages
        keep = [c for c in df.columns if any(x in c for x in ["mass_1_source","mass_2_source","a_1","a_2","chi_eff","chi_p","m1_source","m2_source","source"])]
        if keep:
            df = df[keep]
        out = os.path.join(OUT_DIR, f"{ev}.parquet")
        df.to_parquet(out, index=False)
        print(f"[ok] {ev}: {len(df)} rows -> {out}")
    except Exception as e:
        print(f"[fail] {path}: {e}")

if __name__ == "__main__":
    files = []
    for d in IN_DIRS:
        files += glob.glob(os.path.join(d, "*.h5"))
    files.sort()
    for f in files:
        process_one(f)
