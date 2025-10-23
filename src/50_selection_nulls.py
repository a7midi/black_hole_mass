\
#!/usr/bin/env python
"""
50_selection_nulls.py

Selection-aware nulls using LVK injections (if provided). We compute importance weights
for a smooth baseline in M^2 by matching injection-detected samples to the observed mass range,
then generate mock catalogs to assess the null distribution of the periodicity scan.
"""
from __future__ import annotations
import os, sys, argparse, h5py, numpy as np, pandas as pd, json
from tqdm import tqdm

from common import DERIVED, PLOTS, auto_delta_grid, compute_phase_power, rng

def load_injections(fp: str) -> pd.DataFrame:
    """
    Expect LVK population/injection HDF5 with fields including component masses and a 'found' flag.
    We keep source-frame masses if available; otherwise, detector-frame (approx).
    """
    rows = []
    with h5py.File(fp, "r") as f:
        keys = list(f.keys())
        # Heuristics for populations
        for k in keys:
            g = f[k]
            cols = list(g.keys())
            def arr(name):
                for candidate in [name, name.encode()]:
                    if candidate in g.keys():
                        return g[candidate][...]
                return None
            m1 = g.get("mass1_source") or g.get("mass1")
            m2 = g.get("mass2_source") or g.get("mass2")
            found = g.get("found") or g.get("detected")
            if m1 is None or m2 is None or found is None:
                continue
            m1 = np.array(m1)
            m2 = np.array(m2)
            found = np.array(found).astype(bool)
            rows.append(pd.DataFrame({
                "mass1": m1[found], "mass2": m2[found]
            }))
    if not rows:
        raise RuntimeError("No suitable injection groups found in HDF5")
    df = pd.concat(rows, ignore_index=True)
    df["M2_1"] = df["mass1"]**2
    df["M2_2"] = df["mass2"]**2
    return df

def mock_catalog_from_injections(df_inj: pd.DataFrame, n: int, seed: int = 0) -> np.ndarray:
    R = rng(seed)
    # Draw detected injections uniformly and combine component M^2
    idx = R.integers(0, len(df_inj), size=n)
    M2 = np.concatenate([df_inj["M2_1"].values[idx], df_inj["M2_2"].values[idx]])
    return M2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injections", required=True, help="Path to LVK injections HDF5")
    ap.add_argument("--dataset", choices=["GW","all"], default="GW")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df_obs = pd.read_parquet(os.path.join(DERIVED, "M2_samples.parquet"))
    df_obs = df_obs[df_obs["dataset"].isin(["GW"])] if args.dataset=="GW" else df_obs
    x = df_obs["M2"].values
    x = x[np.isfinite(x)]

    deltas = auto_delta_grid(x, n=160)
    inj = load_injections(args.injections)

    rows = []
    for b in tqdm(range(args.B), desc="selection-aware nulls"):
        M2b = mock_catalog_from_injections(inj, n=len(x)//2, seed=args.seed + b)
        stats = [compute_phase_power(M2b, d)["R"] for d in deltas]
        rows.append({"max_R": float(np.max(stats))})
    df_null = pd.DataFrame(rows)
    df_null.to_csv(os.path.join(PLOTS, f"selection_nulls_{args.dataset}.csv"), index=False)
    print(json.dumps({"mean_max_R": df_null["max_R"].mean(), "std_max_R": df_null["max_R"].std()}, indent=2))

if __name__ == "__main__":
    main()
