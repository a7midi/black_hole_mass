# src/utils/sanity.py
import numpy as np
import pandas as pd
from pesummary.gw.fetch import fetch_open_samples

def load_df(event: str, catalog: str | None = None):
    # robust: let pesummary figure out the right file; 'unpack=True' gives you an HDF5 path too
    data = fetch_open_samples(event, catalog=catalog, read_file=True, unpack=True)
    # pick a label that contains source-frame masses
    labels = list(getattr(data, "labels", data.samples_dict.keys()))
    for label in labels:
        sd = data.samples_dict[label]      # SamplesDict
        keys = set(sd.keys())
        if {"mass_1_source", "mass_2_source"} <= keys:
            return label, pd.DataFrame({k: np.asarray(v) for k, v in sd.items()})
        # if only detector-frame masses exist, convert using redshift
        if {"mass_1", "mass_2", "redshift"} <= keys:
            m1 = np.asarray(sd["mass_1"]) / (1.0 + np.asarray(sd["redshift"]))
            m2 = np.asarray(sd["mass_2"]) / (1.0 + np.asarray(sd["redshift"]))
            df = pd.DataFrame({"mass_1_source": m1, "mass_2_source": m2})
            return label, df
    raise RuntimeError(f"No usable mass columns found in analyses {labels}")

if __name__ == "__main__":
    label, df = load_df("GW190412", catalog="GWTC-2.1")
    print("label:", label)
    print(df.filter(regex="mass_.*_source|redshift").head())
