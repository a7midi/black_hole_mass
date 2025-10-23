\
#!/usr/bin/env python
"""
03_fetch_eht_smbh.py

Create (or update) a simple YAML with SMBH masses for M87* and Sgr A*.
These are optional and can be edited by the user to reflect the latest values.

Usage:
  python src/03_fetch_eht_smbh.py --out data/raw/eht/masses.yaml
"""
from __future__ import annotations
import os, argparse, yaml

from common import RAW_EHT

DEFAULTS = {
  "M87*": {"mass_Msun": 6.5e9, "sigma_Msun": 0.7e9, "reference": "EHT Collaboration (2019, 2021)"},
  "Sgr A*": {"mass_Msun": 4.0e6, "sigma_Msun": 0.2e6, "reference": "EHT Collaboration (2022)"},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RAW_EHT, "masses.yaml"))
    args = ap.parse_args()
    os.makedirs(RAW_EHT, exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w") as f:
            yaml.safe_dump(DEFAULTS, f, sort_keys=True)
        print(f"[done] wrote default EHT masses to {args.out}")
    else:
        print(f"[info] file exists: {args.out} — edit if you wish to update values.")

if __name__ == "__main__":
    main()
