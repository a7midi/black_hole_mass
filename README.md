# Black-Hole Mass–Squared Quantization Tests (Einstein–Memory / Visible Capacity)

This repository implements a fully reproducible pipeline to test the prediction
that black-hole masses occupy discrete \(\sqrt{n}\) levels because the **visible capacity**
\(K\) is integer bits and tracks horizon area, \(A \propto K\), which in the Lorentz limit implies
\(A \propto M^2\).

**Datasets**
- LVK GW posterior samples (GWTC-3 and GWTC-4) — source-frame component masses.
- X-ray binaries from BlackCAT — stellar-mass BHs.
- EHT SMBH masses (M87*, Sgr A*) — optional.

**Analyses**
1. **Periodicity scan** in \(M^2\) using Rayleigh/Kuiper statistics with bootstrap and selection-aware nulls.
2. **Bayesian model comparison**: smooth baseline vs. lattice-mixture \(\sum_n w_n \mathcal{N}(M^2; n\,\Delta, \sigma^2)\)
   with a lattice fraction and **Savage–Dickey Bayes factor** for \(f_{\rm latt}=0\) vs \(>0\).
3. **CDF staircase/Fourier** test: periodic residuals of the empirical CDF of \(M^2\) and a spectral peak near \(1/\Delta\).

**Quickstart**
```bash
# Option A: conda
mamba env create -f environment.yml
mamba activate bhpockets-quant

# Option B: pip (Python 3.11+ recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download and build derived samples
python src/01_fetch_gw_posteriors.py --catalogs GWTC-3-confident GWTC-4-confident --max-events 999
python src/02_fetch_blackcat_xrb.py
python src/03_fetch_eht_smbh.py  # optional, creates data/raw/eht/masses.yaml

# Harmonize and produce M^2 samples
python src/10_prepare_samples.py --n-samples 4000

# Run the three analyses
python src/20_test_periodicity_M2.py --dataset all
python src/30_bayes_mixture_lattice.py --dataset all --draws 3000 --tune 2000
python src/40_cdf_fourier_test.py --dataset all

# (Optional) Generate selection-aware nulls from LVK injections if available
python src/50_selection_nulls.py --injections data/external/LVK_injections.h5
```

All output figures are saved in `plots/`. A complete end-to-end notebook is in
`notebooks/quantization_analysis.ipynb`.

**Parameters**: we avoid tunable scales beyond those fixed by the paper. The only
user-facing choices are grid ranges determined from the data spread (not free constants),
and weakly-informative priors whose scales are expressed *relative to* \(\Delta\)
(e.g., \(\sigma/\Delta\)).

---

## Repository Layout

```
src/
  01_fetch_gw_posteriors.py
  02_fetch_blackcat_xrb.py
  03_fetch_eht_smbh.py
  10_prepare_samples.py
  20_test_periodicity_M2.py
  30_bayes_mixture_lattice.py
  40_cdf_fourier_test.py
  50_selection_nulls.py
notebooks/
  quantization_analysis.ipynb
data/
  raw/
    gw/          # cached LVK posterior files
    xrb/         # BlackCAT table cache
    eht/         # optional YAML for M87*, Sgr A*
  derived/
    M2_samples.parquet
plots/
environment.yml
requirements.txt
README.md
```

---

## Physical interpretation

Let a black-hole pocket \(P\) have **visible capacity** \(K(P) = \sum_{v\in O_P}\log_2|A_v|\) (finite), an integer number of bits.
In quasi-static windows, \(K\) tracks horizon area, and the **Einstein–Memory** law
\(\kappa_R = g_\star (\rho_{\rm mem})_R\) fixes the geometric response. In the Lorentz limit, \(A\propto M^2\),
so \(M^2\propto K\). Detecting a lattice spacing \(\Delta\) in \(M^2\) means **one extra bit of visible capacity corresponds
to an area increment proportional to \(\Delta\)**. We report \(\Delta\) and \(\sigma/\Delta\) and cross-check that different
datasets prefer a consistent \(\Delta\).
