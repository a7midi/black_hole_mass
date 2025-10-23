\
"""
Common utilities for the BH mass-squared quantization analysis.
"""
from __future__ import annotations
import os, re, json, math, gzip
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_GW = os.path.join(DATA_DIR, "raw", "gw")
RAW_XRB = os.path.join(DATA_DIR, "raw", "xrb")
RAW_EHT = os.path.join(DATA_DIR, "raw", "eht")
DERIVED = os.path.join(DATA_DIR, "derived")
PLOTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")

os.makedirs(RAW_GW, exist_ok=True)
os.makedirs(RAW_XRB, exist_ok=True)
os.makedirs(RAW_EHT, exist_ok=True)
os.makedirs(DERIVED, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

def ensure_parquet_engine():
    # Prefer pyarrow if available.
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        return "fastparquet"

def effective_area_proxy(M: np.ndarray, a: Optional[np.ndarray]) -> np.ndarray:
    """
    Kerr horizon area proxy, up to a constant factor: A_eff ∝ M^2 * (1 + sqrt(1 - a^2)).
    If 'a' is None or unavailable, returns M**2.
    """
    if a is None:
        return M**2
    a = np.clip(np.abs(a), 0, 0.999999)
    return (M**2) * (1.0 + np.sqrt(1.0 - a**2))

def frac_part(x: np.ndarray) -> np.ndarray:
    """Return fractional part of x (x - floor(x))."""
    return x - np.floor(x)

def rayleigh_stat(phases: np.ndarray) -> Tuple[float, float]:
    """
    Rayleigh test statistic for non-uniformity on the circle.
    Returns (R, p_value) where R is resultant length / N.
    """
    N = len(phases)
    if N < 5:
        return np.nan, np.nan
    c = np.cos(phases).sum()
    s = np.sin(phases).sum()
    R = np.sqrt(c**2 + s**2) / N
    # Approximate p-value (large N) under uniformity:
    Z = N * (R**2)
    # Using classical approximation:
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*N) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*N**2))
    p = float(np.clip(p, 0.0, 1.0))
    return float(R), p

def kuiper_stat(phases: np.ndarray, n_grid: int = 1024) -> Tuple[float, float]:
    """
    Kuiper test for circular uniformity: returns (V, p_value).
    Uses an approximate p-value valid for moderate to large N.
    """
    N = len(phases)
    if N < 5:
        return np.nan, np.nan
    x = (np.sort(phases) / (2*np.pi)) % 1.0
    i = np.arange(1, N+1)
    D_plus = np.max(i/N - x)
    D_minus = np.max(x - (i-1)/N)
    V = D_plus + D_minus
    # Stephens (1965)-type approximation for p-value
    # Effective lambda
    lam = (np.sqrt(N) + 0.155 + 0.24/np.sqrt(N)) * V
    # Kuiper tail approximation
    p = 0.0
    for j in range(1, 6):
        p += (4*j**2 * lam**2 - 1) * np.exp(-2*j**2 * lam**2)
    p = float(np.clip(2*p, 0.0, 1.0))
    return float(V), p

def compute_phase_power(M2: np.ndarray, delta: float) -> Dict[str, float]:
    """
    For a given lattice spacing Δ, compute circular phases φ = 2π frac(M2 / Δ),
    and return Rayleigh and Kuiper statistics.
    """
    phases = 2*np.pi * frac_part(M2 / delta)
    R, p_R = rayleigh_stat(phases)
    V, p_K = kuiper_stat(phases)
    return {"R": R, "p_rayleigh": p_R, "V": V, "p_kuiper": p_K}

def auto_delta_grid(M2: np.ndarray, n: int = 160):
    """
    Construct a log-spaced Δ grid from the observed dynamic range only;
    no extra scale constants are introduced.
    """
    M2 = np.asarray(M2)
    lo, hi = np.nanpercentile(M2, [5, 95])
    spread = max(hi - lo, 1e-6)
    dmin = spread / 200.0
    dmax = spread / 3.0
    grid = np.geomspace(dmin, dmax, n)
    return grid

def ecdf(x: np.ndarray):
    x = np.sort(np.asarray(x))
    n = x.size
    y = np.arange(1, n+1) / n
    return x, y

def fourier_power_on_ecdf_residuals(x: np.ndarray, nbins: int = 1024):
    """
    Compute a simple discrete Fourier transform of the residuals of the ECDF against a
    linear baseline in the sorted variable space (proxy for periodicity in x).
    """
    xs, F = ecdf(x)
    # Map x to ranks [0,1], residuals wrt linear CDF
    r = F - (xs - xs.min()) / max(xs.max() - xs.min(), 1e-12)
    r = r - r.mean()
    # Zero-pad to power of two for FFT stability
    m = 1 << (int(np.ceil(np.log2(len(r)))))
    pad = m - len(r)
    if pad > 0:
        r = np.pad(r, (0, pad))
    spec = np.fft.rfft(r)
    power = (spec.real**2 + spec.imag**2)
    freqs = np.fft.rfftfreq(len(r), d=1.0)  # arbitrary units
    return freqs, power

def mad(a: np.ndarray) -> float:
    med = np.median(a)
    return np.median(np.abs(a - med))

def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)
