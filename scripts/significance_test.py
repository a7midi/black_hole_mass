import numpy as np, pandas as pd, json

def rayleigh_p(phi):
    N=len(phi); R=np.abs(np.exp(1j*phi).mean()); Z=N*R*R
    p = np.exp(-Z)*(1+(2*Z-Z**2)/(4*N)-(24*Z-132*Z**2+76*Z**3-9*Z**4)/(288*N**2))
    return float(np.clip(p, 0.0, 1.0))

df = pd.read_parquet("data/derived/M2_samples.parquet")
gw = df.query("dataset=='GW'")
events = gw['source'].unique()
rng = np.random.default_rng(2025)

# One fixed draw per event for the observed scan
M2 = (gw.groupby('source', group_keys=False)
        .apply(lambda g: g.sample(1, random_state=int(rng.integers(1<<31))))
        ['M2'].to_numpy())
grid = np.logspace(np.log10(M2.min()/5), np.log10(M2.max()/3), 180)

# Observed max statistic
obs = []
for d in grid:
    ph = 2*np.pi*np.mod(M2/d, 1.0)
    obs.append(-np.log10(max(1e-300, rayleigh_p(ph))))
obs_max = float(np.max(obs))

# Null distribution of the *max over grid*
B = 2000
null_max = np.empty(B)
for b in range(B):
    # Under H0, phases are i.i.d. uniform; draw fresh for each Δ
    mx = 0.0
    for d in grid:
        ph0 = rng.random(M2.size) * 2*np.pi
        mx = max(mx, -np.log10(max(1e-300, rayleigh_p(ph0))))
    null_max[b] = mx

p_global = float((null_max >= obs_max).mean())
print(json.dumps({"obs_max_neglog10p": obs_max, "global_p_value": p_global}))