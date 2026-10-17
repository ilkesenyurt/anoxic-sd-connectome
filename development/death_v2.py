# %% [markdown]
# FlyWire "brain death" simulation - v2 (activity-dependent energy)
# =================================================================
# NEW in v2: energy is USED UP by a resting cost, a cost per spike, and the
# pump's work of clearing K+. When energy runs low, the pump fails: clearance
# stops, leaking K+ builds up, the neuron depolarizes, fires harder (burning
# energy faster), and finally blocks.
# Test: energy at block should now be LOW (v1 was 0.82).
# Still a TOY: K+ is one number per neuron, not real ion chemistry.

# %% Imports
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

SEED = 0
rng = np.random.default_rng(SEED)

# %% Configuration
# ---- files ----
CONNECTIONS_CSV = "connections_princeton.csv"
NEURONS_CSV     = "neurons.csv"
COORDS_CSV      = "coordinates.csv"      # position "[x y z]" in nm
CLASS_CSV       = "classification.csv"

# ---- run mode ----
RUN_MODE     = "death"   # "healthy" (control, oxygen never cut) | "death"
T_ANOX_MS    = 2000.0      # oxygen cutoff (death mode only)
DURATION_MS  = 60000.0
HEALTHY_PEAK = 0.0022        # paste the value the healthy run prints, e.g. 0.0024
BOX_HALF_NM  = None        # None = whole brain; 100_000 = the old 200 um cube

# ---- LIF (Shiu et al. 2024) ----
V_REST0  = -52.0; V_THRESH = -45.0
TAU_M = 20.0; TAU_SYN = 5.0; REFRAC = 2.2; DELAY = 1.8
W_SYN = 0.275; DT = 0.1

# ---- healthy drive ----
TONIC_MEAN = 5.0   # mV  (tuned, ~20 Hz)
NOISE_SD   = 2.0   # mV

# ---- extracellular K+ ----
K0 = 3.0
K_THRESH = 10.0
K_PEAK = 55.0            # IMPOSED cap (fly plateau ~54)
K_LOW_GAIN = 0.3         # guess
K_HIGH_GAIN = 3.0        # guess
K_SPIKE_RELEASE = 0.01   # tuned
K_LEAK_S = 0.3           # NEW: K leaking out per second, balanced by pump when healthy (guess)
K_RELEASE = 8.0          # block dump = K_RELEASE * 5 (guess)
K_REMOVE_S = 1.0         # clearance time constant at full pump power (tuned)
D_K = 1.45e-4             # calibrated in v1 (0.76 mm/min)
K_SPATIAL_KNN = 8

# ---- energy (NEW) ----
E_BASE_USE_S = 70.0   # s to use all energy at resting cost alone -> timing knob (guess)
E_SPIKE_COST = 5e-4   # per spike (guess)
E_PUMP_COST  = 0.01   # per unit of K cleared (guess)
E_FULL = 0.6          # pump full power above this (guess)
E_FAIL = 0.2          # pump stopped below this (guess)
V_PUMP_MV = 5.0       # depolarization when pump stops (guess)
V_BLOCK = -25.0       # guess

SLOW_EVERY = 10       # energy/K updated every 1 ms
USE_NUMBA  = True     # True = compiled loop in fast_sim.py; False = numpy version

# ---- scenario ----
PROPAGATION  = "spatial"     # "spatial" | "synaptic" | "both"
VULN_MODE    = "inhibitory"  # "inhibitory" | "excitatory" | "uniform"
SHUFFLE_CONN = False
SHUFFLE_MODE = "sign"   # "sign": swap inputs only with inputs of the same sign, so every
                        #         neuron keeps its real total excitation and inhibition
                        # "free": old shuffle (any input with any input; rows 27-34)

SYN_FRACTION = 0.5   # in "both"/"synaptic" mode: share of released K sent to
                     # synaptic partners (the rest stays at the neuron) - guess

# %% Load data (no silent fallbacks)
def load_data():
    neurons = pd.read_csv(NEURONS_CSV, usecols=["root_id", "nt_type"])
    coords = pd.read_csv(COORDS_CSV)
    xyz = coords["position"].str.strip("[]").str.split(expand=True).astype(float)
    coords["x"], coords["y"], coords["z"] = xyz[0], xyz[1], xyz[2]
    coords = coords[["root_id", "x", "y", "z"]].drop_duplicates("root_id")
    classes = pd.read_csv(CLASS_CSV, usecols=["root_id", "super_class"])
    meta = neurons.merge(coords, on="root_id", how="inner").merge(
        classes, on="root_id", how="left")

    if BOX_HALF_NM is not None:              # None = whole brain
        cx, cy, cz = meta[["x", "y", "z"]].median()
        h = BOX_HALF_NM
        box = ((meta.x - cx).abs() < h) & ((meta.y - cy).abs() < h) & ((meta.z - cz).abs() < h)
        meta = meta[box].reset_index(drop=True)

    conn = pd.read_csv(CONNECTIONS_CSV, usecols=["pre_root_id", "post_root_id", "syn_count"])
    keep = set(meta["root_id"])
    conn = conn[conn["pre_root_id"].isin(keep) & conn["post_root_id"].isin(keep)].reset_index(drop=True)
    print(len(meta), "neurons,", len(conn), "connections",
          "(whole brain)" if BOX_HALF_NM is None else "in box")
    return meta, conn

# %% Network
def build_network(meta, conn):
    idx = {rid: i for i, rid in enumerate(meta["root_id"].values)}
    N = len(idx)
    sign_map = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "DA": 0.0, "SER": 0.0, "OCT": 0.0}
    nt = meta["nt_type"].fillna("UNK").values
    neuron_sign = np.array([sign_map.get(t, 0.0) for t in nt], dtype=np.float32)
    pre = conn["pre_root_id"].map(idx).values
    post = conn["post_root_id"].map(idx).values
    syn = conn["syn_count"].values.astype(np.float32)
    if SHUFFLE_CONN:
        g = np.random.default_rng(12345)           # own generator: same shuffle every run
        if SHUFFLE_MODE == "free":
            pre = g.permutation(pre)
        else:                                      # "sign": shuffle within each sign group
            pre = pre.copy()
            s = neuron_sign[pre]
            for val in np.unique(s):
                k = np.flatnonzero(s == val)
                pre[k] = pre[k][g.permutation(k.size)]
    w = W_SYN * syn * neuron_sign[pre]
    W = sparse.csr_matrix((w, (post, pre)), shape=(N, N), dtype=np.float32)
    coords = meta[["x", "y", "z"]].values.astype(np.float32)
    return idx, W, neuron_sign, coords, N

def build_vulnerability(neuron_sign):
    """Multiplier on energy use per neuron."""
    vuln = np.ones(len(neuron_sign), dtype=np.float32)
    if VULN_MODE == "inhibitory":
        vuln[neuron_sign < 0] = 2.0; vuln[neuron_sign > 0] = 0.7
    elif VULN_MODE == "excitatory":
        vuln[neuron_sign > 0] = 2.0; vuln[neuron_sign < 0] = 0.7
    vuln *= rng.lognormal(0.0, 0.25, size=len(vuln)).astype(np.float32)
    return vuln

def build_spatial_graph(coords, k=K_SPATIAL_KNN):
    _, nbr = cKDTree(coords).query(coords, k=k + 1)
    nbr = nbr[:, 1:]
    N = coords.shape[0]
    rows = np.repeat(np.arange(N), k)
    data = np.ones(N * k, dtype=np.float32) / k
    return sparse.csr_matrix((data, (rows, nbr.reshape(-1))), shape=(N, N), dtype=np.float32)

# %% Simulation
def simulate_numpy(W, N, vuln, L, t_anox_ms, duration_ms, record_every=10):
    """Original numpy version (kept for checking fast_sim against it)."""
    n_steps = int(duration_ms / DT)
    delay_steps = int(round(DELAY / DT)); refrac_steps = int(round(REFRAC / DT))
    t_anox_step = int(t_anox_ms / DT)
    slow_dt_s = DT * SLOW_EVERY / 1000.0

    v = np.full(N, V_REST0, dtype=np.float32)
    v_rest = np.full(N, V_REST0, dtype=np.float32)
    g_syn = np.zeros(N, dtype=np.float32)
    E = np.ones(N, dtype=np.float32)
    K = np.full(N, K0, dtype=np.float32)
    refr = np.zeros(N, dtype=np.int32)
    blocked = np.zeros(N, dtype=bool)
    spk_count = np.zeros(N, dtype=np.float32)

    block_time = np.full(N, np.nan, dtype=np.float32)
    E_at_block = np.full(N, np.nan, dtype=np.float32)
    K_max = K.copy()

    ring = np.zeros((delay_steps + 1, N), dtype=np.float32)
    tonic = rng.normal(TONIC_MEAN, 0.5, size=N).astype(np.float32)

    W_csc = W.tocsc()
    W_indptr, W_indices, W_data = W_csc.indptr, W_csc.indices, W_csc.data
    absW = abs(W)
    
    # each column = how a neuron shares its K among its synaptic partners (sums to 1)
    col_sums = np.asarray(absW.sum(axis=0)).ravel()
    has_out = col_sums > 0
    col_sums[~has_out] = 1.0
    S = (absW @ sparse.diags(1.0 / col_sums)).tocsc()
    S_indptr, S_indices, S_data = S.indptr, S.indices, S.data
    
    def share(src_idx):
        """How much K each neuron receives when neurons src_idx release 1 unit each."""
        starts = S_indptr[src_idx]
        lens = S_indptr[src_idx + 1] - starts
        pos = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
        return np.bincount(S_indices[pos], weights=S_data[pos], minlength=N)
    
    syn_mode = PROPAGATION in ("synaptic", "both")
    
    noise_scale = np.float32(NOISE_SD * np.sqrt(2 * DT / TAU_M))
    k_remove_slow = slow_dt_s / K_REMOVE_S
    k_leak_slow = K_LEAK_S * slow_dt_s
    base_use_slow = slow_dt_s / E_BASE_USE_S

    rec_t, rec_active, rec_blocked, rec_meanK, rec_meanE, rec_chi = [], [], [], [], [], []
    pop_rate_hist = []

    for step in range(n_steps):
        t_ms = step * DT

        # ===== SLOW (every 1 ms) =====
        if step % SLOW_EVERY == 0:
            # pump power: 1 above E_FULL, 0 below E_FAIL; blocked = no pump
            pump = np.clip((E - E_FAIL) / (E_FULL - E_FAIL), 0.0, 1.0) * (~blocked)
            # K leaks out; the pump clears it
            cleared = k_remove_slow * pump * (K - K0)
            K += k_leak_slow - cleared
            # energy use (only after cutoff)
            if step >= t_anox_step:
                E -= vuln * (base_use_slow + E_SPIKE_COST * spk_count + E_PUMP_COST * cleared)
                np.clip(E, 0.0, 1.0, out=E)
            spk_count[:] = 0.0
            # resting potential
            k_excess = np.maximum(K - K_THRESH, 0.0)
            k_drive = K_LOW_GAIN * np.minimum(K - K0, K_THRESH - K0) + K_HIGH_GAIN * k_excess
            v_rest = V_REST0 + (1.0 - pump) * V_PUMP_MV + k_drive
            # block
            newly_blocked = (v_rest > V_BLOCK) & (~blocked)
            if newly_blocked.any():
                blocked |= newly_blocked
                block_time[newly_blocked] = t_ms
                E_at_block[newly_blocked] = E[newly_blocked]
                dump = K_RELEASE * 5.0
                nb_idx = np.flatnonzero(newly_blocked)
                if syn_mode:
                    self_share = np.where(has_out[nb_idx], 1.0 - SYN_FRACTION, 1.0)
                    K[nb_idx] += dump * self_share
                    src = nb_idx[has_out[nb_idx]]
                    if src.size:
                        K += dump * SYN_FRACTION * share(src)
                else:
                    K[nb_idx] += dump
            # diffusion
            if PROPAGATION in ("spatial", "both"):
                K = (1.0 - D_K * SLOW_EVERY) * K + D_K * SLOW_EVERY * (L @ K)
            np.clip(K, K0, K_PEAK, out=K)
            np.maximum(K_max, K, out=K_max)

        # ===== FAST (every 0.1 ms) =====
        slot = step % (delay_steps + 1)
        g_syn += ring[slot]
        ring[slot] = 0.0
        g_syn -= g_syn * (DT / TAU_SYN)

        not_ref = refr <= 0
        noise = rng.standard_normal(N, dtype=np.float32)
        dv = (-(v - v_rest) + g_syn + tonic) * (DT / TAU_M) + noise_scale * noise
        v[not_ref] += dv[not_ref]
        refr[~not_ref] -= 1

        spiked = (v >= V_THRESH) & (~blocked) & not_ref
        spk_idx = np.flatnonzero(spiked)
        n_spk = spk_idx.size
        if n_spk:
            spk_count[spk_idx] += 1.0
            if syn_mode:
                self_share = np.where(has_out[spk_idx], 1.0 - SYN_FRACTION, 1.0)
                K[spk_idx] += K_SPIKE_RELEASE * self_share
                src = spk_idx[has_out[spk_idx]]
                if src.size:
                    K += K_SPIKE_RELEASE * SYN_FRACTION * share(src)
            else:
                K[spk_idx] += K_SPIKE_RELEASE
            v[spk_idx] = v_rest[spk_idx]
            refr[spk_idx] = refrac_steps
            starts = W_indptr[spk_idx]
            lens = W_indptr[spk_idx + 1] - starts
            pos = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
            out = np.bincount(W_indices[pos], weights=W_data[pos], minlength=N)
            ring[(step + delay_steps) % (delay_steps + 1)] += out
           
        # ===== READOUTS =====
        pop_rate = n_spk / N
        pop_rate_hist.append(pop_rate)
        if step % record_every == 0:
            rec_t.append(t_ms); rec_active.append(pop_rate)
            rec_blocked.append(blocked.mean())
            rec_meanK.append(K.mean()); rec_meanE.append(E.mean())
            if len(pop_rate_hist) > 200:
                win = np.array(pop_rate_hist[-200:])
                rec_chi.append(np.sqrt(win.var() / (win.mean() * (1 - win.mean()) + 1e-9)))
            else:
                rec_chi.append(0.0)

    return dict(t=np.array(rec_t), active=np.array(rec_active),
                blocked=np.array(rec_blocked), meanK=np.array(rec_meanK),
                meanE=np.array(rec_meanE), chi=np.array(rec_chi),
                block_time=block_time, E_at_block=E_at_block,
                K_max=K_max, K_end=K.copy())

def simulate(W, N, vuln, L, t_anox_ms, duration_ms, record_every=10):
    """Run the model. Uses the compiled loop (fast_sim.py) if USE_NUMBA."""
    if not USE_NUMBA:
        return simulate_numpy(W, N, vuln, L, t_anox_ms, duration_ms, record_every)
    import fast_sim
    n_steps = int(duration_ms / DT)
    delay_steps = int(round(DELAY / DT)); refrac_steps = int(round(REFRAC / DT))
    slow_dt_s = DT * SLOW_EVERY / 1000.0

    tonic = rng.normal(TONIC_MEAN, 0.5, size=N)          # same draw as the numpy version
    seed = int(rng.integers(1, 2**62))                   # seed for picking noise from the pool
    noise_pool = rng.standard_normal(2**22).astype(np.float32)   # 4M normal numbers

    W_csc = W.tocsc()
    absW = abs(W)
    col_sums = np.asarray(absW.sum(axis=0)).ravel()
    has_out = col_sums > 0
    col_sums[~has_out] = 1.0
    S = (absW @ sparse.diags(1.0 / col_sums)).tocsc()
    L = L.tocsr()
    f64 = lambda x: np.ascontiguousarray(x, dtype=np.float64)

    out = fast_sim.run(
        n_steps, N, seed, noise_pool,
        W_csc.indptr, W_csc.indices, f64(W_csc.data),
        S.indptr, S.indices, f64(S.data), has_out,
        L.indptr, L.indices, f64(L.data),
        f64(vuln), f64(tonic),
        DT, TAU_M, TAU_SYN, V_REST0, V_THRESH, V_BLOCK, V_PUMP_MV,
        delay_steps, refrac_steps, int(min(t_anox_ms / DT, 2**62)), SLOW_EVERY, record_every,
        K0, K_THRESH, K_PEAK, K_LOW_GAIN, K_HIGH_GAIN, K_SPIKE_RELEASE, K_RELEASE,
        slow_dt_s / K_REMOVE_S, K_LEAK_S * slow_dt_s, slow_dt_s / E_BASE_USE_S,
        E_SPIKE_COST, E_PUMP_COST, E_FULL, E_FAIL,
        D_K, NOISE_SD * np.sqrt(2 * DT / TAU_M),
        PROPAGATION in ("synaptic", "both"), PROPAGATION in ("spatial", "both"), SYN_FRACTION)
    keys = ["t", "active", "blocked", "meanK", "meanE", "chi",
            "block_time", "E_at_block", "K_max", "K_end"]
    return dict(zip(keys, out))

# %% Analysis
def smooth(x, n=50):
    return np.convolve(x, np.ones(n) / n, mode="same")

def analyze(res, meta, coords, t_anox_ms, healthy_peak=None):
    t = res["t"]; act = smooth(res["active"])
    is_death = t_anox_ms < t[-1]

    if not is_death:
        print(f"Control run: peak smoothed activity = {act.max():.4f} -> paste into HEALTHY_PEAK")
    elif healthy_peak is None:
        print("HEALTHY_PEAK not set: run the healthy control first; surge test skipped")
    else:
        after = t >= t_anox_ms
        i = np.argmax(act[after])
        print(f"Peak after cutoff: {act[after][i]:.4f} at {(t[after][i] - t_anox_ms)/1000:.1f} s "
              f"(healthy max {healthy_peak:.4f}) -> surge: {act[after][i] > healthy_peak}")

    bt = res["block_time"]
    df = pd.DataFrame({"super_class": meta["super_class"].values, "block_ms": bt}).dropna()
    order = df.groupby("super_class")["block_ms"].median().sort_values()
    print("\nSilencing order by super_class (median block time, ms):")
    print(order.to_string() if len(order) else "(nothing blocked)")

    print(f"\nLocal K peak (median): {np.median(res['K_max']):.1f} (fly ~54; capped at {K_PEAK:.0f})")
    print(f"K at end of run (mean): {res['K_end'].mean():.1f}")
    if np.isfinite(res["E_at_block"]).any():
        print(f"Energy at block (median): {np.nanmedian(res['E_at_block']):.2f} (v1: 0.82; should be LOW)")

    d, nb = cKDTree(coords).query(coords, k=9)
    dt_pairs = np.abs(bt[:, None] - bt[nb[:, 1:]])
    ok = np.isfinite(dt_pairs) & (dt_pairs > 0)
    if ok.any():
        local = d[:, 1:][ok] / dt_pairs[ok]
        print(f"Local wave speed (median): {np.median(local) * 0.06:.2f} mm/min (fly ~0.77)")

    reached = np.flatnonzero(res["blocked"] >= 0.5)
    if is_death and reached.size:
        print(f"Time to 50% blocked: {(t[reached[0]] - t_anox_ms)/1000:.1f} s after cutoff (fly ~30)")
    else:
        print("50% blocked never reached (or no cutoff in this run)")

def plot(res):
    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for a, key, label in zip(ax, ["active", "blocked", "meanK", "meanE", "chi"],
                             ["active", "blocked", "mean K", "mean energy", "synchrony"]):
        a.plot(res["t"], res[key]); a.set_ylabel(label)
    ax[-1].set_xlabel("time (ms)")
    plt.tight_layout(); plt.show()

# %% RUN
if __name__ == "__main__":
    meta, conn = load_data()
    idx, W, sign, coords, N = build_network(meta, conn)
    vuln = build_vulnerability(sign)
    L = build_spatial_graph(coords)
    t_anox = 1e12 if RUN_MODE == "healthy" else T_ANOX_MS
    res = simulate(W, N, vuln, L, t_anox, DURATION_MS)
    analyze(res, meta, coords, t_anox, None if RUN_MODE == "healthy" else HEALTHY_PEAK)
    plot(res)
    if RUN_MODE == "death":
        label = f"{PROPAGATION}_shuffle{SHUFFLE_CONN}_seed{SEED}"
        np.save(f"block_time_{label}.npy", res["block_time"])
        print("saved", label)


# %% Compare saved runs (run this cell by itself; the runner does this for you)
# Inside the __main__ guard so importing this file (e.g. from runner.py)
# does not try to load these files.
if __name__ == "__main__":
    from scipy.stats import spearmanr
    a = np.load("block_time_spatial_shuffleFalse.npy")
    b = np.load("block_time_both_shuffleFalse.npy")
    c = np.load("block_time_both_shuffleTrue.npy")
    a2 = np.load("block_time_spatial_shuffleFalse2.npy")
    ok = np.isfinite(a) & np.isfinite(a2) & np.isfinite(b) & np.isfinite(c)
    print("A vs A2 (noise ceiling):", spearmanr(a[ok], a2[ok])[0])
    print("A vs B:", spearmanr(a[ok], b[ok])[0])
    print("A vs C:", spearmanr(a[ok], c[ok])[0])
    print("B vs C:", spearmanr(b[ok], c[ok])[0])
