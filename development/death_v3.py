# %% [markdown]
# FlyWire "brain death" simulation - v3 (K in cubes of neuropil)
# ==============================================================
# v2 kept one K value per neuron and mixed it with the 8 nearest neuron points.
# Validation showed that geometry was wrong: dense groups of similar cells
# (Kenyon cells) became islands, and the points were arbitrary marks on each
# neuron (log rows 41-42).
#
# NEW in v3: extracellular K lives in CUBES (default 10 um) of neuropil.
#   - each neuron sits in the cubes that hold its synapses (from the FlyWire
#     synapse table, see flywire_synapse_cubes.ipynb)
#   - it feels the K of those cubes, and its pump clears K there
#   - it releases K where its OUTPUT synapses are
#   - K diffuses between neighbouring cubes with a diffusion constant in um^2/s
# The neuron model (LIF), energy, pump failure and block are the same as v2.
# There is no separate "synaptic route" any more: K released at a synapse
# lands where that synapse is, which is where the partner is.
#
# Conditions (COUPLING):
#   "real" : every neuron uses its own synapse locations
#   "swap" : synapse-location profiles are exchanged between neurons with the
#            same main neuropil (each region keeps its layout, but it is the
#            wrong neuron in each place) - control for "does wiring matter"
# Still a TOY: one K number per cube, no glia, no real ion chemistry.

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
COORDS_CSV      = "coordinates.csv"      # only used for neurons missing from the cube file
CLASS_CSV       = "classification.csv"
CUBE_FILE       = "synapse_cubes_10um.npz"   # made by flywire_synapse_cubes.ipynb

# ---- run mode ----
RUN_MODE     = "death"   # "healthy" | "death" | "trigger" | "bath"
#   healthy : oxygen never cut (control)
#   death   : oxygen cut at T_ANOX_MS
#   trigger : oxygen kept; at TRIGGER_T_MS the cubes within TRIGGER_RADIUS_UM of
#             TRIGGER_AT are set to TRIGGER_K (like a KCl pinprick) - does a wave
#             spread through healthy tissue, and how fast? (Harris: 0.77 mm/min)
#   bath    : oxygen kept; from BATH_T_MS the whole brain sits in extra K
#             (Harris et al.: 5 -> 12.9 mM, SD starts within 14 +- 4.4 s)
T_ANOX_MS    = 2000.0
DURATION_MS  = 60000.0
BOX_HALF_NM  = None      # None = whole brain

# ---- trigger / bath experiments ----
TRIGGER_T_MS      = 2000.0
TRIGGER_AT        = "center"   # "center" (middle of the neuropil) or (x, y, z) in um
TRIGGER_RADIUS_UM = 25.0
TRIGGER_K         = 40.0
TRIGGER_HOLD_MS   = 1000.0     # K is held at TRIGGER_K this long (a single pulse diffuses away in ~0.1 s)
BATH_T_MS         = 2000.0
BATH_DK           = 7.9        # 12.9 - 5.0 mM (Harris et al. 2026)

# ---- LIF (Shiu et al. 2024) ----
V_REST0  = -52.0; V_THRESH = -45.0
TAU_M = 20.0; TAU_SYN = 5.0; REFRAC = 2.2; DELAY = 1.8
W_SYN = 0.275; DT = 0.1

# ---- healthy drive ----
TONIC_MEAN = 5.0
NOISE_SD   = 2.0

# ---- extracellular K+ (same meaning as v2; now per cube) ----
K0 = 3.0
K_THRESH = 10.0
K_PEAK = 55.0            # imposed cap (fly plateau ~54)
K_LOW_GAIN = 0.3
K_HIGH_GAIN = 3.0
K_SPIKE_RELEASE = 0.01   # per spike, in a cube of typical density
K_LEAK_S = 0.3
K_RELEASE = 0.0          # v2 neuron "block dump" (K_RELEASE * 5). OFF in v3.1: the K surge now comes
                         #   from the tissue's spreading depolarization (SD_* below)
K_REMOVE_S = 1.0

# ---- cube geometry (NEW) ----
D_UM2_S     = 770.0      # effective K diffusion constant in brain tissue, um^2/s: free K+ in water
                         #   ~1960 um^2/s divided by tortuosity^2 (~1.6^2) - starting value, may be calibrated
MAX_CUBES   = 128        # at most this many cubes per neuron (real data: median neuron touches 35, 90th percentile 143) ...
CUBE_COVER  = 0.95       # ... or fewer if they already hold 95% of its synapses
CUBE_ACTIVE_K = 20.0     # (unused since v3.1: a cube is "reached" when it goes into SD)

# ---- spreading depolarization of the tissue (NEW in v3.1) ----
SD_ON    = 12.0          # mM: tissue depolarizes when local K passes this (mammals: K ceiling
                         #   ~10-12 mM just before anoxic depolarization, Hansen 1985)
SD_OFF   = 6.0           # mM: tissue recovers when pumps bring K back below this
SD_RATE  = 20.0          # mM/s of K released by depolarized tissue at the start (guess, to calibrate)
SD_TAU_S = 20.0          # s: release fades with this time constant (guess, to calibrate)
SD_PUMP  = 0.2           # K clearance of tissue while it is in SD, relative to normal (guess)
COUPLING    = "real"     # "real" | "swap"
SWAP_SEED   = 12345

# ---- energy ----
E_BASE_USE_S = 70.0
E_SPIKE_COST = 5e-4
E_PUMP_COST  = 0.01
E_FULL = 0.6
E_FAIL = 0.2
V_PUMP_MV = 5.0
V_BLOCK = -25.0
V_UNBLOCK = -40.0        # NEW v3.1: a blocked neuron recovers when its resting potential falls
                         #   back below this (needs K back down and working pumps; never in anoxia)

SLOW_EVERY = 50          # energy, pumps and cube K updated every 5 ms (K changes over seconds)

# ---- scenario ----
VULN_MODE    = "uniform"
SHUFFLE_CONN = False     # shuffles the WIRING (who spikes onto whom), as in v2
SHUFFLE_MODE = "sign"


# %% Load data
def load_data():
    neurons = pd.read_csv(NEURONS_CSV, usecols=["root_id", "nt_type"])
    coords = pd.read_csv(COORDS_CSV)
    xyz = coords["position"].str.strip("[]").str.split(expand=True).astype(float)
    coords["x"], coords["y"], coords["z"] = xyz[0], xyz[1], xyz[2]
    coords = coords[["root_id", "x", "y", "z"]].drop_duplicates("root_id")
    classes = pd.read_csv(CLASS_CSV, usecols=["root_id", "super_class"])
    meta = neurons.merge(coords, on="root_id", how="inner").merge(classes, on="root_id", how="left")
    if BOX_HALF_NM is not None:
        cx, cy, cz = meta[["x", "y", "z"]].median()
        h = BOX_HALF_NM
        box = ((meta.x - cx).abs() < h) & ((meta.y - cy).abs() < h) & ((meta.z - cz).abs() < h)
        meta = meta[box].reset_index(drop=True)

    conn = pd.read_csv(CONNECTIONS_CSV, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
    # main neuropil of each neuron (most synapses, in + out) - used by the "swap" control
    both = pd.concat([conn[["pre_root_id", "neuropil", "syn_count"]].rename(columns={"pre_root_id": "root_id"}),
                      conn[["post_root_id", "neuropil", "syn_count"]].rename(columns={"post_root_id": "root_id"})])
    main = (both.groupby(["root_id", "neuropil"])["syn_count"].sum().reset_index()
                .sort_values("syn_count").drop_duplicates("root_id", keep="last"))
    meta = meta.merge(main[["root_id", "neuropil"]], on="root_id", how="left")
    meta["neuropil"] = meta["neuropil"].fillna("none")

    conn = (conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum())
    keep = set(meta["root_id"])
    conn = conn[conn["pre_root_id"].isin(keep) & conn["post_root_id"].isin(keep)].reset_index(drop=True)
    print(len(meta), "neurons,", len(conn), "connections",
          "(whole brain)" if BOX_HALF_NM is None else "in box")
    return meta, conn


# %% Network (same as v2)
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
        g = np.random.default_rng(12345)
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
    vuln = np.ones(len(neuron_sign), dtype=np.float32)
    if VULN_MODE == "inhibitory":
        vuln[neuron_sign < 0] = 2.0; vuln[neuron_sign > 0] = 0.7
    elif VULN_MODE == "excitatory":
        vuln[neuron_sign > 0] = 2.0; vuln[neuron_sign < 0] = 0.7
    vuln *= rng.lognormal(0.0, 0.25, size=len(vuln)).astype(np.float32)
    return vuln


# %% Cube geometry (NEW)
_GEOM_CACHE = {}

def _top_cubes(n, c, w, N):
    """Keep, per neuron, its biggest cubes until CUBE_COVER of its synapses or
    MAX_CUBES cubes; return CSR (indptr, cube, weight) with rows summing to 1."""
    df = pd.DataFrame({"n": n, "c": c, "w": w.astype(np.float64)})
    df = df.groupby(["n", "c"], as_index=False)["w"].sum()
    df = df.sort_values(["n", "w"], ascending=[True, False])
    tot = df.groupby("n")["w"].transform("sum")
    before = df.groupby("n")["w"].cumsum() - df["w"]          # share already covered before this cube
    rank = df.groupby("n").cumcount()
    df = df[(rank < MAX_CUBES) & (before < CUBE_COVER * tot)]
    df["w"] = df["w"] / df.groupby("n")["w"].transform("sum")
    counts = np.bincount(df["n"].values, minlength=N)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return indptr, df["c"].values.astype(np.int64), df["w"].values.astype(np.float64)


def build_geometry(meta, idx, coords):
    key = (CUBE_FILE, MAX_CUBES, CUBE_COVER, COUPLING, SWAP_SEED, BOX_HALF_NM, len(meta))
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    d = np.load(CUBE_FILE)
    file_ids = d["root_ids"]                                   # sorted
    cube_um = float(d["cube_um"]); dims = d["dims"].astype(np.int64); origin = d["origin_nm"] / 1000.0
    N = len(meta)
    # map cube-file neuron index -> model neuron index
    model_ids = meta["root_id"].values.astype(np.int64)
    pos = np.searchsorted(file_ids, model_ids)
    pos = np.clip(pos, 0, len(file_ids) - 1)
    found = file_ids[pos] == model_ids
    to_model = np.full(len(file_ids), -1, np.int64)
    to_model[pos[found]] = np.flatnonzero(found)

    pre_n = to_model[d["pre_neuron"]];  ok = pre_n >= 0
    pre_n, pre_c, pre_w = pre_n[ok], d["pre_cube"][ok].astype(np.int64), d["pre_count"][ok]
    post_n = to_model[d["post_neuron"]]; ok = post_n >= 0
    post_n, post_c, post_w = post_n[ok], d["post_cube"][ok].astype(np.int64), d["post_count"][ok]

    # presence = all synapses (in + out); outputs = pre synapses only
    P_ptr, P_c, P_w = _top_cubes(np.concatenate([pre_n, post_n]), np.concatenate([pre_c, post_c]),
                                 np.concatenate([pre_w, post_w]), N)

    # neurons without synapses in the file: put them in the cube at their marked point
    missing = np.flatnonzero(np.diff(P_ptr) == 0)
    if missing.size:
        gi = ((coords[missing] / 1000.0 - origin) // cube_um).astype(np.int64)
        gi = np.clip(gi, 0, dims - 1)
        gc = gi[:, 0] + dims[0] * (gi[:, 1] + dims[1] * gi[:, 2])
        n_all = np.concatenate([np.repeat(np.arange(N), np.diff(P_ptr)), missing])
        c_all = np.concatenate([P_c, gc]); w_all = np.concatenate([P_w, np.ones(missing.size)])
        P_ptr, P_c, P_w = _top_cubes(n_all, c_all, w_all, N)
        print(f"  {missing.size} neurons had no synapses in the cube file; placed at their marked point")

    # outputs: only cubes that are also in the neuron's presence set (so K is always released
    # where there is tissue to clear it); neurons with no such outputs release where they sit
    rows = np.repeat(np.arange(N), np.diff(P_ptr))
    pres_set = pd.DataFrame({"n": rows, "c": P_c})
    out = pd.DataFrame({"n": pre_n, "c": pre_c, "w": pre_w}).merge(pres_set, on=["n", "c"])
    has_out = np.zeros(N, bool); has_out[out["n"].unique()] = True
    fill = pres_set[~has_out[pres_set["n"].values]].assign(w=P_w[~has_out[rows]])
    out = pd.concat([out, fill])
    O_ptr, O_c, O_w = _top_cubes(out["n"].values, out["c"].values, out["w"].values, N)

    # control: exchange whole profiles between neurons with the same main neuropil
    if COUPLING == "swap":
        g = np.random.default_rng(SWAP_SEED)
        perm = np.arange(N)
        for _, grp in meta.groupby("neuropil").groups.items():
            grp = np.asarray(grp)
            perm[grp] = grp[g.permutation(grp.size)]
        def permute(ptr, c, w):
            lens = np.diff(ptr)[perm]
            src = np.concatenate([np.arange(ptr[p], ptr[p + 1]) for p in perm])
            return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64), c[src], w[src]
        P_ptr, P_c, P_w = permute(P_ptr, P_c, P_w)
        O_ptr, O_c, O_w = permute(O_ptr, O_c, O_w)
    elif COUPLING != "real":
        raise ValueError(f"unknown COUPLING {COUPLING}")

    # domain = cubes where neurons sit; renumber them 0..C-1
    dom = np.unique(P_c)
    P_idx = np.searchsorted(dom, P_c).astype(np.int64)
    O_idx = np.searchsorted(dom, O_c).astype(np.int64)
    C = dom.size
    m = np.bincount(P_idx, weights=P_w, minlength=C)          # neuron-equivalents per cube
    # reference density = the density of the cube a typical NEURON sits in (weighted by
    # presence), so that for a typical neuron m_rel ~ 1 and leak/clearance rates match v2
    m_ref = float((m ** 2).sum() / m.sum())
    m_rel = m / m_ref

    # 6 neighbours of each cube (only neighbours inside the domain)
    ix = dom % dims[0]; iy = (dom // dims[0]) % dims[1]; iz = dom // (dims[0] * dims[1])
    nb = np.full((C, 6), -1, np.int64)
    for q, (dx, dy, dz) in enumerate([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]):
        jx, jy, jz = ix + dx, iy + dy, iz + dz
        inside = (jx >= 0) & (jx < dims[0]) & (jy >= 0) & (jy < dims[1]) & (jz >= 0) & (jz < dims[2])
        gid = jx + dims[0] * (jy + dims[1] * jz)
        j = np.clip(np.searchsorted(dom, gid), 0, C - 1)
        nb[:, q] = np.where(inside & (dom[j] == gid), j, -1)
    centers_um = origin + (np.stack([ix, iy, iz], 1) + 0.5) * cube_um

    geom = dict(P_ptr=P_ptr, P_idx=P_idx, P_w=P_w, O_ptr=O_ptr, O_idx=O_idx, O_w=O_w,
                m_rel=m_rel, m_ref=m_ref, nb=nb, centers_um=centers_um, cube_um=cube_um,
                cube_ids=dom, dims=dims)
    print(f"  geometry: {C:,} cubes of {cube_um:.0f} um, typical density {m_ref:.2f} neuron-equivalents per cube, "
          f"{np.diff(P_ptr).mean():.1f} cubes per neuron, coupling = {COUPLING}")
    _GEOM_CACHE[key] = geom
    return geom


def build_spatial_graph(coords):
    raise RuntimeError("v3 uses build_geometry(meta, idx, coords); update runner.py")


# %% Simulation
def simulate(W, N, vuln, geom, t_anox_ms, duration_ms, record_every=10):
    import fast_sim_v3
    n_steps = int(duration_ms / DT)
    tonic = rng.normal(TONIC_MEAN, 0.5, size=N)
    seed = int(rng.integers(1, 2**62))
    noise_pool = rng.standard_normal(2**22).astype(np.float32)
    W_csc = W.tocsc()
    f64 = lambda x: np.ascontiguousarray(x, dtype=np.float64)
    # experiments
    trig_step, trig_end, trig_cubes, bath_step = -1, -1, np.zeros(0, np.int64), -1
    if RUN_MODE == "trigger":
        c = geom["centers_um"]
        at = np.median(c, axis=0) if isinstance(TRIGGER_AT, str) else np.asarray(TRIGGER_AT, float)
        near = np.flatnonzero(np.linalg.norm(c - at, axis=1) <= TRIGGER_RADIUS_UM)
        if near.size == 0:
            near = np.array([np.argmin(np.linalg.norm(c - at, axis=1))])
        trig_step, trig_cubes = int(TRIGGER_T_MS / DT), near.astype(np.int64)
        trig_end = int((TRIGGER_T_MS + TRIGGER_HOLD_MS) / DT)
        print(f"  trigger: {near.size} cubes around {np.round(at)} um held at K = {TRIGGER_K} "
              f"from {TRIGGER_T_MS / 1000:.1f} s for {TRIGGER_HOLD_MS / 1000:.1f} s")
    elif RUN_MODE == "bath":
        bath_step = int(BATH_T_MS / DT)
        print(f"  bath: K background +{BATH_DK} from {BATH_T_MS / 1000:.1f} s")
    out = fast_sim_v3.run(
        n_steps, N, seed, noise_pool,
        W_csc.indptr, W_csc.indices, f64(W_csc.data),
        geom["P_ptr"], geom["P_idx"], geom["P_w"],
        geom["O_ptr"], geom["O_idx"], geom["O_w"],
        geom["m_rel"], geom["m_ref"], geom["nb"],
        f64(vuln), f64(tonic),
        DT, TAU_M, TAU_SYN, V_REST0, V_THRESH, V_BLOCK, V_PUMP_MV,
        int(round(DELAY / DT)), int(round(REFRAC / DT)), int(min(t_anox_ms / DT, 2**62)),
        SLOW_EVERY, record_every,
        K0, K_THRESH, K_PEAK, K_LOW_GAIN, K_HIGH_GAIN, K_SPIKE_RELEASE, K_RELEASE,
        K_REMOVE_S, K_LEAK_S, E_BASE_USE_S, E_SPIKE_COST, E_PUMP_COST, E_FULL, E_FAIL,
        D_UM2_S, geom["cube_um"], NOISE_SD * np.sqrt(2 * DT / TAU_M), CUBE_ACTIVE_K,
        trig_step, trig_end, trig_cubes, TRIGGER_K, bath_step, BATH_DK,
        SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP, V_UNBLOCK)
    keys = ["t", "active", "blocked", "meanK", "meanE", "chi", "block_time", "E_at_block",
            "K_max", "K_end", "cube_time", "cube_K_end"]
    return dict(zip(keys, out))


def cube_speed_mm_min(cube_time, geom):
    """Wave speed from cube arrival times, as 1 / |gradient of arrival time|.
    (Taking neighbour distance / time difference over all neighbour pairs
    overestimates speed: pairs not lined up with the wave see a smaller time
    difference. The gradient uses all three directions at once.)"""
    nb = geom["nb"]; h = geom["cube_um"]; t = cube_time
    grad2 = np.zeros(len(t)); n_ok = np.zeros(len(t), int)
    for ax in range(3):
        up, dn = nb[:, 2 * ax], nb[:, 2 * ax + 1]
        tu = np.where(up >= 0, t[np.maximum(up, 0)], np.nan)
        td = np.where(dn >= 0, t[np.maximum(dn, 0)], np.nan)
        g = np.where(np.isfinite(tu) & np.isfinite(td), (tu - td) / (2 * h),
            np.where(np.isfinite(tu), (tu - t) / h, np.where(np.isfinite(td), (t - td) / h, np.nan)))
        ok = np.isfinite(g)
        grad2[ok] += g[ok] ** 2; n_ok += ok
    good = np.isfinite(t) & (n_ok == 3) & (grad2 > 0)
    if not good.any():
        return np.nan
    return float(np.median(1.0 / np.sqrt(grad2[good])) * 60.0)   # um/ms * 60 = mm/min


# %% Analysis
def smooth(x, n=50):
    return np.convolve(x, np.ones(n) / n, mode="same")

def analyze(res, geom, t_anox_ms):
    t = res["t"]; act = smooth(res["active"])
    print(f"Mean firing {res['active'].mean() / (DT / 1000):.1f} Hz; peak smoothed activity {act.max():.4f}")
    print(f"Cube K at end: median {np.median(res['cube_K_end']):.1f}, max {res['cube_K_end'].max():.1f}")
    print(f"Neuron K felt at end: median {np.median(res['K_end']):.1f}, max {res['K_max'].max():.1f}")
    if RUN_MODE in ("trigger", "bath"):
        t0 = TRIGGER_T_MS if RUN_MODE == "trigger" else BATH_T_MS
        ct = res["cube_time"]
        print(f"Cubes that went into SD: {np.isfinite(ct).sum():,} "
              f"of {ct.size:,} ({np.isfinite(ct).mean():.2%})")
        bt = res["block_time"]
        if np.isfinite(bt).any():
            print(f"First neuron blocked {(np.nanmin(bt) - t0) / 1000:.1f} s after the {RUN_MODE}")
            print(f"Neurons blocked: at most {res['blocked'].max():.1%}, at end {res['blocked'][-1]:.1%}")
        else:
            print("No neuron blocked")
        if np.isfinite(ct).any():
            print(f"First SD {(np.nanmin(ct) - t0) / 1000:.1f} s after the {RUN_MODE}"
                  + ("  (Harris KCl: 14 +- 4.4 s)" if RUN_MODE == "bath" else ""))
            reach = np.sort(ct[np.isfinite(ct)] - t0) / 1000
            for q in [0.1, 0.5, 0.9]:
                if len(reach) >= q * ct.size:
                    print(f"  {q:.0%} of cubes in SD by {reach[int(q * ct.size) - 1]:.1f} s")
        if np.isfinite(ct).sum() > 10:
            print(f"Wave speed (cubes): {cube_speed_mm_min(ct, geom):.2f} mm/min (fly ~0.77)")
        return
    if t_anox_ms < t[-1]:
        reached = np.flatnonzero(res["blocked"] >= 0.5)
        print(f"50% blocked: {(t[reached[0]] - t_anox_ms) / 1000:.1f} s after cutoff" if reached.size
              else "50% blocked never reached")
        print(f"Wave speed (cubes): {cube_speed_mm_min(res['cube_time'], geom):.2f} mm/min (fly ~0.77)")
        eb = res["E_at_block"]
        if np.isfinite(eb).any():
            print(f"Energy at block (median): {np.nanmedian(eb):.2f}")

def plot(res):
    fig, ax = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
    for a, key, label in zip(ax, ["active", "blocked", "meanK", "meanE", "chi"],
                             ["active", "blocked", "mean cube K", "mean energy", "synchrony"]):
        a.plot(res["t"], res[key]); a.set_ylabel(label)
    ax[-1].set_xlabel("time (ms)")
    plt.tight_layout(); plt.show()


# %% RUN
if __name__ == "__main__":
    import time
    t_start = time.time()
    meta, conn = load_data()
    idx, W, sign, coords, N = build_network(meta, conn)
    vuln = build_vulnerability(sign)
    geom = build_geometry(meta, idx, coords)
    t_anox = T_ANOX_MS if RUN_MODE == "death" else 1e12
    t_sim = time.time()
    print(f"Loading and setup took {(t_sim - t_start) / 60:.1f} min; simulating {DURATION_MS / 1000:.0f} s ...")
    res = simulate(W, N, vuln, geom, t_anox, DURATION_MS)
    print(f"Simulation took {(time.time() - t_sim) / 60:.1f} min "
          f"(total {(time.time() - t_start) / 60:.1f} min)")
    analyze(res, geom, t_anox)
    plot(res)
    if RUN_MODE != "healthy":
        label = f"v3_{RUN_MODE}_{COUPLING}_shuffle{SHUFFLE_CONN}_seed{SEED}"
        np.save(f"block_time_{label}.npy", res["block_time"])
        np.save(f"cube_time_{label}.npy", res["cube_time"])
        print("saved", label)
