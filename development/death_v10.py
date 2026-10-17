# %% [markdown]
# FlyWire "brain death" simulation - v10 = v9 + firing brake (NA_INACT, RESET_MODE; log row 89), OFF by default (= v9).
# v9 = v8 + glial K buffering + modulator switch (log row 82). OFF by default (= v8).
#   GLIA_ON          : each neuropil cube gets a glial K store (Level 1), a 'shuttle' buffer (MacAulay 2020 review;
#                      Larsen et al. 2014; Nielsen et al. 2025): glia take up excess K FAST while it is high
#                      (GLIA_UPTAKE_S x (K - K0), capped at GLIA_CAP mM) and give it back SLOWLY (GLIA_RELEASE_S);
#                      the neurons' calibrated clearance (unchanged) removes it for good. Uptake is part passive
#                      (GLIA_PASSIVE, Kir-like) and part energy-dependent (glial Na/K pump, also blocked by ouabain);
#                      release is passive. Healthy steady state unchanged (uptake = release, no net removal).
#                      Size: blocking glial Kir4.1 raised activity-evoked [K]o peaks to 122% (Larsen 2014), blocking
#                      glial a2 Na/K-ATPase to 116% (Nielsen 2025); neither changed post-stimulus recovery (neuronal
#                      a3) -> uptake 0.4/s next to neuronal 1/s gives ~135-140% peaks without glia.
#   GLIA_DENSITY     : "uniform" (Kremer et al. 2017: neuropil glia without spatial registration, ~1:1 astrocyte-like :
#                      ensheathing everywhere) or "lamina" (lamina cubes x GLIA_LAMINA_FACTOR; decision record in the
#                      Future directions tab, Oct 2 18:55). Glia energy follows the cube's neuron pump state (proxy).
#   MOD_SIGN         : sign/weight given to dopamine, serotonin, octopamine synapses (v2-v8: 0 = no effect).
#                      Robustness check only: +-0.5 = weakly excitatory / inhibitory.
# FlyWire "brain death" simulation - v8 = v7 + physical SD release (log row 81). OFF by default (= v7).
#   SD_KMAX > 0     : K release of tissue in SD is limited by its driving force, x (SD_KMAX - K)/(SD_KMAX - SD_ON):
#                     unchanged at the wave front, stops by itself near SD_KMAX (fly anoxic plateau ~54 mM)
#                     instead of running on a fixed timer into the K_PEAK cap.
#   SD_PUMP_RECOVER : the pump reduction in SD tissue (x SD_PUMP) fades with the release (SD_TAU_S) instead of
#                     lasting until K < SD_OFF - SD tissue can recover once energy is back.
#   SHUFFLE_MODE "break_recip": rewire E->I loop connections among themselves and I->E loop connections among
#                     themselves: every neuron keeps its degrees AND which cell type it targets; only the
#                     reciprocity of the loops is broken (clean version of "break_loops", row 79).
# FlyWire "brain death" simulation - v7 = v6 + independent validation tests (log row 80)
# v7: everything below is OFF by default, and then v7 gives exactly the v6 result.
#   - T_REOX_MS      : oxygen comes back (Spong et al. 2016: N2 for 60 s, then air; DC potential back to
#                      half amplitude 11.8-16.1 s after N2 off). Energy recovers toward 1 with E_RECOVER_TAU_S.
#   - RUN_MODE "ouabain": oxygen kept; Na/K pumps blocked by OUABAIN_BLOCK (ramped in over OUABAIN_RAMP_S)
#                      from OUABAIN_T_MS (Spong 2016: repetitive SD, first surge 10-70 mM, median 24 mM).
#   - virtual K electrodes: K of PROBE_N random neuropil cubes + the central cube every PROBE_EVERY slow
#                      steps (50 ms), saved to SAVE_PROBES_TO; plus SD events per cube (repetitive SD).
# FlyWire "brain death" simulation - v6 (= v5 + rewiring controls + firing rates per neuron)
# v6 (log row 73): SHUFFLE_MODE "degree" (every neuron keeps its numbers of inputs and outputs),
#   SHUFFLE_SEED (several different rewirings), and per-neuron firing rates before/after oxygen loss.
#   Room-temperature anoxia targets (log rows 70-72): SD speed 2.1-2.5 mm/min, SD 15-50 s after N2,
#   surge from ~10 mM, plateau ~54 mM.
# v5 (log row 65): in v4 the edges of the neuropil are sealed, so edge cubes keep their K and
#   reach threshold first (lamina-first). The exchange at the edges is not known for flies:
#   EDGE_G_S (outer brain surface: cell-body rind + glial sheath / blood-brain barrier) and
#   INNER_G_S (gaps inside the brain: cell-body clusters, tracts) are swept, not fitted.
#   Inside the neuropil a cube swaps K with each neighbour at D / h^2 = 7.7 /s.
# =====================================================================================================
# v4 changes (log rows 62-64):
#   - SD_DELAY_S: a cube goes into SD only after its K stayed above SD_ON this long. With release
#     rate alone the slowest wave that propagates was ~3 mm/min (fly 0.77): below that it fails.
#   - BATH_MODE "surface": KCl enters only through the outer surface of the brain (exchange
#     BATH_G_S per open face per s, bath at BATH_K mM absolute) instead of everywhere at once.
#   - STOP_EARLY: a run ends when the brain is dead (or a trigger has clearly failed) and
#     nothing has changed for STOP_QUIET_MS. Reported numbers are unchanged.
#   - simulate_tissue(): the cube layer alone (no neurons; spikes replaced by a steady release
#     at HEALTHY_RATE_HZ). For calibration only - checked against a full run first.
#
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
RUN_MODE     = "death"   # "healthy" | "death" | "trigger" | "bath" | "ouabain" (v7)
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
TRIGGER_HOLD_MS   = 3000.0     # K is held at TRIGGER_K this long; must exceed SD_DELAY_S (v3: 1000)
BATH_T_MS         = 2000.0
BATH_DK           = 7.9        # 12.9 - 5.0 mM (Harris et al. 2026); used by BATH_MODE "everywhere"
BATH_MODE         = "surface"  # "surface" (v4) | "everywhere" (v3: pumps clear only down to K0 + BATH_DK)
BATH_K            = 12.9       # mM, bath K at the brain surface (absolute; Harris et al. 2026)
BATH_G_S          = 3.0        # 1/s per open face: exchange of a surface cube with the bath (sheath + rind)
BATH_RAMP_S       = 0.0        # s: bath K rises linearly over this time (wash-in); 0 = step
SURF_CLOSE_VOX    = 3          # gaps narrower than ~2x this many cubes inside the brain are not "surface"

# ---- v7 validation tests ----
T_REOX_MS       = None     # ms: oxygen returns (death mode); None = never
E_RECOVER_TAU_S = 2.0      # s: energy recovery time constant after reoxygenation (not known; swept)
OUABAIN_T_MS    = 2000.0   # ouabain mode: pumps start to be blocked here
OUABAIN_BLOCK   = 1.0      # fraction of pump capacity blocked at full effect
OUABAIN_RAMP_S  = 10.0     # s: wash-in
PROBE_N         = 40       # virtual K electrodes: random neuropil cubes (+ the central cube)
PROBE_EVERY     = 10       # slow steps (x 5 ms) between electrode samples
PROBE_SEED      = 7
SAVE_PROBES_TO  = None     # path of an .npz to save the electrode traces to (None = not saved)

# ---- v8 SD physics ----
SD_KMAX         = 0.0      # mM; 0 = off (v7: fixed-time release capped by K_PEAK). On: 54 (fly anoxic plateau)
SD_PUMP_RECOVER = False    # True: pump reduction in SD fades with the release

# ---- v9 glia + modulators ----
GLIA_ON            = False
GLIA_UPTAKE_S      = 0.4     # 1/s next to neuronal clearance 1/s (from the Kir4.1 and a2 block experiments,
                             #   rat hippocampus: mammalian, order of magnitude)
GLIA_RELEASE_S     = 0.05    # 1/s: slow return of glial K (must not change the fast post-stimulus decay) - guessed
GLIA_CAP           = 10.0    # mM of extracellular K the glia can hold before saturating - guessed, swept
GLIA_PASSIVE       = 0.5     # passive share of uptake (Kir effect 22% ~ a2 pump effect 16%) - rough
GLIA_DENSITY       = "uniform"   # "uniform" | "lamina"
GLIA_LAMINA_FACTOR = 1.0     # lamina glia density relative to the rest (estimated by stage13 from Kremer counts)
MOD_SIGN           = 0.0     # weight sign of DA / SER / OCT synapses (0 = silent, as v2-v8; Shiu et al. 2024 use +1:
                             #   their Connectivity_783 file signs every connection +-1, only GABA / Glu negative)
# ---- v9 variant: how extracellular K depolarizes neurons (log row 84) ----
K_COUPLING         = "piecewise" # "piecewise" (v2-v9: K_LOW_GAIN up to K_THRESH, K_HIGH_GAIN above; never sourced)
                                 # "goldman": GHK_SLOPE x log10((K + GHK_ALPHA) / (K0 + GHK_ALPHA)) (membrane physics)
GHK_ALPHA          = 3.5     # mM = (P_Na / P_K) x [Na]o: 0.05 (typical resting neuron) x ~70 mM (adult fly hemolymph
                             #   [Na+] at 25 C, MacMillan et al. 2015; brain extracellular Na behind the BBB not measured)
GHK_SLOPE          = 58.2    # mV per decade at 20 C (RT/F ln10); full K selectivity would give the same slope

# ---- v10 firing brake (log row 89). Reason: with Goldman coupling, after reoxygenation K stayed ~7 mM, v_rest ~ -40 mV
#   was above threshold and the old reset (to v_rest) let neurons fire every refractory period (~306 Hz): no recovery.
NA_INACT   = False   # Na inactivation raises the threshold of depolarized neurons (Platkiewicz & Brette 2010, 2011)
NA_VHALF   = -43.5   # mV, h_inf half point. WEAK: Berger & Crook 2015 (Front Comput Neurosci 9:139) Table 3 gives -38.9 mV
                     #   (Olson et al. 2008, DmNav expressed in Xenopus oocytes, + Lin et al. 2009); splice variants span
                     #   -34.9 to -66.4 mV; their model fits -39.1 / -42.1. Not adult brain neurons. Sensitivity variant only.
NA_KI      = 4.8     # mV, inactivation slope (their 3D model fit 4.9)
NA_KA      = 4.0     # mV, activation slope = ka in the threshold equation (rough; their fit z_m 3 ~ 8.5 mV)
RESET_MODE = "rest"  # "rest": after a spike v -> current v_rest (v2-v9, never sourced)
                     # "fixed": v -> V_RESET (Shiu et al. 2024 LIF: reset -52 mV = rest)
V_RESET    = -52.0

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
SD_ON    = 11.0          # mM: tissue depolarizes when local K passes this (mammals: K ceiling
                         #   ~10-12 mM just before anoxic depolarization, Hansen 1985)
SD_OFF   = 6.0           # mM: tissue recovers when pumps bring K back below this
SD_RATE  = 55.0          # mM/s of K released by depolarized tissue at the start (guess, to calibrate)
SD_TAU_S = 20.0          # s: release fades with this time constant (guess, to calibrate)
SD_PUMP  = 0.2           # K clearance of tissue while it is in SD, relative to normal (guess)
SD_DELAY_S = 0.0         # s: K must stay above SD_ON this long before the cube goes into SD (v4)
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
STOP_EARLY = True        # end the run once nothing changes any more (v4)
STOP_QUIET_MS = 4000.0   # ... i.e. no cube has started SD for this long (and the brain is dead,
                         #     or in trigger/bath mode: SD started and then stopped spreading)
TISSUE_ONLY = False      # (used by stage6) cube layer only, see simulate_tissue
HEALTHY_RATE_HZ = 13.0   # steady spike rate assumed by simulate_tissue (gives the full model's ~3.43 mM)
TISSUE_E_FACTOR = 1.0    # simulate_tissue oxygen loss: energy falls at TISSUE_E_FACTOR / E_BASE_USE_S per s
                         #   x vulnerability (spike and pump costs folded in; fitted to full runs)
TISSUE_ANOX_GAIN = 0.0   # simulate_tissue oxygen loss: spike K release x (1 + gain x pump failure) (fitted)

# ---- K exchange at the edges of the neuropil (v5) ----
EDGE_G_S  = 0.0          # 1/s per face looking OUT of the brain (0 = sealed, as v4)
K_OUT     = 3.0          # mM outside the brain surface (K0 = brain baseline)
INNER_G_S = 0.0          # 1/s per face looking into a gap INSIDE the brain (0 = sealed)
EDGE_EXPORT = 0.0        # v6: active K export by the blood-brain barrier, mM/s per outward face at full
                         #   energy (locust ganglion: 91 pmol/cm2/s ~ 0.5 mM/s per 10 um face if 20% is
                         #   extracellular space; halved during SD; Robertson et al. 2025). 0 = none
K_INNER   = 3.0          # mM in those gaps

# ---- scenario ----
VULN_MODE    = "uniform"
SHUFFLE_CONN = False     # shuffles the WIRING (who spikes onto whom), as in v2
SHUFFLE_MODE = "sign"    # "sign": shuffle WHO sends each connection among neurons of the same sign (v2-v5;
                         #   keeps every neuron's inputs, changes how many outputs each neuron has)
                         # "degree": shuffle WHO receives each connection (v6); every neuron keeps its number
                         #   of outputs AND inputs, and its sign and output weights - only the partners change
SHUFFLE_SEED = 12345     # v6: different seeds give different rewirings (v2-v5 always used 12345)


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
    sign_map = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "DA": MOD_SIGN, "SER": MOD_SIGN, "OCT": MOD_SIGN}
    nt = meta["nt_type"].fillna("UNK").values
    neuron_sign = np.array([sign_map.get(t, 0.0) for t in nt], dtype=np.float32)
    pre = conn["pre_root_id"].map(idx).values
    post = conn["post_root_id"].map(idx).values
    syn = conn["syn_count"].values.astype(np.float32)
    if SHUFFLE_CONN and SHUFFLE_MODE == "sign":
        g = np.random.default_rng(SHUFFLE_SEED)
        pre = pre.copy()
        s = neuron_sign[pre]
        for val in np.unique(s):
            k = np.flatnonzero(s == val)
            pre[k] = pre[k][g.permutation(k.size)]
    elif SHUFFLE_CONN and SHUFFLE_MODE in ("degree", "break_loops", "keep_loops", "break_random", "break_recip"):
        # configuration-model rewiring: permute the receivers of a set of connections. Each neuron
        # keeps its outputs (number, weights, sign) and its number of inputs; partners change.
        #   degree       : all connections (v6)
        #   break_loops  : only the connections that form excitatory<->inhibitory reciprocal pairs
        #                  (a -> b and b -> a, one excitatory, one inhibitory); everything else real
        #   keep_loops   : every connection EXCEPT those loop pairs; the loops stay real
        #   break_random : as many randomly chosen non-loop connections as break_loops rewires
        #                  (control: is it the loops, or rewiring ~10% of anything?)
        g = np.random.default_rng(SHUFFLE_SEED)
        post = post.copy()
        key = pre.astype(np.int64) * N + post
        rev = post.astype(np.int64) * N + pre
        recip = np.isin(rev, key)
        s_pre, s_post = neuron_sign[pre], neuron_sign[post]
        loop = recip & (((s_pre > 0) & (s_post < 0)) | ((s_pre < 0) & (s_post > 0)))
        if SHUFFLE_MODE == "degree":
            sel = np.arange(post.size)
        elif SHUFFLE_MODE == "break_loops":
            sel = np.flatnonzero(loop)
        elif SHUFFLE_MODE == "keep_loops":
            sel = np.flatnonzero(~loop)
        elif SHUFFLE_MODE == "break_random":
            sel = g.choice(np.flatnonzero(~loop), size=int(loop.sum()), replace=False)
        else:                                                    # break_recip: E->I and I->E sets separately
            sel = np.flatnonzero(loop)
        if SHUFFLE_MODE == "break_recip":
            for part in (np.flatnonzero(loop & (s_pre > 0)), np.flatnonzero(loop & (s_pre < 0))):
                post[part] = post[part][g.permutation(part.size)]
        else:
            post[sel] = post[sel][g.permutation(sel.size)]
        for _ in range(10):                                  # remove self-connections by swapping receivers
            selfc = np.flatnonzero(post == pre) if SHUFFLE_MODE == "degree" else sel[post[sel] == pre[sel]]
            if not selfc.size:
                break
            other = (g.integers(0, post.size, selfc.size) if SHUFFLE_MODE == "degree"   # same draws as stage 8
                     else g.choice(sel, selfc.size))
            post[selfc], post[other] = post[other].copy(), post[selfc].copy()
        print(f"  rewiring '{SHUFFLE_MODE}': {sel.size:,} of {post.size:,} connections rewired "
              f"({loop.sum():,} are E-I loop connections)")
    elif SHUFFLE_CONN:
        raise ValueError(f"unknown SHUFFLE_MODE {SHUFFLE_MODE}")
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
    key = (CUBE_FILE, MAX_CUBES, CUBE_COVER, COUPLING, SWAP_SEED, BOX_HALF_NM, len(meta), SURF_CLOSE_VOX)
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

    # brain surface (v4): close small gaps, fill holes, then count each cube's faces that
    # look out of the brain (bath K enters there)
    from scipy import ndimage
    r = int(SURF_CLOSE_VOX)
    occ = np.zeros(tuple(dims + 2 * (r + 1)), bool)
    occ[ix + r + 1, iy + r + 1, iz + r + 1] = True
    body = ndimage.binary_dilation(occ, iterations=r) if r > 0 else occ
    body = ndimage.binary_fill_holes(body)
    if r > 0:
        body = ndimage.binary_erosion(body, iterations=r, border_value=0) | occ
    open_faces = np.zeros(C, np.int64)
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        open_faces += ~body[ix + r + 1 + dx, iy + r + 1 + dy, iz + r + 1 + dz]
    surf_idx = np.flatnonzero(open_faces > 0).astype(np.int64)
    # internal gaps: neighbour positions inside the brain body that hold no neuropil
    in_faces = np.zeros(C, np.int64)
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        jx, jy, jz = ix + r + 1 + dx, iy + r + 1 + dy, iz + r + 1 + dz
        in_faces += body[jx, jy, jz] & ~occ[jx, jy, jz]
    in_idx = np.flatnonzero(in_faces > 0).astype(np.int64)
    o_sum = np.bincount(O_idx, weights=O_w, minlength=C)        # output weight per cube (for tissue-only)
    # v9: dominant neuropil of each cube (main neuropil of the neurons present, weighted), as validate_v3
    codes, names = pd.factorize(meta["neuropil"].astype(str).values)
    rows_p = np.repeat(np.arange(N), np.diff(P_ptr))
    cnp = (pd.DataFrame({"c": P_idx, "n": codes[rows_p], "w": P_w}).groupby(["c", "n"])["w"].sum().reset_index()
           .sort_values("w").drop_duplicates("c", keep="last").set_index("c")["n"])
    cube_region = np.array(["none"] * C, dtype=object)
    cube_region[cnp.index.values] = np.asarray(names)[cnp.values]
    geom = dict(P_ptr=P_ptr, P_idx=P_idx, P_w=P_w, O_ptr=O_ptr, O_idx=O_idx, O_w=O_w,
                m_rel=m_rel, m_ref=m_ref, nb=nb, centers_um=centers_um, cube_um=cube_um,
                cube_ids=dom, dims=dims, surf_idx=surf_idx, surf_open=open_faces[surf_idx],
                o_sum=o_sum, in_idx=in_idx, in_open=in_faces[in_idx], cube_region=cube_region)
    print(f"  internal gaps: {in_idx.size:,} cubes face a gap inside the brain ({in_idx.size / C:.1%})")
    print(f"  surface: {surf_idx.size:,} cubes face the outside of the brain "
          f"({surf_idx.size / C:.1%}; gaps < {2 * r * cube_um:.0f} um closed)")
    print(f"  geometry: {C:,} cubes of {cube_um:.0f} um, typical density {m_ref:.2f} neuron-equivalents per cube, "
          f"{np.diff(P_ptr).mean():.1f} cubes per neuron, coupling = {COUPLING}")
    _GEOM_CACHE[key] = geom
    return geom


def build_spatial_graph(coords):
    raise RuntimeError("v3 uses build_geometry(meta, idx, coords); update runner.py")


# %% Simulation
def _experiment(geom, slow_every):
    """Trigger / bath settings in units of (fast) steps; slow_every converts for run_tissue."""
    trig_step, trig_end, trig_cubes, bath_step, ramp = -1, -1, np.zeros(0, np.int64), -1, 0
    surf_idx, surf_g, bath_dK = np.zeros(0, np.int64), np.zeros(0), BATH_DK
    if RUN_MODE == "trigger":
        c = geom["centers_um"]
        at = np.median(c, axis=0) if isinstance(TRIGGER_AT, str) else np.asarray(TRIGGER_AT, float)
        near = np.flatnonzero(np.linalg.norm(c - at, axis=1) <= TRIGGER_RADIUS_UM)
        if near.size == 0:
            near = np.array([np.argmin(np.linalg.norm(c - at, axis=1))])
        trig_step, trig_cubes = int(TRIGGER_T_MS / DT) // slow_every, near.astype(np.int64)
        trig_end = int((TRIGGER_T_MS + TRIGGER_HOLD_MS) / DT) // slow_every
        print(f"  trigger: {near.size} cubes around {np.round(at)} um held at K = {TRIGGER_K} "
              f"from {TRIGGER_T_MS / 1000:.1f} s for {TRIGGER_HOLD_MS / 1000:.1f} s")
    elif RUN_MODE == "bath":
        bath_step = int(BATH_T_MS / DT) // slow_every
        ramp = int(BATH_RAMP_S * 1000 / DT) // slow_every
        if BATH_MODE == "surface":
            surf_idx = geom["surf_idx"]
            surf_g = BATH_G_S * geom["surf_open"].astype(np.float64)
            bath_dK = BATH_K - K0
            print(f"  bath: {BATH_K} mM at the surface ({surf_idx.size:,} cubes, {BATH_G_S}/s per face) "
                  f"from {BATH_T_MS / 1000:.1f} s, ramp {BATH_RAMP_S:.0f} s")
        elif BATH_MODE == "everywhere":
            print(f"  bath: K background +{BATH_DK} everywhere from {BATH_T_MS / 1000:.1f} s")
        else:
            raise ValueError(f"unknown BATH_MODE {BATH_MODE}")
    edge = (geom["surf_idx"], EDGE_G_S * geom["surf_open"].astype(np.float64), float(K_OUT),
            geom["in_idx"], INNER_G_S * geom["in_open"].astype(np.float64), float(K_INNER))
    if EDGE_G_S <= 0:
        edge = (np.zeros(0, np.int64), np.zeros(0)) + edge[2:]
    if INNER_G_S <= 0:
        edge = edge[:3] + (np.zeros(0, np.int64), np.zeros(0), edge[5])
    if EDGE_EXPORT > 0:
        edge = edge + (geom["surf_idx"], EDGE_EXPORT * geom["surf_open"].astype(np.float64))
    else:
        edge = edge + (np.zeros(0, np.int64), np.zeros(0))
    return trig_step, trig_end, trig_cubes, bath_step, bath_dK, ramp, surf_idx, surf_g, edge


def _stop_mode():
    if not STOP_EARLY or RUN_MODE in ("healthy", "ouabain") or T_REOX_MS is not None:
        return 0
    return 1 if RUN_MODE == "death" else 2


def glia_density(geom):
    C = len(geom["m_rel"])
    if not GLIA_ON:
        return np.zeros(C)
    d = np.ones(C)
    if GLIA_DENSITY == "lamina":
        d[np.isin(geom["cube_region"], ["LA_L", "LA_R"])] = GLIA_LAMINA_FACTOR
    elif GLIA_DENSITY != "uniform":
        raise ValueError(f"unknown GLIA_DENSITY {GLIA_DENSITY}")
    return d


def glia_args(geom):
    """Arguments for the compiled loops: density, uptake rate, capacity, release rate, passive share, neuronal share (1)."""
    dens = glia_density(geom)
    if not GLIA_ON:
        return dens, 0.0, 1.0, 0.0, 0.0, 1.0
    return dens, float(GLIA_UPTAKE_S), float(GLIA_CAP), float(GLIA_RELEASE_S), float(GLIA_PASSIVE), 1.0


def probe_cubes(geom):
    c = geom["centers_um"]
    centre = int(np.argmin(np.linalg.norm(c - np.median(c, axis=0), axis=1)))
    g = np.random.default_rng(PROBE_SEED)
    rest = g.choice(np.delete(np.arange(len(c)), centre), size=min(PROBE_N, len(c) - 1), replace=False)
    return np.concatenate([[centre], rest]).astype(np.int64)


def simulate(W, N, vuln, geom, t_anox_ms, duration_ms, record_every=10):
    import fast_sim_v10 as fast_sim_v6
    n_steps = int(duration_ms / DT)
    tonic = rng.normal(TONIC_MEAN, 0.5, size=N)
    seed = int(rng.integers(1, 2**62))
    noise_pool = rng.standard_normal(2**22).astype(np.float32)
    W_csc = W.tocsc()
    f64 = lambda x: np.ascontiguousarray(x, dtype=np.float64)
    trig_step, trig_end, trig_cubes, bath_step, bath_dK, ramp, surf_idx, surf_g, edge = _experiment(geom, 1)
    out = fast_sim_v6.run(
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
        trig_step, trig_end, trig_cubes, TRIGGER_K,
        bath_step, bath_dK, ramp, surf_idx, f64(surf_g), *edge,
        SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP, SD_DELAY_S, V_UNBLOCK,
        _stop_mode(), STOP_QUIET_MS,
        int(T_REOX_MS / DT) if (T_REOX_MS is not None and RUN_MODE == "death") else -1,
        DT * SLOW_EVERY / 1000.0 / E_RECOVER_TAU_S,
        int(OUABAIN_T_MS / DT) if RUN_MODE == "ouabain" else -1, float(OUABAIN_BLOCK),
        int(OUABAIN_RAMP_S * 1000 / DT), probe_cubes(geom), int(PROBE_EVERY),
        float(SD_KMAX), bool(SD_PUMP_RECOVER),
        *glia_args(geom), 1 if K_COUPLING == "goldman" else 0, float(GHK_ALPHA), float(GHK_SLOPE),
        1 if NA_INACT else 0, float(NA_VHALF), float(NA_KI), float(NA_KA), 1 if RESET_MODE == "fixed" else 0,
        float(V_RESET))
    keys = ["t", "active", "blocked", "meanK", "meanE", "chi", "block_time", "E_at_block",
            "K_max", "K_end", "cube_time", "cube_K_end", "spikes_pre", "spikes_post", "last_step",
            "probe_t", "probe_K", "probe_frac_sd", "sd_count", "probe_glia", "glia_end"]
    res = dict(zip(keys, out))
    if SAVE_PROBES_TO:
        pc = probe_cubes(geom)
        surf = geom["centers_um"][geom["surf_idx"]]
        depth = cKDTree(surf).query(geom["centers_um"][pc])[0] if len(surf) else np.full(pc.size, np.nan)
        np.savez_compressed(SAVE_PROBES_TO, t=res["probe_t"], K=res["probe_K"].astype(np.float32),
                            frac_sd=res["probe_frac_sd"], cube=pc, depth_um=depth,
                            sd_count=res["sd_count"].astype(np.int16), series_t=res["t"],
                            glia_mean=res["probe_glia"].astype(np.float32),
                            active=res["active"].astype(np.float32), blocked=res["blocked"].astype(np.float32),
                            meanK=res["meanK"].astype(np.float32), meanE=res["meanE"].astype(np.float32))
    # v6: firing rate of each neuron before oxygen loss and from oxygen loss to the end of the run (Hz)
    t_end = res.pop("last_step") * DT
    t_cut = min(t_anox_ms, t_end)
    res["rate_pre_hz"] = res.pop("spikes_pre") / max(t_cut / 1000.0, 1e-9)
    res["rate_post_hz"] = res.pop("spikes_post") / max((t_end - t_cut) / 1000.0, 1e-9)
    return res


def simulate_tissue(geom, duration_ms, record_every=10, vuln=None, t_anox_ms=None):
    """Cube layer only (no neurons): all pumps working, spikes replaced by a steady release.
    Only for trigger / bath calibration. Times in the result are in ms like simulate()."""
    import fast_sim_v10 as fast_sim_v6
    slow_ms = DT * SLOW_EVERY
    act_rel = HEALTHY_RATE_HZ * K_SPIKE_RELEASE * geom["o_sum"] / geom["m_ref"]
    trig_step, trig_end, trig_cubes, bath_step, bath_dK, ramp, surf_idx, surf_g, edge = _experiment(geom, SLOW_EVERY)
    out = fast_sim_v6.run_tissue(
        int(duration_ms / slow_ms), slow_ms, geom["m_rel"], geom["nb"],
        np.ascontiguousarray(act_rel, dtype=np.float64), record_every,
        K0, K_PEAK, K_LEAK_S, K_REMOVE_S, D_UM2_S, geom["cube_um"],
        trig_step, trig_end, trig_cubes, TRIGGER_K,
        bath_step, bath_dK, ramp, surf_idx, np.ascontiguousarray(surf_g, dtype=np.float64), *edge,
        geom["P_ptr"], geom["P_idx"], geom["P_w"],
        np.ascontiguousarray(vuln if vuln is not None else np.ones(1), dtype=np.float64),
        int(t_anox_ms / slow_ms) if (RUN_MODE == "death" and t_anox_ms is not None) else -1,
        TISSUE_E_FACTOR / E_BASE_USE_S, E_FULL, E_FAIL, 10, TISSUE_ANOX_GAIN,
        SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP, SD_DELAY_S,
        _stop_mode(), STOP_QUIET_MS, float(SD_KMAX), bool(SD_PUMP_RECOVER),
        *glia_args(geom))
    keys = ["t", "frac_sd", "frac_reached", "meanK", "cube_time", "cube_K_end"]
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
        label = f"v10_{RUN_MODE}_{COUPLING}_shuffle{SHUFFLE_CONN}_seed{SEED}"
        np.save(f"block_time_{label}.npy", res["block_time"])
        np.save(f"cube_time_{label}.npy", res["cube_time"])
        print("saved", label)
