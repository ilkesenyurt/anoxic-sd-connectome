# validate_onset_order.py - compare the model's death wave with Harris et al. 2026
# ==============================================================================
# Uses block-time files you ALREADY have (no new simulation). Two tests:
#   1. How many separate starting points does the wave have?
#      Real flies: 3.0 +/- 0.7 per brain.
#   2. Which brain regions (neuropils) block first?
#      Real flies: usually starts near the mushroom body calyx (MB_CA), then
#      pedunculus (MB_PED) and alpha lobe (MB_VL), then jumps to the other side.
# Neither was used for calibration, so this is a real test.
# Caveat: Harris triggered the wave by cooling/KCl, we by cutting oxygen.
#
# Put this next to death_v2.py and the data, set BLOCK_FILES, run it.

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import death_v2 as m

BLOCK_FILES = [                          # edit: your block-time files
    "block_time_spatial_shuffleFalse.npy",
    "block_time_spatial_shuffleFalse2.npy",
]
ONSET_RADIUS_NM = 50_000   # a start point must block before everything within 50 um
EARLY_FRACTION = 0.10      # only count start points among the first 10% to block

# %% neuron order used by the model (block_time[i] belongs to root_ids[i])
meta, conn = m.load_data()
idx, W, sign, coords, N = m.build_network(meta, conn)
if isinstance(idx, dict):                       # {root_id: row}
    root_ids = np.empty(N, dtype=np.int64)
    for rid, i in idx.items():
        root_ids[i] = rid
else:                                           # already an array of root_ids
    root_ids = np.asarray(idx)

# %% main neuropil of each neuron = where most of its synapses are (in + out)
c = pd.read_csv("connections_princeton.csv",
                usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
both = pd.concat([
    c[["pre_root_id", "neuropil", "syn_count"]].rename(columns={"pre_root_id": "root_id"}),
    c[["post_root_id", "neuropil", "syn_count"]].rename(columns={"post_root_id": "root_id"}),
])
per = both.groupby(["root_id", "neuropil"])["syn_count"].sum().reset_index()
main = per.sort_values("syn_count").drop_duplicates("root_id", keep="last")
neuropil = pd.Series(root_ids).map(main.set_index("root_id")["neuropil"]).fillna("none").values

tree = cKDTree(coords)
rows = []
for f in BLOCK_FILES:
    bt = np.load(f)
    assert len(bt) == N, f"{f} has {len(bt)} neurons, model has {N}"
    ok = np.isfinite(bt)

    # ---- test 1: starting points ----
    # the imaging can't see single neurons, so first blur block times over the
    # 50 nearest neurons; otherwise random neuron-to-neuron jitter looks like
    # dozens of tiny "start points"
    bt_raw = bt
    _, nb50 = tree.query(coords, k=50)
    bt = np.nanmean(np.where(ok[nb50], bt_raw[nb50], np.nan), axis=1)
    bt[~ok] = np.nan
    # a neuron is a start point if it blocks before every neuron within the radius
    early = ok & (bt <= np.nanquantile(bt, EARLY_FRACTION))
    cand = np.flatnonzero(early)
    is_start = np.zeros(N, bool)
    for i in cand:
        nb = tree.query_ball_point(coords[i], ONSET_RADIUS_NM)
        is_start[i] = bt[i] <= np.nanmin(bt[nb])
    starts = np.flatnonzero(is_start)
    # merge start points that sit within the radius of each other
    groups = 0
    left = set(starts.tolist())
    while left:
        i = left.pop(); groups += 1
        for j in tree.query_ball_point(coords[i], ONSET_RADIUS_NM):
            left.discard(j)
    first_regions = pd.Series(neuropil[starts]).value_counts().head(5)

    # ---- test 2: region order ----
    df = pd.DataFrame({"neuropil": neuropil[ok], "bt": bt_raw[ok]})
    order = (df.groupby("neuropil")["bt"].median().sort_values() / 1000).round(2)
    rank = {r: i + 1 for i, r in enumerate(order.index)}
    mb = {r: rank.get(r) for r in order.index if r.startswith("MB_")}

    print(f"\n===== {f} =====")
    print(f"Start points (merged within {ONSET_RADIUS_NM/1000:.0f} um): {groups}   "
          f"[real flies: 3.0 +/- 0.7]")
    print("Regions of the start points:\n" + first_regions.to_string())
    print(f"\nFirst 10 of {len(order)} regions to block (median block time, s):")
    print(order.head(10).to_string())
    print(f"Mushroom body ranks (1 = first): {mb}")
    rows.append(order.rename(f))

# %% do the runs agree with each other on region order?
if len(rows) > 1:
    tab = pd.concat(rows, axis=1).dropna()
    print("\nSpearman correlation of region order between files:")
    print(tab.corr(method="spearman").round(3).to_string())
