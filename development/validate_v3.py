# validate_v3.py - region order and onset sites of the v3 death wave (vs Harris et al. 2026)
# ========================================================================================
# For every death run in the newest runs/*_stage2 folder:
#   - onset sites: cubes whose SD arrival time is the earliest within 30 um, merged
#     within 50 um (real flies: 3.0 +- 0.7 per brain)
#   - region order: median neuron block time per neuropil; where the mushroom body ranks
#     (Harris: SD usually starts near the calyx, then pedunculus and lobes)
#   - inhibitory fraction of a region vs its block time (should be ~0 with uniform vulnerability)
# Prints everything (run via run_runner.bat into validate_v3_log.txt).

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import death_v3 as m

folder = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join("runs", "*_stage2")))[-1]
print("folder:", folder, flush=True)

meta, conn = m.load_data()
idx, W, sign, coords, N = m.build_network(meta, conn)
geom = m.build_geometry(meta, idx, coords)
npil = meta["neuropil"].astype(str).values
centers = geom["centers_um"]
tree = cKDTree(centers)

# dominant neuropil of each cube (weighted by how much of each neuron sits there)
codes, names = pd.factorize(npil)
rows = np.repeat(np.arange(N), np.diff(geom["P_ptr"]))
cube_np = (pd.DataFrame({"c": geom["P_idx"], "n": codes[rows], "w": geom["P_w"]})
             .groupby(["c", "n"])["w"].sum().reset_index()
             .sort_values("w").drop_duplicates("c", keep="last").set_index("c")["n"])
cube_region = np.array(["none"] * len(centers), dtype=object)
cube_region[cube_np.index.values] = names[cube_np.values]

for f in sorted(glob.glob(os.path.join(folder, "block_time_*.npy"))):
    tag = os.path.basename(f)[len("block_time_"):-4]
    if "bath" in tag:
        continue
    bt = np.load(f)
    ct_file = os.path.join(folder, f"cube_time_{tag}.npy")
    print(f"\n===== {tag} =====", flush=True)

    if os.path.exists(ct_file):
        ct = np.load(ct_file)
        ok = np.isfinite(ct)
        print(f"cubes reached by SD: {ok.mean():.1%}")
        if ok.sum() > 10:
            early = np.flatnonzero(ok & (ct <= np.nanquantile(ct, 0.10)))
            starts = []
            for i in early:
                nb = tree.query_ball_point(centers[i], 30.0)
                tn = ct[nb]
                if ct[i] <= np.nanmin(tn):
                    starts.append(i)
            groups, left = [], set(starts)
            while left:
                i = left.pop(); groups.append(i)
                for j in tree.query_ball_point(centers[i], 50.0):
                    left.discard(j)
            print(f"onset sites (merged within 50 um): {len(groups)}   [real flies: 3.0 +- 0.7]")
            first = sorted(groups, key=lambda i: ct[i])[:8]
            for i in first:
                print(f"   {ct[i] / 1000:6.2f} s  {cube_region[i]:10s}  at {np.round(centers[i]).astype(int)} um")
            reg = pd.DataFrame({"r": cube_region[ok], "t": ct[ok] / 1000}).groupby("r")["t"].agg(["median", "size"])
            reg = reg[reg["size"] >= 20].sort_values("median")
            mb = {r: i + 1 for i, r in enumerate(reg.index) if r.startswith("MB_")}
            print(f"region order by cube SD arrival: first 8 of {len(reg)}:")
            print(reg.head(8).round(2).to_string())
            print(f"mushroom body ranks (cubes): {mb}")

    okn = np.isfinite(bt)
    if okn.sum() > 10:
        df = pd.DataFrame({"r": npil[okn], "t": bt[okn] / 1000, "inh": (sign < 0)[okn]})
        g = df.groupby("r").agg(t=("t", "median"), inh=("inh", "mean"), n=("t", "size"))
        g = g[g.n >= 20].sort_values("t")
        mbn = {r: i + 1 for i, r in enumerate(g.index) if r.startswith("MB_")}
        print(f"neurons blocked: {okn.mean():.1%}; region order by neuron block time, first 8 of {len(g)}:")
        print(g.head(8).round(2).to_string())
        print(f"mushroom body ranks (neurons): {mbn}")
        print(f"inhibitory fraction vs block time: rho = {spearmanr(g.inh, g.t)[0]:+.2f}")
print("\nvalidate_v3 done", flush=True)
