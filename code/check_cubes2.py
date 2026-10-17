# check_cubes2.py - seventh review (log row 113). No simulation, about a minute. Splits the direct count of 10-um cubes
# holding synapses (check_cubes.py) into brain neuropil and non-brain (nerve-cord / unassigned) cubes: each occupied
# cube takes the region label of the nearest model cube (runs/geom_<sex>.npz); cubes > 30 um from any model cube are
# counted as 'far' (not occupied by any neuron's main cubes). Then the linear scale implied by the brain-only counts.
import numpy as np
from scipy.spatial import cKDTree

NONBRAIN = ("none", "<unspecified>", "ADMN", "CV", "DMetaN", "DProN", "MesoAN", "MesoLN", "MetaLN", "PDMN", "ProAN",
            "ProCN", "ProLN")
res = {}
for sex, f in (("female", "synapse_cubes_10um.npz"), ("male", "synapse_cubes_male_10um.npz")):
    d = np.load(f)
    g = np.load(f"runs/geom_{sex}.npz")
    dims = d["dims"].astype(np.int64); cu = float(d["cube_um"]); origin = d["origin_nm"] / 1000.0
    c = np.concatenate([d["pre_cube"], d["post_cube"]]).astype(np.int64)
    w = np.concatenate([d["pre_count"], d["post_count"]]).astype(np.int64)
    tot = np.bincount(c, weights=w)
    tree = cKDTree(g["centers_um"])
    nonbrain = np.array([str(r).startswith(NONBRAIN) for r in g["cube_region"]])
    out = {}
    for k in (1, 100):
        ids = np.flatnonzero(tot >= k)
        ix = ids % dims[0]; iy = (ids // dims[0]) % dims[1]; iz = ids // (dims[0] * dims[1])
        cen = origin + (np.stack([ix, iy, iz], 1) + 0.5) * cu
        dist, j = tree.query(cen)
        far = dist > 30.0
        nb = ~far & nonbrain[j]
        brain = ~far & ~nonbrain[j]
        out[k] = dict(all=int(ids.size), brain=int(brain.sum()), nonbrain=int(nb.sum()), far=int(far.sum()))
    res[sex] = out
    print(sex, out, flush=True)
for k in (1, 100):
    for key in ("all", "brain"):
        if not res["female"][k][key]:
            continue
        r = res["male"][k][key] / res["female"][k][key]
        print(f">= {k} synapse(s), {key}: male/female = {r:.3f} -> linear scale {r ** (1 / 3):.3f}")
