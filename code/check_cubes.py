# check_cubes.py - sixth review (log row 111). No simulation, about a minute. Counts the 10-um cubes that contain at
# least one synapse in each dataset (all neurons in the cube file), as a direct measure of the space the neuropil fills.
# The model's cube count (female 23,045, male 30,701) instead follows the occupancy rule (each neuron's largest cubes,
# up to 95% of its synapses or 128 cubes). Male also after keeping a random 77% of synapses (female density).
import numpy as np

for sex, f in (("female", "synapse_cubes_10um.npz"), ("male", "synapse_cubes_male_10um.npz")):
    d = np.load(f)
    c = np.concatenate([d["pre_cube"], d["post_cube"]]).astype(np.int64)
    w = np.concatenate([d["pre_count"], d["post_count"]]).astype(np.int64)
    out = {}
    for frac in ((1.0, 364 / 472) if sex == "male" else (1.0,)):
        ww = np.random.default_rng(0).binomial(w, frac) if frac < 1 else w
        tot = np.bincount(c[ww > 0], weights=ww[ww > 0])
        occ = tot[tot > 0]
        out[f"keep_{frac:.2f}"] = {f">={k} synapse(s)": int((occ >= k).sum()) for k in (1, 5, 20, 100)}
    print(sex, "cube um", float(d["cube_um"]), "dims", d["dims"].tolist(), "synapse sides", int(w.sum()))
    for k, v in out.items():
        print("  ", k, v, flush=True)
