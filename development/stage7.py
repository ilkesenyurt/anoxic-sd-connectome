# stage7.py - v5: how much do the neuropil edges matter? (runs after stage6)
# =========================================================================
# In v4 the neuropil edges are sealed: K at an edge cannot leave, so edge cubes reach
# threshold first (the left lamina, log row 64). Real edges leak, but how fast is not
# known for flies (cell-body rind + glial sheath / blood-brain barrier). Instead of
# fitting the edge rate to the fly onset data (which would use up that test), this
# sweeps it and asks whether the answers change:
#   oxygen loss, real wiring, with the outer edge exchange at 0.5, 2 and 7.7 /s per face
#   (7.7 = as free as between two neuropil cubes), plus swapped / rewired at 2 /s
# Compared with the sealed v4 runs from stage6 (same seed). Validation runs afterwards.

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import runner

SCALE = float(os.environ.get("STAGE7_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE7_WORKERS", "5"))
MODEL = "death_v5"


def log(*a):
    print(*a, flush=True)


def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    return runner.run_condition(cond)


def cond(name, seed, dur_s, **overrides):
    return dict(name=name, seed=seed, overrides=overrides, dur=dur_s * 1000.0 * SCALE)


if __name__ == "__main__":
    log("stage7 started", time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join("runs", "v4_chosen.json")) as f:
        p = dict(json.load(f)["chosen"], SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2)
    log("model settings:", p)
    conds = [
        cond("R_edge0.5", 0, 120, RUN_MODE="death", COUPLING="real", EDGE_G_S=0.5, **p),
        cond("R_edge2",   0, 120, RUN_MODE="death", COUPLING="real", EDGE_G_S=2.0, **p),
        cond("R_edge7.7", 0, 120, RUN_MODE="death", COUPLING="real", EDGE_G_S=7.7, **p),
        cond("S_edge2",   0, 120, RUN_MODE="death", COUPLING="swap", EDGE_G_S=2.0, **p),
        cond("W_edge2",   0, 120, RUN_MODE="death", COUPLING="real", SHUFFLE_CONN=True, EDGE_G_S=2.0, **p),
    ]
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v5edge_stage2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"{len(conds)} runs, {N_WORKERS} at a time -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, stagger=60 * (i % N_WORKERS) * min(SCALE * 10, 1)))
                for i, c in enumerate(conds)]
        for fut in as_completed(futs):
            rows.append(fut.result())
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    cols = [c for c in ["tag", "status", "t50_s", "speed_cube_mm_min", "cubes_reached", "blocked_end",
                        "K_end_mean", "minutes"] if c in summary]
    log("\n" + summary[cols].to_string(index=False))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")

    # compare with the sealed v4 runs (stage6) and between conditions at the same edge setting
    ref = sorted(glob.glob(os.path.join(runner.OUT_DIR, "*_v4_stage2")))
    tags = {r["tag"]: out for r in rows if r.get("status") == "ok"}
    if ref:
        for t in ["R_real_seed0", "R_real_seed1", "S_swap_seed0", "W_rewired_seed0"]:
            if os.path.exists(os.path.join(ref[-1], f"cube_time_{t}.npy")):
                tags["sealed_" + t] = ref[-1]
    ct = {t: np.load(os.path.join(d, f"cube_time_{t.replace('sealed_', '')}.npy")) for t, d in tags.items()}
    bt = {t: np.load(os.path.join(d, f"block_time_{t.replace('sealed_', '')}.npy")) for t, d in tags.items()}
    pairs = [("sealed_R_real_seed0", "sealed_R_real_seed1"),
             ("sealed_R_real_seed0", "R_edge0.5_seed0"), ("sealed_R_real_seed0", "R_edge2_seed0"),
             ("sealed_R_real_seed0", "R_edge7.7_seed0"),
             ("sealed_R_real_seed0", "sealed_S_swap_seed0"), ("sealed_R_real_seed0", "sealed_W_rewired_seed0"),
             ("R_edge2_seed0", "S_edge2_seed0"), ("R_edge2_seed0", "W_edge2_seed0")]
    res = []
    for a, b in pairs:
        if a in ct and b in ct:
            ok = np.isfinite(ct[a]) & np.isfinite(ct[b])
            okn = np.isfinite(bt[a]) & np.isfinite(bt[b])
            res.append(dict(a=a, b=b, rho_cubes=spearmanr(ct[a][ok], ct[b][ok])[0] if ok.sum() > 10 else np.nan,
                            rho_neurons=spearmanr(bt[a][okn], bt[b][okn])[0] if okn.sum() > 10 else np.nan))
    if res:
        res = pd.DataFrame(res)
        log("\nhow similar is the death order (Spearman):\n" + res.round(4).to_string(index=False))
        try:
            res.to_csv(os.path.join(out, "edge_comparison.csv"), index=False)
        except PermissionError:
            pass
    log(f"\nstage7 finished {time.strftime('%Y-%m-%d %H:%M')}")
