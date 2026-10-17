# stage10.py - causal test: are the excitatory<->inhibitory feedback loops what protects the real brain?
# ====================================================================================================
# Row 76: the real wiring has ~70x more E<->I reciprocal loops than degree-preserving random wiring,
# which fires ~2x more and dies earlier, in a different, wiring-specific order (row 78).
# Here only part of the wiring is randomised (every neuron always keeps its numbers of inputs/outputs):
#   break_loops  (2 seeds): only the E<->I loop connections are rewired; the rest stays real
#   keep_loops   (2 seeds): everything EXCEPT the loop connections is rewired; the loops stay real
#   break_random (1 seed) : the same number of random non-loop connections rewired (control)
# Predictions if the loops are the cause: break_loops behaves like random wiring (more activity,
# earlier death); keep_loops behaves like the real brain; break_random stays close to the real brain.
# Oxygen loss with the stage-8 baseline settings; compared with the stage-8 real and rewired runs.

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import runner

SCALE = float(os.environ.get("STAGE10_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE10_WORKERS", "5"))
MODEL = "death_v6"


def log(*a):
    print(*a, flush=True)


def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    return runner.run_condition(cond)


if __name__ == "__main__":
    t0 = time.time()
    log("stage10 started", time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join("runs", "v6_chosen.json")) as f:
        ch = json.load(f)["770"]
    p = dict(RUN_MODE="death", D_UM2_S=770.0, SD_DELAY_S=ch["SD_DELAY_S"], SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2,
             SHUFFLE_CONN=True)
    dur = 90000.0 * SCALE
    conds = [dict(name=f"{mode}_{x}", seed=0, dur=dur, overrides=dict(p, SHUFFLE_MODE=mode, SHUFFLE_SEED=s))
             for mode, x, s in [("break_loops", "a", 1), ("break_loops", "b", 2),
                                ("keep_loops", "a", 1), ("keep_loops", "b", 2), ("break_random", "a", 1)]]
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v6_loops_stage2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"{len(conds)} oxygen-loss runs -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, stagger=45 * i * min(SCALE * 10, 1))) for i, c in enumerate(conds)]
        for fu in as_completed(futs):
            rows.append(fu.result())
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")

    v6 = sorted(glob.glob(os.path.join("runs", "*_v6_robust_stage2")))[-1]
    s8 = pd.read_csv(os.path.join(v6, "summary.csv")).set_index("tag")
    ref = {"real": "base_R_seed0", "rewired_all": "base_Wdeg_a_seed0"}
    ct, info = {}, []
    for name, tag in ref.items():
        ct[name] = np.load(os.path.join(v6, f"cube_time_{tag}.npy"))
        info.append(dict(run=name, rate_pre_hz=s8.loc[tag, "rate_pre_mean_hz"], rate_post_hz=s8.loc[tag, "rate_post_mean_hz"],
                         t50_s=s8.loc[tag, "t50_s"]))
    for _, r in summary.iterrows():
        if r.status != "ok":
            continue
        name = r.tag.replace("_seed0", "")
        ct[name] = np.load(os.path.join(out, f"cube_time_{r.tag}.npy"))
        info.append(dict(run=name, rate_pre_hz=r.rate_pre_mean_hz, rate_post_hz=r.rate_post_mean_hz, t50_s=r.t50_s))

    def rho(a, b):
        x, y = ct[a], ct[b]
        k = np.isfinite(x) & np.isfinite(y)
        return spearmanr(x[k], y[k])[0] if k.sum() > 10 else np.nan

    tab = pd.DataFrame(info)
    tab["first_SD_s"] = [(np.nanmin(ct[n]) - runner.T_ANOX_MS) / 1000 for n in tab.run]
    tab["order_vs_real"] = [rho("real", n) for n in tab.run]
    tab["order_vs_rewired_all"] = [rho("rewired_all", n) for n in tab.run]
    log("\n=== loop test: firing (Hz), timing (s), death-order similarity (1 = identical) ===")
    log(tab.round(3).to_string(index=False))
    log("(reference: real vs real seed 1 = 0.9999; real vs fully rewired = 0.628)")
    try:
        tab.to_csv(os.path.join(out, "loop_test.csv"), index=False)
    except PermissionError:
        pass
    log(f"\nstage10 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")
