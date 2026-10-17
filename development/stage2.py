# stage2.py - overnight follow-up to the v3.2 trigger sweep (runs by itself after runner.py)
# =========================================================================================
# 1. Reads the newest trigger-sweep summary (runs/*/summary.csv with "trig_pump" runs).
# 2. If some setting made SD spread through > 30% of the neuropil cubes, it picks the one whose
#    wave speed is closest to the fly value (0.77 mm/min, Harris et al. 2026) and runs, with it:
#      - the KCl bath test (Harris: SD starts within 14 +- 4.4 s)          40 s
#      - oxygen loss with real synapse locations, seeds 0 and 1           60 s each
#      - oxygen loss with swapped locations (S_swap) and rewired wiring    60 s each
# 3. If no setting spread, it first sweeps the SD threshold (SD_ON 8-11), then continues with 2.
# Everything is printed to stage2_log.txt; results go to runs/<date_time>_stage2/.

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import runner

SCALE = float(os.environ.get("STAGE2_SCALE", "1"))   # <1 only for quick tests
N_WORKERS = 2


def run_one(cond):
    """Run one condition with its own duration (runner uses a global for that)."""
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False            # healthy state of v3 already checked (log row 48)
    return runner.run_condition(cond)


def run_batch(conds, tag):
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_" + tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    print(f"{len(conds)} runs -> {out}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out)) for c in conds]
        for fut in as_completed(futs):
            rows.append(fut.result())
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    cols = [c for c in ["tag", "status", "mode", "t50_s", "speed_cube_mm_min", "cubes_reached",
                        "blocked_max", "blocked_end", "E_at_block_median", "K_end_mean",
                        "surge_peak", "minutes"] if c in summary]
    print("\n" + summary[cols].to_string(index=False), flush=True)
    corr = runner.correlations(rows, out)
    if len(corr):
        try:
            corr.to_csv(os.path.join(out, "correlations.csv"), index=False)
        except PermissionError:
            pass
        print("\n" + corr.to_string(index=False), flush=True)
    for r in rows:
        if r.get("status") == "ERROR":
            print(f"\nERROR in {r['tag']}:\n{r['error']}", flush=True)
    return summary


def cond(name, seed, dur_s, **overrides):
    return dict(name=name, seed=seed, overrides=overrides, dur=dur_s * 1000.0 * SCALE)


def pick(summary):
    """Setting whose SD spread through > 30% of cubes, with speed closest to 0.77 mm/min."""
    good = summary[(summary["status"] == "ok") & (summary["cubes_reached"] > 0.30)]
    if not len(good):
        return None
    best = good.iloc[(good["speed_cube_mm_min"] - 0.77).abs().argsort().values[0]]
    print(f"\nchosen: {best['tag']}; speed {best['speed_cube_mm_min']:.2f} mm/min, "
          f"cubes reached {best['cubes_reached']:.1%}", flush=True)
    return {k[2:]: float(best[k]) for k in ["p_SD_PUMP", "p_SD_RATE", "p_SD_ON"] if k in best}


if __name__ == "__main__":
    print("stage2 started", time.strftime("%Y-%m-%d %H:%M"), flush=True)
    sweeps = [p for p in sorted(glob.glob(os.path.join(runner.OUT_DIR, "*", "summary.csv")))
              if "trig_pump" in open(p).read()]
    if not sweeps:
        raise SystemExit("no trigger sweep found")
    s = pd.read_csv(sweeps[-1])
    print("sweep:", sweeps[-1], flush=True)
    print(s[[c for c in ["tag", "cubes_reached", "blocked_max", "blocked_end",
                         "speed_cube_mm_min"] if c in s]].to_string(index=False), flush=True)
    p = pick(s)
    if p is None:
        # The wave fails at the threshold, not the clearance: a cube next to a depolarized cube
        # (K ~55) settles near ~11.5 mM when its other neighbours are healthy, so SD_ON = 12 can
        # never be crossed along a thin front (log row 55). Sweep the threshold instead.
        print("\nno spread; sweeping the SD threshold (SD_ON)", flush=True)
        conds = [cond(f"trig_on{on:.0f}_pump0.2_rate20", 0, 40, RUN_MODE="trigger",
                      SD_ON=on, SD_PUMP=0.2, SD_RATE=20.0) for on in [8.0, 9.0, 10.0, 11.0]]
        p = pick(run_batch(conds, "sweep_on"))
    if p is None:
        print("\nstill no spreading wave; stopping here for a rethink", flush=True)
    else:
        p.setdefault("SD_ON", 12.0)
        print(f"\nfinal experiments with {p}", flush=True)
        conds = [
            cond("bath",      0, 40, RUN_MODE="bath", **p),
            cond("R_real",    0, 60, RUN_MODE="death", COUPLING="real", **p),
            cond("S_swap",    0, 60, RUN_MODE="death", COUPLING="swap", **p),
            cond("W_rewired", 0, 60, RUN_MODE="death", COUPLING="real", SHUFFLE_CONN=True, **p),
            cond("R_real",    1, 60, RUN_MODE="death", COUPLING="real", **p),
        ]
        run_batch(conds, "stage2")
    print("stage2 finished", time.strftime("%Y-%m-%d %H:%M"), flush=True)
