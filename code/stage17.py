# stage17.py - extras on the FROZEN model (v11 final), run right after stage16 (log row 96). Small outputs only.
#   A. geometry export for figures and onset/region analysis (cube centres, region per cube, surface cubes),
#      female and male -> runs/geom_female.npz, runs/geom_male.npz (~1-2 MB each)
#   B. 5 runs in parallel (~2.5 h):
#      female_reox     : N2 2-62 s, air to 110 s - recovery, virtual electrodes and DC proxy for the figures
#      female_keep     : only the E-I reciprocal loop connections real, everything else rewired (row 84 test,
#                        repeated on the frozen model)
#      female_break    : only the loop connections rewired (receivers permuted across E->I and I->E together)
#      female_noLA_R / female_noLA_W : real and rewired brain WITHOUT lamina neurons (main neuropil LA_L / LA_R):
#                        does the lamina-first onset (open mismatch with Harris 2026) drive the real-vs-rewired result?
# Predictions fixed before the run: keep ~ real (t50 within 2 s of real seed 0 in stage16); break earlier than rewired;
#   noLA: real t50 > rewired t50 by >= 2 s (the main result does not depend on the lamina).
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import runner
import stage16

MODEL = "death_v11"
LAMINA = {"LA_L", "LA_R", "LA(L)", "LA(R)"}
T_ANOX = runner.T_ANOX_MS


def log(*a):
    print(*a, flush=True)


def export_geometry():
    import importlib
    m = importlib.import_module(MODEL)
    for sex, sp in stage16.SEX.items():
        for k, v in {**stage16.FROZEN, **sp}.items():
            setattr(m, k, v)
        if sex == "female":
            for k, v in dict(CONNECTIONS_CSV="connections_princeton.csv", NEURONS_CSV="neurons.csv",
                             COORDS_CSV="coordinates.csv", CLASS_CSV="classification.csv",
                             CUBE_FILE="synapse_cubes_10um.npz").items():
                setattr(m, k, v)
        meta, conn = m.load_data()
        idx, W, sign, coords, N = m.build_network(meta, conn)
        g = m.build_geometry(meta, idx, coords)
        np.savez_compressed(os.path.join("runs", f"geom_{sex}.npz"), centers_um=g["centers_um"].astype(np.float32),
                            cube_region=np.asarray(g["cube_region"]).astype(str), surf_idx=g["surf_idx"],
                            probe_cubes=m.probe_cubes(g))
        log(f"geometry {sex}: {len(g['centers_um']):,} cubes saved")
        m._GEOM_CACHE.clear()


def run_one(cond):
    import importlib
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    runner._DATA.clear()
    time.sleep(cond.pop("stagger", 0))
    m = importlib.import_module(MODEL)
    m._GEOM_CACHE.clear()
    orig = m.load_data
    if cond.pop("drop_lamina", False):
        def load_no_lamina():
            meta, conn = orig()
            keep = ~meta["neuropil"].isin(LAMINA)
            ids = set(meta.loc[keep, "root_id"])
            meta = meta[keep].reset_index(drop=True)
            conn = conn[conn.pre_root_id.isin(ids) & conn.post_root_id.isin(ids)].reset_index(drop=True)
            print(f"  lamina removed: {int((~keep).sum()):,} neurons dropped", flush=True)
            return meta, conn
        m.load_data = load_no_lamina
    cond["overrides"]["SAVE_PROBES_TO"] = os.path.join(cond["out"], f"probes_{cond['name']}_seed{cond['seed']}.npz")
    try:
        return runner.run_condition(cond)
    finally:
        m.load_data = orig
        m._GEOM_CACHE.clear()


def main():
    t0 = time.time()
    log("\n=== A. geometry export ===")
    export_geometry()
    p = dict(stage16.FROZEN, **stage16.SEX["female"])
    t_reox = T_ANOX + 60000.0
    conds = [dict(name="female_reox", seed=0, dur=t_reox + 48000.0,
                  overrides=dict(p, T_REOX_MS=t_reox, STOP_EARLY=False)),
             dict(name="female_keep", seed=0, dur=90000.0,
                  overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="keep_loops", SHUFFLE_SEED=1)),
             dict(name="female_break", seed=0, dur=90000.0,
                  overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="break_loops", SHUFFLE_SEED=1)),
             dict(name="female_noLA_R", seed=0, dur=90000.0, drop_lamina=True, overrides=dict(p)),
             dict(name="female_noLA_W", seed=0, dur=90000.0, drop_lamina=True,
                  overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1))]
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_extras")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"\n=== B. {len(conds)} runs -> {out} ===")
    rows = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, overrides=dict(c["overrides"]), stagger=45 * i))
                for i, c in enumerate(conds)]
        for fu in as_completed(futs):
            r = fu.result()
            rows.append(r)
            log(f"  done {r['tag']} ({r.get('status')})")
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    s = pd.DataFrame(rows)
    runner.save_summary(s, os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")
    cols = [c for c in ["tag", "status", "rate_pre_mean_hz", "t50_s"] if c in s]
    log(s[cols].round(2).to_string(index=False))
    log(f"\nstage17 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage17 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
