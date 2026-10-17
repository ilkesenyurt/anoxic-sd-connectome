# stage12.py - model v8 (physical SD release) + clean loop control (log row 81)
# ============================================================================
# v8 fix (death_v8.py): SD_KMAX 54 (release limited by driving force, stops by itself near the fly
# plateau instead of a timer + cap) and SD_PUMP_RECOVER (pump reduction in SD fades with the release).
#   A. (python stage12.py --tissue, ~10-20 min) speed check, tissue only, KCl trigger, D 770:
#      the v7 physics at delay 0.1 (should give 2.41 mm/min again) and the fix at delays 0-0.2 s.
#      The fix delay closest to the v7 speed is kept -> runs/v8_chosen.json. No day-long calibration.
#   B. (python stage12.py) 5 oxygen-loss runs with the fix:
#      base_R, base_Wdeg_a (same rewiring as stage 8), break_recip a/b (seeds 1, 2: only the reciprocity of
#      the E-I loops broken, every neuron keeps its degrees and which cell type it targets), reox_R (N2 2-62 s,
#      air to 110 s, virtual electrodes: recovery test of stage 11 with the fix).
#   C. comparison with the stage-8 (v6) runs and the stage-11 (v7) recovery.
# Small outputs only.

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import runner

SCALE = float(os.environ.get("STAGE12_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE12_WORKERS", "5"))
MODEL = "death_v8"
FIX = dict(SD_KMAX=54.0, SD_PUMP_RECOVER=True)
BASE = dict(D_UM2_S=770.0, SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2)
T_ANOX = runner.T_ANOX_MS
T_REOX = T_ANOX + 60000.0 * SCALE


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- A
def tissue_part():
    import importlib
    m = importlib.import_module(MODEL)
    defaults = {k: getattr(m, k) for k in dir(m) if k.isupper()}

    def setp(**kw):
        for k, v in defaults.items():
            setattr(m, k, v)
        for k, v in kw.items():
            assert hasattr(m, k), k
            setattr(m, k, v)

    meta, conn = m.load_data()
    idx, W, sign, coords, N = m.build_network(meta, conn)
    geom = m.build_geometry(meta, idx, coords)
    del W, conn
    rows = []
    log("\n=== A. speed check (tissue only, KCl trigger, D 770) ===")
    runs = [("v7 physics", 0.1, {})] + [("fix", d, FIX) for d in [0.0, 0.05, 0.1, 0.15, 0.2]]
    # extra check (not used for the choice): does the wave still spread WITHOUT the pump cut in SD tissue?
    # Nielsen et al. 2025: 40 mM K did not reduce Na/K-ATPase activity, so SD_PUMP 0.2 lacks a K-based reason
    runs += [("fix noCut", d, dict(FIX, SD_PUMP=1.0)) for d in [0.0, 0.05, 0.1]]
    for label, delay, extra in runs:
        setp(**{**dict(RUN_MODE="trigger", SD_DELAY_S=delay, TRIGGER_HOLD_MS=3000.0), **BASE, **extra})
        r = m.simulate_tissue(geom, 60000.0 * SCALE)
        ct = r["cube_time"]
        sp = m.cube_speed_mm_min(ct, geom) if np.isfinite(ct).sum() > 10 else np.nan
        rows.append(dict(model=label, delay=delay, reached=float(np.isfinite(ct).mean()), speed=sp,
                         K_end_median=float(np.median(r["cube_K_end"]))))
        log(f"{label:11s} delay {delay:4.2f} s: reached {rows[-1]['reached']:6.1%}  speed {sp:5.2f} mm/min"
            f"  K at end (median) {rows[-1]['K_end_median']:.1f} mM")
    sw = pd.DataFrame(rows)
    ref = float(sw.speed.iloc[0])
    g = sw[(sw.model == "fix") & (sw.reached > 0.9)]
    if not len(g):
        raise SystemExit("the fix gives no spreading wave; stopping for a rethink")
    best = g.iloc[(g.speed - ref).abs().argsort().values[0]]
    chosen = dict(SD_DELAY_S=float(best.delay), speed=float(best.speed), v7_speed=ref, **FIX)
    log(f"chosen: delay {best.delay:.2f} s -> {best.speed:.2f} mm/min (v7 physics {ref:.2f}; fly room temp 2.1-3.0)")
    with open(os.path.join("runs", "v8_chosen.json"), "w") as f:
        json.dump(chosen, f, indent=1)
    sw.to_csv(os.path.join("runs", "v8_speed_check.csv"), index=False)


# ----------------------------------------------------------------------------- B
def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    cond["overrides"]["SAVE_PROBES_TO"] = os.path.join(cond["out"], f"probes_{cond['name']}_seed0.npz")
    return runner.run_condition(cond)


def full_part():
    t0 = time.time()
    with open(os.path.join("runs", "v8_chosen.json")) as f:
        ch = json.load(f)
    p = dict(BASE, SD_DELAY_S=ch["SD_DELAY_S"], **FIX)
    log(f"\nfix settings: {p}")
    dur = 90000.0 * SCALE
    sh = lambda mode, s: dict(SHUFFLE_CONN=True, SHUFFLE_MODE=mode, SHUFFLE_SEED=s)
    conds = [dict(name="reox_R", dur=T_REOX + 48000.0 * SCALE,
                  overrides=dict(p, RUN_MODE="death", T_REOX_MS=T_REOX, STOP_EARLY=False, E_RECOVER_TAU_S=2.0)),
             dict(name="base_R", dur=dur, overrides=dict(p, RUN_MODE="death")),
             dict(name="base_Wdeg_a", dur=dur, overrides=dict(p, RUN_MODE="death", **sh("degree", 1))),
             dict(name="break_recip_a", dur=dur, overrides=dict(p, RUN_MODE="death", **sh("break_recip", 1))),
             dict(name="break_recip_b", dur=dur, overrides=dict(p, RUN_MODE="death", **sh("break_recip", 2)))]
    for c in conds:
        c["seed"] = 0
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v8_fix")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"{len(conds)} runs -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, overrides=dict(c["overrides"]),
                                        stagger=45 * i * min(SCALE * 10, 1))) for i, c in enumerate(conds)]
        for fu in as_completed(futs):
            rows.append(fu.result())
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")

    # ------------------------------------------------------------------------- C
    v6 = sorted(glob.glob(os.path.join("runs", "*_v6_robust_stage2")))[-1]
    s8 = pd.read_csv(os.path.join(v6, "summary.csv")).set_index("tag")
    ct, info = {}, []
    for name, tag in [("v6 real", "base_R_seed0"), ("v6 rewired_a", "base_Wdeg_a_seed0")]:
        ct[name] = np.load(os.path.join(v6, f"cube_time_{tag}.npy"))
        info.append(dict(run=name, rate_pre_hz=s8.loc[tag, "rate_pre_mean_hz"], t50_s=s8.loc[tag, "t50_s"]))
    for _, r in summary.iterrows():
        if r.status != "ok" or r.tag.startswith("reox"):
            continue
        name = "v8 " + r.tag.replace("_seed0", "")
        ct[name] = np.load(os.path.join(out, f"cube_time_{r.tag}.npy"))
        info.append(dict(run=name, rate_pre_hz=r.rate_pre_mean_hz, t50_s=r.t50_s))

    def rho(a, b):
        if a not in ct or b not in ct:
            return np.nan
        x, y = ct[a], ct[b]
        k = np.isfinite(x) & np.isfinite(y)
        return spearmanr(x[k], y[k])[0] if k.sum() > 10 else np.nan

    tab = pd.DataFrame(info)
    tab["first_SD_s"] = [(np.nanmin(ct[n]) - T_ANOX) / 1000 for n in tab.run]
    for ref in ["v6 real", "v8 base_R", "v8 base_Wdeg_a"]:
        tab[f"order_vs_{ref.replace(' ', '_')}"] = [rho(ref, n) for n in tab.run]
    log("\n=== C1. does the fix change the oxygen-loss results? (firing Hz, timing s, death order 1 = identical) ===")
    log(tab.round(3).to_string(index=False))
    log("(v6: real vs rewired 0.628; loop test keep_loops vs real 0.96, break_loops vs real -0.4)")
    try:
        tab.to_csv(os.path.join(out, "fix_comparison.csv"), index=False)
    except PermissionError:
        pass

    log("\n=== C2. recovery after N2 off with the fix (fly: half recovery 11.8-16.1 s) ===")
    import stage11
    f = os.path.join(out, "probes_reox_R_seed0.npz")
    if os.path.exists(f):
        pr = np.load(f)
        rec = stage11.recovery(pr, T_REOX / 1000.0, T_ANOX / 1000.0)
        log("v8 fix : " + ", ".join(f"{k} {v:.3g}" for k, v in rec.items()))
        e = stage11.electrode(pr, T_ANOX / 1000.0, T_REOX / 1000.0)
        s = e[e.surge]
        if len(s):
            log("electrodes (median): " + ", ".join(f"{c} {s[c].median():.2f}" for c in
                                                   ["onset_s", "K_at_onset", "pre_rise_mM_s", "K_peak"]))
    old = sorted(glob.glob(os.path.join("runs", "*_v7_validation", "probes_reox_R_tau2_seed0.npz")))
    if old:
        rec = stage11.recovery(np.load(old[-1]), T_REOX / 1000.0, T_ANOX / 1000.0)
        log("v7 old : " + ", ".join(f"{k} {v:.3g}" for k, v in rec.items()))
    log(f"\nstage12 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage12 started", time.strftime("%Y-%m-%d %H:%M"), sys.argv[1:])
    if "--tissue" in sys.argv:
        tissue_part()
    else:
        full_part()
