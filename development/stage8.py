# stage8.py - v6: room-temperature recalibration + robustness of the wiring result
# ================================================================================
# Why (log rows 70-72): the old speed target (0.77 mm/min) was cold-induced SD at ~6-7 C.
# At room temperature fly SD travels at 2.1-3.0 mm/min (KCl-triggered, intact flies, two labs); anoxic
# depolarization follows N2 after 8-15 s (DC potential) or 20-50 s (K+ electrode, opened head); the K+
# surge starts at ~10 mM after a slow rise of ~0.2 mM/s; the insect blood-brain barrier actively
# exports K+ (locust ~0.5 mM/s per 10 um face), halved during SD (log rows 70-73).
#
# A. tissue only (minutes): SD onset delay giving ~2.3 mm/min at D = 770 and at D = 400
# B. full model, oxygen loss, 5 at a time, early stop (one round = 5 runs):
#    baseline: real wiring x2 seeds, swapped layout, 2 old-style rewirings ("sign"),
#              3 degree-preserving rewirings ("degree": every neuron keeps its numbers of
#              inputs and outputs - only the partners change)
#    sensitivity (real and one degree-rewired brain each): active K export at the brain surface,
#              hemolymph K leaking in balanced by that export, SD release 100, cell K leak 0.2 mM/s, diffusion
#              D = 400, energy used 2x faster
# C. comparison table + mechanism (firing rate per region vs death time per region)
# Run:  python stage8.py --tissue   then   python stage8.py --full   (stage8.bat does both)

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

SCALE = float(os.environ.get("STAGE8_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE8_WORKERS", "5"))
MODEL = "death_v6"
TARGET = 2.5            # mm/min, room temperature: 2.1-2.5 (Andersen et al. 2025), 3.0 +- 0.7 (Spong et al. 2016)
CHOSEN = os.path.join("runs", "v6_chosen.json")


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
    log("\n=== A. speed at room temperature: SD onset delay x diffusion (SD_ON 11, rate 55, trigger 3 s) ===")
    for D in [770.0, 400.0]:
        for delay in [0.0, 0.1, 0.2, 0.3, 0.4]:
            setp(RUN_MODE="trigger", D_UM2_S=D, SD_DELAY_S=delay, SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2,
                 TRIGGER_HOLD_MS=3000.0)
            r = m.simulate_tissue(geom, 60000.0 * SCALE)
            ct = r["cube_time"]
            sp = m.cube_speed_mm_min(ct, geom) if np.isfinite(ct).sum() > 10 else np.nan
            rows.append(dict(D=D, delay=delay, reached=float(np.isfinite(ct).mean()), speed=sp))
            log(f"D {D:4.0f}  delay {delay:3.1f} s: reached {rows[-1]['reached']:6.1%}  speed {sp:5.2f} mm/min")
    sw = pd.DataFrame(rows)
    chosen = {}
    for D in [770.0, 400.0]:
        g = sw[(sw.D == D) & (sw.reached > 0.9)].copy()
        if not len(g):
            raise SystemExit(f"no spreading wave at D {D}; stopping for a rethink")
        best = g.iloc[(g.speed - TARGET).abs().argsort().values[0]]
        chosen[str(int(D))] = dict(D_UM2_S=D, SD_DELAY_S=float(best.delay), speed=float(best.speed))
        log(f"D {D:.0f}: chosen delay {best.delay:.1f} s -> {best.speed:.2f} mm/min (target {TARGET})")
    setp()
    return chosen, sw


# ----------------------------------------------------------------------------- B
def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    return runner.run_condition(cond)


def cond(name, seed, **overrides):
    return dict(name=name, seed=seed, overrides=dict(RUN_MODE="death", **overrides), dur=90000.0 * SCALE)


def conditions(chosen):
    base = dict(D_UM2_S=770.0, SD_DELAY_S=chosen["770"]["SD_DELAY_S"], SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2)
    d400 = dict(base, D_UM2_S=400.0, SD_DELAY_S=chosen["400"]["SD_DELAY_S"])
    WD = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree")
    WS = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="sign")
    c = [
        # baseline: noise ceiling, layout control, two kinds of rewiring with several seeds
        cond("base_R", 0, **base),
        cond("base_R", 1, **base),
        cond("base_S", 0, COUPLING="swap", **base),
        cond("base_Wsign_a", 0, SHUFFLE_SEED=12345, **WS, **base),
        cond("base_Wsign_b", 0, SHUFFLE_SEED=2, **WS, **base),
        cond("base_Wdeg_a", 0, SHUFFLE_SEED=1, **WD, **base),
        cond("base_Wdeg_b", 0, SHUFFLE_SEED=2, **WD, **base),
        cond("base_Wdeg_c", 0, SHUFFLE_SEED=3, **WD, **base),
    ]
    # sensitivity: one unknown changed at a time, real vs degree-rewired
    variants = {
        "edgeActive": dict(base, EDGE_EXPORT=0.5),           # barrier exports K (locust data), fails with energy
        "edgeBalanced": dict(base, EDGE_EXPORT=0.5, EDGE_G_S=0.02, K_OUT=28.0),  # hemolymph K (28 mM) leaks in slowly,
                                                             # balanced at rest by active export; export fails in anoxia
        "rate100":  dict(base, SD_RATE=100.0),               # more K released by SD tissue
        "leak0.2":  dict(base, K_LEAK_S=0.2),                # slower cell K leak (fly data ~0.2 mM/s)
        "D400":     d400,                                    # slower diffusion (very tight insect ECS), delay re-tuned
        "energyFast": dict(base, E_BASE_USE_S=35.0),         # energy runs out 2x faster (SD 8-15 s after N2 in TPP data)
    }
    for v, p in variants.items():
        c += [cond(f"{v}_R", 0, **p), cond(f"{v}_Wdeg_a", 0, SHUFFLE_SEED=1, **WD, **p)]
    return c


def run_batch(conds, tag):
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_" + tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"{len(conds)} runs, {N_WORKERS} at a time -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1)))
                for i, c in enumerate(conds)]
        for fut in as_completed(futs):
            rows.append(fut.result())
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")
    return out, summary


# ----------------------------------------------------------------------------- C
def analyse(out, summary):
    cols = [c for c in ["tag", "status", "t50_s", "speed_cube_mm_min", "cubes_reached", "blocked_end",
                        "rate_pre_mean_hz", "rate_post_mean_hz", "minutes"] if c in summary]
    log("\n" + summary[cols].to_string(index=False))
    ok = summary[summary.status == "ok"].tag.tolist()
    ct = {t: np.load(os.path.join(out, f"cube_time_{t}.npy")) for t in ok}
    bt = {t: np.load(os.path.join(out, f"block_time_{t}.npy")) for t in ok}
    first = {t: (np.nanmin(ct[t]) - runner.T_ANOX_MS) / 1000 if np.isfinite(ct[t]).any() else np.nan for t in ok}
    log("\nFirst SD after oxygen loss (s) [flies: 8-15 s by DC potential, 20-50 s by K+ electrode]:\n"
        + "  ".join(f"{t.replace('_seed0', '')} {v:.1f}" for t, v in sorted(first.items())))

    def rho(a, b, d):
        x, y = d[a], d[b]
        k = np.isfinite(x) & np.isfinite(y)
        return spearmanr(x[k], y[k])[0] if k.sum() > 10 else np.nan

    def dies(t):
        return float(np.isfinite(ct[t]).mean()) > 0.5

    pairs = [("base_R_seed0", "base_R_seed1", "noise ceiling")]
    for t in ok:
        if t.startswith("base_") and t not in ("base_R_seed0", "base_R_seed1"):
            pairs.append(("base_R_seed0", t, "baseline"))
    for v in ["edgeActive", "edgeBalanced", "rate100", "leak0.2", "D400", "energyFast"]:
        pairs.append((f"{v}_R_seed0", f"{v}_Wdeg_a_seed0", v))
        pairs.append(("base_R_seed0", f"{v}_R_seed0", f"{v} vs baseline (real wiring)"))
    res = []
    for a, b, what in pairs:
        if a in ct and b in ct:
            res.append(dict(what=what, a=a, b=b, a_dies=dies(a), b_dies=dies(b),
                            rho_cubes=rho(a, b, ct), rho_neurons=rho(a, b, bt)))
    res = pd.DataFrame(res)
    log("\nDeath order similarity (Spearman; 1 - rho = how much the order changes):\n" + res.round(4).to_string(index=False))
    try:
        res.to_csv(os.path.join(out, "comparison.csv"), index=False)
    except PermissionError:
        pass

    # mechanism: does a region die earlier when its neurons fire more? (real vs degree-rewired, baseline)
    import importlib
    m = importlib.import_module(MODEL)
    meta, _ = m.load_data()
    npil = meta["neuropil"].astype(str).values
    rows = []
    for t in ok:
        f = os.path.join(out, f"rates_{t}.npz")
        if not os.path.exists(f):
            continue
        r = np.load(f)
        df = pd.DataFrame({"r": npil, "pre": r["pre"], "post": r["post"], "bt": bt[t] / 1000})
        g = df.groupby("r").agg(pre=("pre", "mean"), post=("post", "mean"), bt=("bt", "median"), n=("bt", "size"))
        g = g[g.n >= 50]
        rows.append(dict(tag=t, rho_rate_pre_vs_death=spearmanr(g.pre, g.bt, nan_policy="omit")[0],
                         rho_rate_post_vs_death=spearmanr(g.post, g.bt, nan_policy="omit")[0],
                         mean_rate_pre=df.pre.mean(), mean_rate_post=df.post.mean()))
        if t == "base_R_seed0":
            gR = g
        if t == "base_Wdeg_a_seed0":
            gW = g
    if rows:
        log("\nFiring rate vs regional death time (Spearman over regions; negative = busier regions die earlier):\n"
            + pd.DataFrame(rows).round(3).to_string(index=False))
    if "gR" in locals() and "gW" in locals():
        j = gR.join(gW, lsuffix="_R", rsuffix="_W", how="inner")
        d_rate = j.pre_W - j.pre_R
        d_bt = j.bt_W - j.bt_R
        log(f"\nRewiring: change in regional firing vs change in regional death time: "
            f"rho = {spearmanr(d_rate, d_bt, nan_policy='omit')[0]:+.2f} over {len(j)} regions "
            f"(negative = regions that fire more after rewiring die earlier)")


if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    t0 = time.time()
    if "--full" not in sys.argv:
        log("stage8 (calibration) started", time.strftime("%Y-%m-%d %H:%M"))
        chosen, sw = tissue_part()
        sw.to_csv(os.path.join("runs", time.strftime("%Y%m%d_%H%M%S") + "_v6_tissue_sweep.csv"), index=False)
        with open(CHOSEN, "w") as f:
            json.dump(chosen, f, indent=1)
        log(f"calibration took {(time.time() - t0) / 60:.1f} min")
    if "--tissue" not in sys.argv:
        with open(CHOSEN) as f:
            chosen = json.load(f)
        log(f"\n=== B. full model, oxygen loss, settings {chosen} ===  ({time.strftime('%H:%M')})")
        out, summary = run_batch(conditions(chosen), "v6_robust_stage2")
        analyse(out, summary)
        log(f"\nstage8 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")
