# stage9.py - is it the AMOUNT of activity or the WIRING itself? (activity-matched controls)
# ========================================================================================
# Row 75: degree-preserving rewirings fire ~2x more than the real fly wiring (26 vs 12.6 Hz),
# die 6-15 s earlier and in a different spatial order. Here the activity level is matched:
#   A. calibration (healthy brain, 6 s, no oxygen loss): background drive TONIC_MEAN that makes
#      - the rewired brains fire like the real one (~12.6 Hz)
#      - the real brain fire like the rewired ones (~26.4 Hz)
#   B. oxygen loss (baseline settings of stage 8): 3 rewirings at matched (lower) drive, the
#      real brain at raised drive. Compared with the stage-8 runs of the same folder layout.
#   C. If matched rewirings still die in a different order than the real brain, the specific
#      wiring shapes where death goes; if they become like the real brain, it was the activity level.
# Run: python stage9.py  (stage9.bat runs analyze_balance.py first)

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

SCALE = float(os.environ.get("STAGE9_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE9_WORKERS", "5"))
MODEL = "death_v6"
CAL_MS = 6000.0 * SCALE


def log(*a):
    print(*a, flush=True)


def base_params():
    with open(os.path.join("runs", "v6_chosen.json")) as f:
        ch = json.load(f)["770"]
    return dict(D_UM2_S=770.0, SD_DELAY_S=ch["SD_DELAY_S"], SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2)


WD = lambda seed: dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=seed)


def calib_one(job):
    """Healthy brain (no oxygen loss) for CAL_MS; returns mean firing rate."""
    import importlib
    time.sleep(job.pop("stagger", 0))
    m = importlib.import_module(MODEL)
    for k, v in job["p"].items():
        setattr(m, k, v)
    m.RUN_MODE = "healthy"
    meta, conn = m.load_data()
    m.rng = np.random.default_rng(0)
    idx, W, sign, coords, N = m.build_network(meta, conn)
    vuln = m.build_vulnerability(sign)
    geom = m.build_geometry(meta, idx, coords)
    r = m.simulate(W, N, vuln, geom, 1e12, CAL_MS)
    late = r["t"] >= CAL_MS / 2                         # skip the start-up transient
    rate = float(r["active"][late].mean() / (m.DT / 1000.0))
    return dict(job, rate_hz=rate)


def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    return runner.run_condition(cond)


def pool(fn, jobs):
    out = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fn, dict(j, stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1))) for i, j in enumerate(jobs)]
        for f in as_completed(futs):
            out.append(f.result())
    return out


def interp_drive(rows, target):
    d = pd.DataFrame(rows).sort_values("tonic")
    x, y = d.tonic.values, d.rate_hz.values
    if not (y.min() <= target <= y.max()):
        log(f"  WARNING: target {target:.1f} Hz outside calibrated range {y.min():.1f}-{y.max():.1f} Hz; extrapolating")
        k = np.polyfit(x, y, 1)
        return float((target - k[1]) / k[0])
    return float(np.interp(target, y, x))


if __name__ == "__main__":
    t0 = time.time()
    log("stage9 started", time.strftime("%Y-%m-%d %H:%M"))
    p = base_params()
    v6 = sorted(glob.glob(os.path.join("runs", "*_v6_robust_stage2")))[-1]
    s8 = pd.read_csv(os.path.join(v6, "summary.csv")).set_index("tag")
    r_real = float(s8.loc["base_R_seed0", "rate_pre_mean_hz"])
    r_rew = float(s8.loc[["base_Wdeg_a_seed0", "base_Wdeg_b_seed0", "base_Wdeg_c_seed0"], "rate_pre_mean_hz"].mean())
    log(f"targets from stage 8 (first 2 s): real {r_real:.2f} Hz, rewired {r_rew:.2f} Hz")

    # A. calibration: drive vs firing, healthy brain (all rates measured the same way)
    jobs = [dict(kind="rewired", tonic=t, p=dict(p, TONIC_MEAN=t, **WD(1))) for t in [2.5, 3.25, 4.0, 5.0]]
    jobs += [dict(kind="real", tonic=t, p=dict(p, TONIC_MEAN=t)) for t in [5.0, 6.0, 7.0]]
    log(f"\n=== A. calibration: {len(jobs)} healthy runs of {CAL_MS / 1000:.0f} s ===")
    cal = pool(calib_one, jobs)
    for c in sorted(cal, key=lambda c: (c["kind"], c["tonic"])):
        log(f"  {c['kind']:8s} drive {c['tonic']:.2f}: {c['rate_hz']:.2f} Hz")
    real_c = [c for c in cal if c["kind"] == "real"]
    rew_c = [c for c in cal if c["kind"] == "rewired"]
    real_at5 = [c["rate_hz"] for c in real_c if c["tonic"] == 5.0][0]
    rew_at5 = [c["rate_hz"] for c in rew_c if c["tonic"] == 5.0][0]
    t_rew = interp_drive(rew_c, real_at5)          # rewired turned down to the real level
    t_real = interp_drive(real_c, rew_at5)         # real turned up to the rewired level
    log(f"chosen drive: rewired {t_rew:.2f} (to fire like real, ~{real_at5:.1f} Hz), "
        f"real {t_real:.2f} (to fire like rewired, ~{rew_at5:.1f} Hz); default 5.0")
    with open(os.path.join("runs", "v6_matched_drive.json"), "w") as f:
        json.dump(dict(rewired=t_rew, real=t_real, calibration=[{k: v for k, v in c.items() if k != "p"} for c in cal]), f, indent=1)

    # B. oxygen loss with matched activity
    dur = 90000.0 * SCALE
    conds = [dict(name=f"matched_Wdeg_{x}", seed=0, dur=dur,
                  overrides=dict(RUN_MODE="death", TONIC_MEAN=t_rew, **WD(s), **p)) for x, s in [("a", 1), ("b", 2), ("c", 3)]]
    conds.append(dict(name="raised_R", seed=0, dur=dur, overrides=dict(RUN_MODE="death", TONIC_MEAN=t_real, **p)))
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v6_matched_stage2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"\n=== B. {len(conds)} oxygen-loss runs -> {out}  ({time.strftime('%H:%M')}) ===")
    rows = pool(run_one, [dict(c, out=out) for c in conds])
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    cols = [c for c in ["tag", "status", "t50_s", "cubes_reached", "rate_pre_mean_hz", "rate_post_mean_hz", "minutes"] if c in summary]
    log("\n" + summary[cols].to_string(index=False))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")

    # C. compare with stage 8
    def load(folder, tag):
        f = os.path.join(folder, f"cube_time_{tag}.npy")
        return np.load(f) if os.path.exists(f) else None
    ct = {"real": load(v6, "base_R_seed0"), "real_seed1": load(v6, "base_R_seed1"),
          "rewired_a": load(v6, "base_Wdeg_a_seed0"), "energyFast_real": load(v6, "energyFast_R_seed0")}
    for t in summary.tag:
        ct[t.replace("_seed0", "")] = load(out, t)

    def rho(a, b):
        x, y = ct.get(a), ct.get(b)
        if x is None or y is None:
            return np.nan
        k = np.isfinite(x) & np.isfinite(y)
        return spearmanr(x[k], y[k])[0] if k.sum() > 10 else np.nan

    def first(a):
        x = ct.get(a)
        return (np.nanmin(x) - runner.T_ANOX_MS) / 1000 if x is not None and np.isfinite(x).any() else np.nan

    pairs = [("real", "real_seed1", "noise ceiling"),
             ("real", "rewired_a", "real vs rewired (stage 8, 2x activity)"),
             ("real", "matched_Wdeg_a", "real vs rewired at MATCHED activity"),
             ("real", "matched_Wdeg_b", "real vs rewired at MATCHED activity"),
             ("real", "matched_Wdeg_c", "real vs rewired at MATCHED activity"),
             ("rewired_a", "matched_Wdeg_a", "rewired: normal vs lowered drive"),
             ("real", "raised_R", "real: normal vs raised drive"),
             ("raised_R", "rewired_a", "real at raised drive vs rewired"),
             ("real", "energyFast_real", "real: normal vs energy 2x (dies early, same activity)")]
    res = pd.DataFrame([dict(comparison=w, a=a, b=b, rho=rho(a, b), first_SD_a=first(a), first_SD_b=first(b))
                        for a, b, w in pairs])
    log("\n=== C. death order similarity (Spearman over cubes; 1 = identical) and first SD after N2 (s) ===")
    log(res.round(4).to_string(index=False))
    try:
        res.to_csv(os.path.join(out, "comparison.csv"), index=False)
    except PermissionError:
        pass
    log(f"\nstage9 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")
