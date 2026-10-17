# stage18.py - RATE-MATCHED CONTROL on the frozen model (v11 final), log row 97. Starts after stage17.
# =====================================================================================================================
# Why: at the same drive the rewired brains fire ~2x faster when healthy (female ~20 vs ~9 Hz, male ~20 vs ~11 Hz).
#   A reviewer can say "a brain that fires faster releases more K and dies sooner - that is activity, not wiring".
#   This run separates the two by matching the healthy firing rate.
# Design per sex:
#   1. Calibration (healthy, oxygen kept, 20 s; rate = mean firing per neuron over the last 10 s):
#        real brain at its frozen drive (target rate) and at higher drives; rewired brain (Wdeg1) at its frozen drive
#        (the rewired rate) and at lower drives. Drives found by linear interpolation; if the target lies outside the
#        grid, one extra round 0.3 further out (automatic).
#   2. Death runs (frozen settings, only the drive changed):
#        rate-matched rewired x 3 rewirings (Wdeg1-3, drive lowered so Wdeg1 fires like the real brain)
#        real brain driven UP to the rewired rate x 1 (the reverse comparison)
# Predictions fixed BEFORE the run (Oct 5 00:40):
#   H4 (main) in BOTH sexes the rate-matched rewired brains still die earlier: every matched t50 < every real t50
#        (stage16 real seeds) and mean gap >= 2 s. PASS -> the wiring effect is not just the activity level.
#        FAIL (gap < 2 s) -> the main claim is rephrased: "the real wiring protects mainly by keeping activity low".
#        Reported in all cases: fraction of the stage16 gap that remains = (real - matched) / (real - rewired).
#   H5 the real brain driven up to the rewired rate still dies >= 2 s later than the stage16 rewired brains.
#   Rate used for matching = steady healthy rate (calibration). The death run's own pre-anoxia rate covers only the
#        first 2 s (start-up included), so it is a rough check only.
#   Check (not a hypothesis): the matched group's pre-anoxia rate is within 15% of the real brain's; otherwise the
#        match is reported as imperfect and t50 vs rate is shown for all runs.
#   My expectation, written down: H4 passes in the male (its E->I control already fires at ~20 Hz yet dies like the
#        real brain), uncertain in the female.
#   Known bias, conservative for the wiring claim: a lower drive also keeps neurons further below threshold, which by
#        itself could delay death, so lowering the rewired drive tends to make H4 HARDER to pass.
# Small outputs only.
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
SCALE = float(os.environ.get("STAGE18_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE18_WORKERS", "5"))
CAL_MS = 20000.0
REAL_UP = [3.7, 4.1]           # real brain, higher drives
REW_DOWN = [2.4, 2.7, 3.0]      # rewired brain, lower drives
T_ANOX = runner.T_ANOX_MS


def log(*a):
    print(*a, flush=True)


def calib_one(job):
    """Healthy run: steady firing rate of one network at one drive."""
    time.sleep(job.pop("stagger", 0))
    runner.MODEL_MODULE = MODEL
    runner._DATA.clear()
    m = runner._fresh_module()
    for k, v in job["overrides"].items():
        setattr(m, k, v)
    m._GEOM_CACHE.clear()
    meta, conn = runner._load(m)
    W, coords, N, vuln, L = runner._prepare(m, meta, conn, 0)
    m.RUN_MODE = "healthy"
    dur = CAL_MS * SCALE
    r = m.simulate(W, N, vuln, L, 1e12, dur)
    late = r["t"] >= dur / 2
    return dict(sex=job["sex"], net=job["net"], drive=job["overrides"]["TONIC_MEAN"],
                rate_hz=float(r["active"][late].mean() / (m.DT / 1000.0)),
                blocked_end=float(r["blocked"][-1]), K_max=float(r["K_max"].max()))


def net_overrides(sex, net, drive):
    p = {**stage16.FROZEN, **stage16.SEX[sex], "TONIC_MEAN": float(drive), "RUN_MODE": "death"}
    if net == "rewired":
        p.update(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
    return p


def run_jobs(jobs, fn):
    out = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fn, dict(j, stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1))) for i, j in enumerate(jobs)]
        for fu in as_completed(futs):
            r = fu.result()
            out.append(r)
            log(f"  {r.get('sex', '')} {r.get('net', r.get('tag', ''))} "
                f"{'drive %.3f -> %.2f Hz' % (r['drive'], r['rate_hz']) if 'rate_hz' in r else r.get('status')}")
    return out


def solve(cal, sex, net, target):
    """Drive at which `net` fires at `target` Hz (linear interpolation); None if outside the measured range."""
    c = cal[(cal.sex == sex) & (cal.net == net)].sort_values("drive")
    x, y = c.drive.values, c.rate_hz.values
    if len(x) < 2 or not (y.min() <= target <= y.max()):
        return None
    o = np.argsort(y)
    return float(np.interp(target, y[o], x[o]))


def guess(cal, sex, net, target, width):
    """Two drives around a linear extrapolation from the two measured points closest to the target rate."""
    c = cal[(cal.sex == sex) & (cal.net == net)]
    c = c.iloc[np.argsort(np.abs(c.rate_hz.values - target))[:2]]
    (x1, x2), (y1, y2) = c.drive.values, c.rate_hz.values
    g = x1 + (target - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
    return [float(np.clip(round(g + d, 3), 0.5, 10.0)) for d in (-width, width)]


def calibrate():
    """v2 (Oct 5 03:20): the first attempt (00:59) measured the grid below but the targets lay far outside it
    (rewired rate falls only ~8-10 Hz per drive unit, real only ~3.5-4 Hz per unit). Now: previously measured points
    are reused (calibration.csv of the newest *_v11_ratematch folder), and up to 3 rounds add two drives around a
    linear extrapolation (+-0.2 rewired, +-0.5 real) until the target is bracketed. Hypotheses unchanged."""
    base = {s: stage16.SEX[s]["TONIC_MEAN"] for s in stage16.SEX}
    prev = sorted(glob.glob(os.path.join(runner.OUT_DIR, "*_v11_ratematch", "calibration.csv")))
    cal = pd.read_csv(prev[-1]) if prev else pd.DataFrame(columns=["sex", "net", "drive", "rate_hz"])
    if prev:
        log(f"  reusing {len(cal)} calibration points from {prev[-1]}")

    def have(sex, net, d):
        return bool(((cal.sex == sex) & (cal.net == net) & np.isclose(cal.drive.astype(float), d)).any())

    jobs = []
    for sex in ["male", "female"]:                                   # male first: slower runs
        for d in [base[sex]] + REAL_UP:
            if not have(sex, "real", d):
                jobs.append(dict(sex=sex, net="real", overrides=net_overrides(sex, "real", d)))
        for d in REW_DOWN + [base[sex]]:
            if not have(sex, "rewired", d):
                jobs.append(dict(sex=sex, net="rewired", overrides=net_overrides(sex, "rewired", d)))
    if jobs:
        cal = pd.concat([cal, pd.DataFrame(run_jobs(jobs, calib_one))], ignore_index=True)
    drives = {}
    for rnd in range(4):
        extra = []
        for sex in stage16.SEX:
            c = cal[cal.sex == sex]
            r_real = float(c[(c.net == "real") & np.isclose(c.drive.astype(float), base[sex])].rate_hz.iloc[0])
            r_rew = float(c[(c.net == "rewired") & np.isclose(c.drive.astype(float), base[sex])].rate_hz.iloc[0])
            d_m, d_u = solve(cal, sex, "rewired", r_real), solve(cal, sex, "real", r_rew)
            drives[sex] = dict(real_rate=r_real, rewired_rate=r_rew, matched_rewired_drive=d_m, real_up_drive=d_u)
            if d_m is None:
                extra += [dict(sex=sex, net="rewired", overrides=net_overrides(sex, "rewired", d))
                          for d in guess(cal, sex, "rewired", r_real, 0.2) if not have(sex, "rewired", d)]
            if d_u is None:
                extra += [dict(sex=sex, net="real", overrides=net_overrides(sex, "real", d))
                          for d in guess(cal, sex, "real", r_rew, 0.5) if not have(sex, "real", d)]
        if not extra or rnd == 3:
            break
        log(f"  round {rnd + 2}: target outside the measured range for some cases -> drives "
            + ", ".join(f"{j['sex']} {j['net']} {j['overrides']['TONIC_MEAN']}" for j in extra))
        cal = pd.concat([cal, pd.DataFrame(run_jobs(extra, calib_one))], ignore_index=True)
    return cal, drives


def main():
    t0 = time.time()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_ratematch")
    os.makedirs(out, exist_ok=True)
    log("\n=== 1. calibration (healthy, 20 s) ===")
    cal, drives = calibrate()
    cal.sort_values(["sex", "net", "drive"]).to_csv(os.path.join(out, "calibration.csv"), index=False)
    log(cal.sort_values(["sex", "net", "drive"]).round(3).to_string(index=False))
    for sex, d in drives.items():
        log(f"{sex}: real {d['real_rate']:.2f} Hz, rewired {d['rewired_rate']:.2f} Hz -> matched rewired drive "
            f"{d['matched_rewired_drive']}, real-up drive {d['real_up_drive']}")
    with open(os.path.join(out, "drives.json"), "w") as f:
        json.dump(drives, f, indent=1)

    conds = []
    for sex in ["male", "female"]:
        d = drives[sex]
        if d["matched_rewired_drive"] is not None:
            for k in [1, 2, 3]:
                conds.append(dict(name=f"{sex}_WdegM{k}", seed=0, overrides=dict(
                    net_overrides(sex, "rewired", round(d["matched_rewired_drive"], 3)), SHUFFLE_SEED=k)))
        else:
            log(f"{sex}: no matched drive found - matched runs skipped")
        if d["real_up_drive"] is not None:
            conds.append(dict(name=f"{sex}_Rup", seed=0,
                              overrides=net_overrides(sex, "real", round(d["real_up_drive"], 3))))
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"\n=== 2. {len(conds)} death runs -> {out} ===")
    rows = run_jobs([dict(c, out=out, dur=90000.0 * SCALE, overrides=dict(c["overrides"])) for c in conds],
                    stage16.run_one)
    s = pd.DataFrame(rows)
    runner.save_summary(s, os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")
    analyse(out, s, drives)
    log(f"\nstage18 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


def analyse(out, s, drives):
    main_dir = sorted(glob.glob(os.path.join(runner.OUT_DIR, "*_v11_main")))[-1]
    m16 = pd.read_csv(os.path.join(main_dir, "per_run.csv"))
    if not len(s) or "status" not in s:
        log("no death runs to analyse")
        return
    s = s[s.status == "ok"].set_index("tag")
    verdict = {}
    log("\n=== hypotheses (fixed before the run) ===")
    for sex in stage16.SEX:
        g = m16[m16.sex == sex]
        real, rew = g[g.group == "real"], g[g.group == "rewired"]
        mt = s[[t.startswith(f"{sex}_WdegM") for t in s.index]]
        up = s[[t.startswith(f"{sex}_Rup") for t in s.index]]
        if not len(mt):
            verdict[sex] = dict(H4=None, note="no matched runs")
            continue
        gap = float(real.t50_s.mean() - mt.t50_s.mean())
        orig = float(real.t50_s.mean() - rew.t50_s.mean())
        h4 = bool(mt.t50_s.max() < real.t50_s.min() and gap >= 2.0) if mt.t50_s.notna().all() else None
        h5 = bool(float(up.t50_s.iloc[0]) >= rew.t50_s.mean() + 2.0) if len(up) else None
        rate_ok = bool(abs(mt.rate_pre_mean_hz.mean() / real.rate_pre_hz.mean() - 1) <= 0.15)
        def first(tag):
            c = np.load(os.path.join(out, f"cube_time_{tag}.npy"))
            return float((np.nanmin(c) - T_ANOX) / 1000) if np.isfinite(c).any() else float("nan")
        verdict[sex] = dict(
            H4=h4, H5=h5, gap_matched_s=round(gap, 2), gap_stage16_s=round(orig, 2),
            fraction_of_gap_remaining=round(gap / orig, 2) if orig else None,
            matched_t50=[round(x, 2) for x in mt.t50_s], real_t50_stage16=[round(x, 2) for x in real.t50_s],
            rewired_t50_stage16=[round(x, 2) for x in rew.t50_s],
            real_up_t50=round(float(up.t50_s.iloc[0]), 2) if len(up) else None,
            matched_first_SD=[round(first(t), 2) for t in mt.index],
            rate_pre_real=round(float(real.rate_pre_hz.mean()), 2),
            rate_pre_matched=[round(x, 2) for x in mt.rate_pre_mean_hz],
            rate_pre_real_up=round(float(up.rate_pre_mean_hz.iloc[0]), 2) if len(up) else None,
            rate_pre_rewired_stage16=round(float(rew.rate_pre_hz.mean()), 2),
            rate_match_within_15pct=rate_ok, drives=drives.get(sex))
        log(f"\n{sex}:")
        for k, x in verdict[sex].items():
            log(f"  {k}: {x}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1, default=str)


if __name__ == "__main__":
    log("stage18 started", time.strftime("%Y-%m-%d %H:%M"))
    if "--analyse" in sys.argv:
        out = sorted(glob.glob(os.path.join(runner.OUT_DIR, "*_v11_ratematch")))[-1]
        analyse(out, pd.read_csv(os.path.join(out, "summary.csv")), json.load(open(os.path.join(out, "drives.json"))))
    else:
        main()
