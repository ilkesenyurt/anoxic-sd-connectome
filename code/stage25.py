# stage25.py - seventh review (log row 114). Frozen model v11. Starts automatically after stage24. PC NOT shut down.
# =====================================================================================================================
# WEIGHT-ONLY CONTROL for the glutamate-neutral test. Under the primary rule (stage23, rule W) glutamatergic synapses
# get weight 0 AND every synaptic weight is lowered (W_SYN about 0.11 female, 0.07 male, against 0.275 / 0.2118), so a
# change in the gap could come from either. Here glutamate stays INHIBITORY and only the weights are lowered, to exactly
# the W_SYN that stage23 calibrated for each sex (read from its calibrated.json); the drive is then re-tuned so the real
# brain fires at its frozen healthy rate (11.10 / 12.64 Hz; 20-s healthy runs, late half, interpolation, up to 4 extra
# rounds). Death runs per sex: real seed0 + rewired Wdeg1, Wdeg2 at that W_SYN and drive.
# Reading rule fixed BEFORE any stage23 death result was seen (log row 114), per sex, at the same W_SYN:
#   ratio = gap(glutamate neutral, stage23 rule W) / gap(this control)
#   ratio >= 0.67      -> glutamate's sign does NOT carry the protection
#   0.33 <= ratio < 0.67 -> glutamate's sign carries PART of it
#   ratio < 0.33       -> glutamate's sign carries MOST of it
# Descriptive (no threshold): gap(this control) / gap(frozen model: 6.73 female, 8.67 male) - does the protection shrink
#   as all synapses weaken (network origin)? Healthy rates of the rewired brains reported (runaway check), and firing at
#   -40 mV and slope for every run (Table 3).
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import runner
import stage16
import stage21
import stage22

SCALE = float(os.environ.get("STAGE25_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE25_WORKERS", "5"))
WAIT_FOR = os.environ.get("STAGE25_WAIT", "")
CAL_MS = 20000.0
TARGET = {"female": 11.10, "male": 12.64}
GAP_FROZEN = {"female": 6.73, "male": 8.67}


def log(*a):
    print(*a, flush=True)


def calib_one(job):
    time.sleep(job.pop("stagger", 0))
    m = stage22.fresh(job["sex"], {"W_SYN": job["w"], "TONIC_MEAN": job["drive"]})
    meta, conn = runner._load(m)
    W, coords, N, vuln, L = runner._prepare(m, meta, conn, 0)
    m.RUN_MODE = "healthy"
    dur = CAL_MS * SCALE
    r = m.simulate(W, N, vuln, L, 1e12, dur)
    late = r["t"] >= dur / 2
    return dict(sex=job["sex"], w=job["w"], drive=job["drive"],
                rate_hz=float(r["active"][late].mean() / (m.DT / 1000.0)),
                blocked_end=float(r["blocked"][-1]), K_max=float(r["K_max"].max()))


def death_one(job):
    import importlib
    time.sleep(job.pop("stagger", 0))
    runner.MODEL_MODULE = "death_v11"
    runner.RUN_HEALTHY = False
    runner._DATA.clear()
    runner.T_ANOX_MS = 2000.0
    runner.DEATH_MS = 90000.0 * SCALE
    m = importlib.import_module("death_v11")
    m._GEOM_CACHE.clear()
    ov = dict(job["overrides"])
    ov["SAVE_PROBES_TO"] = os.path.join(job["out"], f"probes_{job['name']}_seed0.npz")
    r = runner.run_condition(dict(name=job["name"], seed=0, overrides=ov, out=job["out"]))
    m._GEOM_CACHE.clear()
    r.update(sex=job["sex"], group=job["group"])
    return r


def pool(jobs, fn):
    res = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(fn, dict(j, stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1))): j
                for i, j in enumerate(jobs)}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                log(f"  ERROR in {j.get('name', j)}: {e!r}")
                continue
            res.append(r)
            if "rate_hz" in r:
                log(f"  {r['sex']} W_SYN {r['w']:.4f}: drive {r['drive']:.3f} -> {r['rate_hz']:.2f} Hz "
                    f"(blocked {r['blocked_end']:.3f}, K max {r['K_max']:.2f})")
            else:
                log(f"  {r.get('sex')} {r.get('tag')} ({r.get('status')}, t50 {r.get('t50_s')})")
    return res


def calibrate(out, wsyn):
    rows = pool([dict(sex=s, w=w, drive=round(stage16.SEX[s]["TONIC_MEAN"] + d, 3))
                 for s, w in wsyn.items() for d in (0.0, 1.5, 3.0)], calib_one)
    found = {}
    for rnd in range(5):
        c = pd.DataFrame(rows)
        todo = []
        for sex, w in wsyn.items():
            if sex in found:
                continue
            cs = c[(c.sex == sex) & (c.blocked_end == 0)].sort_values("drive")
            x, y = cs.drive.values, cs.rate_hz.values
            if len(x) >= 2 and y.min() <= TARGET[sex] <= y.max():
                o = np.argsort(y)
                found[sex] = float(np.interp(TARGET[sex], y[o], x[o]))
                continue
            if rnd == 4 or len(x) < 2:
                continue
            k = np.argsort(np.abs(y - TARGET[sex]))[:2]
            (x1, x2), (y1, y2) = x[k], y[k]
            g = x1 + (TARGET[sex] - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1 + 1.0
            todo += [dict(sex=sex, w=w, drive=float(np.clip(round(g + d, 3), -3, 15))) for d in (-0.3, 0.3)]
        if not todo:
            break
        log(f"  round {rnd + 2}: {[(t['sex'], t['drive']) for t in todo]}")
        rows += pool(todo, calib_one)
    pd.DataFrame(rows).to_csv(os.path.join(out, "calibration.csv"), index=False)
    log(f"  drives (glutamate inhibitory, reduced W_SYN): {found}")
    return found


def wait():
    if not WAIT_FOR or not os.path.exists(WAIT_FOR):
        return
    log(f"waiting for {WAIT_FOR} to report 'finished' ...")
    t0 = time.time()
    while time.time() - t0 < 16 * 3600:
        with open(WAIT_FOR, encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        if "stage24 finished" in txt:
            return
        # fallback only once stage24 is in its death runs (not while it waits for stage23) and its log is stale
        if "=== death runs" in txt and time.time() - os.path.getmtime(WAIT_FOR) > 180 * 60:
            log("  stage24 log unchanged for 180 min during its death runs -> starting anyway")
            return
        time.sleep(60)


def main():
    wait()
    t0 = time.time()
    s23 = sorted(glob.glob(os.path.join(runner.OUT_DIR, "*_v11_stage23")))
    cal23 = {}
    v23 = {}
    if s23:
        p = os.path.join(s23[-1], "calibrated.json")
        if os.path.exists(p):
            cal23 = json.load(open(p))
        p = os.path.join(s23[-1], "verdict.json")
        if os.path.exists(p):
            v23 = json.load(open(p))
    wsyn = {s: float(cal23[f"{s}_W"]) for s in ("female", "male") if f"{s}_W" in cal23}
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage25")
    os.makedirs(out, exist_ok=True)
    log(f"-> {out}\nW_SYN from stage23 rule W: {wsyn}")
    if not wsyn:
        log("no stage23 rule-W calibration found -> nothing to run")
        return
    log("=== calibration (glutamate INHIBITORY, reduced W_SYN, drive re-tuned) ===")
    drives = calibrate(out, wsyn)
    jobs = []
    for sex, drv in drives.items():
        p = {**stage16.FROZEN, **stage16.SEX[sex], "W_SYN": wsyn[sex], "TONIC_MEAN": drv}
        jobs.append(dict(name=f"{sex}_R_ctrlW", sex=sex, group="R_ctrlW", overrides=dict(p), out=out))
        for k in (1, 2):
            jobs.append(dict(name=f"{sex}_Wdeg{k}_ctrlW", sex=sex, group="W_ctrlW", out=out,
                             overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=k)))
    log(f"\n=== death runs: {len(jobs)} ===")
    res = pool(jobs, death_one)
    df = pd.DataFrame(res)
    runner.save_summary(df, os.path.join(out, "summary.csv"))
    meas = []
    for r in res:
        if r.get("status") != "ok":
            continue
        g40, slope = stage21.gain(out, r["tag"])
        meas.append(dict(sex=r["sex"], group=r["group"], run=r["tag"], t50_s=r.get("t50_s"),
                         rate_pre_hz=r.get("rate_pre_mean_hz"), rate_at_m40=g40, slope_hz_per_mV=slope))
    md = pd.DataFrame(meas)
    md.to_csv(os.path.join(out, "measures.csv"), index=False)
    log("\n=== runs ===")
    log(md.round(2).to_string(index=False) if len(md) else "(none)")
    verdict = {}
    for sex in drives:
        s = md[md.sex == sex] if len(md) else md
        rr = s[s.group == "R_ctrlW"]["t50_s"].astype(float).values if len(s) else []
        ww = s[s.group == "W_ctrlW"]["t50_s"].astype(float).values if len(s) else []
        if not (len(rr) and len(ww)):
            continue
        gap_c = float(rr.mean() - ww.mean())
        v = dict(W_SYN=wsyn[sex], drive=round(drives[sex], 3), real_t50=round(float(rr.mean()), 2),
                 rewired_t50=[round(float(x), 2) for x in ww], gap_control_s=round(gap_c, 2),
                 gap_control_over_frozen=round(gap_c / GAP_FROZEN[sex], 2),
                 rate_pre_rewired=[round(float(x), 2) for x in s[s.group == "W_ctrlW"]["rate_pre_hz"]])
        g0 = v23.get(f"{sex}_W", {}).get("gap_s")
        if g0 is not None and gap_c > 0:
            ratio = float(g0) / gap_c
            v.update(gap_glut_neutral_s=g0, ratio=round(ratio, 2),
                     reading=("glutamate's sign does NOT carry the protection" if ratio >= 0.67 else
                              "glutamate's sign carries PART of the protection" if ratio >= 0.33 else
                              "glutamate's sign carries MOST of the protection"))
        verdict[sex] = v
    log("\n=== weight-only control (reading rule fixed before the stage23 death results) ===")
    for sex, v in verdict.items():
        log(f"  {sex}: {json.dumps(v)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    log(f"\nstage25 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage25 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
