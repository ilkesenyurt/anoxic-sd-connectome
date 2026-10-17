# stage23.py - completes the glutamate-neutral test (H10) for the male and adds a second calibration rule that is
# applied identically to BOTH sexes (log row 108). Frozen model v11. PC is NOT shut down.
# =====================================================================================================================
# Background: in stage22 the drive-only re-tuning reached the female target (drive 1.158 mV, H10 PASS, gap 8.2 s) but
# not the male: the search floor (drive 0.2 mV) still gave 15.31 Hz > 12.64 Hz target, so no male death runs.
# RULE D (drive, as pre-registered in stage22, floor lowered): male only. The drive is a constant input current in mV
#    (mean of a per-neuron Gaussian, SD 0.5, plus noise SD 2.0); values below 0 are allowed down to -3 mV (a net
#    hyperpolarising bias). The female already used this rule in stage22 and her floor never bound, so her stage22
#    result IS the rule-D result (re-running it would reproduce it bit for bit with the same seeds).
# RULE W (synaptic weight, drive frozen): BOTH sexes. Drive stays at the frozen value (female 3.36, male 3.418 mV) and
#    W_SYN is re-tuned so the real brain fires at its frozen healthy rate (female 11.10 Hz, male 12.64 Hz; 20-s healthy
#    runs, late half, linear interpolation, up to 4 extra rounds). Rewired brains use the same W_SYN.
# Death runs (90 s, anoxia at 2 s): real seed0 and rewired Wdeg1, Wdeg2 for each sex x rule (male D; female W; male W).
# Hypotheses fixed BEFORE the run:
#    H10b: male, rule D: real t50 - mean rewired t50 >= 2 s.
#    H10c: rule W: real t50 - mean rewired t50 >= 2 s in BOTH sexes.
# Also reported: healthy rate of real and rewired brains, firing at -40 mV mean resting potential, slope (Hz/mV).
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

SCALE = float(os.environ.get("STAGE23_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE23_WORKERS", "5"))
CAL_MS = 20000.0
TARGET = {"female": 11.10, "male": 12.64}
STAGE22_FEMALE_D = {"drive": 1.158, "real_t50": 20.21, "rewired_t50": [12.03, 12.01], "gap_s": 8.2,
                    "rate_pre_real": 9.63, "rate_pre_rewired": 12.29, "H10_PASS": True}
# rule -> (parameter, lower bound, upper bound, starting multipliers/offsets)
RULES = {"D": ("TONIC_MEAN", -3.0, 10.0), "W": ("W_SYN", 0.02, 1.0)}
PLAN = [("male", "D"), ("female", "W"), ("male", "W")]


def log(*a):
    print(*a, flush=True)


def calib_one(job):
    time.sleep(job.pop("stagger", 0))
    par = RULES[job["rule"]][0]
    m = stage22.fresh(job["sex"], {par: job["value"]})
    orig = stage22.glut_neutral(m)
    try:
        meta, conn = runner._load(m)
        W, coords, N, vuln, L = runner._prepare(m, meta, conn, 0)
        m.RUN_MODE = "healthy"
        dur = CAL_MS * SCALE
        r = m.simulate(W, N, vuln, L, 1e12, dur)
    finally:
        m.load_data = orig
    late = r["t"] >= dur / 2
    return dict(sex=job["sex"], rule=job["rule"], value=job["value"],
                rate_hz=float(r["active"][late].mean() / (m.DT / 1000.0)),
                blocked_end=float(r["blocked"][-1]), K_max=float(r["K_max"].max()))


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
                log(f"  ERROR in {j}: {e!r}")
                continue
            res.append(r)
            if "rate_hz" in r:
                log(f"  {r['sex']} rule {r['rule']}: {RULES[r['rule']][0]} {r['value']:.4f} -> {r['rate_hz']:.2f} Hz "
                    f"(blocked {r['blocked_end']:.3f}, K max {r['K_max']:.2f})")
            else:
                log(f"  {r.get('sex')} {r.get('tag')} ({r.get('status')}, t50 {r.get('t50_s')})")
    return res


def starts(sex, rule):
    base = {**stage16.FROZEN, **stage16.SEX[sex]}
    if rule == "D":
        return [-0.8, -0.3, 0.2]                      # stage22: 0.2 -> 15.31, 1.418 -> 18.51 Hz (about 2.6 Hz/mV)
    w = base["W_SYN"]
    return [round(w * f, 4) for f in (1.0, 0.75, 0.5)]


def calibrate(out):
    rows = pool([dict(sex=s, rule=r, value=v) for s, r in PLAN for v in starts(s, r)], calib_one)
    found = {}
    for rnd in range(4):
        c = pd.DataFrame(rows)
        todo = []
        for sex, rule in PLAN:
            if (sex, rule) in found:
                continue
            _, lo, hi = RULES[rule]
            cs = c[(c.sex == sex) & (c.rule == rule) & (c.blocked_end == 0)].sort_values("value")
            x, y = cs.value.values, cs.rate_hz.values
            if len(x) >= 2 and y.min() <= TARGET[sex] <= y.max():
                o = np.argsort(y)
                found[(sex, rule)] = float(np.interp(TARGET[sex], y[o], x[o]))
                continue
            if len(x) >= 2:
                k = np.argsort(np.abs(y - TARGET[sex]))[:2]
                (x1, x2), (y1, y2) = x[k], y[k]
                g = x1 + (TARGET[sex] - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
                step = 0.3 if rule == "D" else 0.1 * abs(g)
            else:
                g, step = (float(x.min()) if len(x) else starts(sex, rule)[-1]) * 0.7, 0.05
            todo += [dict(sex=sex, rule=rule, value=float(np.clip(round(g + d, 4), lo, hi))) for d in (-step, step)]
        if not todo:
            break
        log(f"  round {rnd + 2}: {todo}")
        rows += pool(todo, calib_one)
    pd.DataFrame(rows).to_csv(os.path.join(out, "calibration.csv"), index=False)
    log(f"  calibrated values: { {f'{s}_{r}': round(v, 4) for (s, r), v in found.items()} }")
    for sex, rule in PLAN:
        if (sex, rule) not in found:
            log(f"  NOT CALIBRATED: {sex} rule {rule} (target outside the reachable range) -> no death runs")
    return found


def main():
    t0 = time.time()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage23")
    os.makedirs(out, exist_ok=True)
    log(f"-> {out}\n=== calibration (glutamate neutral, real wiring) ===")
    found = calibrate(out)
    with open(os.path.join(out, "calibrated.json"), "w") as f:
        json.dump({f"{s}_{r}": v for (s, r), v in found.items()}, f, indent=1)
    jobs = []
    for (sex, rule), val in found.items():
        p = {**stage16.FROZEN, **stage16.SEX[sex], RULES[rule][0]: val}
        jobs.append(dict(name=f"{sex}_R_glut0{rule}", sex=sex, group=f"R_{rule}", overrides=dict(p), out=out))
        for k in (1, 2):
            jobs.append(dict(name=f"{sex}_Wdeg{k}_glut0{rule}", sex=sex, group=f"W_{rule}", out=out,
                             overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=k)))
    log(f"\n=== death runs: {len(jobs)} ===")
    res = pool(jobs, stage22.death_one)
    df = pd.DataFrame(res)
    runner.save_summary(df, os.path.join(out, "summary.csv"))
    verdict = {"female_D_from_stage22": STAGE22_FEMALE_D}
    for (sex, rule), val in found.items():
        ok = df[(df.sex == sex) & (df.status == "ok")]
        rr, ww = ok[ok.group == f"R_{rule}"], ok[ok.group == f"W_{rule}"]
        if not (len(rr) and len(ww)):
            continue
        gap = float(rr.t50_s.astype(float).mean() - ww.t50_s.astype(float).mean())
        g = {t: stage21.gain(out, t) for t in ok[ok.group.isin([f"R_{rule}", f"W_{rule}"])].tag}
        verdict[f"{sex}_{rule}"] = dict(
            parameter=RULES[rule][0], value=round(val, 4), real_t50=round(float(rr.t50_s.astype(float).mean()), 2),
            rewired_t50=[round(float(x), 2) for x in ww.t50_s.astype(float)], gap_s=round(gap, 2),
            rate_pre_real=round(float(rr.rate_pre_mean_hz.mean()), 2),
            rate_pre_rewired=round(float(ww.rate_pre_mean_hz.mean()), 2),
            rate_at_m40={t: round(v[0], 1) for t, v in g.items()},
            slope_hz_per_mV={t: round(v[1], 2) for t, v in g.items()},
            PASS=bool(gap >= 2.0))
    v = verdict
    h10b = v.get("male_D", {}).get("PASS")
    h10c = [v.get("female_W", {}).get("PASS"), v.get("male_W", {}).get("PASS")]
    verdict["H10b_male_ruleD"] = "PASS" if h10b else ("FAIL" if h10b is False else "NOT RUN")
    verdict["H10c_ruleW_both"] = ("PASS" if all(x is True for x in h10c) else
                                  "FAIL" if any(x is False for x in h10c) else "NOT RUN")
    log("\n=== verdict (H10b, H10c fixed before the run) ===")
    for k, x in verdict.items():
        log(f"  {k}: {json.dumps(x)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    log(f"\nstage23 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage23 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
