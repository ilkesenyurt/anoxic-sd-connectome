# stage24.py - fifth external review (log row 109). Frozen model v11. Starts automatically after stage23. PC NOT shut down.
# =====================================================================================================================
# A. RATE-MATCHED SPLIT REWIRINGS (H12). In stage21 (test now named H11) WsignI fired at 5.6-5.8 Hz when healthy and
#    WsignE at 13.1 (female) / 21.5 Hz (male), so the WsignI > WsignE difference may be a difference in healthy rate.
#    WsignI1 is driven UP and WsignE1 driven DOWN until each fires at the real brain's steady healthy rate (female
#    11.10 Hz, male 12.64 Hz; 20-s healthy runs, late half, linear interpolation, up to 3 extra rounds), then one death
#    run each per sex (the same rewiring seeds as stage21).
#    H12 (fixed before the run): rate-matched WsignI t50 - rate-matched WsignE t50 >= 1 s in BOTH sexes.
#      PASS -> the inhibitory neurons' input balance carries more of the protection than the excitatory neurons'.
#      FAIL -> the two splits differ mainly in healthy rate; the claim stays at 'each neuron's E/I balance'.
# B. keep_loops REPLICATION (H13). Female keep_loops (only the E-I loop connections real) was one run (stage17).
#    Male keep_loops x 2 rewirings (seeds 1, 2) and a second female rewiring (seed 2), frozen settings.
#    H13 (fixed before the run): every keep_loops run dies no more than 2 s before the mean real brain of its sex
#    (stage16: female 20.51 s, male 22.71 s), i.e. t50 >= real - 2 s.
# C. FEMALE PHOTORECEPTORS AT ZERO WEIGHT (H14). The female transmitter predictions have no histamine class, so 11,153
#    female photoreceptors (classification sub_class 'photo_receptor') carry ACH (3,715), GLUT (986), GABA (303) or no
#    transmitter. Here all of them get synaptic weight 0, as the histaminergic male photoreceptors do. Frozen drive.
#    Real seed0 + rewired Wdeg1, Wdeg2.
#    H14 (fixed before the run): real t50 - mean rewired t50 >= 2 s.
# Reported for every run: healthy rate (2 s before O2 off), t50, first SD, firing at -40 mV and slope (Hz/mV).
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import runner
import stage16
import stage21

MODEL = "death_v11"
SCALE = float(os.environ.get("STAGE24_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE24_WORKERS", "5"))
SEXES = os.environ.get("STAGE24_SEXES", "female,male").split(",")
WAIT_FOR = os.environ.get("STAGE24_WAIT", "")
CAL_MS = 20000.0
TARGET = {"female": 11.10, "male": 12.64}
REAL16 = {"female": 20.51, "male": 22.71}
VARIANTS = {"WsignI": (-1, (0.0, 1.0, 2.0)), "WsignE": (1, (0.0, -0.7, -1.5))}   # receiver sign, drive offsets


def log(*a):
    print(*a, flush=True)


def patch(m, job):
    """Wrap m.load_data for receiver-specific rewiring and/or zero-weight female photoreceptors."""
    orig = m.load_data

    def load():
        meta, conn = orig()
        if job.get("photo0"):
            cls = pd.read_csv(m.CLASS_CSV, usecols=["root_id", "sub_class"])
            ids = set(cls.loc[cls["sub_class"] == "photo_receptor", "root_id"])
            hit = meta["root_id"].isin(ids)
            before = meta.loc[hit, "nt_type"].fillna("UNK").value_counts().to_dict()
            meta = meta.copy()
            meta.loc[hit, "nt_type"] = "PR0"                       # not in sign_map -> synaptic weight 0
            print(f"  photoreceptors set to weight 0: {int(hit.sum()):,} neurons (were {before})", flush=True)
        if job.get("recv") is not None:
            conn = stage21.receiver_conn(meta, conn, job["seed_rw"], job["recv"])
        return meta, conn
    m.load_data = load
    return orig


def calib_one(job):
    time.sleep(job.pop("stagger", 0))
    runner.MODEL_MODULE = MODEL
    runner._DATA.clear()
    m = runner._fresh_module()
    for k, v in {**stage16.FROZEN, **stage16.SEX[job["sex"]], "TONIC_MEAN": job["drive"]}.items():
        setattr(m, k, v)
    m._GEOM_CACHE.clear()
    orig = patch(m, job)
    try:
        meta, conn = runner._load(m)
        W, coords, N, vuln, L = runner._prepare(m, meta, conn, 0)
        m.RUN_MODE = "healthy"
        dur = CAL_MS * SCALE
        r = m.simulate(W, N, vuln, L, 1e12, dur)
    finally:
        m.load_data = orig
    late = r["t"] >= dur / 2
    return dict(sex=job["sex"], var=job["var"], drive=job["drive"],
                rate_hz=float(r["active"][late].mean() / (m.DT / 1000.0)),
                blocked_end=float(r["blocked"][-1]), K_max=float(r["K_max"].max()))


def death_one(job):
    import importlib
    time.sleep(job.pop("stagger", 0))
    runner.MODEL_MODULE = MODEL
    runner.RUN_HEALTHY = False
    runner._DATA.clear()
    runner.T_ANOX_MS = 2000.0
    runner.DEATH_MS = 90000.0 * SCALE
    m = importlib.import_module(MODEL)
    m._GEOM_CACHE.clear()
    orig = patch(m, job)
    ov = dict(job["overrides"])
    ov["SAVE_PROBES_TO"] = os.path.join(job["out"], f"probes_{job['name']}_seed0.npz")
    try:
        r = runner.run_condition(dict(name=job["name"], seed=0, overrides=ov, out=job["out"]))
    finally:
        m.load_data = orig
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
                log(f"  {r['sex']} {r['var']}: drive {r['drive']:.3f} -> {r['rate_hz']:.2f} Hz "
                    f"(blocked {r['blocked_end']:.3f}, K max {r['K_max']:.2f})")
            else:
                log(f"  {r.get('sex')} {r.get('tag')} ({r.get('status')}, t50 {r.get('t50_s')})")
    return res


def calibrate(out):
    jobs = []
    for sex in SEXES:
        base = stage16.SEX[sex]["TONIC_MEAN"]
        for var, (sign, offs) in VARIANTS.items():
            jobs += [dict(sex=sex, var=var, recv=sign, seed_rw=1, drive=round(base + d, 3)) for d in offs]
    rows = pool(jobs, calib_one)
    found = {}
    for rnd in range(4):
        c = pd.DataFrame(rows)
        todo = []
        for sex in SEXES:
            for var, (sign, _) in VARIANTS.items():
                if (sex, var) in found:
                    continue
                cs = c[(c.sex == sex) & (c["var"] == var) & (c.blocked_end == 0)].sort_values("drive")
                x, y = cs.drive.values, cs.rate_hz.values
                if len(x) >= 2 and y.min() <= TARGET[sex] <= y.max():
                    o = np.argsort(y)
                    found[(sex, var)] = float(np.interp(TARGET[sex], y[o], x[o]))
                    continue
                if rnd == 3 or len(x) < 2:
                    continue
                k = np.argsort(np.abs(y - TARGET[sex]))[:2]
                (x1, x2), (y1, y2) = x[k], y[k]
                g = x1 + (TARGET[sex] - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
                todo += [dict(sex=sex, var=var, recv=sign, seed_rw=1, drive=float(np.clip(round(g + d, 3), -3, 10)))
                         for d in (-0.2, 0.2)]
        if not todo:
            break
        log(f"  round {rnd + 2}: {[(t['sex'], t['var'], t['drive']) for t in todo]}")
        rows += pool(todo, calib_one)
    pd.DataFrame(rows).to_csv(os.path.join(out, "calibration.csv"), index=False)
    log(f"  matched drives: { {f'{s}_{v}': round(d, 3) for (s, v), d in found.items()} }")
    for sex in SEXES:
        for var in VARIANTS:
            if (sex, var) not in found:
                log(f"  NOT CALIBRATED: {sex} {var} -> no rate-matched death run")
    return found


def wait():
    if not WAIT_FOR or not os.path.exists(WAIT_FOR):
        return
    log(f"waiting for {WAIT_FOR} to report 'finished' (or to stay unchanged for 150 min) ...")
    t0 = time.time()
    while time.time() - t0 < 10 * 3600:
        with open(WAIT_FOR, encoding="utf-8", errors="ignore") as f:
            if "stage23 finished" in f.read():
                return
        if time.time() - os.path.getmtime(WAIT_FOR) > 150 * 60:
            log("  stage23 log unchanged for 150 min -> starting anyway")
            return
        time.sleep(60)


def main():
    wait()
    t0 = time.time()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage24")
    os.makedirs(out, exist_ok=True)
    log(f"-> {out}\n=== A. calibration of the split rewirings (real-brain healthy rate) ===")
    found = calibrate(out)
    with open(os.path.join(out, "matched_drives.json"), "w") as f:
        json.dump({f"{s}_{v}": d for (s, v), d in found.items()}, f, indent=1)
    jobs = []
    for (sex, var), drv in found.items():
        p = {**stage16.FROZEN, **stage16.SEX[sex], "TONIC_MEAN": drv}
        jobs.append(dict(name=f"{sex}_{var}1M", sex=sex, group=f"{var}M", overrides=p, out=out,
                         recv=VARIANTS[var][0], seed_rw=1))
    for sex, seeds in (("male", (1, 2)), ("female", (2,))):
        if sex in SEXES:
            p = {**stage16.FROZEN, **stage16.SEX[sex]}
            for k in seeds:
                jobs.append(dict(name=f"{sex}_keep{k}", sex=sex, group="keep", out=out,
                                 overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="keep_loops", SHUFFLE_SEED=k)))
    if "female" in SEXES:
        p = {**stage16.FROZEN, **stage16.SEX["female"]}
        jobs.append(dict(name="female_R_photo0", sex="female", group="R_photo0", overrides=dict(p), out=out, photo0=True))
        for k in (1, 2):
            jobs.append(dict(name=f"female_Wdeg{k}_photo0", sex="female", group="W_photo0", out=out, photo0=True,
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
    log(md.round(2).to_string(index=False))
    v = {}
    for sex in SEXES:
        s = md[md.sex == sex] if len(md) else md
        ti = s[s.group == "WsignIM"]["t50_s"].astype(float).values if len(s) else []
        te = s[s.group == "WsignEM"]["t50_s"].astype(float).values if len(s) else []
        if len(ti) and len(te):
            v[f"H12_{sex}"] = dict(WsignI_matched=round(float(ti.mean()), 2), WsignE_matched=round(float(te.mean()), 2),
                                   difference_s=round(float(ti.mean() - te.mean()), 2), PASS=bool(ti.mean() - te.mean() >= 1.0))
        kk = s[s.group == "keep"]["t50_s"].astype(float).values if len(s) else []
        if len(kk):
            v[f"H13_{sex}"] = dict(keep_t50=[round(float(x), 2) for x in kk], real16=REAL16[sex],
                                   PASS=bool((kk >= REAL16[sex] - 2.0).all()))
    if "female" in SEXES and len(md):
        rr = md[(md.sex == "female") & (md.group == "R_photo0")]["t50_s"].astype(float).values
        ww = md[(md.sex == "female") & (md.group == "W_photo0")]["t50_s"].astype(float).values
        if len(rr) and len(ww):
            v["H14_female"] = dict(real=round(float(rr.mean()), 2), rewired=[round(float(x), 2) for x in ww],
                                   gap_s=round(float(rr.mean() - ww.mean()), 2), PASS=bool(rr.mean() - ww.mean() >= 2.0))
    v["H12"] = "PASS" if all(v.get(f"H12_{s}", {}).get("PASS") for s in SEXES) else (
        "FAIL" if any(v.get(f"H12_{s}", {}).get("PASS") is False for s in SEXES) else "NOT RUN")
    log("\n=== verdict (H12, H13, H14 fixed before the run) ===")
    for k, x in v.items():
        log(f"  {k}: {json.dumps(x)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    log(f"\nstage24 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage24 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
