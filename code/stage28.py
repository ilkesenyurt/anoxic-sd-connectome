# stage28.py - log row 124. Frozen model v11. PC is NOT shut down.
# =====================================================================================================================
# check_loops (row 123) found that the stage27 matched set (keepEIs) ran mostly between loop neurons (72% female, 79%
# male of its synapses; rule: between 0.5 and 0.8), but the loop connections attach to neurons with about twice the
# degree (about 690 against 310-400 connections) and are concentrated in the medulla (ME 38% / 35% of loop synapses
# against 27% / 23%). So the loop advantage could still come from WHICH neurons the loop connections link.
#   keepShuf : only the loop connections stay among their own neurons, but who pairs with whom is shuffled: E->I loop
#              connections permute receivers among themselves, I->E loop connections among themselves (as the E->I
#              control), and every other connection is rewired as in keep_loops. Every loop connection keeps its sender,
#              its strength and a receiver from the same set of loop receivers; each neuron keeps its numbers of loop
#              inputs and outputs. Only the reciprocal pairing is broken. Seed 1.
#   Driven to the real steady healthy rate (female 11.10, male 12.64 Hz) as in stage26/27; one death run per sex.
# Reading rule fixed BEFORE the run (log row 124), per sex, recovery R as in stage26/27 and
#   g = (R_shuf - R_s) / (R_loop - R_s), with R_s (keepEIs, stage27) 0.46 / 0.37 and R_loop (keep_loops) 0.97 / 0.73:
#   g >= 0.67 -> the advantage belongs to which neurons the loop connections link (identity / group), not to the pairing
#   g <= 0.33 -> the reciprocal pairing itself carries the advantage
#   otherwise -> both
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import runner
import stage16
import stage20
import stage21

MODEL = "death_v11"
SCALE = float(os.environ.get("STAGE28_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE28_WORKERS", "5"))
SEXES = os.environ.get("STAGE28_SEXES", "female,male").split(",")
CAL_MS = 20000.0
TARGET = {"female": 11.10, "male": 12.64}
REAL16 = {"female": 20.51, "male": 22.71}
WDEGM18 = {"female": 15.36, "male": 15.60}
WDEG16 = {"female": 13.78, "male": 14.04}
VARIANTS = {"keepShuf": (-1.0, 0.0, 1.0)}
R_LOOP = {"female": 0.97, "male": 0.73}
R_S = {"female": 0.46, "male": 0.37}


def log(*a):
    print(*a, flush=True)


def keep_shuf_conn(meta, conn, seed):
    """Loop connections stay among their own neurons with pairs shuffled (E->I and I->E sets separately); all other
    connections are rewired as in keep_loops."""
    g = np.random.default_rng(seed)
    nt = meta["nt_type"].fillna("UNK").values
    sign = np.array([stage20.sign_of_nt(t) for t in nt], dtype=np.int8)
    idx = {r: i for i, r in enumerate(meta["root_id"].values)}
    N = len(idx)
    pi = conn["pre_root_id"].map(idx).values.astype(np.int64)
    qi = conn["post_root_id"].map(idx).values.astype(np.int64)
    syn = conn["syn_count"].values.astype(np.int64)
    sp, so = sign[pi], sign[qi]
    recip = np.isin(qi * N + pi, pi * N + qi)
    e2i = (sp > 0) & (so < 0); i2e = (sp < 0) & (so > 0)
    loop = recip & (e2i | i2e)
    post = qi.copy()
    parts = [np.flatnonzero(loop & e2i), np.flatnonzero(loop & i2e), np.flatnonzero(~loop)]
    for part in parts:
        post[part] = post[part][g.permutation(part.size)]
    for part in parts:                                         # remove self-connections within each set
        for _ in range(10):
            selfc = part[post[part] == pi[part]]
            if not selfc.size:
                break
            other = g.choice(part, selfc.size)
            post[selfc], post[other] = post[other].copy(), post[selfc].copy()
    lp = loop.copy()
    new_recip = np.isin(post[lp] * N + pi[lp], pi[lp] * N + post[lp])
    ids = meta["root_id"].values
    out = conn.copy(); out["post_root_id"] = ids[post]
    log(f"  keepShuf (seed {seed}): loop connections {int(loop.sum()):,} ({syn[loop].sum():,} synapses, mean "
        f"{syn[loop].mean():.2f}; E->I {int((loop & e2i).sum()):,}) shuffled within their sets; still reciprocal after "
        f"shuffling {int(new_recip.sum()):,} ({new_recip.mean():.2%}); other connections rewired {int((~loop).sum()):,}; "
        f"self-connections left {int((post == pi).sum())}")
    return out


def overrides(sex, var, drive=None, seed=1):
    p = {**stage16.FROZEN, **stage16.SEX[sex]}
    if drive is not None:
        p["TONIC_MEAN"] = float(drive)
    if var == "keep":
        p.update(SHUFFLE_CONN=True, SHUFFLE_MODE="keep_loops", SHUFFLE_SEED=seed)
    elif var == "break":
        p.update(SHUFFLE_CONN=True, SHUFFLE_MODE="break_loops", SHUFFLE_SEED=seed)
    return p


def patch(m, job):
    orig = m.load_data
    if job["var"] != "keepShuf":
        return orig

    def load():
        meta, conn = orig()
        return meta, keep_shuf_conn(meta, conn, job.get("seed_rw", 1))
    m.load_data = load
    return orig


def calib_one(job):
    time.sleep(job.pop("stagger", 0))
    runner.MODEL_MODULE = MODEL
    runner._DATA.clear()
    m = runner._fresh_module()
    for k, v in overrides(job["sex"], job["var"], job["drive"]).items():
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
    jobs = [dict(sex=s, var=v, drive=round(stage16.SEX[s]["TONIC_MEAN"] + d, 3))
            for s in SEXES for v, offs in VARIANTS.items() for d in offs]
    rows = pool(jobs, calib_one)
    found = {}
    for rnd in range(5):
        c = pd.DataFrame(rows)
        todo = []
        for sex in SEXES:
            for var in VARIANTS:
                if (sex, var) in found:
                    continue
                cs = c[(c.sex == sex) & (c["var"] == var) & (c.blocked_end == 0)].sort_values("drive")
                x, y = cs.drive.values, cs.rate_hz.values
                if len(x) >= 2 and y.min() <= TARGET[sex] <= y.max():
                    o = np.argsort(y)
                    found[(sex, var)] = float(np.interp(TARGET[sex], y[o], x[o]))
                    continue
                if rnd == 4 or len(x) < 2:
                    continue
                k = np.argsort(np.abs(y - TARGET[sex]))[:2]
                (x1, x2), (y1, y2) = x[k], y[k]
                gs = x1 + (TARGET[sex] - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1 - 1.0
                todo += [dict(sex=sex, var=var, drive=float(np.clip(round(gs + d, 3), -6, 12))) for d in (-0.3, 0.3)]
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


def main():
    t0 = time.time()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage28")
    os.makedirs(out, exist_ok=True)
    log(f"-> {out}\n=== calibration (variants driven to the real healthy rate) ===")
    found = calibrate(out)
    with open(os.path.join(out, "matched_drives.json"), "w") as f:
        json.dump({f"{s}_{v}": d for (s, v), d in found.items()}, f, indent=1)
    jobs = [dict(name=f"{s}_{v}1M", sex=s, var=v, group=f"{v}M", overrides=overrides(s, v, d), out=out, seed_rw=1)
            for (s, v), d in found.items()]
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
    v = {}
    for sex in SEXES:
        s_ = md[md.sex == sex] if len(md) else md
        x = s_[s_.group == "keepShufM"]["t50_s"].astype(float).values if len(s_) else []
        if len(x):
            rec = float((x.mean() - WDEGM18[sex]) / (REAL16[sex] - WDEGM18[sex]))
            gg = (rec - R_S[sex]) / (R_LOOP[sex] - R_S[sex])
            v[f"keepShuf_{sex}"] = dict(t50=round(float(x.mean()), 2), recovery=round(rec, 2), g=round(gg, 2),
                                        reading=("n/a" if not np.isfinite(gg) else
                                                 "which neurons the loop connections link carries the advantage, not the pairing" if gg >= 0.67 else
                                                 "the reciprocal pairing itself carries the advantage" if gg <= 0.33
                                                 else "both the neurons linked and the pairing contribute"))
    log("\n=== verdict (reading rule fixed before the run) ===")
    for k, x in v.items():
        log(f"  {k}: {json.dumps(x)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    log(f"\nstage28 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage28 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
