# stage27.py - follow-up to the ninth review (log row 120). Frozen model v11. PC is NOT shut down.
# =====================================================================================================================
# keepEI (stage26) kept real a random set of non-reciprocal E->I / I->E connections with the same synapse TOTAL as the
# E-I loop connections, but these were more numerous and weaker (about 13 synapses each against 18-19), so it did not
# separate two explanations for why the loop set protects nearly twice as much per synapse: (a) the loop connections
# are stronger, or (b) they form a feedback group linking the same neurons.
#   keepEIs : strength-matched non-loop control - for every loop connection, one non-reciprocal E-I connection with the
#             same direction (E->I or I->E) and the same synapse count is kept real (nearest available count when the
#             pool runs out); everything else, loops included, is rewired (receivers permuted; degrees kept). Seed 1.
#   Driven to the real steady healthy rate (female 11.10, male 12.64 Hz) as in stage26; one death run per sex.
# Reading rule fixed BEFORE the run, per sex, with recoveries from stage26 (keep_loops R_loop: 0.97 F / 0.73 M;
#   keepEI R_EI: 0.51 F / 0.43 M) and recovery R_s = (t50 - t50 rate-matched rewired) / (t50 real - t50 rate-matched
#   rewired), as in stage26:
#   f = (R_s - R_EI) / (R_loop - R_EI)
#   f >= 0.67 -> connection strength explains most of the loop advantage
#   f <= 0.33 -> strength explains little; the feedback-group structure of the loop set explains most of it
#   otherwise -> both contribute
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
SCALE = float(os.environ.get("STAGE27_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE27_WORKERS", "5"))
SEXES = os.environ.get("STAGE27_SEXES", "female,male").split(",")
CAL_MS = 20000.0
TARGET = {"female": 11.10, "male": 12.64}
REAL16 = {"female": 20.51, "male": 22.71}
WDEGM18 = {"female": 15.36, "male": 15.60}
WDEG16 = {"female": 13.78, "male": 14.04}
VARIANTS = {"keepEIs": (-1.0, 0.0, 1.0)}
R_LOOP = {"female": 0.97, "male": 0.73}
R_EI = {"female": 0.51, "male": 0.43}


def log(*a):
    print(*a, flush=True)


def keep_eis_conn(meta, conn, seed):
    """Strength-matched non-loop control: one non-reciprocal E-I connection of the same direction and synapse count per
    loop connection is kept real; everything else is rewired."""
    g = np.random.default_rng(seed)
    nt = dict(zip(meta["root_id"].values, meta["nt_type"].fillna("UNK").values))
    pre = conn["pre_root_id"].values
    post = conn["post_root_id"].values.copy()
    syn = conn["syn_count"].values.astype(np.int64)
    sp = np.array([stage20.sign_of_nt(nt.get(p, "UNK")) for p in pre], dtype=np.int8)
    so = np.array([stage20.sign_of_nt(nt.get(p, "UNK")) for p in post], dtype=np.int8)
    idx = {r: i for i, r in enumerate(meta["root_id"].values)}
    pi = np.array([idx[p] for p in pre], dtype=np.int64); qi = np.array([idx[p] for p in post], dtype=np.int64)
    N = len(idx)
    recip = np.isin(qi * N + pi, pi * N + qi)
    e2i = (sp > 0) & (so < 0); i2e = (sp < 0) & (so > 0)
    loop = recip & (e2i | i2e)
    keep = np.zeros(post.size, bool)
    short = 0
    for dmask in (e2i, i2e):
        need = np.bincount(syn[loop & dmask])                       # loop connections per synapse count
        pool = np.flatnonzero(dmask & ~loop)
        pool = pool[g.permutation(pool.size)]
        by = {}
        for k in pool:
            by.setdefault(int(syn[k]), []).append(k)
        counts = np.array(sorted(by)) if by else np.array([], int)
        for c in np.flatnonzero(need)[::-1]:                        # strongest first
            n = int(need[c])
            for cc in sorted(counts, key=lambda v: (abs(v - c), -v)):  # same count, then nearest
                lst = by.get(int(cc), [])
                take = min(n, len(lst))
                if take:
                    keep[lst[:take]] = True
                    by[int(cc)] = lst[take:]
                    n -= take
                if n == 0:
                    break
            short += n
    sel = np.flatnonzero(~keep)
    post[sel] = post[sel][g.permutation(sel.size)]
    for _ in range(10):
        selfc = sel[post[sel] == pre[sel]]
        if not selfc.size:
            break
        other = g.choice(sel, selfc.size)
        post[selfc], post[other] = post[other].copy(), post[selfc].copy()
    out = conn.copy(); out["post_root_id"] = post
    log(f"  keepEIs (seed {seed}): loop connections {int(loop.sum()):,} ({syn[loop].sum():,} synapses, mean "
        f"{syn[loop].mean():.2f}); kept real {int(keep.sum()):,} non-loop E-I connections ({syn[keep].sum():,} synapses, "
        f"mean {syn[keep].mean():.2f}; E->I {int((keep & e2i).sum()):,} vs loop {int((loop & e2i).sum()):,}); unmatched {short}; "
        f"rewired {sel.size:,} of {post.size:,}; self-connections left {int((post == pre).sum())}")
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
    if job["var"] != "keepEIs":
        return orig

    def load():
        meta, conn = orig()
        return meta, keep_eis_conn(meta, conn, job.get("seed_rw", 1))
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
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage27")
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
        x = s_[s_.group == "keepEIsM"]["t50_s"].astype(float).values if len(s_) else []
        if len(x):
            rec = float((x.mean() - WDEGM18[sex]) / (REAL16[sex] - WDEGM18[sex]))
            f = (rec - R_EI[sex]) / (R_LOOP[sex] - R_EI[sex])
            v[f"keepEIs_{sex}"] = dict(t50=round(float(x.mean()), 2), recovery=round(rec, 2), f=round(f, 2),
                                       reading=("n/a" if not np.isfinite(f) else
                                                "connection strength explains most of the loop advantage" if f >= 0.67 else
                                                "feedback-group structure explains most of the loop advantage" if f <= 0.33
                                                else "both strength and feedback-group structure contribute"))
    log("\n=== verdict (reading rule fixed before the run) ===")
    for k, x in v.items():
        log(f"  {k}: {json.dumps(x)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    log(f"\nstage27 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage27 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
