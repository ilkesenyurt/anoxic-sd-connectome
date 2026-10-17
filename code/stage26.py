# stage26.py - eighth review (log row 117). Frozen model v11. PC is NOT shut down.
# =====================================================================================================================
# The conclusion now rests on keep_loops (only the E-I loop connections real), but keep_loops fires at about half the
# real healthy rate, break_loops is one female run, and keep_loops tests only reciprocal E-I pairs. Rate-matched design
# as in H12: each variant is driven to the real brain's steady healthy rate (female 11.10 Hz, male 12.64 Hz; 20-s
# healthy runs, late half, interpolation, up to 4 extra rounds), then one death run per variant and sex.
#   keep   : keep_loops (SHUFFLE_MODE keep_loops, seed 1) - only the E-I loop connections real.
#   break  : break_loops (seed 1) - only the loop connections rewired, E->I and I->E targets pooled.
#   keepEI : matched non-loop control - a random set of NON-reciprocal E->I and I->E connections with the same total
#            synapse count as the loop connections is kept real (seed 1); everything else, loops included, is
#            rewired (receivers permuted; every neuron keeps its in- and out-degree).
#   Plus: male break_loops at the frozen drive x 2 (seeds 1, 2), as for the female in stage17.
# Recovery at matched rate = (t50_variant - t50_rewired_matched) / (t50_real - t50_rewired_matched), with the stage18
#   rate-matched rewired brains (female 15.36 s, male 15.60 s; means of WdegM1-3) and stage16 real brains (20.51 / 22.71 s).
# Hypotheses fixed BEFORE the run:
#   H15: rate-matched keep_loops recovery >= 0.67 in BOTH sexes (the loop connections carry most of the protection).
#   H16: rate-matched break_loops recovery <= 0.33 in BOTH sexes, and both male frozen-drive break_loops runs die
#        before the mean male rewired brain (14.04 s) (class targeting of loop connections is necessary).
#   H17 (reading rule, not pass/fail): rate-matched keepEI recovery >= 0.67 -> E->I / I->E connections in general carry
#        the protection (broad wording earned); <= 0.33 -> it is specific to the reciprocal loop pairs; between -> partly.
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
SCALE = float(os.environ.get("STAGE26_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE26_WORKERS", "5"))
SEXES = os.environ.get("STAGE26_SEXES", "female,male").split(",")
CAL_MS = 20000.0
TARGET = {"female": 11.10, "male": 12.64}
REAL16 = {"female": 20.51, "male": 22.71}
WDEGM18 = {"female": 15.36, "male": 15.60}
WDEG16 = {"female": 13.78, "male": 14.04}
VARIANTS = {"keep": (0.0, 1.0, 2.0), "break": (0.0, -2.0, -4.0), "keepEI": (-1.0, 0.0, 1.0)}


def log(*a):
    print(*a, flush=True)


def keep_ei_conn(meta, conn, seed):
    """Keep real a random set of non-reciprocal E-I connections with the loop connections' synapse total; rewire the rest."""
    g = np.random.default_rng(seed)
    nt = dict(zip(meta["root_id"].values, meta["nt_type"].fillna("UNK").values))
    pre = conn["pre_root_id"].values
    post = conn["post_root_id"].values.copy()
    syn = conn["syn_count"].values.astype(float)
    sp = np.array([stage20.sign_of_nt(nt.get(p, "UNK")) for p in pre], dtype=np.int8)
    so = np.array([stage20.sign_of_nt(nt.get(p, "UNK")) for p in post], dtype=np.int8)
    idx = {r: i for i, r in enumerate(meta["root_id"].values)}
    pi = np.array([idx[p] for p in pre], dtype=np.int64); qi = np.array([idx[p] for p in post], dtype=np.int64)
    N = len(idx)
    recip = np.isin(qi * N + pi, pi * N + qi)
    ei = ((sp > 0) & (so < 0)) | ((sp < 0) & (so > 0))
    loop = recip & ei
    pool = np.flatnonzero(ei & ~loop)
    order = g.permutation(pool)
    target = syn[loop].sum()
    csum = np.cumsum(syn[order])
    k = int(np.searchsorted(csum, target) + 1)
    keep = np.zeros(post.size, bool); keep[order[:k]] = True
    sel = np.flatnonzero(~keep)
    post[sel] = post[sel][g.permutation(sel.size)]
    for _ in range(10):
        selfc = sel[post[sel] == pre[sel]]
        if not selfc.size:
            break
        other = g.choice(sel, selfc.size)
        post[selfc], post[other] = post[other].copy(), post[selfc].copy()
    out = conn.copy(); out["post_root_id"] = post
    log(f"  keepEI (seed {seed}): loop connections {int(loop.sum()):,} ({target:,.0f} synapses); kept real "
        f"{int(keep.sum()):,} non-loop E-I connections ({syn[keep].sum():,.0f} synapses, pool {pool.size:,}); "
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
    if job["var"] != "keepEI":
        return orig

    def load():
        meta, conn = orig()
        return meta, keep_ei_conn(meta, conn, job.get("seed_rw", 1))
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
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage26")
    os.makedirs(out, exist_ok=True)
    log(f"-> {out}\n=== calibration (variants driven to the real healthy rate) ===")
    found = calibrate(out)
    with open(os.path.join(out, "matched_drives.json"), "w") as f:
        json.dump({f"{s}_{v}": d for (s, v), d in found.items()}, f, indent=1)
    jobs = [dict(name=f"{s}_{v}1M", sex=s, var=v, group=f"{v}M", overrides=overrides(s, v, d), out=out, seed_rw=1)
            for (s, v), d in found.items()]
    if "male" in SEXES:
        for k in (1, 2):
            jobs.append(dict(name=f"male_break{k}", sex="male", var="break", group="break_frozen",
                             overrides=overrides("male", "break", None, seed=k), out=out))
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
        s = md[md.sex == sex] if len(md) else md
        for var in VARIANTS:
            x = s[s.group == f"{var}M"]["t50_s"].astype(float).values if len(s) else []
            if len(x):
                rec = float((x.mean() - WDEGM18[sex]) / (REAL16[sex] - WDEGM18[sex]))
                v[f"{var}_{sex}"] = dict(t50=round(float(x.mean()), 2), recovery=round(rec, 2))
        if sex == "male" and len(s):
            b = s[s.group == "break_frozen"]["t50_s"].astype(float).values
            if len(b):
                v["break_frozen_male"] = dict(t50=[round(float(y), 2) for y in b], rewired16=WDEG16["male"],
                                              all_before_rewired=bool((b < WDEG16["male"]).all()))
    def rec(k):
        return v.get(k, {}).get("recovery")
    h15 = [rec(f"keep_{s}") for s in SEXES]
    v["H15"] = "PASS" if all(r is not None and r >= 0.67 for r in h15) else ("NOT RUN" if any(r is None for r in h15) else "FAIL")
    h16 = [rec(f"break_{s}") for s in SEXES]
    bf = v.get("break_frozen_male", {}).get("all_before_rewired")
    v["H16"] = ("NOT RUN" if any(r is None for r in h16) or bf is None else
                "PASS" if all(r <= 0.33 for r in h16) and bf else "FAIL")
    for s in SEXES:
        r = rec(f"keepEI_{s}")
        if r is not None:
            v[f"H17_{s}"] = ("E-I connections in general carry the protection" if r >= 0.67 else
                             "specific to the reciprocal loop pairs" if r <= 0.33 else "partly general, partly loop-specific")
    log("\n=== verdict (H15, H16 and the H17 reading rule fixed before the run) ===")
    for k, x in v.items():
        log(f"  {k}: {json.dumps(x)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    log(f"\nstage26 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage26 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
