# stage21.py - follow-up to the third external review (log rows 105-106). Frozen model v11. PC is NOT shut down.
# =====================================================================================================================
# A. H9 GLIAL CAPACITY: after pumps fail, passive glial uptake and diffusion are the only K sinks, so the glial store
#    capacity (frozen 10 mM) may set the timing of SD. Real seed0 and rewired Wdeg1 per sex at GLIA_CAP 5 and 20 mM.
#    H9 (fixed before the run): real t50 - rewired t50 >= 2 s in BOTH sexes at BOTH capacities.
# B. RECEIVER-SPECIFIC E/I-PRESERVING REWIRING (descriptive): which receivers' input balance carries the protection?
#      WsignI : inhibitory neurons keep their numbers of excitatory and inhibitory inputs (receivers permuted within
#               sender-sign groups among connections onto inhibitory neurons); all other connections rewired freely.
#      WsignE : the same for excitatory receivers; connections onto inhibitory (and zero-weight) neurons rewired freely.
#    Every neuron keeps its in- and out-degree in both. 2 rewirings per variant per sex.
#    Measure: recovery = (t50_variant - t50_Wdeg) / (t50_Wsign - t50_Wdeg), with the stage16 Wdeg and stage20 Wsign
#    means. Rule fixed before the run: a variant with recovery >= 0.67 'carries most of the protection'; if both lie
#    between 0.33 and 0.67, both halves contribute; recoveries are always reported.
# Also reported for every run: firing at -40 mV mean resting potential and the slope (Hz per mV) during anoxia.
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
import stage20

MODEL = "death_v11"
SCALE = float(os.environ.get("STAGE21_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE21_WORKERS", "5"))
SEXES = os.environ.get("STAGE21_SEXES", "female,male").split(",")
REAL16 = {"female": [20.56, 20.49, 20.48], "male": [22.74, 22.68, 22.70]}
WDEG16 = {"female": [13.85, 13.77, 13.96, 13.61, 13.70], "male": [13.99, 14.09, 14.22, 13.92, 13.96]}
WSIGN20 = {"female": [18.84, 18.76, 18.72, 18.76, 18.77], "male": [19.14, 19.14, 19.09, 19.02, 19.16]}


def log(*a):
    print(*a, flush=True)


def receiver_conn(meta, conn, seed, keep_sign):
    """Receivers of sign keep_sign keep their input counts per sender sign; all other connections rewired freely."""
    g = np.random.default_rng(seed)
    nt = dict(zip(meta["root_id"].values, meta["nt_type"].fillna("UNK").values))
    pre = conn["pre_root_id"].values
    post = conn["post_root_id"].values.copy()
    s_pre = np.array([stage20.sign_of_nt(nt.get(p, "UNK")) for p in pre], dtype=np.int8)
    s_post = np.array([stage20.sign_of_nt(nt.get(p, "UNK")) for p in post], dtype=np.int8)
    inA = s_post == keep_sign
    groups = [np.flatnonzero(inA & (s_pre == v)) for v in (-1, 0, 1)] + [np.flatnonzero(~inA)]
    gid = np.empty(post.size, np.int64)
    for k, idx in enumerate(groups):
        gid[idx] = k
        post[idx] = post[idx][g.permutation(idx.size)]
    for _ in range(10):
        selfc = np.flatnonzero(post == pre)
        if not selfc.size:
            break
        for i in selfc:
            j = g.choice(groups[gid[i]])
            post[i], post[j] = post[j], post[i]
    out = conn.copy()
    out["post_root_id"] = post
    log(f"  receiver-specific rewiring (keep sign counts for receivers of sign {keep_sign}, seed {seed}): "
        f"{len(conn):,} connections, kept-balance set {int(inA.sum()):,}, free set {int((~inA).sum()):,}, "
        f"self-connections left {int((post == pre).sum())}")
    return out


def run_job(job):
    import importlib
    runner.MODEL_MODULE = MODEL
    runner.RUN_HEALTHY = False
    runner._DATA.clear()
    runner.T_ANOX_MS = 2000.0
    runner.DEATH_MS = job["dur"]
    time.sleep(job.get("stagger", 0))
    m = importlib.import_module(MODEL)
    m._GEOM_CACHE.clear()
    orig = m.load_data
    if job.get("recv") is not None:
        def load_recv():
            meta, conn = orig()
            return meta, receiver_conn(meta, conn, job["seed_rw"], job["recv"])
        m.load_data = load_recv
    ov = dict(job["overrides"])
    ov["SAVE_PROBES_TO"] = os.path.join(job["out"], f"probes_{job['name']}_seed0.npz")
    cond = dict(name=job["name"], seed=0, overrides=ov, out=job["out"])
    try:
        r = runner.run_condition(cond)
    finally:
        m.load_data = orig
        m._GEOM_CACHE.clear()
    r.update(sex=job["sex"], group=job["group"])
    return r


def gain(folder, tag):
    f = os.path.join(folder, f"probes_{tag}.npz")
    if not os.path.exists(f):
        return np.nan, np.nan
    p = np.load(f)
    t = (p["series_t"] - 2000.0) / 1000.0
    a = p["active"] / 1e-4
    v = p["meanV"].astype(float)
    blk = p["blocked"]
    start = np.flatnonzero((t > 0) & (blk > 0.005))
    stop = t[start[0]] - 0.5 if start.size else t[-1]
    rr, vv = [], []
    for b0 in np.arange(0, stop, 0.5):
        m = (t >= b0) & (t < b0 + 0.5)
        if m.any():
            rr.append(a[m].mean()); vv.append(v[m].mean())
    if len(vv) < 3:
        return np.nan, np.nan
    vv, rr = np.array(vv), np.array(rr)
    o = np.argsort(vv)
    return float(np.interp(-40.0, vv[o], rr[o], left=np.nan, right=np.nan)), float(np.polyfit(vv, rr, 1)[0])


def analyse(out, rows):
    df = pd.DataFrame(rows)
    runner.save_summary(df, os.path.join(out, "summary.csv"))
    meas = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        g40, slope = gain(out, r["tag"])
        meas.append(dict(sex=r["sex"], group=r["group"], run=r["tag"], t50_s=r.get("t50_s"),
                         rate_pre_hz=r.get("rate_pre_mean_hz"), rate_at_m40=g40, slope_hz_per_mV=slope))
    md = pd.DataFrame(meas)
    md.to_csv(os.path.join(out, "measures.csv"), index=False)
    log("\n=== runs ===")
    log(md.round(2).to_string(index=False))
    verdict = {}
    for sex in SEXES:
        v = {}
        for cap in ("cap5", "cap20"):
            rr = md[(md.sex == sex) & (md.group == f"R_{cap}")]["t50_s"].astype(float).values
            ww = md[(md.sex == sex) & (md.group == f"W_{cap}")]["t50_s"].astype(float).values
            if rr.size and ww.size:
                gap = float(rr.mean() - ww.mean())
                v[f"H9_{cap}"] = dict(real_t50=round(float(rr.mean()), 2), rewired_t50=round(float(ww.mean()), 2),
                                      gap_s=round(gap, 2), PASS=bool(gap >= 2.0))
        wdeg, wsign = np.mean(WDEG16[sex]), np.mean(WSIGN20[sex])
        for grp in ("WsignI", "WsignE"):
            x = md[(md.sex == sex) & (md.group == grp)]["t50_s"].astype(float).values
            if x.size:
                rec = float((x.mean() - wdeg) / (wsign - wdeg))
                label = ("n/a" if not np.isfinite(rec) else
                         "carries most of the protection" if rec >= 0.67 else
                         "contributes" if rec >= 0.33 else "carries little of the protection")
                v[grp] = dict(t50=[round(float(y), 2) for y in x], recovery=round(rec, 2), reading=label,
                              gap_to_real_s=round(float(np.mean(REAL16[sex]) - x.mean()), 2))
        verdict[sex] = v
    log("\n=== verdict (H9 fixed before the run; B descriptive with a pre-set reading rule) ===")
    for sex, v in verdict.items():
        log(f"  {sex}: {json.dumps(v)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1, default=str)


def main():
    t0 = time.time()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage21")
    os.makedirs(out, exist_ok=True)
    jobs = []
    for sex in SEXES:
        p = {**stage16.FROZEN, **stage16.SEX[sex]}
        for cap in (5.0, 20.0):
            tagc = f"cap{int(cap)}"
            jobs.append(dict(name=f"{sex}_R_{tagc}", sex=sex, group=f"R_{tagc}", overrides=dict(p, GLIA_CAP=cap)))
            jobs.append(dict(name=f"{sex}_W_{tagc}", sex=sex, group=f"W_{tagc}",
                             overrides=dict(p, GLIA_CAP=cap, SHUFFLE_CONN=True, SHUFFLE_MODE="degree",
                                            SHUFFLE_SEED=1)))
        for grp, sign in (("WsignI", -1), ("WsignE", 1)):
            for k in (1, 2):
                jobs.append(dict(name=f"{sex}_{grp}{k}", sex=sex, group=grp, overrides=dict(p), recv=sign,
                                 seed_rw=k))
    for i, j in enumerate(jobs):
        j.update(out=out, dur=90000.0 * SCALE, stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1))
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jobs, f, indent=1, default=str)
    log(f"{len(jobs)} runs -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                log(f"  ERROR in {j['name']}: {e!r}")
                continue
            rows.append(r)
            log(f"  done {r['tag']} ({r.get('status')}, t50 {r.get('t50_s')})")
            runner.save_summary(pd.DataFrame(rows), os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r.get('error')}")
    analyse(out, rows)
    log(f"\nstage21 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage21 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
