# stage20.py - follow-up to the second external review (log row 103). Frozen model v11. Starts after stage19.
# =====================================================================================================================
# A. NEW MEASURES on the stage16 main runs (no new simulation): t50 of central-brain neurons only (optic-lobe neurons
#    are ~2/3 of FlyWire), t50 of optic-lobe neurons, time until 50% of cubes are in SD, and the K surge time at the
#    virtual electrodes (median over electrodes of the first time K > 20 mM; comparable with fly K electrodes).
# B. SIGN-STRATIFIED REWIRING ('Wsign'): receivers are permuted separately among connections from excitatory,
#    inhibitory and zero-weight senders. Every neuron keeps its number of outputs (weights, sign) AND its numbers of
#    excitatory and inhibitory inputs; only who they come from changes. 5 rewirings per sex.
#    H6 (fixed before the run): if every Wsign t50 < every stage16 real t50 with mean gap >= 2 s, the protection is NOT
#    explained by each neuron's E/I input balance. If Wsign dies like the real brain (gap < 2 s), the protection comes
#    from the E/I input balance and Discussion 4.2 says so. Always reported: fraction of the stage16 gap that remains.
# C. 20-s WARM-UP: oxygen cut at 20 s instead of 2 s (firing, K and the glial store closer to steady state), real
#    seed0 and rewired Wdeg1 per sex. H7 (fixed before the run): real t50 > rewired t50 by >= 2 s in both sexes.
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
SCALE = float(os.environ.get("STAGE20_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE20_WORKERS", "5"))
SEXES = os.environ.get("STAGE20_SEXES", "female,male").split(",")
WAIT_FOR = os.environ.get("STAGE20_WAIT", "stage19_log.txt")
OPTIC = {"ME", "LO", "LOP", "LA", "AME"}
GAP16 = {"female": 6.74, "male": 8.67}
REAL16 = {"female": [20.56, 20.49, 20.48], "male": [22.74, 22.68, 22.70]}


def log(*a):
    print(*a, flush=True)


def base_of(region):
    r = str(region)
    for s in ("_L", "_R", "(L)", "(R)"):
        if r.endswith(s):
            return r[: -len(s)]
    return r


def wait_for_stage19():
    if not WAIT_FOR or not os.path.exists(WAIT_FOR):
        return
    log(f"waiting for {WAIT_FOR} to report 'stage19 finished' ...")
    t0 = time.time()
    while time.time() - t0 < 8 * 3600:
        try:
            with open(WAIT_FOR, encoding="utf-8", errors="ignore") as f:
                if "stage19 finished" in f.read():
                    log(f"stage19 finished; starting stage20 at {time.strftime('%H:%M')}")
                    time.sleep(30)
                    return
        except OSError:
            pass
        time.sleep(60)
    log("waited 8 h for stage19; starting anyway")


def module_params(sex):
    return {**stage16.FROZEN, **stage16.SEX[sex]}


def sign_of_nt(nt):
    return {"ACH": 1, "GABA": -1, "GLUT": -1}.get(str(nt), 0)


def stratified_conn(meta, conn, seed):
    """Permute receivers within groups of connections whose senders share a sign."""
    g = np.random.default_rng(seed)
    nt = dict(zip(meta["root_id"].values, meta["nt_type"].fillna("UNK").values))
    pre = conn["pre_root_id"].values
    post = conn["post_root_id"].values.copy()
    s = np.array([sign_of_nt(nt.get(p, "UNK")) for p in pre], dtype=np.int8)
    groups = {v: np.flatnonzero(s == v) for v in np.unique(s)}
    for v, k in groups.items():
        post[k] = post[k][g.permutation(k.size)]
    for _ in range(10):                                   # remove self-connections by swaps inside the same group
        selfc = np.flatnonzero(post == pre)
        if not selfc.size:
            break
        for i in selfc:
            j = g.choice(groups[s[i]])
            post[i], post[j] = post[j], post[i]
    out = conn.copy()
    out["post_root_id"] = post
    n_self = int((post == pre).sum())
    log(f"  sign-stratified rewiring (seed {seed}): {len(conn):,} connections, groups "
        f"{ {int(v): int(k.size) for v, k in groups.items()} }, self-connections left {n_self}")
    return out


def run_job(job):
    import importlib
    runner.MODEL_MODULE = MODEL
    runner.RUN_HEALTHY = False
    runner._DATA.clear()
    runner.T_ANOX_MS = job["t_anox"]
    runner.DEATH_MS = job["dur"]
    time.sleep(job.get("stagger", 0))
    m = importlib.import_module(MODEL)
    m._GEOM_CACHE.clear()
    orig = m.load_data
    if job.get("strat_seed") is not None:
        def load_strat():
            meta, conn = orig()
            return meta, stratified_conn(meta, conn, job["strat_seed"])
        m.load_data = load_strat
    ov = dict(job["overrides"])
    ov["SAVE_PROBES_TO"] = os.path.join(job["out"], f"probes_{job['name']}_seed0.npz")
    cond = dict(name=job["name"], seed=0, overrides=ov, out=job["out"])
    try:
        r = runner.run_condition(cond)
    finally:
        m.load_data = orig
        m._GEOM_CACHE.clear()
    r.update(sex=job["sex"], group=job["group"], t_anox_ms=job["t_anox"])
    return r


def measures(sex, folder, tag, t_anox, optic_mask):
    """t50 (all / central / optic), 50% of cubes in SD, median electrode surge time; s after O2 off."""
    def q50(x):
        x = np.where(np.isfinite(x), x, np.inf)
        v = np.quantile(x, 0.5)
        return float(v) if np.isfinite(v) else np.nan
    out = dict(sex=sex, run=tag)
    f = os.path.join(folder, f"block_time_{tag}.npy")
    if os.path.exists(f):
        bt = (np.load(f) - t_anox) / 1000.0
        if bt.size == optic_mask.size:
            out.update(t50_all=q50(bt), t50_central=q50(bt[~optic_mask]), t50_optic=q50(bt[optic_mask]))
    f = os.path.join(folder, f"cube_time_{tag}.npy")
    if os.path.exists(f):
        out["cubes50_s"] = q50((np.load(f) - t_anox) / 1000.0)
    f = os.path.join(folder, f"probes_{tag}.npz")
    if os.path.exists(f):
        p = np.load(f)
        t = (p["t"] - t_anox) / 1000.0
        K = p["K"]
        first = [t[np.argmax(K[:, j] > 20.0)] if (K[:, j] > 20.0).any() else np.nan for j in range(K.shape[1])]
        out["electrode_surge_s"] = float(np.nanmedian(first))
    return out


def optic_masks():
    import importlib
    masks = {}
    for sex in SEXES:
        runner.MODEL_MODULE = MODEL
        runner._DATA.clear()
        m = runner._fresh_module()
        for k, v in module_params(sex).items():
            setattr(m, k, v)
        meta, conn = m.load_data()
        masks[sex] = meta["neuropil"].astype(str).map(base_of).isin(OPTIC).values
        log(f"  {sex}: {masks[sex].sum():,} of {masks[sex].size:,} neurons have an optic-lobe main neuropil "
            f"({masks[sex].mean():.0%})")
        del meta, conn
    runner._DATA.clear()
    return masks


def part_a(out, masks):
    log("\n=== A. new measures on the stage16 main runs ===")
    main = sorted(glob.glob(os.path.join(runner.OUT_DIR, "*_v11_main")))
    if not main:
        log("  stage16 folder not found")
        return
    folder = main[-1]
    rows = []
    for sex in SEXES:
        tags = ([f"{sex}_R_seed{s}" for s in range(3)] + [f"{sex}_Wdeg{k}_seed0" for k in range(1, 6)]
                + [f"{sex}_EI{k}_seed0" for k in (1, 2)])
        for tag in tags:
            r = measures(sex, folder, tag, runner.T_ANOX_MS, masks[sex])
            r["group"] = tag.split("_")[1].rstrip("0123456789")
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "main_runs_new_measures.csv"), index=False)
    log(df.round(2).to_string(index=False))
    g = df.groupby(["sex", "group"]).mean(numeric_only=True).round(2)
    log("\nmeans:\n" + g.to_string())
    for sex in SEXES:
        try:
            R, W = g.loc[(sex, "R")], g.loc[(sex, "Wdeg")]
            log(f"  {sex} real - rewired gap: " + ", ".join(f"{c} {R[c] - W[c]:+.2f} s" for c in g.columns))
        except KeyError:
            pass


def analyse(out, rows, masks):
    df = pd.DataFrame(rows)
    runner.save_summary(df, os.path.join(out, "summary.csv"))
    meas = []
    for r in rows:
        if r.get("status") == "ok":
            mm = measures(r["sex"], out, r["tag"], r["t_anox_ms"], masks[r["sex"]])
            mm.update(group=r["group"], t50_s=r.get("t50_s"), rate_pre_hz=r.get("rate_pre_mean_hz"))
            meas.append(mm)
    md = pd.DataFrame(meas)
    md.to_csv(os.path.join(out, "new_runs_measures.csv"), index=False)
    log("\n=== B/C. runs ===")
    log(md.round(2).to_string(index=False))
    verdict = {}
    for sex in SEXES:
        v = {}
        ws = md[(md["sex"] == sex) & (md["group"] == "Wsign")]["t50_s"].astype(float).values
        real = np.array(REAL16[sex])
        if ws.size:
            gap = float(real.mean() - ws.mean())
            v["H6"] = dict(Wsign_t50=[round(x, 2) for x in ws], real_t50_stage16=list(real), gap_s=round(gap, 2),
                           fraction_of_stage16_gap=round(gap / GAP16[sex], 2),
                           PASS=bool(ws.max() < real.min() and gap >= 2.0))
        wr = md[(md["sex"] == sex) & (md["group"] == "R_warm20")]["t50_s"].astype(float).values
        ww = md[(md["sex"] == sex) & (md["group"] == "W_warm20")]["t50_s"].astype(float).values
        if wr.size and ww.size:
            gap = float(wr.mean() - ww.mean())
            v["H7"] = dict(real_t50=round(float(wr.mean()), 2), rewired_t50=round(float(ww.mean()), 2),
                           gap_s=round(gap, 2), fraction_of_stage16_gap=round(gap / GAP16[sex], 2),
                           PASS=bool(gap >= 2.0))
        verdict[sex] = v
    log("\n=== hypotheses (fixed before the run) ===")
    for sex, v in verdict.items():
        log(f"  {sex}: {json.dumps(v)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1, default=str)


def main():
    wait_for_stage19()
    t0 = time.time()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage20")
    os.makedirs(out, exist_ok=True)
    log("\n=== optic-lobe neurons ===")
    masks = optic_masks()
    part_a(out, masks)
    jobs = []
    for sex in SEXES:
        p = module_params(sex)
        for warm, net in [("R_warm20", "R"), ("W_warm20", "W")]:
            ov = dict(p)
            if net == "W":
                ov.update(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
            jobs.append(dict(name=f"{sex}_{warm}", sex=sex, group=warm, overrides=ov,
                             t_anox=20000.0 * SCALE, dur=(20000.0 + 88000.0) * SCALE))
        for k in range(1, 6):
            jobs.append(dict(name=f"{sex}_Wsign{k}", sex=sex, group="Wsign", overrides=dict(p), strat_seed=k,
                             t_anox=2000.0, dur=90000.0 * SCALE))
    for i, j in enumerate(jobs):
        j.update(out=out, stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1))
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(jobs, f, indent=1, default=str)
    log(f"\n{len(jobs)} runs -> {out}")
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
    analyse(out, rows, masks)
    log(f"\nstage20 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage20 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
