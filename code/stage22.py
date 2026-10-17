# stage22.py - fourth external review (log row 107). Frozen model v11. Runs after stage21 (same .bat). PC NOT shut down.
# =====================================================================================================================
# A. CHECKS FROM EXISTING DATA (no simulation):
#    A1 zero-weight neurons per transmitter, both sexes.
#    A2 photoreceptors: model photoreceptors (histaminergic, zero synaptic weight) still spike and release K, unlike
#       real non-spiking photoreceptors. Share of K release from histaminergic and from all zero-weight neurons in the
#       onset cubes (earliest 1% of SD arrival, stage16 real seed0) vs the whole brain; healthy and anoxic rates.
#    A3 loops: share of each neuron's input synapses that come from E-I loop connections; loop vs non-loop
#       connection strength (synapses per connection).
#    A4 male cube density: keep a random 364/472 = 77% of male synapses (binomial thinning of the per-cube counts) and
#       recount the cubes; if the count falls to about the female's, the 25% excess is a dataset difference.
#    A5 left-right: per optic region and side, cubes, neurons (main neuropil) and synapses, both sexes.
# B. GLUTAMATE NEUTRAL (H10): glutamatergic synapses given weight 0 instead of -1 (the paper's answer now rests on
#    each neuron's E/I balance, and much of the 'I' is glutamate). Drive re-tuned so the real brain fires at its frozen
#    steady healthy rate (female 11.10 Hz, male 12.64 Hz; 20-s healthy runs, linear interpolation, up to 3 extra
#    rounds); rewired brains use the same drive (as in the main experiment). Death runs: real seed0, rewired Wdeg1 and
#    Wdeg2 per sex. H10 (fixed before the run): real t50 - mean rewired t50 >= 2 s in both sexes.
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
SCALE = float(os.environ.get("STAGE22_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE22_WORKERS", "5"))
SEXES = os.environ.get("STAGE22_SEXES", "female,male").split(",")
WAIT_FOR = os.environ.get("STAGE22_WAIT", "")
CAL_MS = 20000.0
TARGET = {"female": 11.10, "male": 12.64}
OPTIC = {"ME", "LO", "LOP", "LA", "AME"}


def log(*a):
    print(*a, flush=True)


def fresh(sex, extra=None):
    runner.MODEL_MODULE = MODEL
    runner._DATA.clear()
    m = runner._fresh_module()
    for k, v in {**stage16.FROZEN, **stage16.SEX[sex], **(extra or {})}.items():
        setattr(m, k, v)
    m._GEOM_CACHE.clear()
    return m


def side_of(r):
    r = str(r)
    return "L" if r.endswith(("_L", "(L)")) else "R" if r.endswith(("_R", "(R)")) else "-"


# ---------------------------------------------------------------------------------------------------- part A
def part_a(out):
    main = sorted(glob.glob(os.path.join(runner.OUT_DIR, "*_v11_main")))[-1]
    res = {}
    for sex in SEXES:
        log(f"\n--- {sex} ---")
        m = fresh(sex)
        meta, conn = m.load_data()
        r = {}
        nt = meta["nt_type"].fillna("UNK")
        r["A1_neurons_by_transmitter"] = nt.value_counts().to_dict()
        sign = nt.map(stage20.sign_of_nt).values
        r["A1_zero_weight_neurons"] = int((sign == 0).sum())
        log(f"A1 transmitters: {r['A1_neurons_by_transmitter']}  zero-weight: {r['A1_zero_weight_neurons']:,}")
        # A3 loops
        idx = {rid: i for i, rid in enumerate(meta["root_id"].values)}
        pre = conn["pre_root_id"].map(idx).values; post = conn["post_root_id"].map(idx).values
        syn = conn["syn_count"].values.astype(float)
        N = len(meta)
        key = pre.astype(np.int64) * N + post; rev = post.astype(np.int64) * N + pre
        recip = np.isin(rev, key)
        sp, so = sign[pre], sign[post]
        loop = recip & (((sp > 0) & (so < 0)) | ((sp < 0) & (so > 0)))
        tot_in = np.bincount(post, weights=syn, minlength=N)
        loop_in = np.bincount(post, weights=syn * loop, minlength=N)
        has = tot_in > 0
        share = np.where(has, loop_in / np.maximum(tot_in, 1), np.nan)
        r["A3_loop_connections"] = int(loop.sum())
        r["A3_loop_share_of_all_input_synapses"] = round(float(syn[loop].sum() / syn.sum()), 3)
        r["A3_syn_per_connection_loop_vs_other"] = [round(float(syn[loop].mean()), 2), round(float(syn[~loop].mean()), 2)]
        r["A3_median_loop_share_per_neuron"] = {"excitatory": round(float(np.nanmedian(share[sign > 0])), 3),
                                                "inhibitory": round(float(np.nanmedian(share[sign < 0])), 3)}
        r["A3_mean_loop_share_per_neuron"] = {"excitatory": round(float(np.nanmean(share[sign > 0])), 3),
                                              "inhibitory": round(float(np.nanmean(share[sign < 0])), 3)}
        log(f"A3 loops: {r['A3_loop_connections']:,} connections, {r['A3_loop_share_of_all_input_synapses']:.1%} of "
            f"input synapses; synapses/connection loop {r['A3_syn_per_connection_loop_vs_other'][0]} vs other "
            f"{r['A3_syn_per_connection_loop_vs_other'][1]}; median per-neuron loop share {r['A3_median_loop_share_per_neuron']}"
            f"; mean {r['A3_mean_loop_share_per_neuron']}")
        # A2 photoreceptors / zero-weight release in onset cubes
        m.rng = np.random.default_rng(0)
        idx2, W, sgn, coords, N2 = m.build_network(meta, conn)
        L = m.build_geometry(meta, idx2, coords)
        f_rates = os.path.join(main, f"rates_{sex}_R_seed0.npz")
        f_ct = os.path.join(main, f"cube_time_{sex}_R_seed0.npy")
        if os.path.exists(f_rates) and os.path.exists(f_ct):
            rt = np.load(f_rates); ct = np.load(f_ct)
            C = len(L["m_rel"]); rows = np.repeat(np.arange(N2), np.diff(L["O_ptr"]))
            hist = (nt.values == "HIST"); zero = (sign == 0)
            onset = np.argsort(np.where(np.isfinite(ct), ct, np.inf))[: max(1, C // 100)]
            out_r = {}
            for phase in ("pre", "post"):
                rate = rt[phase].astype(float)
                def rel(mask):
                    return np.bincount(L["O_idx"], weights=(rate * mask)[rows] * L["O_w"], minlength=C)
                allr, hr, zr = rel(np.ones(N2)), rel(hist.astype(float)), rel(zero.astype(float))
                out_r[phase] = dict(hist_share_onset=round(float(hr[onset].sum() / allr[onset].sum()), 3),
                                    hist_share_brain=round(float(hr.sum() / allr.sum()), 3),
                                    zero_share_onset=round(float(zr[onset].sum() / allr[onset].sum()), 3),
                                    zero_share_brain=round(float(zr.sum() / allr.sum()), 3),
                                    hist_mean_rate_hz=round(float(rate[hist].mean()), 2) if hist.any() else None,
                                    all_mean_rate_hz=round(float(rate.mean()), 2))
            r["A2_release_shares"] = out_r
            log(f"A2 K release shares (pre = healthy, post = anoxia to end of run): {json.dumps(out_r)}")
        # A5 left-right
        raw = pd.read_csv(m.CONNECTIONS_CSV, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
        syn_np = raw.groupby("neuropil")["syn_count"].sum()
        reg = pd.Series(np.asarray(L["cube_region"], dtype=str))
        npl = meta["neuropil"].astype(str)
        lr = {}
        for region in sorted(set(reg.map(stage20.base_of)) & OPTIC):
            for sd in ("L", "R"):
                names = [n for n in set(reg) | set(syn_np.index.astype(str)) if stage20.base_of(n) == region and side_of(n) == sd]
                lr[f"{region}_{sd}"] = dict(cubes=int(reg.isin(names).sum()), neurons=int(npl.isin(names).sum()),
                                            synapses=int(syn_np[syn_np.index.astype(str).isin(names)].sum()))
        r["A5_left_right"] = lr
        log(f"A5 left/right optic: {json.dumps(lr)}")
        # A4 cube count after synapse thinning (male) / reference (female)
        d = np.load(m.CUBE_FILE)
        file_ids = d["root_ids"]; model_ids = meta["root_id"].values.astype(np.int64)
        pos = np.clip(np.searchsorted(file_ids, model_ids), 0, len(file_ids) - 1)
        found = file_ids[pos] == model_ids
        to_model = np.full(len(file_ids), -1, np.int64); to_model[pos[found]] = np.flatnonzero(found)
        n_all = np.concatenate([to_model[d["pre_neuron"]], to_model[d["post_neuron"]]])
        c_all = np.concatenate([d["pre_cube"], d["post_cube"]]).astype(np.int64)
        w_all = np.concatenate([d["pre_count"], d["post_count"]]).astype(np.int64)
        ok = n_all >= 0
        n_all, c_all, w_all = n_all[ok], c_all[ok], w_all[ok]
        cnt = {}
        for frac in ((1.0, 364 / 472) if sex == "male" else (1.0,)):
            w = np.random.default_rng(0).binomial(w_all, frac) if frac < 1 else w_all
            k = w > 0
            ptr, cc, ww = m._top_cubes(n_all[k], c_all[k], w[k].astype(float), N2)
            cnt[f"keep_{frac:.2f}"] = int(np.unique(cc).size)
        r["A4_cube_counts"] = cnt
        log(f"A4 cube counts (presence sets, model neurons): {cnt}")
        res[sex] = r
        del meta, conn, raw, W, L, d
    with open(os.path.join(out, "partA_checks.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)


# ---------------------------------------------------------------------------------------------------- part B
def glut_neutral(m):
    orig = m.load_data

    def load():
        meta, conn = orig()
        meta = meta.copy()
        n = int((meta["nt_type"] == "GLUT").sum())
        meta.loc[meta["nt_type"] == "GLUT", "nt_type"] = "GLUT0"      # maps to weight 0 in build_network
        print(f"  glutamate neutral: {n:,} glutamatergic neurons given synaptic weight 0", flush=True)
        return meta, conn
    m.load_data = load
    return orig


def calib_one(job):
    time.sleep(job.pop("stagger", 0))
    m = fresh(job["sex"], {"TONIC_MEAN": job["drive"]})
    orig = glut_neutral(m)
    try:
        meta, conn = runner._load(m)
        W, coords, N, vuln, L = runner._prepare(m, meta, conn, 0)
        m.RUN_MODE = "healthy"
        dur = CAL_MS * SCALE
        r = m.simulate(W, N, vuln, L, 1e12, dur)
    finally:
        m.load_data = orig
    late = r["t"] >= dur / 2
    return dict(sex=job["sex"], drive=job["drive"], rate_hz=float(r["active"][late].mean() / (m.DT / 1000.0)),
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
    orig = glut_neutral(m)
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
        futs = {ex.submit(fn, dict(j, stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1))): j for i, j in enumerate(jobs)}
        for fu in as_completed(futs):
            try:
                r = fu.result()
            except Exception as e:
                log(f"  ERROR: {e!r}")
                continue
            res.append(r)
            log(f"  {r.get('sex')} " + (f"drive {r['drive']:.3f} -> {r['rate_hz']:.2f} Hz (blocked {r['blocked_end']:.3f},"
                                         f" K max {r['K_max']:.2f})" if "rate_hz" in r else
                                         f"{r.get('tag')} ({r.get('status')}, t50 {r.get('t50_s')})"))
    return res


def part_b(out):
    cal = []
    drives = {}
    for sex in SEXES:
        f = stage16.SEX[sex]["TONIC_MEAN"]
        cal += [dict(sex=sex, drive=round(f - d, 3)) for d in (0.0, 1.0, 2.0)]
    log("\n=== B. calibration (glutamate neutral, real wiring) ===")
    rows = pool(cal, calib_one)
    for rnd in range(3):
        c = pd.DataFrame(rows)
        todo = []
        for sex in SEXES:
            cs = c[(c.sex == sex) & (c.blocked_end == 0)].sort_values("drive")
            x, y = cs.drive.values, cs.rate_hz.values
            if len(x) >= 2 and y.min() <= TARGET[sex] <= y.max():
                o = np.argsort(y)
                drives[sex] = float(np.interp(TARGET[sex], y[o], x[o]))
            elif len(x) >= 2:
                k = np.argsort(np.abs(y - TARGET[sex]))[:2]
                (x1, x2), (y1, y2) = x[k], y[k]
                g = x1 + (TARGET[sex] - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1 - 0.5
                todo += [dict(sex=sex, drive=float(np.clip(round(g + dd, 3), 0.2, 10.0))) for dd in (-0.3, 0.3)]
            else:
                todo += [dict(sex=sex, drive=round(float(cs.drive.min() if len(x) else 1.0) - 1.0, 3))]
        todo = [t for t in todo if t["sex"] not in drives]
        if not todo:
            break
        log(f"  round {rnd + 2}: {todo}")
        rows += pool(todo, calib_one)
    pd.DataFrame(rows).to_csv(os.path.join(out, "glut0_calibration.csv"), index=False)
    log(f"  drives (glutamate neutral): {drives}")
    with open(os.path.join(out, "glut0_drives.json"), "w") as f:
        json.dump(drives, f, indent=1)
    jobs = []
    for sex, drv in drives.items():
        p = {**stage16.FROZEN, **stage16.SEX[sex], "TONIC_MEAN": drv}
        jobs.append(dict(name=f"{sex}_R_glut0", sex=sex, group="R_glut0", overrides=dict(p), out=out))
        for k in (1, 2):
            jobs.append(dict(name=f"{sex}_Wdeg{k}_glut0", sex=sex, group="W_glut0", out=out,
                             overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=k)))
    log(f"\n=== B. death runs: {len(jobs)} ===")
    res = pool(jobs, death_one)
    df = pd.DataFrame(res)
    runner.save_summary(df, os.path.join(out, "summary.csv"))
    verdict = {}
    for sex in drives:
        rr = df[(df.sex == sex) & (df.group == "R_glut0") & (df.status == "ok")]["t50_s"].astype(float)
        ww = df[(df.sex == sex) & (df.group == "W_glut0") & (df.status == "ok")]["t50_s"].astype(float)
        if len(rr) and len(ww):
            gap = float(rr.mean() - ww.mean())
            verdict[sex] = dict(drive=round(drives[sex], 3), real_t50=round(float(rr.mean()), 2),
                                rewired_t50=[round(float(x), 2) for x in ww], gap_s=round(gap, 2),
                                rate_pre_real=round(float(df[(df.sex == sex) & (df.group == 'R_glut0')]['rate_pre_mean_hz'].mean()), 2),
                                rate_pre_rewired=round(float(df[(df.sex == sex) & (df.group == 'W_glut0')]['rate_pre_mean_hz'].mean()), 2),
                                H10_PASS=bool(gap >= 2.0))
    log("\n=== H10 (fixed before the run): gap >= 2 s in both sexes with glutamate neutral ===")
    for sex, v in verdict.items():
        log(f"  {sex}: {json.dumps(v)}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)


def wait():
    if not WAIT_FOR or not os.path.exists(WAIT_FOR):
        return
    log(f"waiting for {WAIT_FOR} to report 'finished' ...")
    t0 = time.time()
    while time.time() - t0 < 10 * 3600:
        with open(WAIT_FOR, encoding="utf-8", errors="ignore") as f:
            if "stage21 finished" in f.read():
                return
        time.sleep(60)


def main():
    wait()
    t0 = time.time()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_stage22")
    os.makedirs(out, exist_ok=True)
    log(f"-> {out}\n=== A. checks from existing data ===")
    try:
        part_a(out)
    except Exception as e:
        import traceback
        log("part A failed:\n" + traceback.format_exc())
    part_b(out)
    log(f"\nstage22 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage22 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
