# stage19.py - two checks asked for by a review of the manuscript draft (log row 102). Frozen model v11, small outputs only.
# =====================================================================================================================
# A. ONSET-SITE ANALYSIS: why does anoxic SD start in the (left) optic lobe?
#    Short death runs that stop ~1 s before the first SD (same seeds and rewiring as stage16), so the K map just
#    before SD is known for every cube. Per cube we compare K with the candidate causes:
#      density  = neuron-equivalents per cube (m_rel): leak (0.3 mM/s x m_rel) and, once pumps fail, the K load
#      release  = K released by firing (0.01 mM per spike x output weight / m_ref), healthy (2 s before O2 off)
#                 and during anoxia (O2 off -> stop)
#    plus left vs right for the optic lobes and the whole brain. Runs: female real seed0 / rewired Wdeg1,
#    male real seed0 / rewired Wdeg1.
# B. SENSITIVITY TO THE SD CLEARANCE CUT (SD_PUMP 0.2 -> 0.35, the upper limit allowed by wave speed; stage15b):
#    real seed0 and rewired Wdeg1 per sex, full 90 s death runs, otherwise frozen settings.
#    PREDICTION fixed before the run: in BOTH sexes real t50 > rewired t50 by >= 2 s at SD_PUMP 0.35.
#    Always reported: the gap as a fraction of the stage16 gap (6.7 s female, 8.7 s male).
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

import runner
import stage16

MODEL = "death_v11"
T_ANOX = runner.T_ANOX_MS
SCALE = float(os.environ.get("STAGE19_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE19_WORKERS", "5"))
SEXES = os.environ.get("STAGE19_SEXES", "female,male").split(",")
SD_PUMP_TEST = 0.35
GAP16 = {"female": 6.74, "male": 8.67}
# stop 1 s before the first SD seen in stage16 (s after O2 off)
STOP_S = {("female", "R"): 17.0, ("female", "W"): 11.5, ("male", "R"): 14.8, ("male", "W"): 12.1}
OPTIC = ("ME", "LO", "LOP", "LA", "AME")


def log(*a):
    print(*a, flush=True)


def side_of(region):
    r = str(region)
    if r.endswith("_L") or r.endswith("(L)"):
        return "L"
    if r.endswith("_R") or r.endswith("(R)"):
        return "R"
    return "-"


def base_of(region):
    r = str(region)
    for s in ("_L", "_R", "(L)", "(R)"):
        if r.endswith(s):
            return r[: -len(s)]
    return r


def module_for(sex, net, sd_pump=None):
    m = runner._fresh_module()
    p = {**stage16.FROZEN, **stage16.SEX[sex]}
    if net == "W":
        p.update(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
    if sd_pump is not None:
        p["SD_PUMP"] = sd_pump
    for k, v in p.items():
        setattr(m, k, v)
    return m


def snapshot(job):
    """Part A: death run stopped ~1 s before the first SD; returns small per-cube arrays."""
    sex, net, out = job["sex"], job["net"], job["out"]
    runner.MODEL_MODULE = MODEL
    runner._DATA.clear()
    time.sleep(job.get("stagger", 0))
    t0 = time.time()
    m = module_for(sex, net)
    m.STOP_EARLY = False
    meta, conn = runner._load(m)
    W, coords, N, vuln, L = runner._prepare(m, meta, conn, 0)
    stop_ms = T_ANOX + STOP_S[(sex, net)] * 1000.0 * SCALE
    res = m.simulate(W, N, vuln, L, T_ANOX, stop_ms)
    # K released per cube by firing (mM/s), from each neuron's rate and its output weights
    O_ptr, O_idx, O_w, m_ref = L["O_ptr"], L["O_idx"], L["O_w"], L["m_ref"]
    C = len(L["m_rel"])
    rows = np.repeat(np.arange(N), np.diff(O_ptr))

    def release(rate):
        return np.bincount(O_idx, weights=rate[rows] * O_w, minlength=C) * m.K_SPIKE_RELEASE / m_ref

    rel_pre = release(res["rate_pre_hz"])
    rel_anox = release(res["rate_post_hz"])
    # optic-lobe neuron counts by side (main neuropil of each neuron), from the data itself
    npl = meta["neuropil"].astype(str)
    nside = pd.DataFrame({"base": npl.map(base_of), "side": npl.map(side_of)})
    counts = nside[nside["base"].isin(OPTIC)].groupby(["base", "side"]).size().to_dict()
    tag = f"{sex}_{net}"
    np.savez_compressed(os.path.join(out, f"kmap_{tag}.npz"), K=res["cube_K_end"].astype(np.float32),
                        m_rel=L["m_rel"].astype(np.float32), rel_pre=rel_pre.astype(np.float32),
                        rel_anox=rel_anox.astype(np.float32), centers=L["centers_um"].astype(np.float32),
                        region=np.asarray(L["cube_region"], dtype=str),
                        sd_started=np.isfinite(res["cube_time"]).astype(np.int8))
    return dict(kind="snap", tag=tag, sex=sex, net=net, stop_s=STOP_S[(sex, net)] * SCALE,
                minutes=round((time.time() - t0) / 60, 1), n_sd_cubes=int(np.isfinite(res["cube_time"]).sum()),
                neuron_counts={f"{b}_{s}": int(v) for (b, s), v in counts.items()},
                rate_pre=float(res["rate_pre_hz"].mean()), rate_anox=float(res["rate_post_hz"].mean()))


def sens(job):
    """Part B: full death run at SD_PUMP 0.35 (otherwise frozen)."""
    sex, net, out = job["sex"], job["net"], job["out"]
    p = {**stage16.FROZEN, **stage16.SEX[sex], "SD_PUMP": SD_PUMP_TEST}
    if net == "W":
        p.update(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
    cond = dict(name=f"{sex}_{net}_pump035", seed=0, overrides=p, out=out,
                dur=90000.0 * SCALE, stagger=job.get("stagger", 0))
    r = stage16.run_one(cond)
    r.update(kind="sens", sex=sex, net=net)
    return r


def smooth(values, centers, radius=30.0):
    """Mean within `radius` um (diffusion spreads K over tens of um within seconds)."""
    tree = cKDTree(centers)
    nb = tree.query_ball_point(centers, radius)
    return np.array([values[i].mean() for i in nb])


def analyse_snap(out, snaps):
    summ = {}
    for s in snaps:
        tag = s["tag"]
        d = np.load(os.path.join(out, f"kmap_{tag}.npz"))
        K, mr, rp, ra, cen, reg = d["K"], d["m_rel"], d["rel_pre"], d["rel_anox"], d["centers"], d["region"]
        load = 0.3 * mr + ra                                       # mM/s added once pumps are gone
        sm = {k: smooth(v, cen) for k, v in dict(m_rel=mr, rel_anox=ra, load=load).items()}
        rho = {k: round(float(spearmanr(K, v)[0]), 3) for k, v in
               dict(m_rel=mr, rel_pre=rp, rel_anox=ra, load=load).items()}
        rho.update({f"{k}_30um": round(float(spearmanr(K, v)[0]), 3) for k, v in sm.items()})
        df = pd.DataFrame(dict(region=reg, base=[base_of(r) for r in reg], side=[side_of(r) for r in reg],
                               K=K, m_rel=mr, rel_pre=rp, rel_anox=ra, load=load))
        reg_tab = (df.groupby(["base", "side"]).agg(n_cubes=("K", "size"), K_median=("K", "median"),
                                                     m_rel_mean=("m_rel", "mean"), rel_anox_mean=("rel_anox", "mean"),
                                                     load_mean=("load", "mean"))
                   .reset_index().sort_values("K_median", ascending=False))
        reg_tab.round(3).to_csv(os.path.join(out, f"onset_regions_{tag}.csv"), index=False)
        hot = df[df["K"] >= np.quantile(df["K"], 0.99)]
        hot_regions = hot["region"].value_counts().head(6).to_dict()
        ratio = {k: round(float(hot[k].mean() / df[k].mean()), 2) for k in ["m_rel", "rel_anox", "load"]}
        opt = df[df["base"].isin(OPTIC)]
        lr = {}
        for part, sub in [("optic", opt), ("whole", df)]:
            g = sub[sub["side"].isin(["L", "R"])].groupby("side")
            lr[part] = {sd: dict(n_cubes=int(len(x)), K_mean=round(float(x["K"].mean()), 3),
                                 m_rel_mean=round(float(x["m_rel"].mean()), 3),
                                 neuron_eq=round(float(x["m_rel"].sum()), 0),
                                 rel_anox_mean=round(float(x["rel_anox"].mean()), 4)) for sd, x in g}
        summ[tag] = dict(stop_s=s["stop_s"], cubes_already_in_SD=s["n_sd_cubes"], K_max=round(float(K.max()), 2),
                         K_mean=round(float(K.mean()), 2), rho_K_vs=rho, hot1pct_regions=hot_regions,
                         hot1pct_vs_mean=ratio, left_right=lr, optic_neurons_by_main_neuropil=s["neuron_counts"],
                         rate_pre_hz=round(s["rate_pre"], 2), rate_anox_hz=round(s["rate_anox"], 2))
        log(f"\n--- {tag} (stopped {s['stop_s']:.1f} s after O2 off; {s['n_sd_cubes']} cubes already in SD) ---")
        log(f"  K mean {K.mean():.2f}, max {K.max():.2f} mM")
        log(f"  Spearman rho, K vs: {rho}")
        log(f"  hottest 1% of cubes: {hot_regions}")
        log(f"  hottest 1% relative to brain mean: {ratio}")
        log(f"  left vs right: {json.dumps(lr)}")
        log(f"  optic neurons by main neuropil: {s['neuron_counts']}")
        log("  top regions by median K:")
        log(reg_tab.head(8).round(3).to_string(index=False))
    with open(os.path.join(out, "onset_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, default=str)


def analyse_sens(out, rows):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "sensitivity_summary.csv"), index=False)
    verdict = {}
    for sex in SEXES:
        def t50(net):
            x = df[(df["sex"] == sex) & (df["net"] == net) & (df["status"] == "ok")]
            return float(x["t50_s"].iloc[0]) if len(x) else np.nan
        r, w = t50("R"), t50("W")
        gap = r - w
        verdict[sex] = dict(SD_PUMP=SD_PUMP_TEST, real_t50=r, rewired_t50=w, gap_s=round(gap, 2),
                            fraction_of_stage16_gap=round(gap / GAP16[sex], 2) if np.isfinite(gap) else None,
                            PASS=bool(np.isfinite(gap) and gap >= 2.0))
    log("\n=== B. sensitivity to the SD clearance cut (prediction: gap >= 2 s in both sexes) ===")
    for sex, v in verdict.items():
        log(f"  {sex}: {v}")
    with open(os.path.join(out, "sensitivity_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1, default=str)


def main():
    t0 = time.time()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_onset_sens")
    os.makedirs(out, exist_ok=True)
    jobs = []
    for sex in SEXES:                      # long runs first, so they start at once
        for net in ["R", "W"]:
            jobs.append(dict(kind="sens", sex=sex, net=net, out=out))
    for sex in SEXES:
        for net in ["R", "W"]:
            jobs.append(dict(kind="snap", sex=sex, net=net, out=out))
    for i, j in enumerate(jobs):
        j["stagger"] = 45 * (i % N_WORKERS) * min(SCALE * 10, 1)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(dict(jobs=jobs, frozen=stage16.FROZEN, sex=stage16.SEX, SD_PUMP_TEST=SD_PUMP_TEST,
                       STOP_S={f"{a}_{b}": v for (a, b), v in STOP_S.items()}), f, indent=1, default=str)
    log(f"{len(jobs)} runs -> {out}")
    snaps, sens_rows = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(sens if j["kind"] == "sens" else snapshot, j): j for j in jobs}
        for fu in as_completed(futs):
            j = futs[fu]
            try:
                r = fu.result()
            except Exception as e:                     # one failure must not lose the others
                log(f"  ERROR in {j['kind']} {j['sex']} {j['net']}: {e!r}")
                continue
            if r["kind"] == "snap":
                snaps.append(r)
                log(f"  done snapshot {r['tag']} ({r['minutes']} min)")
            else:
                sens_rows.append(r)
                log(f"  done {r['tag']} ({r.get('status')}, t50 {r.get('t50_s')})")
    log("\n=== A. onset-site analysis ===")
    if snaps:
        analyse_snap(out, snaps)
    else:
        log("  no snapshot finished")
    if sens_rows:
        analyse_sens(out, sens_rows)
    else:
        log("  no sensitivity run finished")
    log(f"\nstage19 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage19 started", time.strftime("%Y-%m-%d %H:%M"))
    main()
