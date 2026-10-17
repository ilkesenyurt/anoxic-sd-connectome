# stage6.py - v4 calibration (fast, tissue only) + final full-brain batch
# =======================================================================
# A. Tissue-only mode (cube layer without neurons, ~seconds per run):
#    A0. check it against the full model: rerun today's trigger runs (SD_ON 11, SD_RATE 55 and 40,
#        no delay, 1 s trigger) and compare with the full runs in runs/*_sweep_rate
#    A1. wave speed: diffusion D x SD onset delay (trigger held 6 s); pick the setting closest to
#        0.77 mm/min (Harris et al. 2026), preferring the physical D = 770 um^2/s
#    A2. KCl bath through the brain surface: where and when SD starts
#    A3. why the left lamina went first: tissue density / surface share per region
# B. Full model (neurons + cubes), N_WORKERS at a time, early stop, chosen setting:
#    trigger check, surface bath, oxygen loss R_real x3, S_swap x2, W_rewired x2
# C. validate_v3.py on the new folder (run by the .bat afterwards)
# Everything is printed to stage6_log.txt; files go to runs/<date_time>_v4_*.

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import runner

SCALE = float(os.environ.get("STAGE6_SCALE", "1"))   # <1 only for quick tests
N_WORKERS = int(os.environ.get("STAGE6_WORKERS", "5"))
MODEL = "death_v4"
TARGET = 0.77                                        # mm/min


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- A
def tissue_part():
    import death_v4 as m
    defaults = {k: getattr(m, k) for k in dir(m) if k.isupper()}

    def setp(**kw):
        for k, v in defaults.items():
            setattr(m, k, v)
        for k, v in kw.items():
            assert hasattr(m, k), k
            setattr(m, k, v)

    meta, conn = m.load_data()
    idx, W, sign, coords, N = m.build_network(meta, conn)
    geom = m.build_geometry(meta, idx, coords)
    del W, conn

    def trig(dur_s=60.0, **kw):
        setp(RUN_MODE="trigger", **kw)
        t0 = time.time()
        r = m.simulate_tissue(geom, dur_s * 1000.0 * SCALE)
        ct = r["cube_time"]
        reached = float(np.isfinite(ct).mean())
        sp = m.cube_speed_mm_min(ct, geom) if np.isfinite(ct).sum() > 10 else np.nan
        return dict(reached=reached, speed=sp, stop_s=float(r["t"][-1]) / 1000,
                    K_base=float(np.median(r["cube_K_end"][~np.isfinite(ct)])) if (~np.isfinite(ct)).any() else np.nan,
                    sec=time.time() - t0), ct

    # A0 ------------------------------------------------------------------
    log("\n=== A0. tissue-only vs full model (trigger, SD_ON 11, no delay, 1 s trigger) ===")
    full_dir = sorted(glob.glob(os.path.join("runs", "*_sweep_rate")))
    ok_check = True
    checks = []
    for rate in [55.0, 40.0]:
        res, ct = trig(dur_s=40.0, SD_RATE=rate, SD_DELAY_S=0.0, TRIGGER_HOLD_MS=1000.0, STOP_EARLY=False)
        line = f"rate {rate:.0f}: tissue reached {res['reached']:.1%}, speed {res['speed']:.2f} mm/min, " \
               f"healthy K {res['K_base']:.2f} ({res['sec']:.1f} s to run)"
        f = os.path.join(full_dir[-1], f"cube_time_trig_on11_rate{rate:.0f}_seed0.npy") if full_dir else ""
        if f and os.path.exists(f):
            cf = np.load(f)
            spf = m.cube_speed_mm_min(cf, geom) if np.isfinite(cf).sum() > 10 else np.nan
            both = np.isfinite(cf) & np.isfinite(ct)
            rho = spearmanr(cf[both], ct[both])[0] if both.sum() > 10 else np.nan
            dt = np.median(np.abs(cf[both] - ct[both])) / 1000 if both.any() else np.nan
            line += f" | full reached {np.isfinite(cf).mean():.1%}, speed {spf:.2f}; " \
                    f"arrival rho {rho:.3f}, median |dt| {dt:.2f} s"
            same_outcome = (np.isfinite(cf).mean() > 0.5) == (res["reached"] > 0.5)
            close = (res["reached"] < 0.5) or (abs(res["speed"] - spf) / spf < 0.15 and rho > 0.95)
            ok_check &= bool(same_outcome and close)
            checks.append(dict(rate=rate, tissue_reached=res["reached"], full_reached=float(np.isfinite(cf).mean()),
                               tissue_speed=res["speed"], full_speed=spf, rho=rho, median_dt_s=dt))
        else:
            line += " | full run not found"
            ok_check = False
        log(line)
    log("tissue-only mode matches the full model" if ok_check else
        "WARNING: tissue-only mode does NOT match the full model; calibration below is only a guide")

    # A1 ------------------------------------------------------------------
    log("\n=== A1. wave speed: diffusion D x SD onset delay (SD_ON 11, rate 55, trigger 6 s) ===")
    rows = []
    for D in [770.0, 500.0, 300.0]:
        for delay in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            res, _ = trig(dur_s=120.0, D_UM2_S=D, SD_DELAY_S=delay, TRIGGER_HOLD_MS=6000.0)
            rows.append(dict(D=D, delay=delay, **res))
            log(f"D {D:4.0f}  delay {delay:3.1f} s: reached {res['reached']:6.1%}  speed {res['speed']:5.2f} mm/min"
                f"  (stopped at {res['stop_s']:5.1f} s, {res['sec']:.0f} s to run)")
    sweep = pd.DataFrame(rows)
    good = sweep[sweep.reached > 0.9].copy()
    if not len(good):
        raise SystemExit("no setting spread through the brain; stopping for a rethink")
    good["err"] = (good.speed - TARGET).abs()
    near = good[good.err <= 0.1]
    pick = (near.sort_values(["D", "err"], ascending=[False, True]) if len(near) else good.sort_values("err")).iloc[0]
    chosen = dict(D_UM2_S=float(pick.D), SD_DELAY_S=float(pick.delay))
    log(f"chosen: D {pick.D:.0f} um^2/s, delay {pick.delay:.1f} s -> {pick.speed:.2f} mm/min (tissue only)")

    # A2 ------------------------------------------------------------------
    log("\n=== A2. KCl bath (12.9 mM) through the brain surface, tissue only ===")
    cube_region = region_of_cubes(m, meta, geom)
    for G in [1.0, 3.0, 10.0]:
        for ramp in [0.0, 15.0]:
            setp(RUN_MODE="bath", BATH_MODE="surface", BATH_G_S=G, BATH_RAMP_S=ramp, **chosen)
            r = m.simulate_tissue(geom, 90000.0 * SCALE)
            ct = r["cube_time"]; okc = np.isfinite(ct)
            if not okc.any():
                log(f"G {G:4.1f}/s, ramp {ramp:2.0f} s: no SD")
                continue
            first = (np.nanmin(ct) - m.BATH_T_MS) / 1000
            early = okc & (ct <= np.nanmin(ct) + 1000)
            regs = pd.Series(cube_region[early]).value_counts().head(4)
            log(f"G {G:4.1f}/s, ramp {ramp:2.0f} s: first SD {first:5.1f} s after bath (fly 14 +- 4.4), "
                f"reached {okc.mean():.1%}; first-second cubes {early.sum()} in "
                + ", ".join(f"{k} {v}" for k, v in regs.items()))
    log("(note: with constant pumps a surface cube either never reaches SD_ON or does so within ~the delay,"
        " so the 14 s in flies is mostly how fast the bath reaches the tissue - not a calibration target)")

    # A3 ------------------------------------------------------------------
    log("\n=== A3. tissue density by region (why did LA_L go first?) ===")
    surf = np.zeros(len(geom["m_rel"]), bool); surf[geom["surf_idx"]] = True
    df = pd.DataFrame({"r": cube_region, "m": geom["m_rel"], "surf": surf,
                       "nb": (geom["nb"] >= 0).sum(1)})
    g = df.groupby("r").agg(cubes=("m", "size"), density=("m", "median"), sparse=("m", lambda x: (x < 0.2).mean()),
                            surface=("surf", "mean"), neighbours=("nb", "mean"))
    show = [r for r in ["LA_L", "LA_R", "ME_L", "ME_R", "LO_L", "LO_R", "MB_CA_L", "MB_CA_R", "AL_L", "AL_R", "none"]
            if r in g.index]
    log(g.loc[show].round(2).to_string())
    log(f"all cubes: density median {df.m.median():.2f}, sparse (<0.2) {(df.m < 0.2).mean():.1%}, "
        f"surface {df.surf.mean():.1%}, neighbours {df.nb.mean():.2f}")

    setp()                                    # back to the file's values (forked workers inherit them)
    return chosen, ok_check, sweep, checks


def region_of_cubes(m, meta, geom):
    npil = meta["neuropil"].astype(str).values
    codes, names = pd.factorize(npil)
    rows = np.repeat(np.arange(len(npil)), np.diff(geom["P_ptr"]))
    cube_np = (pd.DataFrame({"c": geom["P_idx"], "n": codes[rows], "w": geom["P_w"]})
               .groupby(["c", "n"])["w"].sum().reset_index()
               .sort_values("w").drop_duplicates("c", keep="last").set_index("c")["n"])
    out = np.array(["none"] * len(geom["m_rel"]), dtype=object)
    out[cube_np.index.values] = names[cube_np.values]
    return out


# ----------------------------------------------------------------------------- B
def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))          # don't load the big CSV in all workers at once
    return runner.run_condition(cond)


def run_batch(conds, tag):
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_" + tag)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"{len(conds)} runs, {N_WORKERS} at a time -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, stagger=60 * (i % N_WORKERS) * min(SCALE * 10, 1)))
                for i, c in enumerate(conds)]
        for fut in as_completed(futs):
            rows.append(fut.result())
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    cols = [c for c in ["tag", "status", "mode", "t50_s", "speed_cube_mm_min", "cubes_reached",
                        "blocked_max", "blocked_end", "E_at_block_median", "K_end_mean", "minutes"] if c in summary]
    log("\n" + summary[cols].to_string(index=False))
    corr = runner.correlations(rows, out)
    if len(corr):
        try:
            corr.to_csv(os.path.join(out, "correlations.csv"), index=False)
        except PermissionError:
            pass
        log("\n" + corr.to_string(index=False))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")
    return out


def cond(name, seed, dur_s, **overrides):
    return dict(name=name, seed=seed, overrides=overrides, dur=dur_s * 1000.0 * SCALE)


if __name__ == "__main__":
    # run as two processes (see stage6.bat) so the calibration's memory is freed before 5 full runs start:
    #   python stage6.py --tissue   -> calibration, writes runs/v4_chosen.json
    #   python stage6.py --full     -> full-brain batch with the chosen setting
    os.makedirs("runs", exist_ok=True)
    chosen_file = os.path.join("runs", "v4_chosen.json")
    t0 = time.time()
    if "--full" not in sys.argv:
        log("stage6 (calibration) started", time.strftime("%Y-%m-%d %H:%M"))
        chosen, ok_check, sweep, checks = tissue_part()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sweep.to_csv(os.path.join("runs", f"{stamp}_v4_tissue_sweep.csv"), index=False)
        with open(chosen_file, "w") as f:
            json.dump(dict(chosen=chosen, tissue_matches_full=ok_check, checks=checks), f, indent=1,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
        log(f"\ncalibration took {(time.time() - t0) / 60:.1f} min")
    if "--tissue" not in sys.argv:
        with open(chosen_file) as f:
            chosen = json.load(f)["chosen"]
        p = dict(chosen, SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2)
        log(f"\n=== B. full model with {p} ===  ({time.strftime('%H:%M')})")
        conds = [
            cond("trig",      0, 120, RUN_MODE="trigger", TRIGGER_HOLD_MS=6000.0, **p),
            cond("bath",      0, 90,  RUN_MODE="bath", BATH_MODE="surface", BATH_G_S=3.0, **p),
            cond("R_real",    0, 120, RUN_MODE="death", COUPLING="real", **p),
            cond("S_swap",    0, 120, RUN_MODE="death", COUPLING="swap", **p),
            cond("W_rewired", 0, 120, RUN_MODE="death", COUPLING="real", SHUFFLE_CONN=True, **p),
            cond("R_real",    1, 120, RUN_MODE="death", COUPLING="real", **p),
            cond("S_swap",    1, 120, RUN_MODE="death", COUPLING="swap", SWAP_SEED=54321, **p),
            cond("W_rewired", 1, 120, RUN_MODE="death", COUPLING="real", SHUFFLE_CONN=True, **p),
            cond("R_real",    2, 120, RUN_MODE="death", COUPLING="real", **p),
        ]
        run_batch(conds, "v4_stage2")
        log(f"\nfull batch finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")
