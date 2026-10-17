# stage15.py - final calibration of the death / recovery timing, model v11 (log row 91)
# =====================================================================================
# Model: death_v11 (= v10 + output only: meanV = mean neuron resting potential, the model's DC-potential proxy).
# Final configuration: Goldman K coupling, fixed reset (Shiu), glia on (shuttle 0.4 / 0.05 / 10 / 0.5),
# SD pump cut kept (SD_PUMP 0.2, fading with SD_TAU; row 92: stand-in for SD-front effects not modelled -
# extracellular space shrinking, local ATP drop; with glia and no cut the KCl wave stalls at 3%), SD_KMAX 54, SD_PUMP_RECOVER, drive 3.36 (stage14, reset).
# Rules fixed BEFORE the run (row 91):
#   A1. SD_TAU_S (how long tissue in SD keeps releasing K) from SD DURATION, not from recovery:
#       half-amplitude width of the DC proxy of one KCl-triggered SD (pumps working) at the probe cubes,
#       interpolated to 37 s (locust ouabain SD, young animals, half-amplitude duration 0.62 min;
#       bioRxiv 10.1101/2024.10.10.617596; no fly value found). Tested 15 / 20 / 25 s.
#   A2. SD_DELAY_S: wave speed closest to 2.3 mm/min (Andersen et al. 2025: 2.1-2.5 at room temperature).
#   B.  E_BASE_USE_S (how fast energy runs out) from the DC-proxy onset: time after N2 at which meanV is half way
#       from baseline to its anoxic plateau, interpolated to 11.5 s (Spong et al. 2016, DC half descent 8-15 s).
#       Runs at E_BASE 6, 14, 28 (pump restart 3 s). If 11.5 s is not reachable, no extrapolation: E_BASE 14
#       (middle) is kept, the miss is reported and the choice goes to the user. Fake brain: onset has a floor
#       (~17 s even at E_BASE 6) because K then builds up only as fast as the leak (K_LEAK_S 0.3 mM/s; fly K
#       electrode pre-surge rise ~0.2 mM/s): from 3 to 11 mM takes ~25-40 s, which fits the K-electrode data
#       (surge 20-50 s), not the DC data (8-15 s).
#   C.  Recovery is NOT fitted: DC-proxy half recovery after N2 off (N2 60 s) at the chosen E_BASE (interpolated
#       over the pump-restart-3 s runs), compared to 11.8-16.1 s (Spong 2016). Pump restart time E_RECOVER_TAU_S is
#       unmeasured (literature: seconds): 3 s, plus 10 s and 20 s at E_BASE 14 (shift applied to the chosen E_BASE).
#       Final model: the smallest of 3 / 10 / 20 s that passes; 20 s is beyond the literature bound and is flagged;
#       if none passes, 3 s is kept and recovery is reported as a failed test.
#   D.  Wiring: real vs degree-preserving rewired at E_BASE 14.
# Small outputs only.

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

SCALE = float(os.environ.get("STAGE15_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE15_WORKERS", "5"))
MODEL = "death_v11"
FINAL = dict(D_UM2_S=770.0, SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2, SD_KMAX=54.0, SD_PUMP_RECOVER=True,
             K_COUPLING="goldman", RESET_MODE="fixed", NA_INACT=False, TONIC_MEAN=3.36,
             GLIA_ON=True, GLIA_UPTAKE_S=0.4, GLIA_RELEASE_S=0.05, GLIA_CAP=10.0, GLIA_PASSIVE=0.5)
WD = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
T_ANOX = runner.T_ANOX_MS
T_REOX = T_ANOX + 60000.0 * SCALE
SD_DUR_TARGET = 37.0
SPEED_TARGET = 2.3
ONSET_TARGET = 11.5
REC_RANGE = (11.8, 16.1)
TAUS = [15.0, 20.0, 25.0]
DELAYS = [0.0, 0.05, 0.1, 0.15]
EBASES = [6.0, 14.0, 28.0]
EREC = [3.0, 10.0, 20.0]
E_MID = 14.0


def log(*a):
    print(*a, flush=True)


def ghk_v(K):
    """DC proxy of tissue with working pumps (tissue-only mode): Goldman shift of the resting potential."""
    return 58.2 * np.log10((K + 3.5) / 6.5)


def tissue_one(job):
    import importlib
    time.sleep(job.pop("stagger", 0))
    m = importlib.import_module(MODEL)
    for k, v in dict(FINAL, RUN_MODE="trigger", TRIGGER_HOLD_MS=3000.0, STOP_EARLY=False, **job["p"]).items():
        assert hasattr(m, k), k
        setattr(m, k, v)
    meta, conn = m.load_data()
    idx, W, sign, coords, N = m.build_network(meta, conn)
    geom = m.build_geometry(meta, idx, coords)
    del W, conn
    r = m.simulate_tissue(geom, 150000.0 * min(SCALE * 4, 1))
    ct = r["cube_time"]
    sp = m.cube_speed_mm_min(ct, geom) if np.isfinite(ct).sum() > 10 else np.nan
    t = r["probe_t"] / 1000.0
    V = ghk_v(r["probe_K"])
    widths = []
    trig_end = 5.0
    for q in range(V.shape[1]):
        v = V[:, q]; b = float(v[t < 1.5].mean()); pk = float(v.max())
        if pk - b < 15:
            continue
        above = np.flatnonzero(v > b + (pk - b) / 2)
        if t[above[0]] < trig_end:          # trigger cubes: held high by the experiment, not an SD of their own
            continue
        if above[-1] == len(v) - 1:         # still depolarized at the end: width unknown (longer than the run)
            widths.append(np.inf)
            continue
        widths.append(float(t[above[-1]] - t[above[0]]))
    w = np.array(widths)
    return dict(tau=job["p"]["SD_TAU_S"], delay=job["p"]["SD_DELAY_S"], reached=float(np.isfinite(ct).mean()),
                speed=sp, sd_width_s=float(np.median(w)) if len(w) else np.nan, n_probes=len(w),
                K_peak_median=float(np.median(r["probe_K"].max(0))))


def tissue_part():
    jobs = [dict(p=dict(SD_TAU_S=tau, SD_DELAY_S=0.05)) for tau in TAUS] + \
           [dict(p=dict(SD_TAU_S=20.0, SD_DELAY_S=d)) for d in DELAYS if d != 0.05]
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(tissue_one, dict(j, stagger=30 * (i % N_WORKERS) * min(SCALE * 10, 1)))
                for i, j in enumerate(jobs)]
        for fu in as_completed(futs):
            rows.append(fu.result())
    tab = pd.DataFrame(rows).sort_values(["tau", "delay"])
    log("\n=== A. tissue only, one KCl-triggered SD with working pumps (glia on, SD pump cut 0.2) ===")
    log(tab.round(3).to_string(index=False))
    tab.to_csv(os.path.join("runs", "v11_tissue.csv"), index=False)
    if not (tab.reached > 0.9).any():
        raise SystemExit("no spreading wave in the tissue check: stopping (rethink before the long runs)")
    a = tab[tab.delay == 0.05].sort_values("tau")
    ok = np.isfinite(a.sd_width_s.values)
    if not ok.any():
        raise SystemExit("no SD duration measurable at the probes: stopping")
    if ok.sum() >= 2 and np.all(np.diff(a.sd_width_s.values[ok]) > 0):
        tau = float(np.interp(SD_DUR_TARGET, a.sd_width_s.values[ok], a.tau.values[ok]))
    else:
        tau = float(a.tau.values[np.nanargmin(np.abs(a.sd_width_s.values - SD_DUR_TARGET))])
    tau = round(tau, 1)
    s = tab[tab.tau == 20.0].dropna(subset=["speed"])
    s = s[s.reached > 0.9]
    if not len(s):
        raise SystemExit("no spreading wave in the tissue check: stopping (rethink before the long runs)")
    delay = float(s.delay.values[np.argmin(np.abs(s.speed.values - SPEED_TARGET))])
    log(f"A1 SD_TAU_S {tau} s (SD duration target {SD_DUR_TARGET} s, locust); "
        f"A2 SD_DELAY_S {delay} s (speed {s.speed.values[np.argmin(np.abs(s.speed.values - SPEED_TARGET))]:.2f} "
        f"mm/min, target {SPEED_TARGET}; speed measured at SD_TAU 20 - row 91 fake test: speed ~independent of SD_TAU)")
    return tau, delay


def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    cond["overrides"]["SAVE_PROBES_TO"] = os.path.join(cond["out"], f"probes_{cond['name']}_seed0.npz")
    return runner.run_condition(cond)


def batch(conds, suffix):
    for c in conds:
        c["seed"] = 0
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + suffix)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"{len(conds)} runs -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, overrides=dict(c["overrides"]),
                                        stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1))) for i, c in enumerate(conds)]
        for fu in as_completed(futs):
            rows.append(fu.result())
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")
    return out, summary


def cross(t, y, level, after, below):
    k = np.flatnonzero((t > after) & ((y <= level) if below else (y >= level)))
    return float(t[k[0]]) if len(k) else np.nan


def dc_metrics(p):
    """DC proxy = meanV (mean neuron resting potential). Onset: half way from baseline to the anoxic plateau
    (time after N2 on). Recovery: half way back from the value at N2 off to baseline (time after N2 off)."""
    st = p["series_t"] / 1000.0
    v = p["meanV"].astype(float)
    ta, tr = T_ANOX / 1000.0, T_REOX / 1000.0
    base = float(v[(st > 0.5) & (st < ta)].mean())
    plateau = float(v[(st > ta) & (st < tr)].max())
    onset = cross(st, v, base + (plateau - base) / 2, ta, below=False) - ta
    v_off = float(np.interp(tr, st, v))
    rec = cross(st, v, base + (v_off - base) / 2, tr, below=True) - tr
    back = cross(st, v, base + 0.1 * (v_off - base), tr, below=True) - tr
    return dict(dc_base_mV=base, dc_plateau_mV=plateau, dc_onset_s=onset, dc_at_N2off_mV=v_off,
                dc_half_recovery_s=rec, dc_90pct_recovery_s=back, dc_end_mV=float(v[-1]))


def main():
    t0 = time.time()
    tau, delay = tissue_part()
    p = dict(FINAL, SD_TAU_S=tau, SD_DELAY_S=delay)
    with open(os.path.join("runs", "v11_chosen.json"), "w") as f:
        json.dump(dict(stage="A done", **p), f, indent=1)
    d_reox = T_REOX + 48000.0 * SCALE
    reox = dict(RUN_MODE="death", T_REOX_MS=T_REOX, STOP_EARLY=False)
    grid = [(eb, 3.0) for eb in EBASES] + [(E_MID, er) for er in EREC if er != 3.0]
    conds = [dict(name=f"reox_E{int(eb)}_tau{int(er)}", dur=d_reox,
                  overrides=dict(p, **reox, E_BASE_USE_S=eb, E_RECOVER_TAU_S=er)) for eb, er in grid]
    conds += [dict(name="R_E14", dur=90000.0 * SCALE, overrides=dict(p, RUN_MODE="death", E_BASE_USE_S=E_MID)),
              dict(name="Wdeg_E14", dur=90000.0 * SCALE, overrides=dict(p, RUN_MODE="death", E_BASE_USE_S=E_MID, **WD))]
    out, summary = batch(conds, "_v11_final_cal")

    import stage11
    rows = []
    for eb, er in grid:
        name = f"reox_E{int(eb)}_tau{int(er)}"
        f = os.path.join(out, f"probes_{name}_seed0.npz")
        if not os.path.exists(f):
            continue
        pr = np.load(f)
        row = dict(run=name, E_BASE=eb, E_RECOVER_TAU=er, **dc_metrics(pr))
        rec = stage11.recovery(pr, T_REOX / 1000.0, T_ANOX / 1000.0)
        row.update({f"blk_{k}": v for k, v in rec.items()})
        e = stage11.electrode(pr, T_ANOX / 1000.0, T_REOX / 1000.0)
        s = e[e.surge]
        for c in ["onset_s", "K_at_onset", "pre_rise_mM_s"]:
            row[f"el_{c}"] = float(s[c].median()) if len(s) else np.nan
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "calibration.csv"), index=False)
    log("\n=== B/C. DC proxy (mean neuron resting potential) per run ===")
    cols = ["run", "dc_onset_s", "dc_half_recovery_s", "dc_90pct_recovery_s", "dc_plateau_mV", "dc_end_mV",
            "blk_half_recovery_blocked_s", "blk_K_end", "blk_frac_in_SD_end", "blk_rate_last5s_hz", "el_onset_s",
            "el_K_at_onset"]
    log(tab[[c for c in cols if c in tab]].round(2).to_string(index=False))

    res = dict(SD_TAU_S=tau, SD_DELAY_S=delay)
    g3 = tab[tab.E_RECOVER_TAU == 3.0].sort_values("E_BASE")
    g3 = g3[np.isfinite(g3.dc_onset_s)]
    if len(g3) >= 2:
        x, y = g3.dc_onset_s.values, g3.E_BASE.values
        if ONSET_TARGET < x.min() or ONSET_TARGET > x.max() or np.any(np.diff(x) <= 0):
            eb = E_MID
            log(f"\nB. DC onset target {ONSET_TARGET} s NOT reachable in the tested range "
                f"{ {float(a): round(float(b), 1) for a, b in zip(y, x)} }: E_BASE {eb} kept, decision to the user")
            res["onset_reached"] = False
        else:
            eb = float(np.interp(ONSET_TARGET, x, y))
            res["onset_reached"] = True
            log(f"\nB. E_BASE_USE_S {eb:.1f} (DC onset { {float(a): round(float(b), 1) for a, b in zip(y, x)} } -> target {ONSET_TARGET} s)")
        res["E_BASE_USE_S"] = round(eb, 1)
        rec3 = float(np.interp(eb, g3.E_BASE.values, g3.dc_half_recovery_s.values))
        mid3 = float(g3[g3.E_BASE == E_MID].dc_half_recovery_s.iloc[0]) if (g3.E_BASE == E_MID).any() else np.nan
        log("C. recovery at that E_BASE (not fitted; fly DC half recovery 11.8-16.1 s):")
        res["recovery"] = {}
        for er in EREC:
            if er == 3.0:
                rec = rec3
            else:
                gm = tab[(tab.E_RECOVER_TAU == er) & (tab.E_BASE == E_MID)]
                rec = float(gm.dc_half_recovery_s.iloc[0]) + (rec3 - mid3) if len(gm) else np.nan
            ok = bool(REC_RANGE[0] <= rec <= REC_RANGE[1])
            near = bool(REC_RANGE[0] * 0.75 <= rec <= REC_RANGE[1] * 1.25)
            res["recovery"][f"tau{int(er)}"] = dict(dc_half_recovery_s=round(rec, 1), passes=ok, within_25pct=near)
            log(f"   pump restart {er:4.0f} s: DC half recovery {rec:5.1f} s -> "
                f"{'PASS' if ok else ('near (within 25%)' if near else 'FAIL')}"
                f"{'  (beyond the literature bound of seconds)' if er >= 20 else ''}")
        passing = [er for er in EREC if res["recovery"][f"tau{int(er)}"]["passes"]]
        pick = min(passing) if passing else 3.0
        res["E_RECOVER_TAU_S"] = pick
        log(f"   pre-registered choice: E_RECOVER_TAU_S {pick} s" + ("" if passing else " (no value passes: recovery test FAILED)"))
    with open(os.path.join("runs", "v11_chosen.json"), "w") as f:
        json.dump(dict(res, final=p), f, indent=1, default=str)

    log("\n=== D. wiring at E_BASE 14 (rho = Spearman of cube SD arrival times) ===")
    ct = {}
    for n in ["R_E14", "Wdeg_E14"]:
        f = os.path.join(out, f"cube_time_{n}_seed0.npy")
        if os.path.exists(f):
            ct[n] = np.load(f)
    if len(ct) == 2:
        x, y = ct["R_E14"], ct["Wdeg_E14"]
        k = np.isfinite(x) & np.isfinite(y)
        sm = summary.set_index("tag")
        log(f"real vs rewired rho {spearmanr(x[k], y[k])[0]:.3f}; first SD real {(np.nanmin(x) - T_ANOX) / 1000:.1f} s, "
            f"rewired {(np.nanmin(y) - T_ANOX) / 1000:.1f} s; t50 real {sm.loc['R_E14_seed0', 't50_s']:.1f}, "
            f"rewired {sm.loc['Wdeg_E14_seed0', 't50_s']:.1f}; healthy rate real {sm.loc['R_E14_seed0', 'rate_pre_mean_hz']:.2f} Hz")
    log(f"\nstage15 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage15 started", time.strftime("%Y-%m-%d %H:%M"), sys.argv[1:])
    main()
