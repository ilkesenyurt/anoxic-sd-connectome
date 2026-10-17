# stage11.py - independent validation tests (log row 80): things the model was NOT tuned to
# ===========================================================================================
# Model death_v7 (= v6 + reoxygenation, ouabain, virtual K electrodes; identical to v6 when they are off).
# 5 runs, real wiring unless noted, stage-8 baseline settings (D 770, SD delay 0.1 s):
#   reox_R_tau2      N2 from 2 s to 62 s, then air until 110 s (Spong et al. 2016: N2 for 60 s);
#                    energy comes back with a 2 s time constant
#   reox_R_tau10     same, energy comes back with 10 s (energy recovery speed is not known)
#   reox_Wdeg_tau2   same as the first, degree-preserving rewired brain (does wiring change recovery?)
#   ouabain_0.8 / 0.9  oxygen kept; 80% / 90% of Na/K pump capacity blocked (10 s wash-in), 60 s
# Fly targets (Drosophila brain, room temperature):
#   A. virtual K electrode during anoxia: slow K rise before the surge (~0.2 mM/s), surge from ~10 mM,
#      surge 20-50 s after N2 with K electrodes (8-15 s from the DC potential)    [log rows 70-72]
#   B. recovery after N2 off: DC potential back to half amplitude in 11.8-16.1 s
#      (Spong et al. 2016, N2 60 s); the brain recovers fully                   [not tuned]
#   C. ouabain: SD is repetitive (events with recovery in between, ~10 min of activity);
#      first surge 10-70 mM, median 24 mM (Spong et al. 2016)                     [not tuned]
# Small outputs only: summary.csv, validation.csv, probes_<tag>.npz (~1 MB each), stage11_log.txt.

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import runner

SCALE = float(os.environ.get("STAGE11_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE11_WORKERS", "5"))
MODEL = "death_v7"
T_ANOX = runner.T_ANOX_MS                  # 2000 ms
T_REOX = T_ANOX + 60000.0 * SCALE          # N2 for 60 s (Spong 2016)


def log(*a):
    print(*a, flush=True)


def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    cond["overrides"]["SAVE_PROBES_TO"] = os.path.join(cond["out"], f"probes_{cond['name']}_seed0.npz")
    return runner.run_condition(cond)


# ---------------------------------------------------------------- analysis helpers
def first_cross(t, y, level, after=-np.inf, below=False):
    k = np.flatnonzero((t >= after) & ((y <= level) if below else (y >= level)))
    return t[k[0]] if k.size else np.nan


def electrode(p, t_anox_s, t_end_s):
    """Per virtual electrode: surge onset (steepest rise), K at onset, rise rate in the 10 s before."""
    t = p["t"] / 1000.0
    rows = []
    for q in range(p["K"].shape[1]):
        k = p["K"][:, q].astype(float)
        w = (t >= t_anox_s) & (t <= t_end_s)
        if k[w].max() < 20:                                   # no surge at this electrode
            rows.append(dict(probe=q, surge=False)); continue
        dk = np.gradient(k, t)
        i_max = np.flatnonzero(w)[np.argmax(dk[w])]
        # onset = last time before the steepest point where the rise was still slower than 2 mM/s
        j = i_max
        while j > 0 and dk[j] > 2.0 and t[j] > t_anox_s:
            j -= 1
        t_on = t[j]
        def K_at(tt):
            return float(np.interp(tt, t, k))
        base = float(k[(t >= t_anox_s - 1.5) & (t < t_anox_s)].mean())
        rows.append(dict(probe=q, surge=True, depth_um=float(p["depth_um"][q]), K_base=base,
                         onset_s=t_on - t_anox_s, K_at_onset=K_at(t_on),
                         pre_rise_mM_s=(K_at(t_on - 1) - K_at(t_on - 11)) / 10.0,
                         max_slope_mM_s=float(dk[i_max]), K_peak=float(k[w].max())))
    return pd.DataFrame(rows)


def recovery(p, t_reox_s, t_anox_s):
    """After N2 off: half recovery of the 'DC-like' signal (fraction of neurons blocked) and of K."""
    st = p["series_t"] / 1000.0; bl = p["blocked"].astype(float)
    t = p["t"] / 1000.0; Km = np.median(p["K"], axis=1)
    b0 = float(np.interp(t_reox_s, st, bl)); k0 = float(np.interp(t_reox_s, t, Km))
    kb = float(np.median(p["K"][(t > 0.5) & (t < 2.0)]))
    if b0 < 0.05:                                             # brain was not down at N2 off
        return dict(blocked_at_N2_off=b0, half_recovery_blocked_s=np.nan, half_recovery_K_s=np.nan,
                    blocked_end=float(bl[-1]))
    out = dict(blocked_at_N2_off=b0, K_at_N2_off=k0,
               half_recovery_blocked_s=first_cross(st, bl, b0 / 2, after=t_reox_s, below=True) - t_reox_s,
               half_recovery_K_s=first_cross(t, Km, (k0 + kb) / 2, after=t_reox_s, below=True) - t_reox_s,
               K_back_below_6_s=first_cross(t, Km, 6.0, after=t_reox_s, below=True) - t_reox_s,
               blocked_end=float(bl[-1]), frac_in_SD_end=float(p["frac_sd"][-1]), K_end=float(Km[-1]))
    act = p["active"].astype(float) / (0.1 / 1000.0)        # population rate per neuron (Hz); DT 0.1 ms
    out["rate_last5s_hz"] = float(act[st >= st[-1] - 5].mean())
    out["rate_before_N2_hz"] = float(act[(st > 0.5) & (st < t_anox_s)].mean())
    return out


if __name__ == "__main__":
    t0 = time.time()
    log("stage11 started", time.strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join("runs", "v6_chosen.json")) as f:
        ch = json.load(f)["770"]
    p = dict(D_UM2_S=770.0, SD_DELAY_S=ch["SD_DELAY_S"], SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2)
    reox = dict(p, RUN_MODE="death", T_REOX_MS=T_REOX, STOP_EARLY=False)
    d_reox = T_REOX + 48000.0 * SCALE
    d_ouab = 60000.0 * SCALE
    WD = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)     # = stage-8 rewired_a
    conds = [dict(name="reox_R_tau2", dur=d_reox, overrides=dict(reox, E_RECOVER_TAU_S=2.0)),
             dict(name="reox_R_tau10", dur=d_reox, overrides=dict(reox, E_RECOVER_TAU_S=10.0)),
             dict(name="reox_Wdeg_tau2", dur=d_reox, overrides=dict(reox, E_RECOVER_TAU_S=2.0, **WD)),
             dict(name="ouabain_0.8", dur=d_ouab, overrides=dict(p, RUN_MODE="ouabain", OUABAIN_BLOCK=0.8)),
             dict(name="ouabain_0.9", dur=d_ouab, overrides=dict(p, RUN_MODE="ouabain", OUABAIN_BLOCK=0.9))]
    for c in conds:
        c["seed"] = 0
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v7_validation")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"{len(conds)} runs -> {out}  (N2 {T_ANOX / 1000:.0f}-{T_REOX / 1000:.0f} s, reox runs end at {d_reox / 1000:.0f} s)")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, overrides=dict(c["overrides"]),
                                        stagger=45 * i * min(SCALE * 10, 1))) for i, c in enumerate(conds)]
        for fu in as_completed(futs):
            rows.append(fu.result())
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows).sort_values("tag")
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")

    def probes(name):
        f = os.path.join(out, f"probes_{name}_seed0.npz")
        return np.load(f) if os.path.exists(f) else None

    ta, tr = T_ANOX / 1000.0, T_REOX / 1000.0
    res = []
    # A. virtual K electrodes during anoxia (real brain)
    pr = probes("reox_R_tau2")
    if pr is not None:
        e = electrode(pr, ta, tr)
        s = e[e.surge]
        log(f"\n=== A. virtual K electrodes, real brain, during N2 ({len(s)} of {len(e)} electrodes saw a surge) ===")
        if len(s):
            q = s[["onset_s", "K_base", "K_at_onset", "pre_rise_mM_s", "max_slope_mM_s", "K_peak", "depth_um"]].quantile([0.25, 0.5, 0.75])
            log(q.round(2).to_string())
            log("central electrode: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, (int, float, np.floating)) else f"{k} {v}" for k, v in e.iloc[0].to_dict().items()))
            log("fly: surge 20-50 s after N2 (K electrode), from ~10 mM, after a ~0.2 mM/s rise;"
                " peak ~54 mM is imposed by the model cap (K_PEAK 55) - NOT a test")
            for c in ["onset_s", "K_at_onset", "pre_rise_mM_s"]:
                res.append(dict(test="A electrode " + c, model=float(s[c].median()),
                                target={"onset_s": "20-50", "K_at_onset": "~10", "pre_rise_mM_s": "~0.2"}[c]))
            e.to_csv(os.path.join(out, "electrodes_reox_R_tau2.csv"), index=False)
    # sanity check: until N2 off the real-brain run must repeat the stage-8 baseline exactly (v7 = v6)
    try:
        v6 = sorted(glob.glob(os.path.join("runs", "*_v6_robust_stage2")))[-1]
        a = np.load(os.path.join(v6, "cube_time_base_R_seed0.npy"))
        b = np.load(os.path.join(out, "cube_time_reox_R_tau2_seed0.npy"))
        k = np.isfinite(a) & (a < T_REOX)
        log(f"\ncheck: SD times before N2 off identical to stage-8 base_R: {np.array_equal(a[k], b[k])}"
            f" ({k.sum():,} cubes)")
    except Exception as ex:
        log(f"\ncheck against stage 8 not possible: {ex}")
    # B. recovery after N2 off
    log("\n=== B. recovery after N2 off (fly DC potential: half recovery 11.8-16.1 s; full recovery) ===")
    rec = {}
    for n in ["reox_R_tau2", "reox_R_tau10", "reox_Wdeg_tau2"]:
        pr = probes(n)
        if pr is None:
            continue
        rec[n] = recovery(pr, tr, ta)
        log(f"{n}: " + ", ".join(f"{k} {v:.3g}" for k, v in rec[n].items()))
        res.append(dict(test=f"B half recovery (blocked) {n}", model=rec[n]["half_recovery_blocked_s"], target="11.8-16.1 s"))
        res.append(dict(test=f"B half recovery (K) {n}", model=rec[n]["half_recovery_K_s"], target="(no fly number)"))
        res.append(dict(test=f"B fully recovered {n}", model=rec[n]["blocked_end"], target="~0 blocked"))
    # C. ouabain
    log("\n=== C. ouabain (fly: repetitive SD; first surge 10-70 mM, median 24 mM) ===")
    for n in ["ouabain_0.8", "ouabain_0.9"]:
        pr = probes(n)
        if pr is None:
            continue
        t = pr["t"] / 1000.0
        f = os.path.join(out, f"cube_time_{n}_seed0.npy")
        ct = np.load(f) if os.path.exists(f) else np.full(1, np.nan)
        sdc = pr["sd_count"]
        lat = (np.nanmin(ct) - T_ANOX) / 1000.0 if np.isfinite(ct).any() else np.nan
        peak = np.median(pr["K"].max(0))
        rep = float((sdc >= 2).mean())
        log(f"{n}: cubes in SD {np.isfinite(ct).mean():.1%}, first SD {lat:.1f} s after ouabain start,"
            f" electrode peak K median {peak:.1f} mM, cubes with >= 2 SD events {rep:.1%},"
            f" blocked at end {pr['blocked'][-1]:.1%}")
        res.append(dict(test=f"C first SD latency {n}", model=lat, target="(minutes, injections; not comparable)"))
        res.append(dict(test=f"C surge peak {n}", model=peak, target="median 24 (10-70) mM"))
        res.append(dict(test=f"C repetitive SD {n}", model=rep, target="repetitive (yes)"))
    tab = pd.DataFrame(res)
    log("\n=== validation summary ===")
    log(tab.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    try:
        tab.to_csv(os.path.join(out, "validation.csv"), index=False)
    except PermissionError:
        pass
    log(f"\nstage11 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")
