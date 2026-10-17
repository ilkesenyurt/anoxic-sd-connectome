# stage14.py - model v10: firing brake for Goldman K coupling (log row 89)
# ======================================================================
# Problem (row 88): with Goldman coupling, after reoxygenation K stayed ~7.2 mM, 99.9% of tissue stayed in SD and
# unblocked neurons fired ~306 Hz. Fake-brain diagnosis (row 89): the cause is the v2-v9 reset rule (after a spike
# v -> the neuron's CURRENT resting potential). Once K pushes v_rest above threshold (-45 mV), the neuron fires again
# right after every refractory period. Shiu et al. 2024 reset to a fixed -52 mV. Na inactivation alone did not fix it.
# Variants (Goldman coupling, v8 SD fix, settings otherwise as stage13 --goldman):
#   reset   : RESET_MODE "fixed" (Shiu's LIF, V_RESET -52)
#   resetNA : + NA_INACT (threshold rises with Na inactivation; Platkiewicz & Brette 2010; fly DmNav values)
#   A (~30 min) healthy drive calibration for both (20 s healthy runs, target = v8 real-brain rate)
#   B (~3 h)    reox_R, R, Wdeg for both variants + 1 preview of the final model (resetNA + glia + no SD pump cut)
# Decision rule, fixed BEFORE the run (revised Oct 3 05:00, before any result): the final model uses the reset alone if it
#   passes recovery (pass = tissue leaves SD and K < 6 mM by the end; half recovery 11.8-16.1 s is a later calibration
#   target). resetNA is a sensitivity check only: its fly values are weak (V1/2 from oocyte-expressed DmNav, splice
#   variants span -34.9 to -66.4 mV). resetNA becomes the choice only if the reset alone fails and resetNA passes.
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

SCALE = float(os.environ.get("STAGE14_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE14_WORKERS", "5"))
MODEL = "death_v10"
BASE = dict(D_UM2_S=770.0, SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2, SD_KMAX=54.0, SD_PUMP_RECOVER=True,
            K_COUPLING="goldman")
VARIANTS = {"reset": dict(RESET_MODE="fixed", NA_INACT=False),
            "resetNA": dict(RESET_MODE="fixed", NA_INACT=True)}
GLIA = dict(GLIA_ON=True, GLIA_UPTAKE_S=0.4, GLIA_RELEASE_S=0.05, GLIA_CAP=10.0, GLIA_PASSIVE=0.5)
WD = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
T_ANOX = runner.T_ANOX_MS
T_REOX = T_ANOX + 60000.0 * SCALE
DRIVES = [2.5, 3.0, 3.5, 4.0]


def log(*a):
    print(*a, flush=True)


def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    cond["overrides"]["SAVE_PROBES_TO"] = os.path.join(cond["out"], f"probes_{cond['name']}_seed0.npz")
    return runner.run_condition(cond)


def calib_one(job):
    """Healthy brain (no oxygen loss) for 20 s: mean rate in the 2nd half, max K, blocked."""
    import importlib
    time.sleep(job.pop("stagger", 0))
    m = importlib.import_module(MODEL)
    for k, v in job["p"].items():
        assert hasattr(m, k), k
        setattr(m, k, v)
    m.RUN_MODE = "healthy"
    meta, conn = m.load_data()
    m.rng = np.random.default_rng(0)
    idx, W, sign, coords, N = m.build_network(meta, conn)
    vuln = m.build_vulnerability(sign)
    geom = m.build_geometry(meta, idx, coords)
    dur = 20000.0 * SCALE
    r = m.simulate(W, N, vuln, geom, 1e12, dur)
    late = r["t"] >= dur / 2
    return dict(variant=job["variant"], tonic=job["tonic"], rate_hz=float(r["active"][late].mean() / (m.DT / 1000.0)),
                K_cube_max=float(r["cube_K_end"].max()), K_cube_median=float(np.median(r["cube_K_end"])),
                blocked_end=float(r["blocked"][-1]), cubes_in_SD=float(np.isfinite(r["cube_time"]).mean()))


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


def burst(p):
    """Peak population rate (1 s smoothing) between N2 on and N2 off / end, and its time after N2 on."""
    st = p["series_t"] / 1000.0
    act = p["active"].astype(float) / (0.1 / 1000.0)
    n = max(int(round(1.0 / max(np.median(np.diff(st)), 1e-6))), 1)
    sm = np.convolve(act, np.ones(n) / n, mode="same")
    t_end = T_REOX / 1000.0
    k = (st > T_ANOX / 1000.0) & (st < t_end)
    i = np.argmax(np.where(k, sm, -1))
    return float(sm[i]), float(st[i] - T_ANOX / 1000.0)


def main():
    t0 = time.time()
    with open(os.path.join("runs", "v8_chosen.json")) as f:
        ch = json.load(f)
    base = dict(BASE, SD_DELAY_S=ch["SD_DELAY_S"])
    v8 = sorted(glob.glob(os.path.join("runs", "*_v8_fix")))
    target = float(pd.read_csv(os.path.join(v8[-1], "summary.csv")).set_index("tag").loc["base_R_seed0",
                                                                                         "rate_pre_mean_hz"]) if v8 else 12.6
    log(f"\n=== A. healthy drive per variant (target {target:.2f} Hz = v8 real brain; v9 Goldman: drive 3.0 -> 12.70) ===")
    jobs = [dict(variant=v, tonic=d, p=dict(base, **kw, TONIC_MEAN=d)) for v, kw in VARIANTS.items() for d in DRIVES]
    cal = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(calib_one, dict(j, stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1)))
                for i, j in enumerate(jobs)]
        for fu in as_completed(futs):
            cal.append(fu.result())
    cal = pd.DataFrame(cal).sort_values(["variant", "tonic"])
    cal["stable"] = (cal.blocked_end == 0) & (cal.cubes_in_SD == 0) & (cal.K_cube_max < 10)
    log(cal.round(3).to_string(index=False))
    cal.to_csv(os.path.join("runs", "v10_drive_calibration.csv"), index=False)
    drive = {}
    for v in VARIANTS:
        ok = cal[(cal.variant == v) & cal.stable]
        if not len(ok):
            log(f"{v}: no stable healthy brain -> variant skipped")
            continue
        # linear interpolation of rate vs drive to the target (drive is continuous); falls back to the nearest
        x, y = ok.tonic.values, ok.rate_hz.values
        d = float(np.interp(target, y, x)) if len(x) > 1 and np.all(np.diff(y) > 0) else \
            float(x[np.abs(y - target).argmin()])
        drive[v] = round(d, 3)
        log(f"{v}: drive {drive[v]} (interpolated to {target:.2f} Hz)")
    if not drive:
        return
    with open(os.path.join("runs", "v10_chosen.json"), "w") as f:
        json.dump(dict(drive=drive, target_hz=target, base=base), f, indent=1)

    dur = 90000.0 * SCALE
    d_reox = T_REOX + 48000.0 * SCALE
    reox = dict(RUN_MODE="death", T_REOX_MS=T_REOX, STOP_EARLY=False)
    conds = []
    for v, kw in VARIANTS.items():
        if v not in drive:
            continue
        p = dict(base, **kw, TONIC_MEAN=drive[v])
        conds.append(dict(name=f"reox_R_{v}", dur=d_reox, overrides=dict(p, **reox)))
    if "reset" in drive:     # preview of the final model: brake + glia + no SD pump cut (row 85: wave spreads without it)
        p = dict(base, **VARIANTS["reset"], TONIC_MEAN=drive["reset"], **GLIA, SD_PUMP=1.0)
        conds.append(dict(name="reox_R_finalprev", dur=d_reox, overrides=dict(p, **reox)))
    for v, kw in VARIANTS.items():
        if v not in drive:
            continue
        p = dict(base, **kw, TONIC_MEAN=drive[v])
        conds += [dict(name=f"R_{v}", dur=dur, overrides=dict(p, RUN_MODE="death")),
                  dict(name=f"Wdeg_{v}", dur=dur, overrides=dict(p, RUN_MODE="death", **WD))]
    out, summary = batch(conds, "_v10_brake")

    # ------------------------------------------------------------------ results
    ct, info = {}, {}
    for _, r in summary.iterrows():
        if r.status == "ok":
            n = r.tag.replace("_seed0", "")
            ct[n] = np.load(os.path.join(out, f"cube_time_{r.tag}.npy"))
            info[n] = r
    g9 = sorted(glob.glob(os.path.join("runs", "*_v9_goldman")))
    if g9:
        for n, tag in [("v9gold R", "R_goldman_seed0"), ("v9gold Wdeg", "Wdeg_goldman_seed0")]:
            f = os.path.join(g9[-1], f"cube_time_{tag}.npy")
            if os.path.exists(f):
                ct[n] = np.load(f)

    def rho(a, b):
        if a not in ct or b not in ct:
            return np.nan
        x, y = ct[a], ct[b]
        k = np.isfinite(x) & np.isfinite(y)
        return spearmanr(x[k], y[k])[0] if k.sum() > 10 else np.nan

    def fsd(a):
        return (np.nanmin(ct[a]) - T_ANOX) / 1000 if a in ct and np.isfinite(ct[a]).any() else np.nan

    log("\n=== B1. death and wiring (rho = Spearman of cube SD arrival times; v6-v9 real vs rewired 0.63-0.76) ===")
    rows = [dict(comparison="v9 Goldman (no brake): real vs rewired", rho=rho("v9gold R", "v9gold Wdeg"),
                 first_SD_real=fsd("v9gold R"), first_SD_rewired=fsd("v9gold Wdeg"))]
    for v in VARIANTS:
        rows.append(dict(comparison=f"{v}: real vs rewired", rho=rho(f"R_{v}", f"Wdeg_{v}"),
                         first_SD_real=fsd(f"R_{v}"), first_SD_rewired=fsd(f"Wdeg_{v}"),
                         rate_real=info[f"R_{v}"].rate_pre_mean_hz if f"R_{v}" in info else np.nan,
                         t50_real=info[f"R_{v}"].t50_s if f"R_{v}" in info else np.nan,
                         t50_rewired=info[f"Wdeg_{v}"].t50_s if f"Wdeg_{v}" in info else np.nan))
        rows.append(dict(comparison=f"real: v9 Goldman vs {v} (death order)", rho=rho("v9gold R", f"R_{v}")))
    rows.append(dict(comparison="real: reset vs resetNA", rho=rho("R_reset", "R_resetNA")))
    tab = pd.DataFrame(rows)
    log(tab.round(3).to_string(index=False))
    tab.to_csv(os.path.join(out, "comparison.csv"), index=False)

    log("\n=== B2. recovery after N2 off (fly: half recovery 11.8-16.1 s; v9 Goldman: K stuck 7.24, SD 99.9%, 306 Hz) ===")
    import stage11
    rec_rows = []
    for name in [f"reox_R_{v}" for v in VARIANTS] + ["reox_R_finalprev"]:
        f = os.path.join(out, f"probes_{name}_seed0.npz")
        if not os.path.exists(f):
            continue
        pr = np.load(f)
        rec = stage11.recovery(pr, T_REOX / 1000.0, T_ANOX / 1000.0)
        bmax, bt = burst(pr)
        rec.update(run=name, burst_peak_hz=bmax, burst_t_s=bt)
        e = stage11.electrode(pr, T_ANOX / 1000.0, T_REOX / 1000.0)
        s = e[e.surge]
        for c in ["onset_s", "K_at_onset", "pre_rise_mM_s"]:
            rec[f"el_{c}"] = float(s[c].median()) if len(s) else np.nan
        rec["passes"] = bool(rec.get("frac_in_SD_end", 1) < 0.05 and rec.get("K_end", 99) < 6.0)
        rec_rows.append(rec)
        log(f"{name}: " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()
                                    if k != "run"))
    rr = pd.DataFrame(rec_rows)
    rr.to_csv(os.path.join(out, "recovery.csv"), index=False)
    if len(rr):
        ok = set(rr.run[rr.passes])
        choice = "reset" if "reox_R_reset" in ok else ("resetNA" if "reox_R_resetNA" in ok else None)
        log(f"\npre-registered choice: {choice or 'NONE passes -> rethink'}")
        with open(os.path.join("runs", "v10_chosen.json")) as f:
            j = json.load(f)
        j["brake"] = choice
        with open(os.path.join("runs", "v10_chosen.json"), "w") as f:
            json.dump(j, f, indent=1)
    log(f"\nstage14 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage14 started", time.strftime("%Y-%m-%d %H:%M"), sys.argv[1:])
    main()
