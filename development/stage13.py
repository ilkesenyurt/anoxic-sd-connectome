# stage13.py - glia (Level 1 uniform + lamina) and neuromodulator robustness check, model v9 (log row 82)
# =====================================================================================================
# Run AFTER stage12 (needs runs/v8_chosen.json). Builds on the v8 fix (SD_KMAX 54, SD_PUMP_RECOVER).
#   --prep  (~20 min) A1 lamina: FlyWire lamina coverage + lamina glia density factor from Kremer et al. 2017
#                     A2 speed check with glia (tissue only, KCl trigger): re-match the SD onset delay to the v8 speed
#   (none)  (~4 h)    B  6 oxygen-loss runs with glia: real + rewired at default glia, real + rewired at the
#                        strongest glia setting (uptake 1.0/s, cap 30, all passive), real with lamina density, recovery (real)
#   --mod   (~2 h)    C  modulator synapses excitatory (+1, Shiu et al. 2024 convention) / weakly inhibitory (-0.5)
#   --goldman (~3 h)  D  K depolarizes neurons by the Goldman equation instead of the unsourced piecewise gains:
#                        healthy stability + drive calibration (20 s healthy runs), then real, rewired, recovery
# Glia settings are NOT tuned to the validation targets: 'shuttle', uptake 0.4/s from mammalian block
# experiments (Larsen 2014, Nielsen 2025), CAP 10 and PASSIVE 0.5 rough; and the question is whether the wiring result survives across them.
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

SCALE = float(os.environ.get("STAGE13_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE13_WORKERS", "5"))
MODEL = "death_v9"
BASE = dict(D_UM2_S=770.0, SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2, SD_KMAX=54.0, SD_PUMP_RECOVER=True)
GLIA = dict(GLIA_ON=True, GLIA_UPTAKE_S=0.4, GLIA_RELEASE_S=0.05, GLIA_CAP=10.0, GLIA_PASSIVE=0.5)
GLIA_STRONG = dict(GLIA, GLIA_UPTAKE_S=1.0, GLIA_CAP=30.0, GLIA_PASSIVE=1.0)   # bracket: maximal glia influence
T_ANOX = runner.T_ANOX_MS
T_REOX = T_ANOX + 60000.0 * SCALE

# Kremer et al. 2017 (Glia 65:606), adult CNS counts. Neuropil glia = astrocyte-like + neuropil ensheathing.
KREMER = dict(ALG=4600, EGN=3100, lamina_epithelial=470, lamina_marginal=100)
BRAIN_SHARE = 0.85      # ASSUMPTION: share of CNS neuropil glia in the brain (rest in the ventral nerve cord),
                        # taken ~ proportional to neuron numbers; not measured
LAMINA_PER_SIDE = True  # ASSUMPTION: Kremer's lamina counts are per side (main text; Table S2 not available)


def log(*a):
    print(*a, flush=True)


def fresh(m, defaults, **kw):
    for k, v in defaults.items():
        setattr(m, k, v)
    for k, v in kw.items():
        assert hasattr(m, k), k
        setattr(m, k, v)


# ----------------------------------------------------------------------------- A
def prep():
    import importlib
    m = importlib.import_module(MODEL)
    defaults = {k: getattr(m, k) for k in dir(m) if k.isupper()}
    with open(os.path.join("runs", "v8_chosen.json")) as f:
        ch8 = json.load(f)
    meta, conn = m.load_data()
    idx, W, sign, coords, N = m.build_network(meta, conn)
    geom = m.build_geometry(meta, idx, coords)
    del W, conn
    reg = geom["cube_region"]
    la = np.isin(reg, ["LA_L", "LA_R"])
    n_la_neurons = int(meta["neuropil"].isin(["LA_L", "LA_R"]).sum())
    log("\n=== A1. lamina ===")
    log(f"neurons whose main neuropil is the lamina: {n_la_neurons:,} "
        f"(LA_L {int((meta.neuropil == 'LA_L').sum()):,}, LA_R {int((meta.neuropil == 'LA_R').sum()):,})")
    log(f"lamina cubes: {la.sum():,} of {la.size:,} ({la.mean():.1%}); LA_L {int((reg == 'LA_L').sum())}, "
        f"LA_R {int((reg == 'LA_R').sum())}")
    log("(a complete lamina has ~800 columns per side with L1-L5 and other neurons; low numbers = incomplete coverage)")
    lam_glia = (KREMER["lamina_epithelial"] + KREMER["lamina_marginal"]) * (2 if LAMINA_PER_SIDE else 1)
    rest_glia = (KREMER["ALG"] + KREMER["EGN"]) * BRAIN_SHARE - lam_glia
    dens_la = lam_glia / max(la.sum(), 1)
    dens_rest = rest_glia / max((~la).sum(), 1)
    factor = float(dens_la / dens_rest) if la.sum() else 1.0
    log(f"neuropil glia per cube: lamina {dens_la:.3f}, rest {dens_rest:.3f} -> lamina factor {factor:.2f}")
    log(f"assumptions: lamina counts per side = {LAMINA_PER_SIDE}, brain share of CNS glia = {BRAIN_SHARE}")
    out = dict(lamina_cubes=int(la.sum()), lamina_neurons=n_la_neurons, factor=factor,
               lamina_per_side=LAMINA_PER_SIDE, brain_share=BRAIN_SHARE, kremer=KREMER)

    log("\n=== A2. speed check with glia (tissue only, KCl trigger, D 770) ===")
    rows = []
    runs = [("v8 fix", ch8["SD_DELAY_S"], {})] + [("glia", d, GLIA) for d in [0.0, 0.025, 0.05, 0.075, 0.1, 0.15]]
    for label, delay, extra in runs:
        fresh(m, defaults, RUN_MODE="trigger", SD_DELAY_S=delay, TRIGGER_HOLD_MS=3000.0, **BASE, **extra)
        r = m.simulate_tissue(geom, 60000.0 * SCALE)
        ct = r["cube_time"]
        sp = m.cube_speed_mm_min(ct, geom) if np.isfinite(ct).sum() > 10 else np.nan
        rows.append(dict(model=label, delay=delay, reached=float(np.isfinite(ct).mean()), speed=sp))
        log(f"{label:7s} delay {delay:4.2f} s: reached {rows[-1]['reached']:6.1%}  speed {sp:5.2f} mm/min")
    sw = pd.DataFrame(rows)
    ref = float(sw.speed.iloc[0])
    g = sw[(sw.model == "glia") & (sw.reached > 0.9)]
    if not len(g):
        raise SystemExit("glia stop the wave in healthy tissue; stopping for a rethink")
    best = g.iloc[(g.speed - ref).abs().argsort().values[0]]
    out.update(SD_DELAY_S=float(best.delay), speed=float(best.speed), v8_speed=ref)
    log(f"chosen delay with glia: {best.delay:.2f} s -> {best.speed:.2f} mm/min (v8 {ref:.2f})")
    with open(os.path.join("runs", "v9_chosen.json"), "w") as f:
        json.dump(out, f, indent=1)


# ----------------------------------------------------------------------------- B, C
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


def compare(out, summary, pairs, extra_refs=()):
    ct, info = {}, {}
    for name, folder, tag in extra_refs:
        f = os.path.join(folder, f"cube_time_{tag}.npy")
        if os.path.exists(f):
            ct[name] = np.load(f)
    for _, r in summary.iterrows():
        if r.status != "ok":
            continue
        name = r.tag.replace("_seed0", "")
        ct[name] = np.load(os.path.join(out, f"cube_time_{r.tag}.npy"))
        info[name] = r

    def rho(a, b):
        if a not in ct or b not in ct:
            return np.nan
        x, y = ct[a], ct[b]
        k = np.isfinite(x) & np.isfinite(y)
        return spearmanr(x[k], y[k])[0] if k.sum() > 10 else np.nan

    rows = []
    for a, b, what in pairs:
        rows.append(dict(comparison=what, a=a, b=b, rho=rho(a, b),
                         first_SD_a=(np.nanmin(ct[a]) - T_ANOX) / 1000 if a in ct else np.nan,
                         first_SD_b=(np.nanmin(ct[b]) - T_ANOX) / 1000 if b in ct else np.nan,
                         rate_a=info[a].rate_pre_mean_hz if a in info else np.nan,
                         rate_b=info[b].rate_pre_mean_hz if b in info else np.nan,
                         t50_a=info[a].t50_s if a in info else np.nan, t50_b=info[b].t50_s if b in info else np.nan))
    tab = pd.DataFrame(rows)
    log(tab.round(3).to_string(index=False))
    try:
        tab.to_csv(os.path.join(out, "comparison.csv"), index=False)
    except PermissionError:
        pass
    return ct


def main_part():
    t0 = time.time()
    with open(os.path.join("runs", "v9_chosen.json")) as f:
        ch = json.load(f)
    p = dict(BASE, SD_DELAY_S=ch["SD_DELAY_S"], RUN_MODE="death")
    WD = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
    dur = 90000.0 * SCALE
    conds = [dict(name="reox_R_glia", dur=T_REOX + 48000.0 * SCALE,
                  overrides=dict(p, **GLIA, T_REOX_MS=T_REOX, STOP_EARLY=False, E_RECOVER_TAU_S=2.0)),
             dict(name="R_glia", dur=dur, overrides=dict(p, **GLIA)),
             dict(name="Wdeg_glia", dur=dur, overrides=dict(p, **GLIA, **WD)),
             dict(name="R_gliaStrong", dur=dur, overrides=dict(p, **GLIA_STRONG)),
             dict(name="Wdeg_gliaStrong", dur=dur, overrides=dict(p, **GLIA_STRONG, **WD)),
             dict(name="R_glia_lamina", dur=dur, overrides=dict(p, **GLIA, GLIA_DENSITY="lamina",
                                                              GLIA_LAMINA_FACTOR=ch["factor"]))]
    log(f"\nglia settings: default {GLIA}; strong {GLIA_STRONG}; lamina factor {ch['factor']:.2f}; delay {ch['SD_DELAY_S']}")
    out, summary = batch(conds, "_v9_glia")
    v8 = sorted(glob.glob(os.path.join("runs", "*_v8_fix")))
    refs = [("v8 R", v8[-1], "base_R_seed0"), ("v8 Wdeg", v8[-1], "base_Wdeg_a_seed0")] if v8 else []
    log("\n=== B. does the wiring result survive glia? (v8 without glia: see stage12; v6 real vs rewired 0.628) ===")
    ct = compare(out, summary, [("v8 R", "v8 Wdeg", "no glia (v8): real vs rewired"),
                                ("R_glia", "Wdeg_glia", "glia default: real vs rewired"),
                                ("R_gliaStrong", "Wdeg_gliaStrong", "glia strong: real vs rewired"),
                                ("v8 R", "R_glia", "real: no glia vs glia"),
                                ("R_glia", "R_gliaStrong", "real: glia default vs strong"),
                                ("R_glia", "R_glia_lamina", "real: uniform vs lamina glia")], refs)
    # lamina prediction (decision record Oct 2 18:55): denser lamina glia -> later lamina SD, onset away from lamina
    import importlib
    m = importlib.import_module(MODEL)
    meta, conn = m.load_data()
    idx, W, sign, coords, N = m.build_network(meta, conn)
    geom = m.build_geometry(meta, idx, coords)
    la = np.isin(geom["cube_region"], ["LA_L", "LA_R"])
    log("\n=== B2. lamina prediction: lamina SD time and onset site, uniform vs lamina glia ===")
    for n in ["R_glia", "R_glia_lamina"]:
        if n in ct:
            x = ct[n]
            first = int(np.nanargmin(x))
            log(f"{n}: lamina cubes median SD {np.nanmedian(x[la]) / 1000 - T_ANOX / 1000:.2f} s, rest "
                f"{np.nanmedian(x[~la]) / 1000 - T_ANOX / 1000:.2f} s; first SD in {geom['cube_region'][first]}")
    log("\n=== B3. recovery with glia (fly: half recovery 11.8-16.1 s) ===")
    import stage11
    f = os.path.join(out, "probes_reox_R_glia_seed0.npz")
    if os.path.exists(f):
        rec = stage11.recovery(np.load(f), T_REOX / 1000.0, T_ANOX / 1000.0)
        log("glia: " + ", ".join(f"{k} {v:.3g}" for k, v in rec.items()))
    log(f"\nstage13 main finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


def mod_part():
    t0 = time.time()
    with open(os.path.join("runs", "v8_chosen.json")) as f:
        ch = json.load(f)
    p = dict(BASE, SD_DELAY_S=ch["SD_DELAY_S"], RUN_MODE="death")
    WD = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
    dur = 90000.0 * SCALE
    # +1.0 = the convention of Shiu et al. 2024 (their Connectivity_783 signs every connection +-1: only GABA and
    # glutamate negative, so DA / SER / OCT excitatory at full weight); -0.5 = weakly inhibitory alternative
    conds = [dict(name=f"{w}_mod{s:+.1f}", dur=dur, overrides=dict(p, MOD_SIGN=s, **(WD if w == "Wdeg" else {})))
             for s in [1.0, -0.5] for w in ["R", "Wdeg"]]
    out, summary = batch(conds, "_v9_mod")
    v8 = sorted(glob.glob(os.path.join("runs", "*_v8_fix")))
    refs = [("v8 R", v8[-1], "base_R_seed0"), ("v8 Wdeg", v8[-1], "base_Wdeg_a_seed0")] if v8 else []
    log("\n=== C. modulator synapses silent (v8) vs weakly excitatory / inhibitory ===")
    compare(out, summary, [("v8 R", "v8 Wdeg", "silent: real vs rewired"),
                           ("R_mod+1.0", "Wdeg_mod+1.0", "excitatory (Shiu convention): real vs rewired"),
                           ("R_mod-0.5", "Wdeg_mod-0.5", "inhibitory: real vs rewired"),
                           ("v8 R", "R_mod+1.0", "real: silent vs excitatory"),
                           ("v8 R", "R_mod-0.5", "real: silent vs inhibitory")], refs)
    log(f"\nstage13 mod finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


def calib_one(job):
    """Healthy brain (no oxygen loss) for 20 s with Goldman coupling: mean rate in the 2nd half, max K, blocked."""
    import importlib
    time.sleep(job.pop("stagger", 0))
    m = importlib.import_module(MODEL)
    for k, v in job["p"].items():
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
    return dict(tonic=job["tonic"], rate_hz=float(r["active"][late].mean() / (m.DT / 1000.0)),
                K_cube_max=float(r["cube_K_end"].max()), K_cube_median=float(np.median(r["cube_K_end"])),
                blocked_end=float(r["blocked"][-1]), cubes_in_SD=float(np.isfinite(r["cube_time"]).mean()))


def goldman_part():
    t0 = time.time()
    with open(os.path.join("runs", "v8_chosen.json")) as f:
        ch = json.load(f)
    p = dict(BASE, SD_DELAY_S=ch["SD_DELAY_S"], K_COUPLING="goldman")
    v8 = sorted(glob.glob(os.path.join("runs", "*_v8_fix")))
    target = float(pd.read_csv(os.path.join(v8[-1], "summary.csv")).set_index("tag").loc["base_R_seed0", "rate_pre_mean_hz"]) \
        if v8 else 12.6
    log(f"\n=== D1. Goldman coupling: healthy stability and drive (target rate {target:.2f} Hz = v8 real brain) ===")
    jobs = [dict(tonic=tn, p=dict(p, TONIC_MEAN=tn)) for tn in [3.0, 3.5, 4.0, 4.5, 5.0]]
    cal = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(calib_one, dict(j, stagger=45 * i * min(SCALE * 10, 1))) for i, j in enumerate(jobs)]
        for fu in as_completed(futs):
            cal.append(fu.result())
    cal = pd.DataFrame(cal).sort_values("tonic")
    cal["stable"] = (cal.blocked_end == 0) & (cal.cubes_in_SD == 0) & (cal.K_cube_max < 10)
    log(cal.round(3).to_string(index=False))
    ok = cal[cal.stable]
    if not len(ok):
        log("NO stable healthy brain with Goldman coupling at any drive: with physical K coupling the model's K release /"
            " clearance balance runs away. Finding to report; death runs skipped.")
        return
    best = ok.iloc[(ok.rate_hz - target).abs().argsort().values[0]]
    tonic = float(best.tonic)
    log(f"chosen drive {tonic} -> {best.rate_hz:.2f} Hz (target {target:.2f})")
    WD = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
    dur = 90000.0 * SCALE
    conds = [dict(name="reox_R_goldman", dur=T_REOX + 48000.0 * SCALE,
                  overrides=dict(p, TONIC_MEAN=tonic, RUN_MODE="death", T_REOX_MS=T_REOX, STOP_EARLY=False)),
             dict(name="R_goldman", dur=dur, overrides=dict(p, TONIC_MEAN=tonic, RUN_MODE="death")),
             dict(name="Wdeg_goldman", dur=dur, overrides=dict(p, TONIC_MEAN=tonic, RUN_MODE="death", **WD))]
    out, summary = batch(conds, "_v9_goldman")
    refs = [("v8 R", v8[-1], "base_R_seed0"), ("v8 Wdeg", v8[-1], "base_Wdeg_a_seed0")] if v8 else []
    log("\n=== D2. does the wiring result survive Goldman coupling? ===")
    compare(out, summary, [("v8 R", "v8 Wdeg", "piecewise (v8): real vs rewired"),
                           ("R_goldman", "Wdeg_goldman", "Goldman: real vs rewired"),
                           ("v8 R", "R_goldman", "real: piecewise vs Goldman")], refs)
    import stage11
    f = os.path.join(out, "probes_reox_R_goldman_seed0.npz")
    if os.path.exists(f):
        pr = np.load(f)
        rec = stage11.recovery(pr, T_REOX / 1000.0, T_ANOX / 1000.0)
        log("recovery (Goldman): " + ", ".join(f"{k} {v:.3g}" for k, v in rec.items()))
        e = stage11.electrode(pr, T_ANOX / 1000.0, T_REOX / 1000.0)
        s = e[e.surge]
        if len(s):
            log("electrodes (median): " + ", ".join(f"{c} {s[c].median():.2f}" for c in
                                                   ["onset_s", "K_at_onset", "pre_rise_mM_s"]))
    log(f"\nstage13 goldman finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage13 started", time.strftime("%Y-%m-%d %H:%M"), sys.argv[1:])
    if "--prep" in sys.argv:
        prep()
    elif "--mod" in sys.argv:
        mod_part()
    elif "--goldman" in sys.argv:
        goldman_part()
    else:
        main_part()
