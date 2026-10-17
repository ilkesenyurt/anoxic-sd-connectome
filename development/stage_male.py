# stage_male.py - PREVIEW replication on the male CNS connectome (brain part), log row 94
# ======================================================================================
# Same model and settings as the stage15 final-calibration runs (v11: Goldman, fixed reset, glia, SD pump cut 0.2,
# SD_TAU 19.6, SD_DELAY 0, E_BASE 14, pump restart 20 s). Only the input files change, plus the healthy drive,
# which is refitted so the male brain fires at the same healthy rate as the female model (12.64 Hz target).
# PREVIEW: the final model is frozen only after stage15/15b; this run is repeated on the frozen model later.
#
# Predictions written BEFORE the run (from the female FlyWire results, rows 85-90):
#   (revised 23:40 Oct 3, before any male data existed, after the female final-calibration result: female at the same
#    settings gives order rho 0.941 and real later by 6.1 s (first SD) / 7.2 s (t50); the old P1 threshold 0.9 came
#    from earlier model versions and would have failed for the female brain too)
#   P1  real vs rewired death order similar but not identical, like the female: rho between 0.85 and 0.98
#   P2  (main) the real brain is more protected: first SD AND t50 at least 2 s later in real than in rewired
#   P3  healthy brain stable at the fitted drive (no block, no SD, cube K < 10 mM)
# Pipeline difference fixed BEFORE the run (row 94): synapse detection differs between FlyEM (male) and FlyWire (female).
#   If the mean number of input synapses per neuron (pairs >= 5 synapses, brain only) differs by more than 20%,
#   the male synapse weight is scaled so the mean synaptic input per neuron matches the female:
#   W_SYN_male = 0.275 x (female synapses per neuron / male synapses per neuron). Then the drive is refitted as planned.
#   Histamine (HIST, 3.9%: mostly photoreceptors R7/R8, absent from FlyWire's transmitter set) gets sign 0 (silent),
#   like DA/SER/OCT.
# Needs in this folder (from malecns_to_model_inputs.ipynb on Colab): synapse_cubes_male_10um.npz,
# connections_male.csv, neurons_male.csv, coordinates_male.csv, classification_male.csv
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

SCALE = float(os.environ.get("MALE_SCALE", "1"))
N_WORKERS = int(os.environ.get("MALE_WORKERS", "5"))
MODEL = "death_v11"
DEFAULT_FILES = dict(CONNECTIONS_CSV="connections_princeton.csv", NEURONS_CSV="neurons.csv",
                     COORDS_CSV="coordinates.csv", CLASS_CSV="classification.csv", CUBE_FILE="synapse_cubes_10um.npz")
FILES = dict(CONNECTIONS_CSV="connections_male.csv", NEURONS_CSV="neurons_male.csv",
             COORDS_CSV="coordinates_male.csv", CLASS_CSV="classification_male.csv",
             CUBE_FILE="synapse_cubes_male_10um.npz")
FINAL = dict(D_UM2_S=770.0, SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2, SD_KMAX=54.0, SD_PUMP_RECOVER=True,
             K_COUPLING="goldman", RESET_MODE="fixed", NA_INACT=False,
             GLIA_ON=True, GLIA_UPTAKE_S=0.4, GLIA_RELEASE_S=0.05, GLIA_CAP=10.0, GLIA_PASSIVE=0.5,
             SD_TAU_S=19.6, SD_DELAY_S=0.0, E_BASE_USE_S=14.0, E_RECOVER_TAU_S=20.0)
WD = dict(SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=1)
DRIVES = [2.5, 3.0, 3.5, 4.0]
TARGET_HZ = 12.64
T_ANOX = runner.T_ANOX_MS


def log(*a):
    print(*a, flush=True)


def check_files():
    missing = [f for f in FILES.values() if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"missing input files: {missing} (run malecns_to_model_inputs.ipynb on Colab first)")
    import importlib
    m = importlib.import_module(MODEL)
    for k, v in {**FILES, **FINAL}.items():
        setattr(m, k, v)
    meta, conn = m.load_data()
    log(f"male brain: {len(meta):,} neurons, {len(conn):,} connected pairs, {int(conn.syn_count.sum()):,} synapses")
    log("transmitters:", meta.nt_type.value_counts().to_dict())
    log("female FlyWire for comparison: 139,255 neurons, 3,732,460 pairs")
    idx, W, sign, coords, N = m.build_network(meta, conn)
    geom = m.build_geometry(meta, idx, coords)
    log(f"geometry: {len(geom['centers_um']):,} cubes (female 23,045)")
    male_spn = float(conn.syn_count.sum()) / N
    for k in FILES:                                   # female defaults back
        setattr(m, k, DEFAULT_FILES[k])
    fmeta, fconn = m.load_data()
    fem_spn = float(fconn.syn_count.sum()) / len(fmeta)
    ratio = fem_spn / male_spn
    w = 0.275 * ratio if abs(ratio - 1.0) > 0.2 else 0.275
    log(f"input synapses per neuron: male {male_spn:.0f}, female {fem_spn:.0f} -> ratio {ratio:.3f}; W_SYN_male {w:.4f}"
        f" ({'scaled' if w != 0.275 else 'unchanged, within 20%'})")
    return w


def calib_one(job):
    import importlib
    time.sleep(job.pop("stagger", 0))
    m = importlib.import_module(MODEL)
    for k, v in {**FILES, **FINAL, "W_SYN": job["w_syn"], "TONIC_MEAN": job["tonic"], "RUN_MODE": "healthy"}.items():
        assert hasattr(m, k), k
        setattr(m, k, v)
    meta, conn = m.load_data()
    m.rng = np.random.default_rng(0)
    idx, W, sign, coords, N = m.build_network(meta, conn)
    vuln = m.build_vulnerability(sign)
    geom = m.build_geometry(meta, idx, coords)
    dur = 20000.0 * SCALE
    r = m.simulate(W, N, vuln, geom, 1e12, dur)
    late = r["t"] >= dur / 2
    return dict(tonic=job["tonic"], rate_hz=float(r["active"][late].mean() / (m.DT / 1000.0)),
                K_cube_max=float(r["cube_K_end"].max()), blocked_end=float(r["blocked"][-1]),
                cubes_in_SD=float(np.isfinite(r["cube_time"]).mean()))


def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    time.sleep(cond.pop("stagger", 0))
    cond["overrides"]["SAVE_PROBES_TO"] = os.path.join(cond["out"], f"probes_{cond['name']}_seed0.npz")
    return runner.run_condition(cond)


def main():
    t0 = time.time()
    log("\n=== 0. input check ===")
    w_syn = check_files()
    FINAL["W_SYN"] = w_syn
    log(f"\n=== 1. healthy drive (target {TARGET_HZ} Hz) ===")
    cal = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(calib_one, dict(tonic=d, w_syn=w_syn, stagger=45 * i * min(SCALE * 10, 1))) for i, d in enumerate(DRIVES)]
        for fu in as_completed(futs):
            cal.append(fu.result())
    cal = pd.DataFrame(cal).sort_values("tonic")
    cal["stable"] = (cal.blocked_end == 0) & (cal.cubes_in_SD == 0) & (cal.K_cube_max < 10)
    log(cal.round(3).to_string(index=False))
    ok = cal[cal.stable]
    log(f"P3 healthy brain stable at some drive: {'YES' if len(ok) else 'NO'}")
    if not len(ok):
        return
    x, y = ok.tonic.values, ok.rate_hz.values
    drive = float(np.interp(TARGET_HZ, y, x)) if len(x) > 1 and np.all(np.diff(y) > 0) else float(x[np.abs(y - TARGET_HZ).argmin()])
    drive = round(drive, 3)
    log(f"male drive {drive} (female 3.36)")

    p = dict(FILES, **FINAL, TONIC_MEAN=drive, RUN_MODE="death")
    conds = [dict(name="male_R", dur=90000.0 * SCALE, overrides=dict(p)),
             dict(name="male_Wdeg", dur=90000.0 * SCALE, overrides=dict(p, **WD))]
    for c in conds:
        c["seed"] = 0
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_male_preview")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    rows = []
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, overrides=dict(c["overrides"]), stagger=45 * i * min(SCALE * 10, 1)))
                for i, c in enumerate(conds)]
        for fu in as_completed(futs):
            rows.append(fu.result())
    summary = pd.DataFrame(rows).set_index("tag")
    runner.save_summary(summary.reset_index(), os.path.join(out, "summary.csv"))
    for _, r in summary.iterrows():
        if r.get("status") == "ERROR":
            log(f"ERROR: {r['error']}")
            return
    a = np.load(os.path.join(out, "cube_time_male_R_seed0.npy"))
    b = np.load(os.path.join(out, "cube_time_male_Wdeg_seed0.npy"))
    k = np.isfinite(a) & np.isfinite(b)
    rho = spearmanr(a[k], b[k])[0] if k.sum() > 10 else np.nan
    f_r, f_w = (np.nanmin(a) - T_ANOX) / 1000, (np.nanmin(b) - T_ANOX) / 1000
    t_r, t_w = summary.loc["male_R_seed0", "t50_s"], summary.loc["male_Wdeg_seed0", "t50_s"]
    log("\n=== 2. male real vs rewired (female, same settings, stage15 D at E_BASE 14: see log row 93) ===")
    log(f"death order rho {rho:.3f}; first SD real {f_r:.1f} s, rewired {f_w:.1f} s; t50 real {t_r:.1f}, rewired {t_w:.1f}; "
        f"healthy rate real {summary.loc['male_R_seed0', 'rate_pre_mean_hz']:.2f} Hz")
    log(f"P1 order similar but not identical (0.85 <= rho <= 0.98; female 0.941): {'YES' if 0.85 <= rho <= 0.98 else 'NO'}")
    d1, d2 = f_r - f_w, t_r - t_w
    log(f"P2 real more protected by >= 2 s (female +6.1 / +7.2 s): first SD {d1:+.1f} s, t50 {d2:+.1f} s -> "
        f"{'YES' if (d1 >= 2 and d2 >= 2) else ('PARTLY' if (d1 > 0 or d2 > 0) else 'NO')}")
    log(f"\nstage_male finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage_male started", time.strftime("%Y-%m-%d %H:%M"))
    main()
