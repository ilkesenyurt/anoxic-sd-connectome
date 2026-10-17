# runner.py - batch runner for the FlyWire death simulation
# =========================================================
# Runs a list of conditions. For EACH condition it:
#   1. runs a healthy control (oxygen never cut) and checks it is stable,
#   2. if stable, runs the death run with the same random draws,
#   3. saves block times + time series, and adds one row to a summary table.
# At the end it compares every pair of death runs (Spearman correlation of
# per-neuron block times).
#
# Nothing in your simulation file needs to change. The runner imports it and
# sets its parameters for each condition, then puts them back afterwards.
#
# HOW TO RUN
#   - In Spyder: keep N_WORKERS = 1 (runs one condition after another).
#   - Parallel (faster): set N_WORKERS = 3 or 4 and run from a terminal
#     (Anaconda Prompt) in this folder:   python runner.py
#     (Parallel runs don't work reliably inside Spyder on Windows.)
#
# OUTPUT (in runs/<date_time>/)
#   row_<tag>.json     one run's results, saved by the run itself (backup)
#   summary.csv        one row per condition: healthy checks + death metrics
#   correlations.csv   Spearman rho for every pair of death runs
#   settings.json      every parameter value used (for the log)
#   block_time_<tag>.npy, series_<tag>.npz   raw results per condition

import os
# one CPU thread per simulation, so parallel runs don't fight over cores
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib
import json
import time
import traceback
from datetime import datetime
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

# %% Settings -----------------------------------------------------------------
MODEL_MODULE = "death_v3"   # your simulation file name, without .py (death_v2 still works)
OUT_DIR      = "runs"
N_WORKERS    = 2            # 1 in Spyder; 3-4 from a terminal
HEALTHY_MS   = 30000.0      # length of each healthy control (K settles in ~20 s)
DEATH_MS     = 40000.0      # length of each death (or trigger/bath) run
T_ANOX_MS    = 2000.0       # oxygen cutoff in the death run
RUN_HEALTHY  = False        # False = skip the healthy control (only for calibration sweeps,
                            #   once the healthy state of that model version is known to be fine)


def condition(name, seed, **overrides):
    """One experiment: a name, a random seed, and the parameters to change."""
    return dict(name=name, seed=seed, overrides=overrides)


# ---- the list of conditions to run (edit this) ----
# Example: the wiring experiment with 3 seeds each.
CONDITIONS = []
# v3.2 calibration sweep: local K pulse in a healthy brain ("trigger").
# v3.1 sweep (SD_RATE 10-40, SD_ON 12/15) never spread beyond ~1.5% of cubes: healthy pumps in the
# depolarized cubes kept clearing K. v3.2 weakens clearance of tissue in SD (SD_PUMP).
# Harris et al.: SD spreads through the whole brain at 0.77 mm/min and the tissue recovers.
# (RUN_HEALTHY = False, DEATH_MS = 40000 for this sweep.)
for pump in [0.1, 0.2, 0.4]:
    for rate in [10.0, 20.0]:
        CONDITIONS.append(condition(f"trig_pump{pump}_rate{rate:.0f}", 0,
                                    RUN_MODE="trigger", SD_PUMP=pump, SD_RATE=rate, SD_ON=12.0))
# The v3 wiring experiment (after calibration):
# for seed in [0, 1]:
#     CONDITIONS += [
#         condition("R_real",    seed, COUPLING="real"),
#         condition("S_swap",    seed, COUPLING="swap"),
#         condition("W_rewired", seed, COUPLING="real", SHUFFLE_CONN=True),
#     ]
# Example for later: SYN_FRACTION robustness
# for f in [0.25, 0.75]:
#     CONDITIONS.append(condition(f"B_syn{f}", 0, PROPAGATION="both", SYN_FRACTION=f))


# %% Helpers ------------------------------------------------------------------
_DEFAULTS = {}   # the parameter values in your file, remembered at first import
_DATA = {}       # loaded data, reused between conditions in the same process


def _fresh_module():
    """Import the simulation and reset every UPPERCASE parameter to the value in
    the file, so one condition's changes never leak into the next."""
    m = importlib.import_module(MODEL_MODULE)
    if not _DEFAULTS:
        _DEFAULTS.update({k: getattr(m, k) for k in dir(m) if k.isupper()})
    for k, v in _DEFAULTS.items():
        setattr(m, k, v)
    return m


def _load(m):
    key = getattr(m, "BOX_HALF_NM", None)
    if key not in _DATA:
        _DATA[key] = m.load_data()
    return _DATA[key]


def _prepare(m, meta, conn, seed):
    """Build the network with a given seed. Called twice per condition so the
    healthy control and the death run get identical random draws."""
    m.rng = np.random.default_rng(seed)
    idx, W, sign, coords, N = m.build_network(meta, conn)
    vuln = m.build_vulnerability(sign)
    if hasattr(m, "build_geometry"):                  # v3: K in cubes
        L = m.build_geometry(meta, idx, coords)
    else:                                             # v2: K per neuron, kNN mixing
        L = m.build_spatial_graph(coords)
    return W, coords, N, vuln, L


def _smooth(x, n=50):
    return np.convolve(x, np.ones(n) / n, mode="same")


def healthy_metrics(res, m):
    k_max = float(res["K_max"].max())
    blocked_end = float(res["blocked"][-1])
    return dict(
        h_peak=float(_smooth(res["active"]).max()),
        h_rate_hz=float(res["active"].mean() / (m.DT / 1000.0)),   # mean firing per neuron
        h_blocked_end=blocked_end,
        h_K_median_end=float(np.median(res["K_end"])),
        h_K_max=k_max,
        healthy_ok=bool(blocked_end == 0 and k_max < m.K_THRESH),
    )


def death_metrics(res, coords, t_anox_ms, healthy_peak, m=None, geom=None):
    t = res["t"]
    act = _smooth(res["active"])
    after = t >= t_anox_ms
    i = int(np.argmax(act[after]))
    out = dict(
        surge_peak=float(act[after][i]),
        surge_t_s=float((t[after][i] - t_anox_ms) / 1000),
        surge=bool(act[after][i] > healthy_peak),
        blocked_end=float(res["blocked"][-1]),
        blocked_max=float(res["blocked"].max()),
        K_end_mean=float(res["K_end"].mean()),
        K_max_median=float(np.median(res["K_max"])),
    )
    reached = np.flatnonzero(res["blocked"] >= 0.5)
    out["t50_s"] = float((t[reached[0]] - t_anox_ms) / 1000) if reached.size else np.nan

    eb = res["E_at_block"]
    out["E_at_block_median"] = float(np.nanmedian(eb)) if np.isfinite(eb).any() else np.nan

    # neighbour-based wave speed (nm/ms * 0.06 = mm/min)
    bt = res["block_time"]
    d, nb = cKDTree(coords).query(coords, k=9)
    dt_pairs = np.abs(bt[:, None] - bt[nb[:, 1:]])
    ok = np.isfinite(dt_pairs) & (dt_pairs > 0)
    out["speed_mm_min"] = float(np.median(d[:, 1:][ok] / dt_pairs[ok]) * 0.06) if ok.any() else np.nan
    # NOTE: this neighbour-pair speed overestimates a plane wave ~2x (pairs not lined up
    # with the wave see small time differences). v3 adds the gradient-based cube speed:
    if m is not None and "cube_time" in res and hasattr(m, "cube_speed_mm_min"):
        out["speed_cube_mm_min"] = m.cube_speed_mm_min(res["cube_time"], geom)
        out["cubes_reached"] = float(np.isfinite(res["cube_time"]).mean())
    return out


# %% One condition ------------------------------------------------------------
def run_condition(cond):
    tag = f"{cond['name']}_seed{cond['seed']}"
    row = dict(tag=tag, name=cond["name"], seed=cond["seed"],
               **{f"p_{k}": v for k, v in cond["overrides"].items()})
    t0 = time.time()
    try:
        m = _fresh_module()
        for k, v in cond["overrides"].items():
            if not hasattr(m, k):
                raise AttributeError(f"{MODEL_MODULE} has no parameter called {k}")
            setattr(m, k, v)

        meta, conn = _load(m)

        # 1. healthy control (no oxygen loss, no trigger)
        mode = getattr(m, "RUN_MODE", "death")
        if mode == "healthy":
            mode = "death"
        print(f"  {time.strftime('%H:%M')}  {tag}: healthy control started", flush=True)
        W, coords, N, vuln, L = _prepare(m, meta, conn, cond["seed"])
        if hasattr(m, "RUN_MODE"):
            m.RUN_MODE = "healthy"
        if RUN_HEALTHY:
            res_h = m.simulate(W, N, vuln, L, 1e12, HEALTHY_MS)
            row.update(healthy_metrics(res_h, m))
            if not row["healthy_ok"]:
                row["status"] = "healthy control FAILED - death run skipped"
                return row
        else:
            row.update(h_peak=np.nan, healthy_ok=None)

        # 2. death run, same random draws as the control
        print(f"  {time.strftime('%H:%M')}  {tag}: healthy passed, death run started", flush=True)
        W, coords, N, vuln, L = _prepare(m, meta, conn, cond["seed"])
        if hasattr(m, "RUN_MODE"):
            m.RUN_MODE = mode                  # "death", or v3 "trigger" / "bath" (oxygen kept)
        row["mode"] = mode
        res_d = m.simulate(W, N, vuln, L, T_ANOX_MS if mode == "death" else 1e12, DEATH_MS)
        row.update(death_metrics(res_d, coords, T_ANOX_MS, row["h_peak"], m, L))

        out = cond["out"]
        np.save(os.path.join(out, f"block_time_{tag}.npy"), res_d["block_time"])
        if "cube_time" in res_d:
            np.save(os.path.join(out, f"cube_time_{tag}.npy"), res_d["cube_time"])
        if "rate_pre_hz" in res_d:                               # v6: firing rate per neuron
            np.savez_compressed(os.path.join(out, f"rates_{tag}.npz"),
                                pre=res_d["rate_pre_hz"].astype(np.float32),
                                post=res_d["rate_post_hz"].astype(np.float32))
            row["rate_pre_mean_hz"] = float(np.mean(res_d["rate_pre_hz"]))
            row["rate_post_mean_hz"] = float(np.mean(res_d["rate_post_hz"]))
        np.savez_compressed(os.path.join(out, f"series_{tag}.npz"),
                            **{k: res_d[k] for k in ["t", "active", "blocked",
                                                     "meanK", "meanE", "chi"]})
        row["status"] = "ok"
    except Exception:
        row["status"] = "ERROR"
        row["error"] = traceback.format_exc(limit=3)
    finally:
        row["minutes"] = round((time.time() - t0) / 60, 1)
        print(f"[{row['status']}] {tag}  ({row['minutes']} min)", flush=True)
        # each run also saves its own row, so nothing is lost if summary.csv
        # cannot be written (e.g. open in Excel)
        try:
            with open(os.path.join(cond["out"], f"row_{tag}.json"), "w") as f:
                json.dump(row, f, indent=1,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o))
        except Exception as e:
            print(f"  could not save row_{tag}.json: {e}", flush=True)
    return row


def save_summary(rows, path):
    """Write summary.csv; if it is locked (open in Excel), warn and carry on."""
    try:
        pd.DataFrame(rows).to_csv(path, index=False)
    except PermissionError:
        print(f"  WARNING: could not write {path} (open in Excel?). "
              "Results are safe in the row_*.json files; close the file and it "
              "will be written after the next run.", flush=True)


# %% Pairwise comparison ------------------------------------------------------
def correlations(rows, out):
    """Spearman correlation of block times (per neuron) for every pair of runs.
    v3 also compares cube arrival times (where the wave is when), which is the
    right comparison when neurons were moved (the "swap" control)."""
    tags = sorted(r["tag"] for r in rows if r.get("status") == "ok")
    bts = {t: np.load(os.path.join(out, f"block_time_{t}.npy")) for t in tags}
    cts = {}
    for t in tags:
        f = os.path.join(out, f"cube_time_{t}.npy")
        if os.path.exists(f):
            cts[t] = np.load(f)
    pairs = []
    for a, b in combinations(tags, 2):
        x, y = bts[a], bts[b]
        ok = np.isfinite(x) & np.isfinite(y)
        rho = spearmanr(x[ok], y[ok])[0] if ok.sum() > 10 else np.nan
        row = dict(a=a, b=b, spearman=rho, n_neurons=int(ok.sum()))
        if a in cts and b in cts and len(cts[a]) == len(cts[b]):
            ok = np.isfinite(cts[a]) & np.isfinite(cts[b])
            row["spearman_cubes"] = spearmanr(cts[a][ok], cts[b][ok])[0] if ok.sum() > 10 else np.nan
        pairs.append(row)
    return pd.DataFrame(pairs)


# %% Main ---------------------------------------------------------------------
if __name__ == "__main__":
    out = os.path.join(OUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    # record every parameter value used, for the calibration log
    _fresh_module()
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(dict(defaults={k: repr(v) for k, v in _DEFAULTS.items()},
                       conditions=CONDITIONS, HEALTHY_MS=HEALTHY_MS,
                       DEATH_MS=DEATH_MS, T_ANOX_MS=T_ANOX_MS), f, indent=2)

    conds = [dict(c, out=out) for c in CONDITIONS]
    print(f"{len(conds)} conditions, {N_WORKERS} at a time -> {out}")

    rows = []
    summary_path = os.path.join(out, "summary.csv")
    if N_WORKERS == 1:
        for c in conds:
            rows.append(run_condition(c))
            save_summary(rows, summary_path)                       # saved after every run
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(run_condition, c) for c in conds]
            for fut in as_completed(futures):
                rows.append(fut.result())
                save_summary(rows, summary_path)

    summary = pd.DataFrame(rows).sort_values("tag")
    save_summary(summary, summary_path)
    cols = [c for c in ["tag", "status", "healthy_ok", "h_rate_hz", "t50_s",
                        "speed_mm_min", "speed_cube_mm_min", "cubes_reached", "blocked_max",
                        "E_at_block_median", "K_end_mean", "blocked_end", "surge", "minutes"] if c in summary]
    print("\n", summary[cols].to_string(index=False))

    corr = correlations(rows, out)
    if len(corr):
        try:
            corr.to_csv(os.path.join(out, "correlations.csv"), index=False)
        except PermissionError:
            print("  WARNING: correlations.csv is open elsewhere; printed below only")
        print("\nSpearman correlation of block times:")
        print(corr.to_string(index=False))

    for r in rows:
        if r.get("status") == "ERROR":
            print(f"\nERROR in {r['tag']}:\n{r['error']}")
