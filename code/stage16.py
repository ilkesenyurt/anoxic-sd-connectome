# stage16.py - MAIN EXPERIMENT on the frozen model (v11 final, log row 95): does the real wiring change how the brain
# dies? Female (FlyWire) and male (FlyEM male CNS, brain only), with repeats.
# =====================================================================================================================
# Frozen settings (row 95): v11, Goldman K coupling, fixed reset, glia (0.4/0.05/10/0.5), SD pump cut 0.2,
#   SD_TAU 19.6 s, SD_DELAY 0, E_BASE_USE_S 6 (Andersen 2025 onset rule), pump restart 20 s, D 770, SD_ON 11, SD_RATE 55.
#   Female: drive 3.36, W_SYN 0.275. Male: drive 3.418, W_SYN 0.2118 (pipeline scaling, row 94).
# Per sex (10 runs): real brain x 3 noise seeds (0,1,2); degree-preserving rewired x 5 rewirings (seed 0);
#   E->I loop control 'break_recip' x 2 (row 85: E-I targeting kept, reciprocity broken).
# Hypotheses fixed BEFORE the run:
#   H1 (main) in BOTH sexes the real brain dies later: t50 of every real run > t50 of every rewired run
#        (complete separation, one-sided Mann-Whitney p = 1/56 = 0.018 for 3 vs 5) and mean gap >= 2 s.
#   H2 the death ORDER of real vs rewired differs more than real vs real (other noise seeds):
#        mean rho(real0, rewired_k) < mean rho(real0, real_s).
#   H3 break_recip behaves like the real brain (row 85): its t50 is closer to the real mean than to the rewired mean.
#   Secondary (reported, no threshold): first SD, onset spread, male vs female gap sizes.
# Small outputs only.
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import runner

SCALE = float(os.environ.get("STAGE16_SCALE", "1"))
N_WORKERS = int(os.environ.get("STAGE16_WORKERS", "5"))
MODEL = "death_v11"
FROZEN = dict(D_UM2_S=770.0, SD_ON=11.0, SD_RATE=55.0, SD_PUMP=0.2, SD_KMAX=54.0, SD_PUMP_RECOVER=True,
              K_COUPLING="goldman", RESET_MODE="fixed", NA_INACT=False,
              GLIA_ON=True, GLIA_UPTAKE_S=0.4, GLIA_RELEASE_S=0.05, GLIA_CAP=10.0, GLIA_PASSIVE=0.5,
              SD_TAU_S=19.6, SD_DELAY_S=0.0, E_BASE_USE_S=6.0, E_RECOVER_TAU_S=20.0, RUN_MODE="death")
SEX = {"female": dict(TONIC_MEAN=3.36, W_SYN=0.275),
       "male": dict(TONIC_MEAN=3.418, W_SYN=0.2118, CONNECTIONS_CSV="connections_male.csv",
                    NEURONS_CSV="neurons_male.csv", COORDS_CSV="coordinates_male.csv",
                    CLASS_CSV="classification_male.csv", CUBE_FILE="synapse_cubes_male_10um.npz")}
T_ANOX = runner.T_ANOX_MS


def log(*a):
    print(*a, flush=True)


def run_one(cond):
    runner.MODEL_MODULE = MODEL
    runner.DEATH_MS = cond.pop("dur")
    runner.RUN_HEALTHY = False
    runner._DATA.clear()                      # the data cache is keyed by box only: never reuse across sexes
    time.sleep(cond.pop("stagger", 0))
    cond["overrides"]["SAVE_PROBES_TO"] = os.path.join(cond["out"], f"probes_{cond['name']}_seed{cond['seed']}.npz")
    return runner.run_condition(cond)


def conditions():
    conds = []
    for sex, sp in SEX.items():
        p = dict(FROZEN, **sp)
        for s in [0, 1, 2]:
            conds.append(dict(name=f"{sex}_R", seed=s, overrides=dict(p)))
        for k in [1, 2, 3, 4, 5]:
            conds.append(dict(name=f"{sex}_Wdeg{k}", seed=0,
                              overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="degree", SHUFFLE_SEED=k)))
        for k in [1, 2]:
            conds.append(dict(name=f"{sex}_EI{k}", seed=0,
                              overrides=dict(p, SHUFFLE_CONN=True, SHUFFLE_MODE="break_recip", SHUFFLE_SEED=k)))
    return conds


def analyse(out, summary):
    sm = summary.set_index("tag")
    rows, verdict = [], {}
    for sex in SEX:
        def ct(tag):
            f = os.path.join(out, f"cube_time_{tag}.npy")
            return np.load(f) if os.path.exists(f) else None

        def rho(a, b):
            if a is None or b is None:
                return np.nan
            k = np.isfinite(a) & np.isfinite(b)
            return spearmanr(a[k], b[k])[0] if k.sum() > 10 else np.nan

        def first(tag):
            c = ct(tag)
            return (np.nanmin(c) - T_ANOX) / 1000 if c is not None and np.isfinite(c).any() else np.nan

        groups = {"real": [f"{sex}_R_seed{s}" for s in [0, 1, 2]],
                  "rewired": [f"{sex}_Wdeg{k}_seed0" for k in [1, 2, 3, 4, 5]],
                  "EI": [f"{sex}_EI{k}_seed0" for k in [1, 2]]}
        g = {}
        for name, tags in groups.items():
            tags = [t for t in tags if t in sm.index and sm.loc[t, "status"] == "ok"]
            g[name] = dict(tags=tags, t50=np.array([sm.loc[t, "t50_s"] for t in tags], float),
                           first=np.array([first(t) for t in tags], float))
            for t in tags:
                rows.append(dict(sex=sex, group=name, run=t, t50_s=sm.loc[t, "t50_s"], first_SD_s=first(t),
                                 rate_pre_hz=sm.loc[t, "rate_pre_mean_hz"]))
        r0 = ct(f"{sex}_R_seed0")
        rho_rr = [rho(r0, ct(t)) for t in g["real"]["tags"][1:]]
        rho_rw = [rho(r0, ct(t)) for t in g["rewired"]["tags"]]
        rho_re = [rho(r0, ct(t)) for t in g["EI"]["tags"]]
        tr, tw, te = g["real"]["t50"], g["rewired"]["t50"], g["EI"]["t50"]
        p = mannwhitneyu(tr, tw, alternative="greater").pvalue if len(tr) and len(tw) else np.nan
        gap = float(tr.mean() - tw.mean()) if len(tr) and len(tw) else np.nan
        h1 = bool(len(tr) and len(tw) and tr.min() > tw.max() and gap >= 2.0)
        h2 = bool(np.nanmean(rho_rw) < np.nanmean(rho_rr)) if rho_rr and rho_rw else None
        h3 = bool(abs(te.mean() - tr.mean()) < abs(te.mean() - tw.mean())) if len(te) and len(tr) and len(tw) else None
        verdict[sex] = dict(H1=h1, H2=h2, H3=h3, t50_gap_s=round(gap, 2), p_one_sided=float(p),
                            real_t50=f"{tr.mean():.1f} +- {tr.std(ddof=1) if len(tr) > 1 else 0:.1f}",
                            rewired_t50=f"{tw.mean():.1f} +- {tw.std(ddof=1) if len(tw) > 1 else 0:.1f}",
                            EI_t50=f"{te.mean():.1f}" if len(te) else "nan",
                            first_SD_real=f"{np.nanmean(g['real']['first']):.1f}",
                            first_SD_rewired=f"{np.nanmean(g['rewired']['first']):.1f}",
                            rho_real_vs_real=[round(x, 3) for x in rho_rr],
                            rho_real_vs_rewired=[round(x, 3) for x in rho_rw],
                            rho_real_vs_EI=[round(x, 3) for x in rho_re])
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out, "per_run.csv"), index=False)
    log("\n=== per run ===")
    log(tab.round(2).to_string(index=False))
    log("\n=== hypotheses (fixed before the run) ===")
    for sex, v in verdict.items():
        log(f"\n{sex}:")
        for k, x in v.items():
            log(f"  {k}: {x}")
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1, default=str)


def main():
    t0 = time.time()
    conds = conditions()
    out = os.path.join(runner.OUT_DIR, time.strftime("%Y%m%d_%H%M%S") + "_v11_main")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "settings.json"), "w") as f:
        json.dump(conds, f, indent=1, default=str)
    log(f"{len(conds)} runs -> {out}")
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(run_one, dict(c, out=out, dur=90000.0 * SCALE, overrides=dict(c["overrides"]),
                                        stagger=45 * (i % N_WORKERS) * min(SCALE * 10, 1))) for i, c in enumerate(conds)]
        for fu in as_completed(futs):
            r = fu.result()
            rows.append(r)
            log(f"  done {r['tag']} ({r.get('status')})")
            runner.save_summary(rows, os.path.join(out, "summary.csv"))
    summary = pd.DataFrame(rows)
    runner.save_summary(summary, os.path.join(out, "summary.csv"))
    for r in rows:
        if r.get("status") == "ERROR":
            log(f"\nERROR in {r['tag']}:\n{r['error']}")
    analyse(out, summary)
    log(f"\nstage16 finished {time.strftime('%Y-%m-%d %H:%M')} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    log("stage16 started", time.strftime("%Y-%m-%d %H:%M"))
    if "--analyse" in sys.argv:
        out = sorted(glob.glob(os.path.join(runner.OUT_DIR, "*_v11_main")))[-1]
        analyse(out, pd.read_csv(os.path.join(out, "summary.csv")))
    else:
        main()
