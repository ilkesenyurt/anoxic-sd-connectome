# stage15b.py - how much does the guessed SD pump cut matter? (log row 93) ~5 min, tissue only
# SD_PUMP = pump strength in SD tissue (0.2 = the old guess; 0.5 ~ "half", the extracellular space roughly halves
# during SD). Final configuration otherwise (glia on, SD_TAU 19.6, KCl trigger). Wave reach, speed, SD duration.
import os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

MODEL = "death_v11"
BASE = dict(D_UM2_S=770.0, SD_ON=11.0, SD_RATE=55.0, SD_KMAX=54.0, SD_PUMP_RECOVER=True, SD_TAU_S=19.6,
            RUN_MODE="trigger", TRIGGER_HOLD_MS=3000.0, STOP_EARLY=False,
            GLIA_ON=True, GLIA_UPTAKE_S=0.4, GLIA_RELEASE_S=0.05, GLIA_CAP=10.0, GLIA_PASSIVE=0.5)
JOBS = [dict(SD_PUMP=sp, SD_DELAY_S=d) for sp in [0.2, 0.35, 0.5, 0.75] for d in [0.0, 0.05]]


def one(p):
    import importlib
    m = importlib.import_module(MODEL)
    for k, v in {**BASE, **p}.items():
        assert hasattr(m, k), k
        setattr(m, k, v)
    meta, conn = m.load_data()
    idx, W, sign, coords, N = m.build_network(meta, conn)
    geom = m.build_geometry(meta, idx, coords)
    del W, conn
    r = m.simulate_tissue(geom, 100000.0)
    ct = r["cube_time"]
    sp = m.cube_speed_mm_min(ct, geom) if np.isfinite(ct).sum() > 10 else np.nan
    t = r["probe_t"] / 1000.0
    V = 58.2 * np.log10((r["probe_K"] + 3.5) / 6.5)
    w = []
    for q in range(V.shape[1]):
        v = V[:, q]; b = float(v[t < 1.5].mean()); pk = float(v.max())
        if pk - b < 15:
            continue
        a = np.flatnonzero(v > b + (pk - b) / 2)
        if t[a[0]] < 5.0 or a[-1] == len(v) - 1:
            continue
        w.append(float(t[a[-1]] - t[a[0]]))
    return dict(SD_PUMP=p["SD_PUMP"], delay=p["SD_DELAY_S"], reached=float(np.isfinite(ct).mean()), speed=sp,
                sd_width_s=float(np.median(w)) if w else np.nan, K_peak_median=float(np.median(r["probe_K"].max(0))))


if __name__ == "__main__":
    print("stage15b started", time.strftime("%Y-%m-%d %H:%M"), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("W", "6"))) as ex:
        for fu in as_completed([ex.submit(one, j) for j in JOBS]):
            rows.append(fu.result())
    tab = pd.DataFrame(rows).sort_values(["SD_PUMP", "delay"])
    print("\n=== SD pump cut sensitivity (fly speed 2.1-2.5 mm/min; locust SD duration 37 s) ===", flush=True)
    print(tab.round(3).to_string(index=False), flush=True)
    tab.to_csv(os.path.join("runs", "v11_sd_pump_sensitivity.csv"), index=False)
    print("stage15b finished", time.strftime("%Y-%m-%d %H:%M"), flush=True)
