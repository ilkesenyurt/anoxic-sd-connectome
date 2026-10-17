# stage15diag.py - why did the KCl wave stop in stage15 A (reached 3%)? (log row 92) ~5 min, tissue only
# Same trigger as stage13 A2 / stage12. Compares glia x SD pump cut, and model v9 vs v11 code.
import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

BASE = dict(D_UM2_S=770.0, SD_ON=11.0, SD_RATE=55.0, SD_KMAX=54.0, SD_PUMP_RECOVER=True, SD_TAU_S=20.0,
            RUN_MODE="trigger", TRIGGER_HOLD_MS=3000.0)
GLIA = dict(GLIA_ON=True, GLIA_UPTAKE_S=0.4, GLIA_RELEASE_S=0.05, GLIA_CAP=10.0, GLIA_PASSIVE=0.5)
JOBS = [("v9  glia + cut (stage13: 2.19)", "death_v9", dict(GLIA, SD_PUMP=0.2, SD_DELAY_S=0.0)),
        ("v9  noglia + noCut (row 85: 2.22)", "death_v9", dict(SD_PUMP=1.0, SD_DELAY_S=0.05)),
        ("v9  glia + noCut", "death_v9", dict(GLIA, SD_PUMP=1.0, SD_DELAY_S=0.0)),
        ("v11 glia + cut", "death_v11", dict(GLIA, SD_PUMP=0.2, SD_DELAY_S=0.0)),
        ("v11 noglia + noCut", "death_v11", dict(SD_PUMP=1.0, SD_DELAY_S=0.05)),
        ("v11 glia + noCut", "death_v11", dict(GLIA, SD_PUMP=1.0, SD_DELAY_S=0.0)),
        ("v11 glia + noCut, STOP_EARLY on", "death_v11", dict(GLIA, SD_PUMP=1.0, SD_DELAY_S=0.0, STOP_EARLY=True))]


def one(job):
    label, mod, p = job
    import importlib
    m = importlib.import_module(mod)
    for k, v in {**BASE, "STOP_EARLY": False, **p}.items():
        assert hasattr(m, k), k
        setattr(m, k, v)
    meta, conn = m.load_data()
    idx, W, sign, coords, N = m.build_network(meta, conn)
    geom = m.build_geometry(meta, idx, coords)
    del W, conn
    t0 = time.time()
    r = m.simulate_tissue(geom, 60000.0)
    ct = r["cube_time"]
    sp = m.cube_speed_mm_min(ct, geom) if np.isfinite(ct).sum() > 10 else np.nan
    return dict(run=label, reached=float(np.isfinite(ct).mean()), speed=sp, meanK_end=float(r["meanK"][-1]),
                maxK_end=float(r["cube_K_end"].max()), t_end_s=float(r["t"][-1]) / 1000, sim_s=time.time() - t0)


if __name__ == "__main__":
    print("stage15diag started", time.strftime("%Y-%m-%d %H:%M"), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("W", "7"))) as ex:
        for fu in as_completed([ex.submit(one, j) for j in JOBS]):
            rows.append(fu.result())
    print(pd.DataFrame(rows).sort_values("run").round(3).to_string(index=False), flush=True)
    print("stage15diag finished", time.strftime("%Y-%m-%d %H:%M"), flush=True)
