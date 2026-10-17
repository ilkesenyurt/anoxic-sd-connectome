# recover_summary.py - rebuild the summary and correlations of a runner batch
# ==========================================================================
# Use when a batch finished its runs but could not write summary.csv (for
# example because the file was open in Excel). Everything is recomputed from
# the files each run saved: block_time_<tag>.npy and series_<tag>.npz.
#
# Recovered: time to 50% blocked, wave speed, surge (peak activity after the
# cutoff and when), mean K and mean energy at the end, all pairwise Spearman
# correlations of block times.
# NOT recoverable: healthy-control numbers (firing rate, max K), energy at
# block, per-neuron K peak. A run that has a block_time file passed its healthy
# control (the death run only starts if it did).
#
# Set RUN_DIR and run this next to death_v2.py (it reloads neuron positions).
# BOX_HALF_NM in death_v2.py must match the batch (None for whole brain).

import glob
import os
import numpy as np
import pandas as pd

import runner                      # reuses the same speed and correlation code
import death_v2 as m

RUN_DIR = os.path.join("runs", "20260929_230813")   # edit
T_ANOX_MS = runner.T_ANOX_MS

meta, conn = m.load_data()
coords = meta[["x", "y", "z"]].values.astype(np.float32)

rows = []
for f in sorted(glob.glob(os.path.join(RUN_DIR, "block_time_*.npy"))):
    tag = os.path.basename(f)[len("block_time_"):-len(".npy")]
    bt = np.load(f)
    s = np.load(os.path.join(RUN_DIR, f"series_{tag}.npz"))
    assert len(bt) == len(coords), f"{tag}: {len(bt)} neurons vs {len(coords)} loaded - check BOX_HALF_NM"
    res = dict(t=s["t"], active=s["active"], blocked=s["blocked"], block_time=bt,
               E_at_block=np.full(len(bt), np.nan),       # not saved
               K_end=np.full(len(bt), np.nan), K_max=np.full(len(bt), np.nan))
    row = dict(tag=tag, status="ok (recovered)")
    row.update(runner.death_metrics(res, coords, T_ANOX_MS, healthy_peak=np.inf))
    for k in ["surge", "E_at_block_median", "K_end_mean", "K_max_median"]:
        row.pop(k, None)                                  # need data that was not saved
    row["meanK_end"] = float(s["meanK"][-1])
    row["meanE_end"] = float(s["meanE"][-1])
    rows.append(row)
    print("recovered", tag)

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(RUN_DIR, "summary_recovered.csv"), index=False)
cols = ["tag", "t50_s", "speed_mm_min", "surge_peak", "surge_t_s", "meanK_end",
        "meanE_end", "blocked_end"]
print("\n" + summary[cols].to_string(index=False))

tags = summary["tag"].tolist()
corr = runner.correlations([dict(tag=t, status="ok") for t in tags], RUN_DIR)
corr.to_csv(os.path.join(RUN_DIR, "correlations.csv"), index=False)
print("\nSpearman correlation of block times:")
print(corr.to_string(index=False))
