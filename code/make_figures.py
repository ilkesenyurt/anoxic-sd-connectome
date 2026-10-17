# make_figures.py - report figures for the FlyWire anoxic-death simulation (frozen model v11, log rows 95-100).
# Uses only the small result files in runs\ (summary/per_run csv, series/probes npz, cube_time npy, geom npz).
# Usage:  python make_figures.py [runs_folder] [output_folder]      (defaults: runs  figures)
# Output: one PDF (vector, for the paper) and one PNG (300 dpi, for viewing) per figure.
import glob
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)
MAIN = sorted(glob.glob(os.path.join(RUNS, "*_v11_main")))[-1]
EXTRA = sorted(glob.glob(os.path.join(RUNS, "*_v11_extras")))[-1]
RATE = sorted(glob.glob(os.path.join(RUNS, "*_v11_ratematch")))[-1]
S20 = sorted(glob.glob(os.path.join(RUNS, "*_v11_stage20")))[-1]
SENS = sorted(glob.glob(os.path.join(RUNS, "*_v11_onset_sens")))[-1]
LATER = [sorted(g)[-1] for g in (glob.glob(os.path.join(RUNS, f"*_v11_stage{k}")) for k in (21, 22, 23, 24, 25, 26, 27, 28)) if g]
T_ANOX = 2000.0           # ms, oxygen cut in every death run
T_REOX = 62000.0          # ms, air back in the reoxygenation run
DT_S = 0.1e-3             # model time step (s): firing rate (Hz) = recorded 'active' / DT_S

# ---- style: one palette, colour = identity, text in ink colours ------------------------------------------------
REAL, REW, CTRL = "#2a78d6", "#eb6834", "#1baf7a"           # validated categorical slots 1-3 (dataviz palette)
WSI, WSE = "#7a5bc4", "#c08a00"                              # receiver-specific E/I-preserving rewirings
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]   # dark -> light
CMAP_TIME = LinearSegmentedColormap.from_list("arrival", BLUE_RAMP)       # early (died first) = dark
CMAP_TIME.set_bad("#f0efec")
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
    "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150, "axes.titlelocation": "left"})
W2 = 7.2                   # double-column width (inches)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def letter(ax, s, x=-0.16, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def grid_y(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ---- data ---------------------------------------------------------------------------------------------------------
def summ(folder):
    s = pd.read_csv(os.path.join(folder, "summary.csv"))
    return s.set_index("tag")


S = pd.concat([summ(MAIN), summ(EXTRA), summ(RATE), summ(S20), *[summ(f) for f in LATER],
               pd.read_csv(os.path.join(SENS, "sensitivity_summary.csv")).set_index("tag")])
NEWM = pd.read_csv(os.path.join(S20, "main_runs_new_measures.csv")).set_index("run")
GEOM = {sex: dict(np.load(os.path.join(RUNS, f"geom_{sex}.npz"))) for sex in ("female", "male")}


def folder_of(tag):
    for f in (MAIN, EXTRA, RATE, S20, SENS, *LATER):
        if os.path.exists(os.path.join(f, f"cube_time_{tag}.npy")) or os.path.exists(os.path.join(f, f"probes_{tag}.npz")):
            return f
    raise FileNotFoundError(tag)


def cube_time(tag):
    return (np.load(os.path.join(folder_of(tag), f"cube_time_{tag}.npy")) - T_ANOX) / 1000.0


def series(tag):
    d = np.load(os.path.join(folder_of(tag), f"series_{tag}.npz"))
    return {k: d[k] for k in d.files}


def probes(tag):
    d = np.load(os.path.join(folder_of(tag), f"probes_{tag}.npz"))
    return {k: d[k] for k in d.files}


def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def t50(tag):
    return float(S.loc[tag, "t50_s"])


NONBRAIN = ("none", "<unspecified>", "ADMN", "CV", "DMetaN", "DProN", "MesoAN", "MesoLN", "MetaLN", "PDMN", "ProAN",
            "ProCN", "ProLN")


def brain_cubes(sex):
    """Cubes of brain neuropil: nerve-labelled, unassigned and unnamed cubes excluded (log row 110)."""
    return ~np.array([r.startswith(NONBRAIN) for r in GEOM[sex]["cube_region"]])


def first_sd(tag):
    ct = cube_time(tag)
    return float(np.nanmin(ct[brain_cubes(tag.split("_")[0])]))


TAGS = {sex: dict(real=[f"{sex}_R_seed{s}" for s in range(3)],
                  rewired=[f"{sex}_Wdeg{k}_seed0" for k in range(1, 6)],
                  ei=[f"{sex}_EI{k}_seed0" for k in (1, 2)],
                  matched=[f"{sex}_WdegM{k}_seed0" for k in (1, 2, 3)],
                  up=[f"{sex}_Rup_seed0"],
                  wsign=[f"{sex}_Wsign{k}_seed0" for k in range(1, 6)],
                  rwarm=[f"{sex}_R_warm20_seed0"], wwarm=[f"{sex}_W_warm20_seed0"],
                  rpump=[f"{sex}_R_pump035_seed0"], wpump=[f"{sex}_W_pump035_seed0"],
                  wsignI=[f"{sex}_WsignI{k}_seed0" for k in (1, 2)], wsignE=[f"{sex}_WsignE{k}_seed0" for k in (1, 2)],
                  rcap5=[f"{sex}_R_cap5_seed0"], wcap5=[f"{sex}_W_cap5_seed0"],
                  rcap20=[f"{sex}_R_cap20_seed0"], wcap20=[f"{sex}_W_cap20_seed0"],
                  # glutamate neutral: stage22 (female, drive rule) and stage23 (male drive rule; weight rule both)
                  rglutD=[f"{sex}_R_glut0_seed0", f"{sex}_R_glut0D_seed0"],
                  wglutD=[f"{sex}_Wdeg{k}_glut0{x}_seed0" for k in (1, 2) for x in ("", "D")],
                  rglutW=[f"{sex}_R_glut0W_seed0"], wglutW=[f"{sex}_Wdeg{k}_glut0W_seed0" for k in (1, 2)],
                  rctrl=[f"{sex}_R_ctrlW_seed0"], wctrl=[f"{sex}_Wdeg{k}_ctrlW_seed0" for k in (1, 2)],
                  rphoto=[f"{sex}_R_photo0_seed0"], wphoto=[f"{sex}_Wdeg{k}_photo0_seed0" for k in (1, 2)],
                  keep=[f"{sex}_keep{x}_seed0" for x in ("", "1", "2")],
                  wsignIM=[f"{sex}_WsignI1M_seed0"], wsignEM=[f"{sex}_WsignE1M_seed0"],
                  keepM=[f"{sex}_keep1M_seed0"], keepEIM=[f"{sex}_keepEI1M_seed0"], keepEIsM=[f"{sex}_keepEIs1M_seed0"], keepShufM=[f"{sex}_keepShuf1M_seed0"],
                  brk=[f"{sex}_break{x}_seed0" for x in ("", "1", "2")])
        for sex in ("female", "male")}
for _sex in TAGS:                                     # keep only runs that exist (stage23 may still be running)
    for _g in ("rglutD", "wglutD", "rglutW", "wglutW", "rctrl", "wctrl", "rphoto", "wphoto", "keep", "wsignIM", "wsignEM", "keepM", "keepEIM", "keepEIsM", "keepShufM", "brk"):
        TAGS[_sex][_g] = [t for t in TAGS[_sex][_g] if t in S.index]
SEXLAB = {"female": "Female (FlyWire)", "male": "Male (FlyEM male CNS)"}


def front_map(sex, ct, res=10.0):
    """Front view: earliest SD arrival along the anterior-posterior axis for every (x, y) pixel."""
    c = GEOM[sex]["centers_um"]
    x, y = c[:, 0], c[:, 1]
    ix = np.round((x - x.min()) / res).astype(int)
    iy = np.round((y - y.min()) / res).astype(int)
    img = np.full((iy.max() + 1, ix.max() + 1), np.inf)
    v = np.where(np.isfinite(ct), ct, np.inf)
    np.minimum.at(img, (iy, ix), v)
    inside = np.zeros_like(img, bool)
    inside[iy, ix] = True
    img[~np.isfinite(img)] = np.nan
    img = np.ma.masked_where(~inside, img)
    ext = [0, (ix.max() + 1) * res / 1000, (iy.max() + 1) * res / 1000, 0]     # mm; y down = ventral
    return img, ext


def draw_map(ax, sex, ct, vmin, vmax, title):
    img, ext = front_map(sex, ct)
    im = ax.imshow(img, cmap=CMAP_TIME, vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(title)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    x0 = ext[1] * 0.03                                            # 100 um scale bar
    ax.plot([x0, x0 + 0.1], [ext[2] * 0.97] * 2, color=INK, lw=1.5, solid_capstyle="butt")
    ax.text(x0 + 0.05, ext[2] * 0.93, "100 µm", ha="center", va="bottom", fontsize=6, color=INK2)
    return im


# =================================================================================================================
def fig1():
    """The model brain dies like a fly brain (female, real wiring)."""
    tag = "female_R_seed0"
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.05, 1.0], hspace=0.55, wspace=0.42)
    sr = series(tag)
    t = (sr["t"] - T_ANOX) / 1000
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, smooth(sr["active"], 200) / DT_S, color=REAL, lw=1)
    ax.axvline(0, color=MUTED, lw=0.6, ls="--")
    ax.set(xlim=(-2, 30), xlabel="Time after O₂ off (s)", ylabel="Mean firing rate (Hz)",
           title="Firing rises, then stops")
    grid_y(ax); letter(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, sr["blocked"], color=REAL)
    ax.axvline(0, color=MUTED, lw=0.6, ls="--")
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=":")
    ax.annotate(f"t50 = {t50(tag):.1f} s", (t50(tag), 0.5), (t50(tag) - 14, 0.72), fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set(xlim=(-2, 30), ylim=(-0.02, 1.02), xlabel="Time after O₂ off (s)", ylabel="Fraction of neurons blocked",
           title="Depolarization block")
    grid_y(ax); letter(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    pr = probes(tag)
    tp = (pr["t"] - T_ANOX) / 1000
    for j in range(pr["K"].shape[1]):
        ax.plot(tp, pr["K"][:, j], color="#b7d3f6", lw=0.5)
    ax.plot([], [], color="#b7d3f6", lw=0.8, label="one electrode")
    ax.plot(tp, np.median(pr["K"], axis=1), color=REAL, lw=1.4, label="median")
    ax.axvline(0, color=MUTED, lw=0.6, ls="--")
    ax.set(xlim=(-2, 30), xlabel="Time after O₂ off (s)", ylabel="Extracellular K⁺ (mM)",
           title="K⁺ at 41 virtual electrodes")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.6))
    grid_y(ax); letter(ax, "c")

    ax = fig.add_subplot(gs[1, 0:2])
    ct = cube_time(tag)
    im = draw_map(ax, "female", ct, np.nanpercentile(ct, 1), np.nanpercentile(ct, 99),
                  "Where SD arrives first (front view)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("SD arrival (s after O₂ off)"); cb.outline.set_visible(False)
    letter(ax, "d", x=-0.07)

    # e: model vs fly measurements (calibration targets)
    ax = fig.add_subplot(gs[1, 2])
    p = probes("female_reox_seed0")
    st, v = p["series_t"] / 1000, p["meanV"].astype(float)
    ta, tr = T_ANOX / 1000, T_REOX / 1000
    base = v[(st > 0.5) & (st < ta)].mean(); plat = v[(st > ta) & (st < tr)].max()
    onset = st[np.flatnonzero((st > ta) & (v >= base + (plat - base) / 2))[0]] - ta
    voff = np.interp(tr, st, v)
    rec = st[np.flatnonzero((st > tr) & (v <= base + (voff - base) / 2))[0]] - tr
    rows = [  # label, fly low, fly point, fly high, model, source
        ("Wave speed (mm/min)", 2.1, None, 2.5, 2.19, "Andersen 2025"),
        ("SD onset after N₂ (s; model: DC midpoint)", 10, None, 22, onset, "Andersen 2025"),
        ("DC half-recovery (s)", 11.76, None, 16.13, rec, "Spong 2016"),
    ]
    ax.set_xlim(0, 1); ax.set_ylim(len(rows) - 0.4, -0.7); ax.axis("off")
    ax.set_title("Model vs fly data")
    for i, (lab, lo, pt, hi, mod, src) in enumerate(rows):
        span = max(hi, mod) * 1.2
        xs = lambda q: 0.02 + 0.96 * q / span
        ax.text(0.0, i - 0.42, lab, fontsize=6.5, color=INK, va="center")
        ax.plot([xs(0), xs(span)], [i, i], color=GRID, lw=0.8)
        ax.plot([xs(lo), xs(hi)], [i, i], color="#c3c2b7", lw=6, solid_capstyle="butt")
        if pt is not None:
            ax.plot(xs(pt), i, "|", color=INK2, ms=8, mew=1)
        ax.plot(xs(mod), i, "o", color=REAL, ms=5.5, mec="white", mew=0.7, zorder=3)
        ax.text(0.0, i + 0.3, f"fly: {src}", fontsize=5.5, color=MUTED, va="center")
        ax.text(1.0, i + 0.3, f"model {mod:.1f}", fontsize=5.8, color=REAL, va="center", ha="right")
    ax.set_ylim(len(rows) + 0.15, -0.7)
    ax.text(0.0, len(rows) + 0.1, "grey band = fly range (onset: 35 → 15 °C);\nall three were calibration targets",
            fontsize=5.5, color=MUTED, va="bottom")
    letter(ax, "e", x=-0.12)
    save(fig, "Fig1_model_dies_like_a_fly_brain")


# =================================================================================================================
def strip(ax, groups, key, ylab):
    """groups: list of (label, tags, colour, marker)."""
    rng = np.random.default_rng(0)
    for i, (lab, tags, col, mk) in enumerate(groups):
        vals = np.array([key(t) for t in tags])
        x = i + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=22, color=col, marker=mk, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [vals.mean()] * 2, color=INK, lw=1, zorder=4)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylab)
    grid_y(ax)


def fig2():
    """Main result: the real wiring delays death, in both connectomes."""
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1, 1], hspace=0.6,
                                                                       wspace=0.45))
    for r, sex in enumerate(("female", "male")):
        T = TAGS[sex]
        ax = axs[r, 0]
        for grp, col, ls, lab in (("rewired", REW, "-", "rewired (5)"),
                                  ("ei", CTRL, "--", "E→I control (2)"),
                                  ("real", REAL, "-", "real (3 seeds)")):
            for k, tag in enumerate(T[grp]):
                sr = series(tag)
                ax.plot((sr["t"] - T_ANOX) / 1000, sr["blocked"], color=col, ls=ls, lw=1,
                        label=lab if k == 0 else None)
        ax.axhline(0.5, color=MUTED, lw=0.6, ls=":")
        ax.set(xlim=(0, 30), ylim=(-0.02, 1.02), xlabel="Time after O₂ off (s)", ylabel="Fraction blocked",
               title=SEXLAB[sex])
        grid_y(ax)
        if r == 0:
            ax.legend(loc="upper left", handlelength=1.6, fontsize=6, bbox_to_anchor=(0.0, 1.02))
        letter(ax, "ab"[r], x=-0.12)
        groups = [("real", T["real"], REAL, "o"), ("rewired", T["rewired"], REW, "s"), ("E→I\ncontrol", T["ei"], CTRL, "D")]
        ax = axs[r, 1]
        strip(ax, groups, t50, "t50, half the brain blocked (s)")
        gap = np.mean([t50(t) for t in T["real"]]) - np.mean([t50(t) for t in T["rewired"]])
        ax.set_title(f"Half-death: real +{gap:.1f} s")
        ax.set_ylim(10, 26)
        letter(ax, "cd"[r], x=-0.3)
        ax = axs[r, 2]
        strip(ax, groups, first_sd, "First SD in brain neuropil (s)")
        gap = np.mean([first_sd(t) for t in T["real"]]) - np.mean([first_sd(t) for t in T["rewired"]])
        ax.set_title(f"First SD: real +{gap:.1f} s")
        ax.set_ylim(10, 26)
        letter(ax, "ef"[r], x=-0.3)
    fig.text(0.5, -0.02, "Real vs rewired t50: complete separation in both sexes, one-sided Mann-Whitney p = 0.018 "
             "(3 vs 5, smallest possible). Bars = means.", ha="center", fontsize=6, color=INK2)
    save(fig, "Fig2_real_wiring_delays_death")


# =================================================================================================================
def region_medians(sex, ct):
    reg = pd.Series(GEOM[sex]["cube_region"]).str.replace(r"[_(]?[LR]\)?$", "", regex=True)
    return pd.Series(ct).groupby(reg.values).median()


def fig3():
    """Where the brain dies: wiring shifts the timing, the spatial order is mostly kept."""
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.0, 0.06, 1.15], hspace=0.42, wspace=0.12)
    a, b = cube_time("female_R_seed0"), cube_time("female_Wdeg1_seed0")
    vmin, vmax = np.nanpercentile(np.r_[a, b], 1), np.nanpercentile(np.r_[a, b], 99)
    ax = fig.add_subplot(gs[0, 0])
    im = draw_map(ax, "female", a, vmin, vmax, "Female, real wiring")
    letter(ax, "a", x=-0.06)
    ax2 = fig.add_subplot(gs[1, 0])
    draw_map(ax2, "female", b, vmin, vmax, "Female, rewired (rewiring 1)")
    letter(ax2, "b", x=-0.06)
    cax = fig.add_subplot(gs[:, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("SD arrival (s after O₂ off), same scale in a and b"); cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    ax.set_position(ax.get_position().translated(0.06, 0))
    k = np.isfinite(a) & np.isfinite(b)
    ax.hexbin(a[k], b[k], gridsize=40, cmap=LinearSegmentedColormap.from_list("h", ["#cde2fb", "#0d366b"]),
              mincnt=1, bins="log", linewidths=0)
    lim = [min(a[k].min(), b[k].min()) - 0.5, max(a[k].max(), b[k].max()) + 0.5]
    ax.plot(lim, lim, color=MUTED, lw=0.6, ls="--")
    ax.set(xlim=lim, ylim=lim, xlabel="Real: SD arrival (s)", ylabel="Rewired: SD arrival (s)")
    rho = spearmanr(a[k], b[k])[0]
    ax.set_title(f"All {k.sum():,} cubes: ρ = {rho:.2f}")
    ax.text(0.04, 0.95, "dashed = same time;\nall cubes die earlier\nwhen rewired", transform=ax.transAxes,
            va="top", fontsize=5.8, color=INK2)
    ax.set_aspect("equal")
    letter(ax, "c", x=-0.32)

    ax = fig.add_subplot(gs[1, 2])
    ax.set_position(ax.get_position().translated(0.06, 0))
    ra = region_medians("female", a); rb = region_medians("female", b)
    j = ra.index.intersection(rb.index)
    ax.scatter(ra[j], rb[j], s=12, color=REAL, edgecolor="white", linewidth=0.5, zorder=3)
    for name, dx, dy in (("LA", 5, -4), ("ME", 5, -6), ("AL", 5, -2), ("LO", -12, 4)):
        if name in j:
            ax.annotate(name, (ra[name], rb[name]), (dx, dy), textcoords="offset points", fontsize=5.5, color=INK2)
    lo = min(ra[j].min(), rb[j].min()) - 0.5; hi = max(ra[j].max(), rb[j].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.6, ls="--")
    rr = spearmanr(ra[j], rb[j])[0]
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Real: median arrival (s)", ylabel="Rewired: median arrival (s)",
           title=f"{len(j)} brain regions (L+R pooled): ρ = {rr:.2f}")
    ax.set_aspect("equal")
    grid_y(ax)
    letter(ax, "d", x=-0.32)
    save(fig, "Fig3_where_the_brain_dies")


# =================================================================================================================
def fig4():
    """Mechanism: which feature of the wiring protects (loop controls and E/I-input-preserving rewiring)."""
    conds = [("Real wiring", TAGS["female"]["real"], REAL, "o"),
             ("Only E–I loops real\n(keep_loops)", TAGS["female"]["keep"], CTRL, "^"),
             ("E→I targeting kept,\nloops broken (E→I control)", TAGS["female"]["ei"], CTRL, "D"),
             ("E/I input counts kept,\npartners rewired", TAGS["female"]["wsign"], CTRL, "P"),
             ("E/I counts kept for\ninhibitory neurons only", TAGS["female"]["wsignI"], WSI, "X"),
             ("E/I counts kept for\nexcitatory neurons only", TAGS["female"]["wsignE"], WSE, "h"),
             ("Fully rewired", TAGS["female"]["rewired"], REW, "s"),
             ("Only E–I loops rewired\n(break_loops)", ["female_break_seed0"], REW, "v")]
    fig = plt.figure(figsize=(W2, 10.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 0.75], height_ratios=[1.05, 1, 0.3, 0.8], hspace=0.45, wspace=0.15)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    sub = gs[1, :].subgridspec(1, 2, wspace=0.28)
    axs += [fig.add_subplot(sub[0, 0]), fig.add_subplot(sub[0, 1])]
    lax = fig.add_subplot(gs[2, :]); lax.axis("off")
    eax = fig.add_subplot(gs[3, :].subgridspec(1, 2, width_ratios=[0.32, 1])[0, 1])
    ax = axs[0]
    for i, (lab, tags, col, mk) in enumerate(conds):
        v = [t50(t) for t in tags]
        ax.scatter(v, [i] * len(v), color=col, marker=mk, s=24, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([c[0] for c in conds], fontsize=6)
    ax.invert_yaxis()
    ax.set(xlabel="t50 (s after O₂ off)", title="Time to half-death (female)", xlim=(11, 23))
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    letter(ax, "a", x=-0.75)
    ax = axs[1]
    for i, (lab, tags, col, mk) in enumerate(conds):
        v = [S.loc[t, "rate_pre_mean_hz"] for t in tags]
        ax.scatter(v, [i] * len(v), color=col, marker=mk, s=24, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set(xlabel="Healthy firing (Hz, 2 s before O₂ off)", title="Healthy firing rate", xlim=(0, 45))
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    letter(ax, "b", x=-0.08)
    # c, d: every condition, both sexes; x = healthy rate (c) or firing at -40 mV during anoxia (d)
    grp_style = [("real", REAL, "o"), ("ei", CTRL, "D"), ("wsign", CTRL, "P"), ("wsignI", WSI, "X"),
                 ("wsignE", WSE, "h"), ("wsignIM", WSI, "d"), ("wsignEM", WSE, "p"), ("keep", CTRL, "^"),
                 ("keepM", CTRL, "*"), ("keepEIM", "#8a8a8a", "8"), ("keepEIsM", "#5f5f5f", "H"), ("keepShufM", CTRL, "o"), ("up", REAL, ">"), ("matched", REW, "<"),
                 ("rewired", REW, "s"), ("brk", REW, "v")]
    extra = {"female": [], "male": []}
    def groups_of(sex):
        return [(g, TAGS[sex][g], c, m) for g, c, m in grp_style if TAGS[sex][g]] + extra[sex]
    for ax, xf, xlab, title, xlim in (
            (axs[2], lambda t: S.loc[t, "rate_pre_mean_hz"], "Healthy firing (Hz, 2 s before O₂ off)",
             "Death vs healthy rate", (0, 45)),
            (axs[3], rate_at, "Firing at −40 mV during anoxia (Hz)", "Death vs firing at the same depolarization",
             (45, 105))):
        txt = []
        for sex, filled in (("female", True), ("male", False)):
            mx, my = [], []
            for g, tags, col, mk in groups_of(sex):
                xs, ys = [xf(t) for t in tags], [t50(t) for t in tags]
                ax.scatter(xs, ys, marker=mk, s=22, facecolor=col if filled else "white", edgecolor=col,
                           linewidth=0.9, zorder=3)
                mx.append(np.nanmean(xs)); my.append(np.mean(ys))
            rho = pd.Series(mx).corr(pd.Series(my), method="spearman")
            txt.append(f"{sex}: ρ = {rho:.2f} ({len(mx)} conditions)")
        ax.text(0.97, 0.97 if ax is axs[3] else 0.38, "Spearman, condition means\n" + "\n".join(txt),
                transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=INK2)
        ax.set(xlabel=xlab, ylabel="t50 (s)", xlim=xlim, ylim=(11, 25.5), title=title)
        grid_y(ax)
    axs[2].annotate("male E→I control:\nfast firing, late death", (20.2, 23.8), (26, 20.5), fontsize=5.8,
                    color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    axs[2].text(0.03, 0.03, "filled: female; open: male\n▶ real driven up; ◀ rewired slowed down", transform=axs[2].transAxes,
                fontsize=5.6, color=INK2, va="bottom")
    letter(axs[2], "c", x=-0.18)
    letter(axs[3], "d", x=-0.18)
    # legend for c, d (marker = condition; filled female, open male)
    names = {"real": "Real wiring", "ei": "E→I control", "wsign": "E/I input counts kept",
             "wsignI": "E/I counts kept, inhibitory receivers", "wsignE": "E/I counts kept, excitatory receivers",
             "wsignIM": "  same, inhibitory, at real rate", "wsignEM": "  same, excitatory, at real rate",
             "keep": "Only E–I loops real (keep_loops)", "keepM": "  keep_loops at real rate",
             "keepShufM": "  loops kept, partners shuffled, at real rate", "keepEIM": "  non-loop E–I, equal synapses, at real rate",
             "keepEIsM": "  non-loop E–I, strength-matched, at real rate", "up": "Real, driven up to rewired rate",
             "matched": "Rewired, slowed to real rate", "rewired": "Fully rewired", "brk": "Only E–I loops rewired (break_loops)"}
    order = ["real", "up", "ei", "wsign", "wsignI", "wsignIM", "wsignE", "wsignEM", "keep", "keepM", "keepShufM",
             "keepEIM", "keepEIsM", "rewired", "matched", "brk"]
    sty = {g: (c, m) for g, c, m in grp_style}
    hs = [plt.Line2D([], [], ls="", marker=sty[g][1], mfc=sty[g][0], mec=sty[g][0], ms=4.5, label=names[g]) for g in order]
    lax.legend(handles=hs, loc="upper left", ncol=4, fontsize=5.6, frameon=False, handletextpad=0.3, columnspacing=1.2,
               labelspacing=0.5, title="Markers in c and d (filled: female; open: male)", title_fontsize=6.2,
               alignment="left", bbox_to_anchor=(0.0, 1.05))
    # e: recovery at the real healthy rate (share of the real-minus-rewired gap kept; rewired brains slowed to the
    # same rate as comparison: female 15.36 s, male 15.60 s; real 20.51 / 22.71 s)
    base_m = {"female": 15.36, "male": 15.60}          # rate-matched rewired brains (stage18, WdegM1-3 means)
    real_m = {"female": 20.51, "male": 22.71}          # real brains (stage16 means); as in the stage26-28 scripts
    rows = [("keep_loops (only loops real)", "keepM", CTRL), ("loops kept, partners shuffled", "keepShufM", CTRL),
            ("non-loop E–I, equal synapse total", "keepEIM", "#8a8a8a"),
            ("non-loop E–I, strength-matched", "keepEIsM", "#5f5f5f"),
            ("E/I counts kept, inhibitory receivers", "wsignIM", WSI), ("E/I counts kept, excitatory receivers", "wsignEM", WSE)]
    eax.axvspan(0.67, 1.15, color="#eef6f1", lw=0); eax.axvspan(-0.1, 0.33, color="#fbefe9", lw=0)
    for x0 in (0.33, 0.67):
        eax.axvline(x0, color=MUTED, lw=0.6, ls="--")
    for i, (lab, g, col) in enumerate(rows):
        for sex, dy, filled in (("female", -0.13, True), ("male", 0.13, False)):
            tg = TAGS[sex][g]
            if not tg:
                continue
            r = (np.mean([t50(t) for t in tg]) - base_m[sex]) / (real_m[sex] - base_m[sex])
            eax.plot(r, i + dy, "o", ms=5, mfc=col if filled else "white", mec=col, mew=1.0, zorder=3)
            eax.text(r + 0.03, i + dy, f"{r:.2f}", fontsize=5.3, color=INK2, va="center")
    eax.set_yticks(range(len(rows))); eax.set_yticklabels([r[0] for r in rows], fontsize=6)
    eax.set_xlim(-0.1, 1.15)
    eax.set(xlabel="Share of the protection kept at the real healthy rate", title="Recovery when firing rates are matched (filled: female; open: male)")
    eax.text(0.69, -0.62, "most kept (≥ 0.67)", fontsize=5.5, color=INK2, va="center")
    eax.text(-0.08, -0.62, "little kept (≤ 0.33)", fontsize=5.5, color=INK2, va="center")
    eax.set_ylim(len(rows) - 0.5, -0.9)
    eax.grid(axis="x", color=GRID, lw=0.5); eax.set_axisbelow(True)
    eax.tick_params(axis="y", length=0)
    letter(eax, "e", x=-0.42)
    save(fig, "Fig4_which_wiring_feature_protects")


# =================================================================================================================
def gain_curve(tag, binw=0.5):
    """Mean firing rate vs mean resting potential in 0.5-s bins from O2 off until neurons start to block."""
    p = probes(tag)
    t = (p["series_t"] - T_ANOX) / 1000.0
    a = p["active"] / DT_S
    v = p["meanV"].astype(float)
    blk = p["blocked"]
    stop = t[np.flatnonzero((t > 0) & (blk > 0.005))[0]] - 0.5
    rr, vv = [], []
    for b0 in np.arange(0, stop, binw):
        m = (t >= b0) & (t < b0 + binw)
        rr.append(a[m].mean()); vv.append(v[m].mean())
    return np.array(vv), np.array(rr)


def rate_at(tag, v0=-40.0):
    vv, rr = gain_curve(tag)
    o = np.argsort(vv)
    return float(np.interp(v0, vv[o], rr[o], left=np.nan, right=np.nan))


def fig5():
    """Activity: gain vs drive, rate-matched control, and firing vs depolarization during anoxia."""
    cal = pd.read_csv(os.path.join(RATE, "calibration.csv"))
    drv = json.load(open(os.path.join(RATE, "drives.json")))
    frozen = {"female": 3.36, "male": 3.418}
    fig = plt.figure(figsize=(W2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.62, wspace=0.45)
    axs = [fig.add_subplot(gs[0, k]) for k in range(3)]
    ax = axs[0]
    for sex, ls, mk in (("female", "-", "o"), ("male", "--", "s")):
        for net, col in (("real", REAL), ("rewired", REW)):
            c = cal[(cal.sex == sex) & (cal.net == net)].sort_values("drive")
            ax.plot(c.drive, c.rate_hz, ls=ls, color=col, marker=mk, ms=3,
                    mfc=col if sex == "female" else "white", lw=1)
        ax.axvline(frozen[sex], color=MUTED, lw=0.5, ls=ls)
    sl = {(s_, n): np.polyfit(*cal[(cal.sex == s_) & (cal.net == n)][["drive", "rate_hz"]].values.T, 1)[0]
          for s_ in ("female", "male") for n in ("real", "rewired")}
    ax.text(0.97, 0.03, f"Hz per mV of drive\nreal {sl['female', 'real']:.1f} / {sl['male', 'real']:.1f}\n"
            f"rewired {sl['female', 'rewired']:.1f} / {sl['male', 'rewired']:.1f}\n(female / male)",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=5.8, color=INK2)
    ax.text(frozen["male"] + 0.1, 23, "frozen drive", fontsize=5.5, color=MUTED, va="top")
    ax.text(1.2, 35, "solid, filled: female\ndashed, open: male", fontsize=5.5, color=INK2, va="top")
    ax.text(1.55, 5.2, "rewired", fontsize=6, color=REW); ax.text(5.2, 17.5, "real", fontsize=6, color=REAL)
    ax.set(xlabel="Tonic drive (mV)", ylabel="Healthy firing, steady state (Hz)",
           title="Gain: firing vs drive", ylim=(0, 36))
    grid_y(ax)
    letter(ax, "a", x=-0.25)
    for j, sex in enumerate(("female", "male")):
        ax = axs[j + 1]
        d = drv[sex]
        r_lo, r_hi = d["real_rate"], d["rewired_rate"]
        pts = {"real": (r_lo, [t50(t) for t in TAGS[sex]["real"]]),
               "up": (r_hi, [t50(t) for t in TAGS[sex]["up"]]),
               "rewired": (r_hi, [t50(t) for t in TAGS[sex]["rewired"]]),
               "matched": (r_lo, [t50(t) for t in TAGS[sex]["matched"]])}
        for (a_, b_), col, lab in ((("real", "up"), REAL, "real wiring"), (("matched", "rewired"), REW, "rewired")):
            xa, ya = pts[a_]; xb, yb = pts[b_]
            ax.plot([xa, xb], [np.mean(ya), np.mean(yb)], color=col, lw=1.2, label=lab)
        for key, col, filled in (("real", REAL, True), ("up", REAL, False), ("rewired", REW, True),
                                 ("matched", REW, False)):
            x, ys = pts[key]
            jit = np.linspace(-0.35, 0.35, len(ys)) if len(ys) > 1 else [0]
            ax.scatter(np.asarray(jit) + x, ys, s=20, marker="o" if col == REAL else "s",
                       facecolor=col if filled else "white", edgecolor=col, linewidth=0.9, zorder=3)
        orig = np.mean(pts["real"][1]) - np.mean(pts["rewired"][1])
        gap_lo = np.mean(pts["real"][1]) - np.mean(pts["matched"][1])
        gap_hi = np.mean(pts["up"][1]) - np.mean(pts["rewired"][1])
        ax.annotate("", (r_lo - 1.2, np.mean(pts["matched"][1])), (r_lo - 1.2, np.mean(pts["real"][1])),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
        ax.text(r_lo - 1.8, (np.mean(pts["real"][1]) + np.mean(pts["matched"][1])) / 2,
                f"{gap_lo:.1f} s\n({gap_lo / orig:.0%})", ha="right", va="center", fontsize=5.8, color=INK2)
        ax.annotate("", (r_hi + 1.2, np.mean(pts["rewired"][1])), (r_hi + 1.2, np.mean(pts["up"][1])),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
        ax.text(r_hi + 1.8, (np.mean(pts["up"][1]) + np.mean(pts["rewired"][1])) / 2,
                f"{gap_hi:.1f} s\n({gap_hi / orig:.0%})", ha="left", va="center", fontsize=5.8, color=INK2)
        ax.set(xlabel="Healthy firing, steady state (Hz)", ylabel="t50 (s)" if j == 0 else None,
               title=SEXLAB[sex], xlim=(r_lo - 9, r_hi + 9), ylim=(12, 24.5))
        grid_y(ax)
        if j == 0:
            ax.legend(loc="upper right", handlelength=1.5)
        letter(ax, "bc"[j], x=-0.2)
    # d: firing vs depolarization during anoxia
    ax = fig.add_subplot(gs[1, 0:2])
    groups = (("real", REAL, "real wiring"), ("ei", CTRL, "E→I control"),
              ("wsign", "#7fcfb0", "E/I input counts kept"), ("rewired", REW, "rewired"))
    for sex, ls in (("female", "-"), ("male", "--")):
        for grp, col, lab in groups:
            for k, tag in enumerate(TAGS[sex][grp]):
                vv, rr = gain_curve(tag)
                ax.plot(vv, rr, ls=ls, color=col, lw=0.9, alpha=0.9,
                        label=lab if (sex == "female" and k == 0) else None)
    ax.axvline(-40, color=MUTED, lw=0.6, ls=":")
    ax.set(xlabel="Mean resting potential during anoxia (mV)", ylabel="Mean firing rate (Hz)",
           title="Firing as neurons depolarize (O₂ off until block starts)", xlim=(-52, -27), ylim=(0, 125))
    ax.legend(loc="upper left", ncol=2, handlelength=1.8)
    ax.text(0.99, 0.03, "solid: female; dashed: male", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.8, color=INK2)
    grid_y(ax)
    letter(ax, "d", x=-0.08)
    # e: rate at -40 mV per run
    ax = fig.add_subplot(gs[1, 2])
    groups = groups[:3] + (("wsignI", WSI, ""), ("wsignE", WSE, "")) + groups[3:]
    for i, (grp, col, lab) in enumerate(groups):
        for sex, filled, dx in (("female", True, -0.12), ("male", False, 0.12)):
            v = [rate_at(t) for t in TAGS[sex][grp]]
            ax.scatter(np.full(len(v), i + dx), v, s=16, marker="o", facecolor=col if filled else "white",
                       edgecolor=col, linewidth=0.9, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["real", "E→I control", "E/I kept (all)", "E/I kept (I only)", "E/I kept (E only)", "rewired"],
                       fontsize=5.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set(ylabel="Firing at −40 mV (Hz)", title="At the same depolarization", ylim=(40, 105))
    ax.text(0.03, 0.97, "filled: female\nopen: male", transform=ax.transAxes, va="top", fontsize=5.6, color=INK2)
    grid_y(ax)
    letter(ax, "e", x=-0.3)
    fig.text(0.5, 0.505, "In b–c: filled = frozen drive; open = drive changed to match the other wiring's healthy rate "
             "(rewired: drive ×0.5; real: drive ×2). Percentages = share of the original gap that remains.",
             ha="center", fontsize=5.8, color=INK2)
    save(fig, "Fig5_activity_and_gain")


# =================================================================================================================
def gap_ci(real_tags, rew_tags):
    a = np.array([t50(t) for t in real_tags]); b = np.array([t50(t) for t in rew_tags])
    diffs = (a[:, None] - b[None, :]).ravel()
    return a.mean() - b.mean(), diffs.min(), diffs.max(), len(a), len(b)


def fig6():
    """Robustness: real minus rewired t50 across every test."""
    def gp(ra, rb):
        a_ = np.asarray(ra, float); b_ = np.asarray(rb, float)
        d = (a_[:, None] - b_[None, :]).ravel()
        return a_.mean() - b_.mean(), d.min(), d.max(), len(a_), len(b_)
    T = lambda tags: [t50(t) for t in tags]
    def col(sex, c):
        return [NEWM.loc[t, c] for t in TAGS[sex]["real"]], [NEWM.loc[t, c] for t in TAGS[sex]["rewired"]]
    rows = [("Main experiment", None, None)]
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}", T(TAGS[sex]["real"]), T(TAGS[sex]["rewired"])))
    rows += [("Other measures of death", None, None)]
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, central-brain t50", *col(sex, "t50_central")))
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, K⁺ surge at electrodes", *col(sex, "electrode_surge_s")))
    rows += [("Model and protocol", None, None),
             ("Female, lamina removed", T(["female_noLA_R_seed0"]), T(["female_noLA_W_seed0"]))]
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, 20-s warm-up", T(TAGS[sex]["rwarm"]), T(TAGS[sex]["wwarm"])))
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, clearance cut 0.35", T(TAGS[sex]["rpump"]), T(TAGS[sex]["wpump"])))
    rows += [("Glial capacity (frozen: 10 mM)", None, None)]
    for cap in ("5", "20"):
        for sex, lab in (("female", "Female"), ("male", "Male")):
            rows.append((f"{lab}, capacity {cap} mM", T(TAGS[sex]["rcap" + cap]), T(TAGS[sex]["wcap" + cap])))
    for sex, lab in (("female", "Female"),):
        rows.append((f"{lab}, photoreceptors at weight 0", T(TAGS[sex]["rphoto"]), T(TAGS[sex]["wphoto"])))
    rows += [("Glutamate (healthy rate restored)", None, None)]
    for key, how in (("glutW", "neutral, all weights lowered (primary)"), ("ctrl", "inhibitory, same weights (control)"),
                     ("glutD", "neutral, drive lowered")):
        for sex, lab in (("female", "Female"), ("male", "Male")):
            r_, w_ = TAGS[sex]["r" + key], TAGS[sex]["w" + key]
            if r_ and w_:
                rows.append((f"{lab}, glutamate {how}", T(r_), T(w_)))
    rows += [("Activity matched", None, None)]
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, rewired slowed to real rate", T(TAGS[sex]["real"]), T(TAGS[sex]["matched"])))
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, real sped up to rewired rate", T(TAGS[sex]["up"]), T(TAGS[sex]["rewired"])))
    rows += [("Rewired keeping E/I input counts", None, None)]
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, real minus E/I-preserving", T(TAGS[sex]["real"]), T(TAGS[sex]["wsign"])))
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, real minus E/I kept for inhibitory only", T(TAGS[sex]["real"]), T(TAGS[sex]["wsignI"])))
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, real minus E/I kept for excitatory only", T(TAGS[sex]["real"]), T(TAGS[sex]["wsignE"])))
    for sex, lab in (("female", "Female"), ("male", "Male")):
        rows.append((f"{lab}, real minus inhibitory only, rate-matched", T(TAGS[sex]["real"]), T(TAGS[sex]["wsignIM"])))
        rows.append((f"{lab}, real minus excitatory only, rate-matched", T(TAGS[sex]["real"]), T(TAGS[sex]["wsignEM"])))
    fig, ax = plt.subplots(figsize=(5.4, 0.2 * len(rows) + 0.6))
    labels = []
    for i, (lab, ra, rb) in enumerate(rows):
        if ra is None:
            labels.append(lab)
            ax.axhline(i, color=GRID, lw=0.5)
            continue
        labels.append("   " + lab)
        if not ra:
            continue
        g, lo, hi, na, nb = gp(ra, rb)
        ax.plot([lo, hi], [i, i], color="#9ec5f4", lw=4, solid_capstyle="round")
        dot = (WSI if "inhibitory only" in lab else WSE if "excitatory only" in lab else   # descriptive, no threshold
               "#8a8a8a" if "control" in lab else
               REAL if g >= 2 else REW)
        ax.plot(g, i, "o", color=dot, ms=5, mec="white", mew=0.6, zorder=3)
        ax.text(12.0, i, f"+{g:.1f} s  ({na} vs {nb})", va="center", fontsize=5.8, color=INK2)
    ax.axvline(0, color=INK2, lw=0.6)
    ax.axvline(2, color=MUTED, lw=0.6, ls="--")
    ax.text(2.1, 0.0, "pre-registered\nthreshold (2 s)", fontsize=5.5, color=MUTED, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=6)
    for tl, (lab, ra, rb) in zip(ax.get_yticklabels(), rows):
        if ra is None:
            tl.set_fontweight("bold"); tl.set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(-0.5, 11.9)
    ax.set_xlabel("Real minus comparison t50 (s); dot = mean gap, bar = range over run pairs")
    ax.set_title("Real minus rewired, across every test", pad=4)
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "Fig6_robustness")


# =================================================================================================================
def fig7():
    """Recovery after 60 s of anoxia (female, real wiring)."""
    p = probes("female_reox_seed0")
    st = (p["series_t"] - T_ANOX) / 1000
    v = p["meanV"].astype(float)
    tr = (T_REOX - T_ANOX) / 1000
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.45))
    for ax in axs:
        ax.axvspan(0, tr, color="#f0efec", lw=0)
        ax.set_xlim(-2, 106)
        ax.set_xlabel("Time after O₂ off (s)")
        grid_y(ax)
    ax = axs[0]
    ax.plot(st, v, color=REAL)
    base = v[(st > -1.5) & (st < 0)].mean(); voff = np.interp(tr, st, v)
    half = base + (voff - base) / 2
    k = np.flatnonzero((st > tr) & (v <= half))[0]
    ax.plot(st[k], v[k], "o", color=INK, ms=3.5)
    ax.annotate(f"half-recovery\n{st[k] - tr:.1f} s after air\n(fly 11.8–16.1 s)", (st[k], v[k]), (st[k] + 6, v[k] + 18),
                fontsize=5.8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(tr / 2, -45, "N₂, 60 s", ha="center", fontsize=6, color=INK2)
    ax.set(ylabel="Mean membrane potential (mV)", title="DC shift and recovery")
    letter(ax, "a", x=-0.28)
    ax = axs[1]
    tp = (p["t"] - T_ANOX) / 1000
    for j in range(p["K"].shape[1]):
        ax.plot(tp, p["K"][:, j], color="#b7d3f6", lw=0.5)
    ax.plot(tp, np.median(p["K"], axis=1), color=REAL, lw=1.4)
    ax.set(ylabel="Extracellular K⁺ (mM)", title="K⁺ at 41 virtual electrodes")
    letter(ax, "b", x=-0.25)
    ax = axs[2]
    sr = series("female_reox_seed0")
    ts = (sr["t"] - T_ANOX) / 1000
    ax.plot(ts, sr["blocked"], color=REAL)
    ax.plot(ts, sr["meanE"], color=INK2, ls="--", lw=1)
    ax.text(40, 0.93, "fraction blocked", fontsize=6, color=REAL, ha="center", va="top")
    ax.text(103, 0.78, "mean\nenergy", fontsize=6, color=INK2, ha="right", va="top")
    ax.set(ylim=(-0.02, 1.05), ylabel="Fraction / energy (0–1)", title="Block and energy")
    letter(ax, "c", x=-0.25)
    save(fig, "Fig7_recovery_after_60s_anoxia")


# =================================================================================================================
def figS1():
    """Male maps, real vs rewired."""
    a, b = cube_time("male_R_seed0"), cube_time("male_Wdeg1_seed0")
    vmin, vmax = np.nanpercentile(np.r_[a, b], 1), np.nanpercentile(np.r_[a, b], 99)
    fig, axs = plt.subplots(2, 1, figsize=(5.2, 4.6), gridspec_kw=dict(hspace=0.25))
    im = draw_map(axs[0], "male", a, vmin, vmax, "Male, real wiring")
    draw_map(axs[1], "male", b, vmin, vmax, "Male, rewired (rewiring 1)")
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.01)
    cb.set_label("SD arrival (s after O₂ off)"); cb.outline.set_visible(False)
    save(fig, "FigS1_male_spread_maps")


def figS2():
    """Virtual electrode K+ in real vs rewired (both sexes)."""
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.3))
    for ax, sex in zip(axs, ("female", "male")):
        for grp, col in (("rewired", REW), ("real", REAL)):
            for k, tag in enumerate(TAGS[sex][grp]):
                p = probes(tag)
                ax.plot((p["t"] - T_ANOX) / 1000, np.median(p["K"], axis=1), color=col, lw=1,
                        label=("real" if grp == "real" else "rewired") if k == 0 else None)
        ax.set(xlim=(-2, 30), xlabel="Time after O₂ off (s)", ylabel="Median K⁺ at 41 electrodes (mM)",
               title=SEXLAB[sex])
        grid_y(ax)
    axs[0].legend(loc="upper left")
    save(fig, "FigS2_K_electrodes_real_vs_rewired")


def figS3():
    """Mean firing during anoxia, real vs rewired (both sexes)."""
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.3))
    for ax, sex in zip(axs, ("female", "male")):
        for grp, col in (("rewired", REW), ("real", REAL)):
            for k, tag in enumerate(TAGS[sex][grp]):
                sr = series(tag)
                ax.plot((sr["t"] - T_ANOX) / 1000, smooth(sr["active"], 200) / DT_S, color=col, lw=0.9,
                        label=grp if k == 0 else None)
        ax.set(xlim=(-2, 30), xlabel="Time after O₂ off (s)", ylabel="Mean firing rate (Hz)", title=SEXLAB[sex])
        grid_y(ax)
    axs[0].legend(loc="upper right")
    save(fig, "FigS3_firing_during_anoxia")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, figS1, figS2, figS3):
        f()
