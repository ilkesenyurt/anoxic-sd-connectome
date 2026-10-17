# check_loops.py - log row 123. No simulation, about 2-4 min. Frozen model v11 data loading.
# =====================================================================================================================
# Question (after stage27): keepEIs kept real non-loop E-I connections matched to the loop connections in direction and
# synapse count, and it protected no more than a random non-loop set. So strength does not explain the loop advantage,
# but the loop set may still differ from the matched set in WHICH neurons it involves (cell class, region, degree).
# This script measures that overlap, using exactly the keepEIs selection of stage27 (seed 1).
# Decision rule fixed BEFORE running (log row 123):
#   share of keepEIs synapses whose sender AND receiver are both loop neurons
#     >= 0.80 -> the matched set already ran between the same neurons; the identity explanation is weak; no stage28,
#                one sentence in the paper.
#     <  0.50 -> identity is not ruled out; run stage28 (neuron-matched control).
#     between -> report the numbers and decide with the class / region comparison (lean towards running stage28).
# Also reported: share of E and I neurons that are loop neurons; class and region make-up (by synapses) of the loop set
# vs the keepEIs set with total-variation distance; mean degree of their neurons; and how many non-loop E-I connections
# run between loop neurons (the pool a neuron-matched control would draw from) and how well it can be strength-matched.
import os

import numpy as np
import pandas as pd

import runner
import stage16
import stage20


def log(*a):
    print(*a, flush=True)


def eis_mask(syn, loop, e2i, i2e, seed):
    """Same selection as stage27.keep_eis_conn (same random stream), returning only the kept mask."""
    g = np.random.default_rng(seed)
    keep = np.zeros(syn.size, bool)
    for dmask in (e2i, i2e):
        need = np.bincount(syn[loop & dmask])
        pool = np.flatnonzero(dmask & ~loop)
        pool = pool[g.permutation(pool.size)]
        by = {}
        for k in pool:
            by.setdefault(int(syn[k]), []).append(k)
        counts = np.array(sorted(by)) if by else np.array([], int)
        for c in np.flatnonzero(need)[::-1]:
            n = int(need[c])
            for cc in sorted(counts, key=lambda v: (abs(v - c), -v)):
                lst = by.get(int(cc), [])
                take = min(n, len(lst))
                if take:
                    keep[lst[:take]] = True
                    by[int(cc)] = lst[take:]
                    n -= take
                if n == 0:
                    break
    return keep


def make_up(labels_pre, labels_post, syn, mask, top=8):
    s = pd.concat([pd.Series(syn[mask], index=labels_pre[mask]), pd.Series(syn[mask], index=labels_post[mask])])
    return s.groupby(level=0).sum() / s.sum()


def tv(a, b):
    k = a.index.union(b.index)
    return 0.5 * float(np.abs(a.reindex(k, fill_value=0) - b.reindex(k, fill_value=0)).sum())


def strength_match_share(syn, need_mask, pool_mask):
    """Share of the needed connections that a pool can match with the SAME synapse count."""
    need = np.bincount(syn[need_mask])
    have = np.bincount(syn[pool_mask], minlength=need.size)[:need.size]
    return float(np.minimum(need, have).sum() / max(need.sum(), 1))


def main():
    for sex in os.environ.get("CHECK_SEXES", "female,male").split(","):
        runner.MODEL_MODULE = "death_v11"
        runner._DATA.clear()
        m = runner._fresh_module()
        for k, v in {**stage16.FROZEN, **stage16.SEX[sex]}.items():
            setattr(m, k, v)
        meta, conn = runner._load(m)
        log(f"\n===== {sex} =====")
        ids = meta["root_id"].values
        idx = {r: i for i, r in enumerate(ids)}
        N = len(ids)
        sign = np.array([stage20.sign_of_nt(t) for t in meta["nt_type"].fillna("UNK").values], dtype=np.int8)
        pi = conn["pre_root_id"].map(idx).values.astype(np.int64)
        qi = conn["post_root_id"].map(idx).values.astype(np.int64)
        syn = conn["syn_count"].values.astype(np.int64)
        sp, so = sign[pi], sign[qi]
        recip = np.isin(qi * N + pi, pi * N + qi)
        e2i = (sp > 0) & (so < 0)
        i2e = (sp < 0) & (so > 0)
        loop = recip & (e2i | i2e)
        keep = eis_mask(syn, loop, e2i, i2e, 1)
        log(f"loop connections {int(loop.sum()):,} ({syn[loop].sum():,} syn); keepEIs {int(keep.sum()):,} "
            f"({syn[keep].sum():,} syn) [stage27 log: 353,630 / 537,672 kept]")

        inL = np.zeros(N, bool)
        inL[pi[loop]] = True
        inL[qi[loop]] = True
        nE, nI = int((sign > 0).sum()), int((sign < 0).sum())
        log(f"loop neurons: {int(inL.sum()):,} = {(inL & (sign > 0)).sum() / nE:.1%} of {nE:,} E and "
            f"{(inL & (sign < 0)).sum() / nI:.1%} of {nI:,} I neurons")
        allEI = e2i | i2e
        log(f"share of ALL E-I synapses that are loop synapses: {syn[loop].sum() / syn[allEI].sum():.1%}")

        both = inL[pi] & inL[qi]
        either = inL[pi] | inL[qi]
        for nm, msk in (("keepEIs", keep), ("all non-loop E-I", allEI & ~loop)):
            w = syn[msk]
            log(f"{nm}: synapses with sender AND receiver loop neurons {w[both[msk]].sum() / w.sum():.1%}; "
                f"with at least one {w[either[msk]].sum() / w.sum():.1%}  (connections: {both[msk].mean():.1%} / "
                f"{either[msk].mean():.1%})")
        nl = np.zeros(N, bool); nl[pi[loop]] = True; nl[qi[loop]] = True
        nk = np.zeros(N, bool); nk[pi[keep]] = True; nk[qi[keep]] = True
        log(f"neurons used: loop set {int(nl.sum()):,}, keepEIs set {int(nk.sum()):,}, both {int((nl & nk).sum()):,} "
            f"(Jaccard {(nl & nk).sum() / (nl | nk).sum():.2f})")
        share = syn[keep & both].sum() / syn[keep].sum()
        verdict = ("identity explanation weak (>= 0.80): no stage28" if share >= 0.80 else
                   "identity not ruled out (< 0.50): run stage28" if share < 0.50 else
                   "between: decide with class / region comparison")
        log(f"DECISION RULE: share = {share:.3f} -> {verdict}")

        deg = np.bincount(pi, minlength=N) + np.bincount(qi, minlength=N)
        for nm, msk in (("loop", loop), ("keepEIs", keep)):
            log(f"mean degree of neurons at either end ({nm}, synapse-weighted): "
                f"{np.average(deg[pi[msk]], weights=syn[msk]):.0f} senders, {np.average(deg[qi[msk]], weights=syn[msk]):.0f} receivers")

        for col in ("super_class", "neuropil"):
            if col not in meta:
                continue
            lab = meta[col].fillna("unknown").astype(str).values
            a = make_up(lab[pi], lab[qi], syn, loop)
            b = make_up(lab[pi], lab[qi], syn, keep)
            t = pd.DataFrame({"loop": a, "keepEIs": b}).fillna(0).sort_values("loop", ascending=False)
            log(f"\n{col} make-up by synapses (sender + receiver), total-variation distance {tv(a, b):.2f}:")
            log((t.head(10) * 100).round(1).to_string())

        pool_both = allEI & ~loop & both
        for dn, d in (("E->I", e2i), ("I->E", i2e)):
            log(f"stage28 pool {dn}: non-loop connections between loop neurons {int((pool_both & d).sum()):,} "
                f"for {int((loop & d).sum()):,} loop connections; same-count match possible for "
                f"{strength_match_share(syn, loop & d, pool_both & d):.1%}")
    log("\ncheck_loops finished")


if __name__ == "__main__":
    main()
