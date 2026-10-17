# analyze_balance.py - why is randomly rewired wiring twice as active as the real fly wiring?
# ===========================================================================================
# Compares the real FlyWire network with its degree-preserving rewirings (SHUFFLE_MODE
# "degree", seeds 1-3; every neuron keeps its numbers of inputs and outputs and its output
# weights and sign; only the partners change). Prints (log row 76):
#   1. input balance per neuron: excitatory vs inhibitory input strength, how well they match
#   2. who inhibits whom: inhibitory input onto excitatory vs inhibitory neurons; E<->I loops
#   3. recurrent gain: leading eigenvalue of the weight matrix (runaway tendency)
#   4. firing in the oxygen-loss runs (runs/*_v6_robust_stage2): which neurons got busier
# Small text output only.

import glob
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

import death_v6 as m


def log(*a):
    print(*a, flush=True)


def net(meta, conn, shuffle, seed):
    m.SHUFFLE_CONN = shuffle
    m.SHUFFLE_MODE = "degree"
    m.SHUFFLE_SEED = seed
    m.rng = np.random.default_rng(0)
    idx, W, sign, coords, N = m.build_network(meta, conn)
    return W.tocsr().astype(np.float64), sign


def metrics(W, sign, name):
    N = W.shape[0]
    Wp = W.multiply(W > 0).tocsr(); Wn = (-W).multiply(W < 0).tocsr()
    E_in = np.asarray(Wp.sum(1)).ravel(); I_in = np.asarray(Wn.sum(1)).ravel()
    exc, inh = sign > 0, sign < 0
    has = (E_in + I_in) > 0
    out = dict(network=name)
    out["E_I_input_match_rho"] = spearmanr(E_in[has], I_in[has])[0]          # balance: high = matched
    out["frac_net_excitatory"] = float(np.mean(E_in[has] > I_in[has]))
    out["median_net_input"] = float(np.median((E_in - I_in)[has]))
    out["I_onto_exc_mean"] = float(I_in[exc].mean()); out["I_onto_inh_mean"] = float(I_in[inh].mean())
    out["E_onto_inh_mean"] = float(E_in[inh].mean()); out["E_onto_exc_mean"] = float(E_in[exc].mean())
    # E<->I loops: excitatory a -> inhibitory b with b -> a back (feedback inhibition)
    A = (W != 0).astype(np.int8).tocsr()                                    # A[post, pre]
    AT = A.T.tocsr()
    recip = A.multiply(AT).tocsr()                                          # pairs connected both ways
    rows, cols = recip.nonzero()                                            # post=rows, pre=cols
    ei = np.sum(exc[cols] & inh[rows])                                      # E pre -> I post, reciprocated
    n_ei = np.sum(exc[A.nonzero()[1]] & inh[A.nonzero()[0]])
    out["EI_feedback_loops"] = int(ei)
    out["EI_loop_fraction"] = float(ei / max(n_ei, 1))
    out["reciprocal_pairs"] = int(recip.nnz // 2)
    return out, E_in, I_in


if __name__ == "__main__":
    t0 = time.time()
    log("analyze_balance started", time.strftime("%Y-%m-%d %H:%M"))
    meta, conn = m.load_data()
    rows = []; inputs = {}
    for name, shuffle, seed in [("real", False, 0), ("rewired_a", True, 1), ("rewired_b", True, 2), ("rewired_c", True, 3)]:
        W, sign = net(meta, conn, shuffle, seed)
        r, E_in, I_in = metrics(W, sign, name)
        inputs[name] = (E_in, I_in)
        try:
            from scipy.sparse.linalg import eigs
            ev = eigs(W, k=1, which="LR", maxiter=600, tol=1e-3, return_eigenvectors=False)
            r["leading_eigenvalue_real"] = float(np.real(ev[0]))
        except Exception as e:
            r["leading_eigenvalue_real"] = np.nan
            log(f"  eigenvalue for {name} failed: {e}")
        rows.append(r)
        log(f"  {name} done ({(time.time() - t0) / 60:.1f} min)")
    tab = pd.DataFrame(rows).set_index("network").T
    log("\n=== 1-3. network balance: real vs degree-preserving rewirings ===")
    log(tab.to_string(float_format=lambda x: f"{x:.4g}"))

    # 4. firing in the oxygen-loss runs
    folders = sorted(glob.glob(os.path.join("runs", "*_v6_robust_stage2")))
    if folders:
        f = folders[-1]
        sc = meta["super_class"].fillna("unknown").astype(str).values
        nt = meta["nt_type"].fillna("UNK").astype(str).values
        R = np.load(os.path.join(f, "rates_base_R_seed0.npz"))["pre"]
        Wr = np.load(os.path.join(f, "rates_base_Wdeg_a_seed0.npz"))["pre"]
        df = pd.DataFrame({"nt": nt, "super_class": sc, "real_hz": R, "rewired_hz": Wr})
        log("\n=== 4. firing before oxygen loss (Hz), real vs rewired_a, by transmitter ===")
        g = df.groupby("nt").agg(n=("real_hz", "size"), real=("real_hz", "mean"), rewired=("rewired_hz", "mean"))
        log(g.round(2).to_string())
        log("\nby super class:")
        g = df.groupby("super_class").agg(n=("real_hz", "size"), real=("real_hz", "mean"), rewired=("rewired_hz", "mean"))
        log(g.sort_values("n", ascending=False).round(2).to_string())
        log(f"\nsilent neurons (<0.5 Hz): real {np.mean(R < 0.5):.1%}, rewired {np.mean(Wr < 0.5):.1%};"
            f" very active (>50 Hz): real {np.mean(R > 50):.1%}, rewired {np.mean(Wr > 50):.1%}")
        E0, I0 = inputs["real"]; E1, I1 = inputs["rewired_a"]
        d_net = (E1 - I1) - (E0 - I0)
        d_rate = Wr - R
        log(f"per neuron: change in net input vs change in firing, Spearman rho = {spearmanr(d_net, d_rate)[0]:+.2f}")
        log(f"per neuron: change in inhibitory input vs change in firing, rho = {spearmanr(I1 - I0, d_rate)[0]:+.2f}")
        log(f"per neuron: change in excitatory input vs change in firing, rho = {spearmanr(E1 - E0, d_rate)[0]:+.2f}")
    log(f"\nanalyze_balance finished ({(time.time() - t0) / 60:.1f} min)")
