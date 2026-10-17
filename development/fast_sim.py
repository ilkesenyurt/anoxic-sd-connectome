# fast_sim.py - compiled version of the simulation loop in death_v2.py
# =====================================================================
# Same model, same order of operations as death_v2.simulate_numpy(), but the
# loop over neurons is compiled to machine code with Numba (@njit). Instead of
# ~15 separate numpy passes over all neurons per time step (each making a
# temporary array), every neuron is updated in ONE pass.
#
# Differences from the numpy version:
#   - noise: drawing normal random numbers was the slowest part (~80% of the
#     time). Instead, a pool of 4 million normal numbers is drawn once with
#     numpy, and each step picks entries at random positions using a very
#     fast generator (xorshift). Each draw is still a draw from a normal
#     distribution; results are not bit-identical to the numpy version and
#     must agree within seed-to-seed noise (checked)
#   - state is float64 instead of float32 (no effect beyond rounding)
# Parameters are passed in as arguments (not read as globals), so the runner
# can still change them per condition.
#
# Needs:  conda install numba   (or: pip install numba)

import numpy as np
from numba import njit


@njit(cache=True)
def run(n_steps, N, seed, noise_pool,
        W_indptr, W_indices, W_data,            # W as CSC: column j = outputs of neuron j
        S_indptr, S_indices, S_data, has_out,   # K-sharing matrix as CSC
        L_indptr, L_indices, L_data,            # spatial neighbour matrix as CSR
        vuln, tonic,
        DT, TAU_M, TAU_SYN, V_REST0, V_THRESH, V_BLOCK, V_PUMP_MV,
        delay_steps, refrac_steps, t_anox_step, slow_every, record_every,
        K0, K_THRESH, K_PEAK, K_LOW_GAIN, K_HIGH_GAIN, K_SPIKE_RELEASE, K_RELEASE,
        k_remove_slow, k_leak_slow, base_use_slow,
        E_SPIKE_COST, E_PUMP_COST, E_FULL, E_FAIL,
        D_K, noise_scale, syn_mode, spatial_mode, syn_fraction):
    # fast random index generator (xorshift64*) into the noise pool
    x = np.uint64(seed) | np.uint64(1)
    pool_shift = np.uint64(64 - int(np.log2(noise_pool.size)))   # pool size must be 2^n
    mult = np.uint64(2685821657736338717)
    s12 = np.uint64(12); s25 = np.uint64(25); s27 = np.uint64(27)

    # ---- state ----
    v = np.full(N, V_REST0)
    v_rest = np.full(N, V_REST0)
    g_syn = np.zeros(N)
    E = np.ones(N)
    K = np.full(N, K0)
    refr = np.zeros(N, np.int64)
    blocked = np.zeros(N, np.bool_)
    spk_count = np.zeros(N)
    block_time = np.full(N, np.nan)
    E_at_block = np.full(N, np.nan)
    K_max = K.copy()
    ring = np.zeros((delay_steps + 1, N), np.float32)   # float32: half the memory, fewer cache misses
    LK = np.zeros(N)
    newly = np.zeros(N, np.bool_)
    new_list = np.empty(N, np.int64)
    spk_list = np.empty(N, np.int64)
    n_blocked = 0

    # ---- recordings ----
    n_rec = (n_steps + record_every - 1) // record_every
    rec_t = np.zeros(n_rec); rec_active = np.zeros(n_rec); rec_blocked = np.zeros(n_rec)
    rec_meanK = np.zeros(n_rec); rec_meanE = np.zeros(n_rec); rec_chi = np.zeros(n_rec)
    hist = np.zeros(200)          # population rate of the last 200 steps (for synchrony)
    rec_i = 0

    syn_decay = DT / TAU_SYN
    dt_m = DT / TAU_M
    dump = K_RELEASE * 5.0
    a_diff = D_K * slow_every
    n_slots = delay_steps + 1

    for step in range(n_steps):
        t_ms = step * DT

        # ===== SLOW (every 1 ms): pump, energy, resting potential, block, K diffusion =====
        if step % slow_every == 0:
            after_cutoff = step >= t_anox_step
            for i in range(N):
                pump = (E[i] - E_FAIL) / (E_FULL - E_FAIL)
                if pump < 0.0:
                    pump = 0.0
                elif pump > 1.0:
                    pump = 1.0
                if blocked[i]:
                    pump = 0.0
                cleared = k_remove_slow * pump * (K[i] - K0)
                K[i] += k_leak_slow - cleared
                if after_cutoff:
                    E[i] -= vuln[i] * (base_use_slow + E_SPIKE_COST * spk_count[i]
                                       + E_PUMP_COST * cleared)
                    if E[i] < 0.0:
                        E[i] = 0.0
                    elif E[i] > 1.0:
                        E[i] = 1.0
                spk_count[i] = 0.0
                k_ex = K[i] - K_THRESH
                if k_ex < 0.0:
                    k_ex = 0.0
                k_low = K[i] - K0
                if k_low > K_THRESH - K0:
                    k_low = K_THRESH - K0
                v_rest[i] = V_REST0 + (1.0 - pump) * V_PUMP_MV + K_LOW_GAIN * k_low + K_HIGH_GAIN * k_ex
                newly[i] = (v_rest[i] > V_BLOCK) and (not blocked[i])

            # newly blocked neurons: record, then dump K (to self and synaptic partners)
            n_new = 0
            for i in range(N):
                if newly[i]:
                    blocked[i] = True
                    block_time[i] = t_ms
                    E_at_block[i] = E[i]
                    new_list[n_new] = i
                    n_new += 1
            n_blocked += n_new
            for q in range(n_new):
                j = new_list[q]
                if syn_mode and has_out[j]:
                    K[j] += dump * (1.0 - syn_fraction)
                    for p in range(S_indptr[j], S_indptr[j + 1]):
                        K[S_indices[p]] += dump * syn_fraction * S_data[p]
                else:
                    K[j] += dump

            # spatial diffusion (uses K of all neurons BEFORE this update)
            if spatial_mode:
                for i in range(N):
                    s = 0.0
                    for p in range(L_indptr[i], L_indptr[i + 1]):
                        s += L_data[p] * K[L_indices[p]]
                    LK[i] = s
                for i in range(N):
                    K[i] = (1.0 - a_diff) * K[i] + a_diff * LK[i]
            for i in range(N):
                if K[i] < K0:
                    K[i] = K0
                elif K[i] > K_PEAK:
                    K[i] = K_PEAK
                if K[i] > K_max[i]:
                    K_max[i] = K[i]

        # ===== FAST (every 0.1 ms): synapses, membrane, spikes =====
        slot = step % n_slots
        out_slot = (step + delay_steps) % n_slots
        n_spk = 0
        for i in range(N):
            g = g_syn[i] + ring[slot, i]
            ring[slot, i] = 0.0
            g -= g * syn_decay
            g_syn[i] = g
            if refr[i] <= 0:
                x ^= x >> s12
                x ^= x << s25
                x ^= x >> s27
                noise = noise_pool[(x * mult) >> pool_shift]
                v[i] += (-(v[i] - v_rest[i]) + g + tonic[i]) * dt_m + noise_scale * noise
                if v[i] >= V_THRESH and not blocked[i]:
                    spk_list[n_spk] = i
                    n_spk += 1
            else:
                refr[i] -= 1

        for q in range(n_spk):
            j = spk_list[q]
            spk_count[j] += 1.0
            if syn_mode and has_out[j]:
                K[j] += K_SPIKE_RELEASE * (1.0 - syn_fraction)
                for p in range(S_indptr[j], S_indptr[j + 1]):
                    K[S_indices[p]] += K_SPIKE_RELEASE * syn_fraction * S_data[p]
            else:
                K[j] += K_SPIKE_RELEASE
            v[j] = v_rest[j]
            refr[j] = refrac_steps
            for p in range(W_indptr[j], W_indptr[j + 1]):
                ring[out_slot, W_indices[p]] += np.float32(W_data[p])

        # ===== READOUTS =====
        pop_rate = n_spk / N
        hist[step % 200] = pop_rate
        if step % record_every == 0:
            sK = 0.0
            sE = 0.0
            for i in range(N):
                sK += K[i]
                sE += E[i]
            rec_t[rec_i] = t_ms
            rec_active[rec_i] = pop_rate
            rec_blocked[rec_i] = n_blocked / N
            rec_meanK[rec_i] = sK / N
            rec_meanE[rec_i] = sE / N
            if step + 1 > 200:
                m = hist.mean()
                var = ((hist - m) ** 2).mean()
                rec_chi[rec_i] = np.sqrt(var / (m * (1.0 - m) + 1e-9))
            rec_i += 1

    return (rec_t, rec_active, rec_blocked, rec_meanK, rec_meanE, rec_chi,
            block_time, E_at_block, K_max, K)
