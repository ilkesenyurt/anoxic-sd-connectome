# fast_sim_v3.py - compiled simulation loop for death model v3
# ============================================================
# v3: extracellular K+ lives in CUBES of neuropil (e.g. 10 um), not in neurons.
#   - each neuron "sits" in the cubes that hold its synapses (weights P, sum 1)
#   - it FEELS the K of those cubes (weighted average) and its pump clears K there
#   - it RELEASES K where its OUTPUT synapses are (weights O, sum 1)
#   - K diffuses between neighbouring cubes (6 neighbours, real diffusion
#     constant D in um^2/s); cubes without synapses are outside the domain
#   - v3.1: spreading depolarization is a property of the TISSUE in each cube:
#     when cube K passes SD_ON, the cube depolarizes, its K clearance weakens (SD_PUMP)
#     and it releases K (SD_RATE, fading with SD_TAU_S). Neurons feel it through
#     the cube K. (A single neuron averages ~20 cubes, so neuron-level block alone
#     could not start or carry a wave - log row 51.)
#   - leak, pump clearance and release scale with how much tissue a cube holds
#     (m_c = neuron-equivalents in the cube = sum of P; m_rel = m_c / m_ref, where
#     m_ref is the density of the cube a typical neuron sits in). Release per
#     neuron is divided by m_ref, so a cube of typical density behaves like one
#     v2 neuron. A healthy cube sits near the
#     same K whatever its density, but dense cubes rise FASTER when pumps fail
# The spiking part (LIF, synapses, noise, energy per neuron) is the same as v2.

import numpy as np
from numba import njit


@njit(cache=True)
def run(n_steps, N, seed, noise_pool,
        W_indptr, W_indices, W_data,           # spikes: W as CSC (column j = outputs of j)
        P_indptr, P_idx, P_w,                  # neuron -> cubes it sits in (CSR, rows sum 1)
        O_indptr, O_idx, O_w,                  # neuron -> cubes it releases into (CSR, rows sum 1)
        m_rel, m_ref, nb,                      # cube tissue amount (/typical), the typical density, cube neighbours (C x 6, -1 = none)
        vuln, tonic,
        DT, TAU_M, TAU_SYN, V_REST0, V_THRESH, V_BLOCK, V_PUMP_MV,
        delay_steps, refrac_steps, t_anox_step, slow_every, record_every,
        K0, K_THRESH, K_PEAK, K_LOW_GAIN, K_HIGH_GAIN, K_SPIKE_RELEASE, K_RELEASE,
        K_REMOVE_S, K_LEAK_S, E_BASE_USE_S, E_SPIKE_COST, E_PUMP_COST, E_FULL, E_FAIL,
        D_UM2_S, CUBE_UM, noise_scale, CUBE_ACTIVE_K,
        trig_step, trig_end_step, trig_cubes, trig_K,   # local K clamp (like a KCl pinprick) from trig_step to trig_end_step
        bath_step, bath_dK,                     # KCl bath: from bath_step pumps clear only down to K0 + bath_dK
        SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP, V_UNBLOCK):   # spreading depolarization of the TISSUE in each cube
    C = m_rel.size
    slow_dt = DT * slow_every / 1000.0                       # s
    # ---- random numbers: xorshift index into a pool of normal numbers ----
    x = np.uint64(seed) | np.uint64(1)
    pool_shift = np.uint64(64 - int(np.log2(noise_pool.size)))
    mult = np.uint64(2685821657736338717)
    s12 = np.uint64(12); s25 = np.uint64(25); s27 = np.uint64(27)

    # ---- neuron state ----
    v = np.full(N, V_REST0); v_rest = np.full(N, V_REST0); g_syn = np.zeros(N)
    E = np.ones(N); refr = np.zeros(N, np.int64); blocked = np.zeros(N, np.bool_)
    spk_count = np.zeros(N); pump_n = np.ones(N); Kn = np.full(N, K0)
    block_time = np.full(N, np.nan); E_at_block = np.full(N, np.nan); Kn_max = Kn.copy()
    ring = np.zeros((delay_steps + 1, N), np.float32)
    spk_list = np.empty(N, np.int64)
    n_blocked = 0

    # ---- cube state ----
    K = np.full(C, K0); newK = np.empty(C); pump_c = np.zeros(C); pres = np.zeros(C)
    cube_time = np.full(C, np.nan)                           # first time the cube goes into SD
    sd = np.zeros(C, np.bool_)                               # cube tissue currently in SD
    sd_t = np.zeros(C)                                       # when it entered SD (ms)
    n_sd_events = 0
    # diffusion: explicit, split into substeps if needed for stability
    a = D_UM2_S * slow_dt / (CUBE_UM * CUBE_UM)
    n_sub = int(a / 0.15) + 1
    a_sub = a / n_sub

    # ---- recordings ----
    n_rec = (n_steps + record_every - 1) // record_every
    rec_t = np.zeros(n_rec); rec_active = np.zeros(n_rec); rec_blocked = np.zeros(n_rec)
    rec_meanK = np.zeros(n_rec); rec_meanE = np.zeros(n_rec); rec_chi = np.zeros(n_rec)
    hist = np.zeros(200); rec_i = 0
    meanK = K0; meanE = 1.0

    syn_decay = DT / TAU_SYN; dt_m = DT / TAU_M; n_slots = delay_steps + 1
    dump = K_RELEASE * 5.0

    for step in range(n_steps):
        t_ms = step * DT

        # ================= SLOW: energy, pumps, block, cube K =================
        if step % slow_every == 0:
            after = step >= t_anox_step
            # --- neurons feel the K of their cubes (all read BEFORE any dump this step) ---
            for i in range(N):
                s = 0.0
                for p in range(P_indptr[i], P_indptr[i + 1]):
                    kc = K[P_idx[p]]
                    s += P_w[p] * (K_PEAK if kc > K_PEAK else kc)
                Kn[i] = s
                if s > Kn_max[i]:
                    Kn_max[i] = s
            # --- neurons: pump, energy, resting potential, block ---
            sE = 0.0
            for i in range(N):
                s = Kn[i]
                pump = (E[i] - E_FAIL) / (E_FULL - E_FAIL)
                pump = 0.0 if pump < 0.0 else (1.0 if pump > 1.0 else pump)
                pump_n[i] = pump                     # (blocked neurons keep their energy-limited pump;
                                                     #  the SD tissue weakening is applied per cube)
                cleared = slow_dt / K_REMOVE_S * pump * (s - K0)   # this neuron's pump work
                if after:
                    E[i] -= vuln[i] * (slow_dt / E_BASE_USE_S + E_SPIKE_COST * spk_count[i]
                                       + E_PUMP_COST * cleared)
                    E[i] = 0.0 if E[i] < 0.0 else (1.0 if E[i] > 1.0 else E[i])
                sE += E[i]
                spk_count[i] = 0.0
                k_ex = s - K_THRESH
                k_ex = 0.0 if k_ex < 0.0 else k_ex
                k_low = s - K0
                k_low = K_THRESH - K0 if k_low > K_THRESH - K0 else k_low
                v_rest[i] = V_REST0 + (1.0 - pump) * V_PUMP_MV + K_LOW_GAIN * k_low + K_HIGH_GAIN * k_ex
                if v_rest[i] > V_BLOCK and not blocked[i]:
                    blocked[i] = True
                    n_blocked += 1
                    if np.isnan(block_time[i]):              # first block only
                        block_time[i] = t_ms
                        E_at_block[i] = E[i]
                    for p in range(O_indptr[i], O_indptr[i + 1]):   # dump K where it releases (off in v3.1)
                        K[O_idx[p]] += dump * O_w[p] / m_ref
                elif blocked[i] and v_rest[i] < V_UNBLOCK:
                    blocked[i] = False                       # recovers once K is back down (needs energy)
                    n_blocked -= 1
            meanE = sE / N
            # --- cubes: average pump of the neurons present ---
            for c in range(C):
                pump_c[c] = 0.0
                pres[c] = 0.0
            for i in range(N):
                for p in range(P_indptr[i], P_indptr[i + 1]):
                    pump_c[P_idx[p]] += P_w[p] * pump_n[i]
                    pres[P_idx[p]] += P_w[p]
            # --- experimental triggers (off unless set) ---
            if trig_step >= 0 and step >= trig_step and step < trig_end_step:
                for q in range(trig_cubes.size):
                    if K[trig_cubes[q]] < trig_K:
                        K[trig_cubes[q]] = trig_K
            Kb = K0 + bath_dK if (bath_step >= 0 and step >= bath_step) else K0   # level pumps clear down to
            # --- cubes: leak in, pump out (exact exponential decay), scaled by tissue ---
            for c in range(C):
                pc_ = pump_c[c] / pres[c] if pres[c] > 0.0 else 0.0
                if sd[c]:
                    pc_ *= SD_PUMP                   # depolarized tissue clears K poorly (v3.2)
                K[c] += K_LEAK_S * slow_dt * m_rel[c]
                K[c] = Kb + (K[c] - Kb) * np.exp(-slow_dt / K_REMOVE_S * pc_ * m_rel[c])
                # SD: once local K passes SD_ON the tissue depolarizes and releases K
                # (regenerative), fading with time constant SD_TAU_S; it recovers when
                # pumps bring K back below SD_OFF
                if sd[c]:
                    K[c] += SD_RATE * np.exp(-(t_ms - sd_t[c]) / (SD_TAU_S * 1000.0)) * slow_dt * m_rel[c]
                    if K[c] < SD_OFF and t_ms - sd_t[c] > SD_TAU_S * 1000.0:
                        sd[c] = False
                elif K[c] > SD_ON:
                    sd[c] = True
                    sd_t[c] = t_ms
                    n_sd_events += 1
                    if np.isnan(cube_time[c]):
                        cube_time[c] = t_ms
            # --- diffusion between neighbouring cubes ---
            for _ in range(n_sub):
                for c in range(C):
                    lap = 0.0
                    for q in range(6):
                        j = nb[c, q]
                        if j >= 0:
                            lap += K[j] - K[c]
                    newK[c] = K[c] + a_sub * lap
                for c in range(C):
                    K[c] = newK[c]
            sK = 0.0
            for c in range(C):
                if K[c] < K0:
                    K[c] = K0
                elif K[c] > K_PEAK:
                    K[c] = K_PEAK
                sK += K[c]
            meanK = sK / C

        # ================= FAST: synapses, membrane, spikes =================
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
                v[i] += (-(v[i] - v_rest[i]) + g + tonic[i]) * dt_m \
                        + noise_scale * noise_pool[(x * mult) >> pool_shift]
                if v[i] >= V_THRESH and not blocked[i]:
                    spk_list[n_spk] = i
                    n_spk += 1
            else:
                refr[i] -= 1
        for q in range(n_spk):
            j = spk_list[q]
            spk_count[j] += 1.0
            for p in range(O_indptr[j], O_indptr[j + 1]):          # K released at its output synapses
                K[O_idx[p]] += K_SPIKE_RELEASE * O_w[p] / m_ref
            v[j] = v_rest[j]
            refr[j] = refrac_steps
            for p in range(W_indptr[j], W_indptr[j + 1]):
                ring[out_slot, W_indices[p]] += np.float32(W_data[p])

        # ================= READOUTS =================
        pop_rate = n_spk / N
        hist[step % 200] = pop_rate
        if step % record_every == 0:
            rec_t[rec_i] = t_ms; rec_active[rec_i] = pop_rate
            rec_blocked[rec_i] = n_blocked / N
            rec_meanK[rec_i] = meanK; rec_meanE[rec_i] = meanE
            if step + 1 > 200:
                m = hist.mean()
                rec_chi[rec_i] = np.sqrt(((hist - m) ** 2).mean() / (m * (1.0 - m) + 1e-9))
            rec_i += 1

    return (rec_t, rec_active, rec_blocked, rec_meanK, rec_meanE, rec_chi,
            block_time, E_at_block, Kn_max, Kn, cube_time, K)
