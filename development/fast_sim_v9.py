# fast_sim_v9.py - compiled loops for death model v9 (= v8 + glial K buffering, rows 82-83; Goldman K coupling
#   variant, row 84: k_mode 1)
# v9 adds to cube_step (OFF when glia_dens is all zero -> identical to v8):
#   per cube a glial K store G (in mM of extracellular K taken up):
#     'shuttle': fast uptake g_up x density x (K - K0) x (1 - G/g_cap) x (g_pas + (1 - g_pas) x pump), slow passive
#     release g_rel x G back to the extracellular space. Glia do not remove K for good: the neurons' clearance
#     (unchanged, n_share = 1) does.
# fast_sim_v8.py - compiled loops for death model v8 (= v7 + physical SD release / pump recovery; log row 81)
# v8 adds to cube_step (both OFF by default -> identical to v7):
#   - sd_kmax > 0: K release by tissue in SD is limited by its driving force: x (sd_kmax - K) / (sd_kmax - SD_ON).
#     At the wave front (K ~ SD_ON) release is unchanged; it stops by itself near sd_kmax (fly plateau ~54 mM)
#     instead of running into the K_PEAK cap.
#   - sd_pump_rec = 1: the pump reduction in SD (x SD_PUMP) fades with the release (time constant SD_TAU_S)
#     instead of lasting until K < SD_OFF. Same at the wave front; lets SD tissue recover once energy is back.
# fast_sim_v7.py - compiled loops for death model v7 (= v6 + validation features; log row 80)
# v7 adds, all OFF by default (then identical to v6):
#   - reoxygenation: after t_reox_step energy is no longer used and recovers toward 1 with
#     e_recover = slow_dt / E_RECOVER_TAU_S per slow step (pumps restart, tissue can recover)
#   - ouabain: pump capacity x (1 - block x ramp) from ouab_step (energy untouched)
#   - virtual K electrodes: K of probe cubes and the fraction of cubes in SD every probe_every slow steps
#   - sd_count: how many separate SD events each cube had (repetitive SD)
# (v6 header follows)
# fast_sim_v6.py - compiled simulation loops for death model v6 (= v5 + spike counts per neuron)
# =============================================================
# v5 adds (see death_v5.py):
#   - edge exchange: cubes at the brain's OUTER surface swap K with an outside level k_out
#     (rate edge_g per cube = rate per open face x faces), cubes next to INTERNAL gaps
#     (cell-body clusters, tracts) with k_in (rate in_g). v4 = both sealed (rates 0).
#   - run_tissue oxygen loss: pumps fail on the full model's energy schedule (each
#     neuron's energy falls at e_rate x its vulnerability after t_anox), without spikes.
#
# Same model as fast_sim_v3 (K in 10 um cubes of neuropil, tissue spreading
# depolarization per cube, LIF neurons) plus:
#   - SD_DELAY_S : a cube goes into SD only after its K has stayed above SD_ON for
#                  this long (tissue needs time to depolarize; slows the wave).
#                  0 = old behaviour.
#   - surface bath: with surf_idx/surf_g given, K enters only through the brain's
#                  outer surface (exchange rate surf_g per cube, 1/s) from a bath
#                  at K_ext = K0 (+ bath_dK from bath_step). Pumps clear to K0.
#                  bath_ramp_steps: K_ext rises linearly over this many steps (wash-in).
#                  With no surface cubes given, the old "bath everywhere" applies.
#   - early stop : stop_mode 1 = stop when >= 99.9% of neurons are blocked and no
#                  cube has started SD for STOP_QUIET_MS; stop_mode 2 = also stop when
#                  SD has started somewhere and then no new cube for STOP_QUIET_MS
#                  (a trigger that fails, or a finished wave). 0 = run to the end.
#   - run_tissue : the cube layer alone, without neurons (all pumps working, spikes
#                  replaced by a steady release act_rel per cube). For calibration:
#                  ~100x faster. Checked against a full run before use.
# The cube update is shared by both loops (cube_step), so they cannot drift apart.

import numpy as np
from numba import njit


@njit(cache=True)
def cube_step(K, newK, pump_frac, m_rel, nb, sd, sd_t, above_t, cube_time, sd_count,
              t_ms, slow_dt, Kb, K_ext, surf_idx, surf_g, K_in, in_idx, in_g, extra, ej_idx, ej,
              K0, K_PEAK, K_LEAK_S, K_REMOVE_S,
              SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP, SD_DELAY_MS, a_sub, n_sub, sd_kmax, sd_pump_rec,
              G, glia_dens, g_up, g_cap, g_rel, g_pas, n_share):
    """One slow step of the cube layer. Returns (new SD starts, cubes now in SD, mean K)."""
    C = K.size
    n_new = 0
    # leak in, pump out (exact exponential decay), steady extra release (tissue-only mode)
    for c in range(C):
        pc_ = pump_frac[c]
        if sd[c]:
            if sd_pump_rec:
                pc_ *= 1.0 - (1.0 - SD_PUMP) * np.exp(-(t_ms - sd_t[c]) / (SD_TAU_S * 1000.0))
            else:
                pc_ *= SD_PUMP
        K[c] += (K_LEAK_S * m_rel[c] + extra[c]) * slow_dt
        K[c] = Kb + (K[c] - Kb) * np.exp(-slow_dt / K_REMOVE_S * pc_ * m_rel[c] * n_share)
    # glial K buffering (v9, 'shuttle'; MacAulay 2020, Larsen et al. 2014, Nielsen et al. 2025): glia take up
    # excess K fast while it is high (uptake g_up x density x (K - K0) x room left, part passive (Kir-like),
    # part energy-dependent (glial a2 pump)) and give it back slowly (g_rel x G, passive); the neurons' calibrated
    # clearance (unchanged) removes it for good. Healthy steady state: uptake = release, no net removal.
    if g_up > 0.0:
        for c in range(C):
            gd = glia_dens[c]
            if gd <= 0.0:
                continue
            ex = K[c] - K0
            if ex < 0.0:
                ex = 0.0
            room = 1.0 - G[c] / g_cap
            if room < 0.0:
                room = 0.0
            up = g_up * gd * ex * room * (g_pas + (1.0 - g_pas) * pump_frac[c]) * slow_dt
            if up > ex:
                up = ex
            rel = g_rel * G[c] * slow_dt
            if rel > G[c]:
                rel = G[c]
            K[c] += rel - up
            G[c] += up - rel
    # bath exchange through the brain surface
    for q in range(surf_idx.size):
        c = surf_idx[q]
        K[c] = K_ext + (K[c] - K_ext) * np.exp(-surf_g[q] * slow_dt)
    # active K export through the blood-brain barrier (v6): ej mM/s per cube at full energy,
    # scaled by the cube's pump state, halved while the cube is in SD (locust: Robertson et al. 2025)
    for q in range(ej_idx.size):
        c = ej_idx[q]
        out = ej[q] * pump_frac[c] * (0.5 if sd[c] else 1.0) * slow_dt
        K[c] = K[c] - out if K[c] - out > K0 else K0
    # exchange with internal gaps (cell-body clusters, tracts)
    for q in range(in_idx.size):
        c = in_idx[q]
        K[c] = K_in + (K[c] - K_in) * np.exp(-in_g[q] * slow_dt)
    # tissue SD
    n_sd = 0
    for c in range(C):
        if sd[c]:
            rel = SD_RATE * np.exp(-(t_ms - sd_t[c]) / (SD_TAU_S * 1000.0)) * slow_dt * m_rel[c]
            if sd_kmax > 0.0:
                f = (sd_kmax - K[c]) / (sd_kmax - SD_ON)
                rel *= 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)
            K[c] += rel
            if K[c] < SD_OFF and t_ms - sd_t[c] > SD_TAU_S * 1000.0:
                sd[c] = False
                above_t[c] = np.nan
        elif K[c] > SD_ON:
            if np.isnan(above_t[c]):
                above_t[c] = t_ms
            if t_ms - above_t[c] >= SD_DELAY_MS:
                sd[c] = True
                sd_t[c] = t_ms
                n_new += 1
                sd_count[c] += 1
                if np.isnan(cube_time[c]):
                    cube_time[c] = t_ms
        else:
            above_t[c] = np.nan
        if sd[c]:
            n_sd += 1
    # diffusion between neighbouring cubes
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
    return n_new, n_sd, sK / C


@njit(cache=True)
def _diff_substeps(D_UM2_S, slow_dt, CUBE_UM):
    a = D_UM2_S * slow_dt / (CUBE_UM * CUBE_UM)
    n_sub = int(a / 0.15) + 1
    return a / n_sub, n_sub


@njit(cache=True)
def run_tissue(n_slow, slow_dt_ms, m_rel, nb, act_rel, record_every,
               K0, K_PEAK, K_LEAK_S, K_REMOVE_S, D_UM2_S, CUBE_UM,
               trig_step, trig_end_step, trig_cubes, trig_K,
               bath_step, bath_dK, bath_ramp_steps, surf_idx, surf_g,
               edge_idx, edge_g, k_out, in_idx, in_g, k_in, ej_idx, ej,
               P_indptr, P_idx, P_w, vuln, t_anox_step, e_rate, E_FULL, E_FAIL, pump_every, anox_gain,
               SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP, SD_DELAY_S,
               stop_mode, stop_quiet_ms, sd_kmax, sd_pump_rec, glia_dens, g_up, g_cap, g_rel, g_pas, n_share):
    """Cube layer only. Steps are SLOW steps (slow_dt_ms each). Returns
    (t, frac_in_SD, frac_ever_SD, meanK, cube_time, K)."""
    C = m_rel.size
    slow_dt = slow_dt_ms / 1000.0
    a_sub, n_sub = _diff_substeps(D_UM2_S, slow_dt, CUBE_UM)
    K = np.full(C, K0); newK = np.empty(C)
    sd = np.zeros(C, np.bool_); sd_t = np.zeros(C); above_t = np.full(C, np.nan)
    cube_time = np.full(C, np.nan); sd_count = np.zeros(C, np.int64); G = np.zeros(C)
    pump_frac = np.ones(C); pres = np.zeros(C); extra = act_rel.copy()
    n_rec = (n_slow + record_every - 1) // record_every
    rec_t = np.zeros(n_rec); rec_sd = np.zeros(n_rec); rec_ever = np.zeros(n_rec); rec_K = np.zeros(n_rec)
    rec_i = 0; n_ever = 0; last_new = -1e18
    use_surf = surf_idx.size > 0
    for s in range(n_slow):
        t_ms = s * slow_dt_ms
        if trig_step >= 0 and s >= trig_step and s < trig_end_step:
            for q in range(trig_cubes.size):
                if K[trig_cubes[q]] < trig_K:
                    K[trig_cubes[q]] = trig_K
        if t_anox_step >= 0 and s >= t_anox_step and (s - t_anox_step) % pump_every == 0:
            dt_an = (s - t_anox_step) * slow_dt
            for c in range(C):
                pump_frac[c] = 0.0; pres[c] = 0.0
            for i in range(vuln.size):
                E = 1.0 - vuln[i] * e_rate * dt_an
                pmp = (E - E_FAIL) / (E_FULL - E_FAIL)
                pmp = 0.0 if pmp < 0.0 else (1.0 if pmp > 1.0 else pmp)
                for p in range(P_indptr[i], P_indptr[i + 1]):
                    pump_frac[P_idx[p]] += P_w[p] * pmp
                    pres[P_idx[p]] += P_w[p]
            for c in range(C):
                pump_frac[c] = pump_frac[c] / pres[c] if pres[c] > 0.0 else 0.0
                # starving neurons depolarize and fire more: more K released (fitted gain)
                extra[c] = act_rel[c] * (1.0 + anox_gain * (1.0 - pump_frac[c]))
        bath_on = bath_step >= 0 and s >= bath_step
        if use_surf:
            Kb = K0
            K_ext = k_out
            if bath_on:
                fr = 1.0 if bath_ramp_steps <= 0 else min(1.0, (s - bath_step) / bath_ramp_steps)
                K_ext = k_out + (K0 + bath_dK - k_out) * fr
            si, sg = surf_idx, surf_g
        else:
            Kb = K0 + bath_dK if bath_on else K0
            K_ext = k_out
            si, sg = edge_idx, edge_g
        n_new, n_sd, mK = cube_step(K, newK, pump_frac, m_rel, nb, sd, sd_t, above_t, cube_time, sd_count,
                                    t_ms, slow_dt, Kb, K_ext, si, sg, k_in, in_idx, in_g, extra, ej_idx, ej,
                                    K0, K_PEAK, K_LEAK_S, K_REMOVE_S,
                                    SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP, SD_DELAY_S * 1000.0,
                                    a_sub, n_sub, sd_kmax, sd_pump_rec,
                                    G, glia_dens, g_up, g_cap, g_rel, g_pas, n_share)
        if n_new > 0:
            n_ever = 0
            for c in range(C):
                if not np.isnan(cube_time[c]):
                    n_ever += 1
            last_new = t_ms
        if s % record_every == 0:
            rec_t[rec_i] = t_ms; rec_sd[rec_i] = n_sd / C; rec_ever[rec_i] = n_ever / C; rec_K[rec_i] = mK
            rec_i += 1
        if stop_mode >= 1 and n_ever > 0 and t_ms - last_new > stop_quiet_ms \
                and s > trig_end_step and (bath_step < 0 or s > bath_step) \
                and (t_anox_step < 0 or n_ever >= 0.995 * C):     # oxygen loss: only once the brain is dead
            break
    return rec_t[:rec_i], rec_sd[:rec_i], rec_ever[:rec_i], rec_K[:rec_i], cube_time, K


@njit(cache=True)
def run(n_steps, N, seed, noise_pool,
        W_indptr, W_indices, W_data,
        P_indptr, P_idx, P_w,
        O_indptr, O_idx, O_w,
        m_rel, m_ref, nb,
        vuln, tonic,
        DT, TAU_M, TAU_SYN, V_REST0, V_THRESH, V_BLOCK, V_PUMP_MV,
        delay_steps, refrac_steps, t_anox_step, slow_every, record_every,
        K0, K_THRESH, K_PEAK, K_LOW_GAIN, K_HIGH_GAIN, K_SPIKE_RELEASE, K_RELEASE,
        K_REMOVE_S, K_LEAK_S, E_BASE_USE_S, E_SPIKE_COST, E_PUMP_COST, E_FULL, E_FAIL,
        D_UM2_S, CUBE_UM, noise_scale, CUBE_ACTIVE_K,
        trig_step, trig_end_step, trig_cubes, trig_K,
        bath_step, bath_dK, bath_ramp_steps, surf_idx, surf_g,
        edge_idx, edge_g, k_out, in_idx, in_g, k_in, ej_idx, ej,
        SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP, SD_DELAY_S, V_UNBLOCK,
        stop_mode, stop_quiet_ms,
        t_reox_step, e_recover, ouab_step, ouab_block, ouab_ramp_steps, probe_idx, probe_every, sd_kmax, sd_pump_rec,
        glia_dens, g_up, g_cap, g_rel, g_pas, n_share, k_mode, ghk_alpha, ghk_slope):
    C = m_rel.size
    slow_dt = DT * slow_every / 1000.0
    n_slow_tot = n_steps // slow_every + 1
    n_prb = n_slow_tot // probe_every + 1
    prb_t = np.zeros(n_prb); prb_K = np.zeros((n_prb, probe_idx.size)); prb_fsd = np.zeros(n_prb)
    prb_i = 0; slow_i = 0; n_sd = 0
    sd_count = np.zeros(C, np.int64); G = np.zeros(C); prb_G = np.zeros(n_prb)
    x = np.uint64(seed) | np.uint64(1)
    pool_shift = np.uint64(64 - int(np.log2(noise_pool.size)))
    mult = np.uint64(2685821657736338717)
    s12 = np.uint64(12); s25 = np.uint64(25); s27 = np.uint64(27)

    v = np.full(N, V_REST0); v_rest = np.full(N, V_REST0); g_syn = np.zeros(N)
    E = np.ones(N); refr = np.zeros(N, np.int64); blocked = np.zeros(N, np.bool_)
    spk_count = np.zeros(N); pump_n = np.ones(N); Kn = np.full(N, K0)
    spikes_pre = np.zeros(N); spikes_post = np.zeros(N)   # v6: spikes before / after oxygen loss
    block_time = np.full(N, np.nan); E_at_block = np.full(N, np.nan); Kn_max = Kn.copy()
    ring = np.zeros((delay_steps + 1, N), np.float32)
    spk_list = np.empty(N, np.int64)
    n_blocked = 0

    K = np.full(C, K0); newK = np.empty(C); pump_c = np.zeros(C); pres = np.zeros(C)
    pump_frac = np.zeros(C)
    cube_time = np.full(C, np.nan)
    sd = np.zeros(C, np.bool_); sd_t = np.zeros(C); above_t = np.full(C, np.nan)
    a_sub, n_sub = _diff_substeps(D_UM2_S, slow_dt, CUBE_UM)
    extra = np.zeros(C)
    use_surf = surf_idx.size > 0
    n_ever = 0; last_new = -1e18

    n_rec = (n_steps + record_every - 1) // record_every
    rec_t = np.zeros(n_rec); rec_active = np.zeros(n_rec); rec_blocked = np.zeros(n_rec)
    rec_meanK = np.zeros(n_rec); rec_meanE = np.zeros(n_rec); rec_chi = np.zeros(n_rec)
    hist = np.zeros(200); rec_i = 0
    meanK = K0; meanE = 1.0

    syn_decay = DT / TAU_SYN; dt_m = DT / TAU_M; n_slots = delay_steps + 1
    dump = K_RELEASE * 5.0
    trig_end_ms = trig_end_step * DT
    bath_ms = bath_step * DT

    for step in range(n_steps):
        t_ms = step * DT

        if step % slow_every == 0:
            # early stop (checked at slow steps only)
            if stop_mode >= 1 and t_ms - last_new > stop_quiet_ms and n_ever > 0:
                if n_blocked >= 0.999 * N:
                    break
                if stop_mode >= 2 and (trig_step < 0 or t_ms > trig_end_ms) \
                        and (bath_step < 0 or t_ms > bath_ms):
                    break
            after = step >= t_anox_step and (t_reox_step < 0 or step < t_reox_step)
            reox = t_reox_step >= 0 and step >= t_reox_step
            pscale = 1.0
            if ouab_step >= 0 and step >= ouab_step:
                fr = 1.0 if ouab_ramp_steps <= 0 else min(1.0, (step - ouab_step) / ouab_ramp_steps)
                pscale = 1.0 - ouab_block * fr
            for i in range(N):
                s = 0.0
                for p in range(P_indptr[i], P_indptr[i + 1]):
                    kc = K[P_idx[p]]
                    s += P_w[p] * (K_PEAK if kc > K_PEAK else kc)
                Kn[i] = s
                if s > Kn_max[i]:
                    Kn_max[i] = s
            sE = 0.0
            for i in range(N):
                s = Kn[i]
                pump = (E[i] - E_FAIL) / (E_FULL - E_FAIL)
                pump = 0.0 if pump < 0.0 else (1.0 if pump > 1.0 else pump)
                pump *= pscale
                pump_n[i] = pump
                cleared = slow_dt / K_REMOVE_S * pump * (s - K0)
                if after:
                    E[i] -= vuln[i] * (slow_dt / E_BASE_USE_S + E_SPIKE_COST * spk_count[i]
                                       + E_PUMP_COST * cleared)
                    E[i] = 0.0 if E[i] < 0.0 else (1.0 if E[i] > 1.0 else E[i])
                elif reox:
                    E[i] += (1.0 - E[i]) * e_recover
                sE += E[i]
                spk_count[i] = 0.0
                k_ex = s - K_THRESH
                k_ex = 0.0 if k_ex < 0.0 else k_ex
                k_low = s - K0
                k_low = K_THRESH - K0 if k_low > K_THRESH - K0 else k_low
                if k_mode == 1:
                    # Goldman (GHK) shift of the resting potential relative to K0 (v9 variant):
                    # slope x log10((K + a) / (K0 + a)), a = (P_Na / P_K) x [Na]o; K_i and Na_i cancel out
                    k_term = ghk_slope * np.log10((s + ghk_alpha) / (K0 + ghk_alpha))
                else:
                    k_term = K_LOW_GAIN * k_low + K_HIGH_GAIN * k_ex
                v_rest[i] = V_REST0 + (1.0 - pump) * V_PUMP_MV + k_term
                if v_rest[i] > V_BLOCK and not blocked[i]:
                    blocked[i] = True
                    n_blocked += 1
                    if np.isnan(block_time[i]):
                        block_time[i] = t_ms
                        E_at_block[i] = E[i]
                    for p in range(O_indptr[i], O_indptr[i + 1]):
                        K[O_idx[p]] += dump * O_w[p] / m_ref
                elif blocked[i] and v_rest[i] < V_UNBLOCK:
                    blocked[i] = False
                    n_blocked -= 1
            meanE = sE / N
            for c in range(C):
                pump_c[c] = 0.0
                pres[c] = 0.0
            for i in range(N):
                for p in range(P_indptr[i], P_indptr[i + 1]):
                    pump_c[P_idx[p]] += P_w[p] * pump_n[i]
                    pres[P_idx[p]] += P_w[p]
            for c in range(C):
                pump_frac[c] = pump_c[c] / pres[c] if pres[c] > 0.0 else 0.0
            if trig_step >= 0 and step >= trig_step and step < trig_end_step:
                for q in range(trig_cubes.size):
                    if K[trig_cubes[q]] < trig_K:
                        K[trig_cubes[q]] = trig_K
            bath_on = bath_step >= 0 and step >= bath_step
            if use_surf:
                Kb = K0
                K_ext = k_out
                if bath_on:
                    fr = 1.0 if bath_ramp_steps <= 0 else min(1.0, (step - bath_step) / bath_ramp_steps)
                    K_ext = k_out + (K0 + bath_dK - k_out) * fr
                si, sg = surf_idx, surf_g
            else:
                Kb = K0 + bath_dK if bath_on else K0
                K_ext = k_out
                si, sg = edge_idx, edge_g
            n_new, n_sd, meanK = cube_step(K, newK, pump_frac, m_rel, nb, sd, sd_t, above_t, cube_time, sd_count,
                                           t_ms, slow_dt, Kb, K_ext, si, sg, k_in, in_idx, in_g, extra, ej_idx, ej,
                                           K0, K_PEAK, K_LEAK_S, K_REMOVE_S,
                                           SD_ON, SD_OFF, SD_RATE, SD_TAU_S, SD_PUMP,
                                           SD_DELAY_S * 1000.0, a_sub, n_sub, sd_kmax, sd_pump_rec,
                                    G, glia_dens, g_up, g_cap, g_rel, g_pas, n_share)
            if n_new > 0:
                n_ever += n_new
                last_new = t_ms
            if slow_i % probe_every == 0 and prb_i < n_prb:
                prb_t[prb_i] = t_ms
                for q in range(probe_idx.size):
                    prb_K[prb_i, q] = K[probe_idx[q]]
                prb_fsd[prb_i] = n_sd / C
                prb_G[prb_i] = G.mean()
                prb_i += 1
            slow_i += 1

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
            if step < t_anox_step:
                spikes_pre[j] += 1.0
            else:
                spikes_post[j] += 1.0
            for p in range(O_indptr[j], O_indptr[j + 1]):
                K[O_idx[p]] += K_SPIKE_RELEASE * O_w[p] / m_ref
            v[j] = v_rest[j]
            refr[j] = refrac_steps
            for p in range(W_indptr[j], W_indptr[j + 1]):
                ring[out_slot, W_indices[p]] += np.float32(W_data[p])

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

    return (rec_t[:rec_i], rec_active[:rec_i], rec_blocked[:rec_i], rec_meanK[:rec_i],
            rec_meanE[:rec_i], rec_chi[:rec_i],
            block_time, E_at_block, Kn_max, Kn, cube_time, K, spikes_pre, spikes_post, step,
            prb_t[:prb_i], prb_K[:prb_i], prb_fsd[:prb_i], sd_count, prb_G[:prb_i], G)
