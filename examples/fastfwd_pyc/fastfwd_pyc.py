# -*- coding: utf-8 -*-
"""FastFwd packet processing engine using Cycle-Aware API.

A high-performance packet forwarding engine demonstrating:
- CycleAwareQueue for issue/completion queues
- CycleAwareReg for wheels, ROBs, and history tracking
- Complex scheduling with timing wheels
"""
from __future__ import annotations

from dataclasses import dataclass

from pycircuit import (
    CycleAwareCircuit,
    CycleAwareDomain,
    CycleAwareQueue,
    CycleAwareReg,
    CycleAwareSignal,
    compile_cycle_aware,
    mux,
    ca_cat,
)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@dataclass(frozen=True)
class _Wheel:
    valid: list[CycleAwareReg]
    seq: list[CycleAwareReg]


@dataclass(frozen=True)
class _Rob:
    valid: list[CycleAwareReg]
    data: list[CycleAwareReg]


@dataclass(frozen=True)
class _Hist:
    valid: list[CycleAwareReg]
    seq: list[CycleAwareReg]
    data: list[CycleAwareReg]


def _wheel_read(
    m: CycleAwareCircuit,
    valid: list[CycleAwareReg],
    seq: list[CycleAwareReg],
    *,
    idx: CycleAwareSignal,
    zero_seq: CycleAwareSignal,
) -> tuple[CycleAwareSignal, CycleAwareSignal]:
    """Read (valid, seq) at dynamic idx from wheel."""
    out_v = m.ca_const(0, width=1)
    out_s = zero_seq
    for s in range(len(valid)):
        is_s = idx.eq(s)
        out_v = mux(is_s, valid[s].out(), out_v)
        out_s = mux(is_s, seq[s].out(), out_s)
    return out_v, out_s


def _wheel_slot_busy(
    m: CycleAwareCircuit,
    valid: list[CycleAwareReg],
    *,
    idx: CycleAwareSignal,
) -> CycleAwareSignal:
    busy = m.ca_const(0, width=1)
    for s in range(len(valid)):
        busy = busy | (idx.eq(s) & valid[s].out())
    return busy


def _wheel_update(
    m: CycleAwareCircuit,
    valid: list[CycleAwareReg],
    seq: list[CycleAwareReg],
    *,
    set_en: CycleAwareSignal,
    set_idx: CycleAwareSignal,
    set_seq: CycleAwareSignal,
    clear_en: CycleAwareSignal,
    clear_idx: CycleAwareSignal,
) -> None:
    """Update wheel with one optional set and one optional clear."""
    one = m.ca_const(1, width=1)
    zero = m.ca_const(0, width=1)
    
    for s in range(len(valid)):
        v_cur = valid[s].out()
        s_cur = seq[s].out()

        do_clear = clear_en & clear_idx.eq(s)
        do_set = set_en & set_idx.eq(s)

        v_next = mux(do_set, one, mux(do_clear, zero, v_cur))
        s_next = mux(do_set, set_seq, s_cur)

        valid[s].set(v_next)
        seq[s].set(s_next)


def _rob_update(
    m: CycleAwareCircuit,
    rob: _Rob,
    *,
    exp_seq: CycleAwareSignal,
    commit_pop: CycleAwareSignal,
    ins_fire: CycleAwareSignal,
    ins_seq: CycleAwareSignal,
    ins_data: CycleAwareSignal,
    depth: int,
) -> None:
    """Update per-lane ROB."""
    one = m.ca_const(1, width=1)
    zero1 = m.ca_const(0, width=1)
    zero_data = m.ca_const(0, width=ins_data.width)
    
    delta = (ins_seq - exp_seq) >> 2

    v_ins: list[CycleAwareSignal] = []
    d_ins: list[CycleAwareSignal] = []
    for i in range(depth):
        hit = ins_fire & delta.eq(i)
        v_ins.append(mux(hit, one, rob.valid[i].out()))
        d_ins.append(mux(hit, ins_data, rob.data[i].out()))

    for i in range(depth):
        if i + 1 < depth:
            v_next = mux(commit_pop, v_ins[i + 1], v_ins[i])
            d_next = mux(commit_pop, d_ins[i + 1], d_ins[i])
        else:
            v_next = mux(commit_pop, zero1, v_ins[i])
            d_next = mux(commit_pop, zero_data, d_ins[i])
        rob.valid[i].set(v_next)
        rob.data[i].set(d_next)


def _hist_shift_insert(
    m: CycleAwareCircuit,
    hist: _Hist,
    *,
    k: CycleAwareSignal,
    seqs: list[CycleAwareSignal],
    datas: list[CycleAwareSignal],
) -> None:
    """Shift history down by k (0..4) and insert up to 4 newest entries."""
    depth = len(hist.valid)
    one = m.ca_const(1, width=1)
    
    for i in range(depth):
        v0 = hist.valid[i].out()
        s0 = hist.seq[i].out()
        d0 = hist.data[i].out()

        # Defaults: no shift (k=0)
        v_next = v0
        s_next = s0
        d_next = d0

        # k==1
        if i == 0:
            v_k1 = one
            s_k1 = seqs[0]
            d_k1 = datas[0]
        else:
            v_k1 = hist.valid[i - 1].out()
            s_k1 = hist.seq[i - 1].out()
            d_k1 = hist.data[i - 1].out()

        # k==2
        if i == 0:
            v_k2, s_k2, d_k2 = one, seqs[1], datas[1]
        elif i == 1:
            v_k2, s_k2, d_k2 = one, seqs[0], datas[0]
        else:
            v_k2 = hist.valid[i - 2].out()
            s_k2 = hist.seq[i - 2].out()
            d_k2 = hist.data[i - 2].out()

        # k==3
        if i == 0:
            v_k3, s_k3, d_k3 = one, seqs[2], datas[2]
        elif i == 1:
            v_k3, s_k3, d_k3 = one, seqs[1], datas[1]
        elif i == 2:
            v_k3, s_k3, d_k3 = one, seqs[0], datas[0]
        else:
            v_k3 = hist.valid[i - 3].out()
            s_k3 = hist.seq[i - 3].out()
            d_k3 = hist.data[i - 3].out()

        # k==4
        if i == 0:
            v_k4, s_k4, d_k4 = one, seqs[3], datas[3]
        elif i == 1:
            v_k4, s_k4, d_k4 = one, seqs[2], datas[2]
        elif i == 2:
            v_k4, s_k4, d_k4 = one, seqs[1], datas[1]
        elif i == 3:
            v_k4, s_k4, d_k4 = one, seqs[0], datas[0]
        else:
            v_k4 = hist.valid[i - 4].out()
            s_k4 = hist.seq[i - 4].out()
            d_k4 = hist.data[i - 4].out()

        is1 = k.eq(1)
        is2 = k.eq(2)
        is3 = k.eq(3)
        is4 = k.eq(4)

        v_next = mux(is4, v_k4, mux(is3, v_k3, mux(is2, v_k2, mux(is1, v_k1, v_next))))
        s_next = mux(is4, s_k4, mux(is3, s_k3, mux(is2, s_k2, mux(is1, s_k1, s_next))))
        d_next = mux(is4, d_k4, mux(is3, d_k3, mux(is2, d_k2, mux(is1, d_k1, d_next))))

        hist.valid[i].set(v_next)
        hist.seq[i].set(s_next)
        hist.data[i].set(d_next)


def _build_fastfwd_impl(
    m: CycleAwareCircuit,
    domain: CycleAwareDomain,
    eng_per_lane: int,
    lane_q_depth: int,
    eng_q_depth: int,
    rob_depth: int,
    seq_w: int,
    wheel: int,
    hist_depth: int,
) -> None:
    total_eng = 4 * eng_per_lane
    wheel_bits = (wheel - 1).bit_length()
    bundle_w = seq_w + 128 + 5
    comp_w = seq_w + 128

    # Input signals
    pkt_in_vld = [domain.create_signal(f"lane{i}_pkt_in_vld", width=1) for i in range(4)]
    pkt_in_data = [domain.create_signal(f"lane{i}_pkt_in_data", width=128) for i in range(4)]
    pkt_in_ctrl = [domain.create_signal(f"lane{i}_pkt_in_ctrl", width=5) for i in range(4)]
    fwded_vld = [domain.create_signal(f"fwded{e}_pkt_data_vld", width=1) for e in range(total_eng)]
    fwded_data = [domain.create_signal(f"fwded{e}_pkt_data", width=128) for e in range(total_eng)]

    # Output registers
    bkpr_r = m.ca_reg("pkt_in_bkpr", domain=domain, width=1, init=0)
    out_vld_r = [m.ca_reg(f"lane{i}_pkt_out_vld", domain=domain, width=1, init=0) for i in range(4)]
    out_data_r = [m.ca_reg(f"lane{i}_pkt_out_data", domain=domain, width=128, init=0) for i in range(4)]

    # Global registers
    cycle = m.ca_reg("cycle", domain=domain, width=16, init=0)
    cycle.set(cycle.out() + 1)
    cycle_mod = cycle.out()[0:wheel_bits]

    seq_alloc = m.ca_reg("seq_alloc", domain=domain, width=seq_w, init=0)
    commit_lane = m.ca_reg("commit_lane", domain=domain, width=2, init=0)
    exp_seq = [m.ca_reg(f"lane{i}__exp_seq", domain=domain, width=seq_w, init=i) for i in range(4)]

    # Per-lane issue queues
    issue_q: list[CycleAwareQueue] = []
    for lane in range(4):
        issue_q.append(m.ca_queue(f"lane{lane}__issue_q", domain=domain, width=bundle_w, depth=lane_q_depth))

    # Per-engine wheels + completion queues
    wheels: list[_Wheel] = []
    comp_q: list[CycleAwareQueue] = []
    for e in range(total_eng):
        wv = [m.ca_reg(f"eng{e}__wheel_v{s}", domain=domain, width=1, init=0) for s in range(wheel)]
        ws = [m.ca_reg(f"eng{e}__wheel_seq{s}", domain=domain, width=seq_w, init=0) for s in range(wheel)]
        wheels.append(_Wheel(valid=wv, seq=ws))
        comp_q.append(m.ca_queue(f"eng{e}__comp_q", domain=domain, width=comp_w, depth=eng_q_depth))

    # Per-lane ROBs
    robs: list[_Rob] = []
    for lane in range(4):
        rv = [m.ca_reg(f"lane{lane}__rob_v{i}", domain=domain, width=1, init=0) for i in range(rob_depth)]
        rd = [m.ca_reg(f"lane{lane}__rob_d{i}", domain=domain, width=128, init=0) for i in range(rob_depth)]
        robs.append(_Rob(valid=rv, data=rd))

    # History
    hist = _Hist(
        valid=[m.ca_reg(f"hist_v{i}", domain=domain, width=1, init=0) for i in range(hist_depth)],
        seq=[m.ca_reg(f"hist_seq{i}", domain=domain, width=seq_w, init=0) for i in range(hist_depth)],
        data=[m.ca_reg(f"hist_d{i}", domain=domain, width=128, init=0) for i in range(hist_depth)],
    )

    # Shadow counts for backpressure
    shadow_cnt = [m.ca_reg(f"lane{lane}__iq_cnt", domain=domain, width=16, init=0) for lane in range(4)]

    # INPUT: accept packets
    bkpr = bkpr_r.out()
    accept = ~bkpr

    eff_v = [pkt_in_vld[i] & accept for i in range(4)]
    zero_seq = m.ca_const(0, width=seq_w)
    one_seq = m.ca_const(1, width=seq_w)
    inc = [mux(eff_v[i], one_seq, zero_seq) for i in range(4)]

    base = seq_alloc.out()
    seq_lane = [base, base + inc[0], base + inc[0] + inc[1], base + inc[0] + inc[1] + inc[2]]
    total_inc = inc[0] + inc[1] + inc[2] + inc[3]
    seq_alloc.set(base + total_inc)

    # Map packets to issue queues
    push_v = [m.ca_const(0, width=1) for _ in range(4)]
    push_d = [m.ca_const(0, width=bundle_w) for _ in range(4)]

    for i in range(4):
        seq_i = seq_lane[i]
        lane_i = seq_i[0:2]
        bundle_i = ca_cat(seq_i, pkt_in_data[i], pkt_in_ctrl[i])
        for lane in range(4):
            hit = eff_v[i] & lane_i.eq(lane)
            push_v[lane] = push_v[lane] | hit
            push_d[lane] = mux(hit, bundle_i, push_d[lane])

    push_fire = []
    for lane in range(4):
        push_fire.append(issue_q[lane].push(push_d[lane], when=push_v[lane]))

    # DISPATCH: issue queues -> FE
    fwd_vld = [m.ca_const(0, width=1) for _ in range(total_eng)]
    fwd_data = [m.ca_const(0, width=128) for _ in range(total_eng)]
    fwd_lat = [m.ca_const(0, width=2) for _ in range(total_eng)]
    fwd_dp_vld = [m.ca_const(0, width=1) for _ in range(total_eng)]
    fwd_dp_data = [m.ca_const(0, width=128) for _ in range(total_eng)]

    lane_pop = [m.ca_const(0, width=1) for _ in range(4)]
    dispatch_fire_eng = [m.ca_const(0, width=1) for _ in range(total_eng)]
    dispatch_slot_eng = [m.ca_const(0, width=wheel_bits) for _ in range(total_eng)]
    dispatch_seq_eng = [m.ca_const(0, width=seq_w) for _ in range(total_eng)]

    # Dependency lookup helper
    def dep_lookup(dep_seq: CycleAwareSignal) -> tuple[CycleAwareSignal, CycleAwareSignal]:
        found_h = m.ca_const(0, width=1)
        data_h = m.ca_const(0, width=128)
        for idx in range(hist_depth):
            match = hist.valid[idx].out() & hist.seq[idx].out().eq(dep_seq)
            found_h = found_h | match
            data_h = mux(match, hist.data[idx].out(), data_h)

        dep_lane = dep_seq[0:2]
        found_r = m.ca_const(0, width=1)
        data_r = m.ca_const(0, width=128)
        for lane in range(4):
            is_lane = dep_lane.eq(lane)
            delta = (dep_seq - exp_seq[lane].out()) >> 2
            for s in range(rob_depth):
                match = is_lane & robs[lane].valid[s].out() & delta.eq(s)
                found_r = found_r | match
                data_r = mux(match, robs[lane].data[s].out(), data_r)

        found = found_r | found_h
        data = mux(found_r, data_r, data_h)
        return found, data

    for lane in range(4):
        qv = issue_q[lane].out_valid
        qb = issue_q[lane].out_data

        ctrl = qb[0:5]
        data_i = qb[5:133]
        seq_i = qb[133:133 + seq_w]

        lat = ctrl[0:2]
        dp = ctrl[2:5]

        dp_present = dp.ne(0)
        dep_seq = seq_i - dp.zext(width=seq_w)
        dep_found, dep_data = dep_lookup(dep_seq)
        dep_ok = (~dp_present) | dep_found

        # Completion slot
        lat3 = lat.zext(width=wheel_bits)
        slot = cycle_mod + lat3 + m.ca_const(2, width=wheel_bits)

        # Pick an engine
        eng_base = lane * eng_per_lane
        chosen = [m.ca_const(0, width=1) for _ in range(eng_per_lane)]
        any_free = m.ca_const(0, width=1)
        for k in range(eng_per_lane):
            e = eng_base + k
            busy = _wheel_slot_busy(m, wheels[e].valid, idx=slot)
            free = ~busy
            take = free & ~any_free
            any_free = any_free | free
            chosen[k] = take

        dispatch_ok = dep_ok & any_free
        pop = issue_q[lane].pop(when=dispatch_ok)
        lane_pop[lane] = pop.fire

        for k in range(eng_per_lane):
            e = eng_base + k
            fire_e = pop.fire & chosen[k]
            dispatch_fire_eng[e] = fire_e
            dispatch_slot_eng[e] = mux(fire_e, slot, dispatch_slot_eng[e])
            dispatch_seq_eng[e] = mux(fire_e, seq_i, dispatch_seq_eng[e])

            fwd_vld[e] = mux(fire_e, m.ca_const(1, width=1), fwd_vld[e])
            fwd_data[e] = mux(fire_e, data_i, fwd_data[e])
            fwd_lat[e] = mux(fire_e, lat, fwd_lat[e])
            fwd_dp_vld[e] = mux(fire_e, dp_present, fwd_dp_vld[e])
            fwd_dp_data[e] = mux(fire_e, dep_data, fwd_dp_data[e])

    # COMPLETE: FE -> completion queues
    for e in range(total_eng):
        wv_cur, ws_cur = _wheel_read(m, wheels[e].valid, wheels[e].seq, idx=cycle_mod, zero_seq=zero_seq)
        comp_v = wv_cur & fwded_vld[e]
        comp_bus = ca_cat(ws_cur, fwded_data[e])
        fire = comp_q[e].push(comp_bus, when=comp_v)

        _wheel_update(
            m,
            wheels[e].valid,
            wheels[e].seq,
            set_en=dispatch_fire_eng[e],
            set_idx=dispatch_slot_eng[e],
            set_seq=dispatch_seq_eng[e],
            clear_en=fire,
            clear_idx=cycle_mod,
        )

    # MERGE: completion queues -> ROBs
    ins_fire_lane = [m.ca_const(0, width=1) for _ in range(4)]
    ins_seq_lane = [m.ca_const(0, width=seq_w) for _ in range(4)]
    ins_data_lane = [m.ca_const(0, width=128) for _ in range(4)]

    for lane in range(4):
        eng_base = lane * eng_per_lane
        take = [m.ca_const(0, width=1) for _ in range(eng_per_lane)]
        any_take = m.ca_const(0, width=1)
        sel_bus = m.ca_const(0, width=comp_w)

        for k in range(eng_per_lane):
            e = eng_base + k
            vb = comp_q[e].out_valid
            db = comp_q[e].out_data
            seq_c = db[128:128 + seq_w]
            delta = (seq_c - exp_seq[lane].out()) >> 2
            in_range = delta.lt(rob_depth)
            cand = vb & in_range & ~any_take
            take[k] = cand
            any_take = any_take | cand
            sel_bus = mux(cand, db, sel_bus)

        for k in range(eng_per_lane):
            e = eng_base + k
            comp_q[e].pop(when=take[k])

        ins_fire_lane[lane] = any_take
        ins_seq_lane[lane] = sel_bus[128:128 + seq_w]
        ins_data_lane[lane] = sel_bus[0:128]

    # COMMIT: ROB -> output
    start = commit_lane.out()
    lane_ready = [robs[l].valid[0].out() for l in range(4)]
    lane_data0 = [robs[l].data[0].out() for l in range(4)]

    def prefix(start_lane: int) -> tuple[list[CycleAwareSignal], CycleAwareSignal]:
        v = []
        ok = m.ca_const(1, width=1)
        for k in range(4):
            lane = (start_lane + k) & 3
            ok = ok & lane_ready[lane]
            v.append(ok)
        cnt = mux(v[0], m.ca_const(1, width=3), m.ca_const(0, width=3))
        cnt = mux(v[1], m.ca_const(2, width=3), cnt)
        cnt = mux(v[2], m.ca_const(3, width=3), cnt)
        cnt = mux(v[3], m.ca_const(4, width=3), cnt)
        return v, cnt

    v0, c0 = prefix(0)
    v1, c1 = prefix(1)
    v2, c2 = prefix(2)
    v3, c3 = prefix(3)

    is0 = start.eq(0)
    is1 = start.eq(1)
    is2 = start.eq(2)
    is3 = start.eq(3)

    commit_cnt = mux(is3, c3, mux(is2, c2, mux(is1, c1, c0)))

    out_v = [m.ca_const(0, width=1) for _ in range(4)]
    out_d = [m.ca_const(0, width=128) for _ in range(4)]

    commit_v_s = [v0, v1, v2, v3]
    for phys in range(4):
        vv = m.ca_const(0, width=1)
        dd = m.ca_const(0, width=128)
        for s in range(4):
            for k in range(4):
                lane = (s + k) & 3
                if lane != phys:
                    continue
                hit = commit_v_s[s][k]
                cond = start.eq(s) & hit
                vv = mux(cond, m.ca_const(1, width=1), vv)
                dd = mux(cond, lane_data0[phys], dd)
        out_v[phys] = vv
        out_d[phys] = dd

    for i in range(4):
        out_vld_r[i].set(out_v[i])
        out_data_r[i].set(mux(out_v[i], out_d[i], m.ca_const(0, width=128)))

    commit_pop = out_v
    next_lane = (start + commit_cnt[0:2])[0:2]
    commit_lane.set(next_lane)

    for lane in range(4):
        inc4 = mux(commit_pop[lane], m.ca_const(4, width=seq_w), m.ca_const(0, width=seq_w))
        exp_seq[lane].set(exp_seq[lane].out() + inc4)

    for lane in range(4):
        _rob_update(
            m,
            robs[lane],
            exp_seq=exp_seq[lane].out(),
            commit_pop=commit_pop[lane],
            ins_fire=ins_fire_lane[lane],
            ins_seq=ins_seq_lane[lane],
            ins_data=ins_data_lane[lane],
            depth=rob_depth,
        )

    # Build commit seqs/datas for history
    commit_seq_slots = [m.ca_const(0, width=seq_w) for _ in range(4)]
    commit_data_slots = [m.ca_const(0, width=128) for _ in range(4)]
    for k in range(4):
        lane_k = (start + m.ca_const(k, width=2))[0:2]
        s = m.ca_const(0, width=seq_w)
        d = m.ca_const(0, width=128)
        for lane in range(4):
            is_lane = lane_k.eq(lane)
            s = mux(is_lane, exp_seq[lane].out(), s)
            d = mux(is_lane, lane_data0[lane], d)
        commit_seq_slots[k] = s
        commit_data_slots[k] = d

    _hist_shift_insert(m, hist, k=commit_cnt, seqs=commit_seq_slots, datas=commit_data_slots)

    # BACKPRESSURE
    bkpr_next = m.ca_const(0, width=1)
    for lane in range(4):
        push_i = push_fire[lane].zext(width=16)
        pop_i = lane_pop[lane].zext(width=16)
        cnt_next = shadow_cnt[lane].out() + push_i - pop_i
        shadow_cnt[lane].set(cnt_next)
        near_full = cnt_next.ge(lane_q_depth - 2)
        bkpr_next = bkpr_next | near_full
    bkpr_r.set(bkpr_next)

    # OUTPUTS
    m.output("pkt_in_bkpr", bkpr_r.out().sig)
    for i in range(4):
        m.output(f"lane{i}_pkt_out_vld", out_vld_r[i].out().sig)
        m.output(f"lane{i}_pkt_out_data", out_data_r[i].out().sig)

    for e in range(total_eng):
        m.output(f"fwd{e}_pkt_data_vld", fwd_vld[e].sig)
        m.output(f"fwd{e}_pkt_data", fwd_data[e].sig)
        m.output(f"fwd{e}_pkt_lat", fwd_lat[e].sig)
        m.output(f"fwd{e}_pkt_dp_vld", fwd_dp_vld[e].sig)
        m.output(f"fwd{e}_pkt_dp_data", fwd_dp_data[e].sig)


def fastfwd_pyc(m: CycleAwareCircuit, domain: CycleAwareDomain) -> None:
    """FastFwd with default parameters."""
    _build_fastfwd_impl(
        m, domain,
        eng_per_lane=2,
        lane_q_depth=32,
        eng_q_depth=8,
        rob_depth=16,
        seq_w=16,
        wheel=8,
        hist_depth=8,
    )


if __name__ == "__main__":
    circuit = compile_cycle_aware(fastfwd_pyc, name="fastfwd_pyc")
    print(circuit.emit_mlir())
