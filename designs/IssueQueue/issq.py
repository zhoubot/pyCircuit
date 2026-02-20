from __future__ import annotations

import sys
from pathlib import Path

from pycircuit import Circuit, compile, module, u

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from issq_config import (  # noqa: E402
    _alloc_field,
    _count_ones,
    _derive_cfg,
    _entry_spec,
    _lane_lt,
    _not1,
    _onehot_mux,
    _ready_lookup,
    _slot_select,
    _uop_spec,
    _wake_hit,
)

@module
def build(
    m: Circuit,
    *,
    entries: int = 16,
    ptag_count: int = 64,
    payload_width: int = 32,
    enq_ports: int = 2,
    issue_ports: int = 2,
    init_ready_mask: int = 0,
):
    cfg = _derive_cfg(
        m,
        entries=entries,
        ptag_count=ptag_count,
        payload_width=payload_width,
        enq_ports=enq_ports,
        issue_ports=issue_ports,
        init_ready_mask=init_ready_mask,
    )

    e = int(cfg.entries)
    p = int(cfg.ptag_count)
    ptag_w = int(cfg.ptag_width)
    payload_w = int(cfg.payload_width)
    n_enq = int(cfg.enq_ports)
    n_issue = int(cfg.issue_ports)
    occ_w = int(cfg.occupancy_width)
    issue_cnt_w = int(cfg.issue_count_width)
    issued_total_w = int(cfg.issued_total_width)

    clk = m.clock("clk")
    rst = m.reset("rst")

    uop_spec = _uop_spec(m, cfg)
    entry_spec = _entry_spec(m, cfg)

    enq_valid = [m.input(f"enq{k}_valid", width=1) for k in range(n_enq)]
    enq_uops = [m.inputs(uop_spec, prefix=f"enq{k}_") for k in range(n_enq)]

    entry_state = [
        m.state(entry_spec, clk=clk, rst=rst, prefix=f"ent{i}_", init=0)
        for i in range(e)
    ]

    age_state = [
        [m.out(f"age_{i}_{j}", clk=clk, rst=rst, width=1, init=u(1, 0)) for j in range(e)]
        for i in range(e)
    ]

    ready_state = [
        m.out(
            f"ready_ptag_{t}",
            clk=clk,
            rst=rst,
            width=1,
            init=u(1, (int(cfg.init_ready_mask) >> t) & 1),
        )
        for t in range(p)
    ]

    issued_total_q = m.out("issued_total_q", clk=clk, rst=rst, width=issued_total_w, init=u(issued_total_w, 0))

    cur = []
    for i in range(e):
        s = entry_state[i]
        cur.append(
            {
                "valid": s["valid"].read(),
                "src0_valid": s["uop.src0.valid"].read(),
                "src0_ptag": s["uop.src0.ptag"].read(),
                "src0_ready": s["uop.src0.ready"].read(),
                "src1_valid": s["uop.src1.valid"].read(),
                "src1_ptag": s["uop.src1.ptag"].read(),
                "src1_ready": s["uop.src1.ready"].read(),
                "dst_valid": s["uop.dst.valid"].read(),
                "dst_ptag": s["uop.dst.ptag"].read(),
                "dst_ready": s["uop.dst.ready"].read(),
                "payload": s["uop.payload"].read(),
            }
        )

    entry_ready = [cur[i]["valid"] & cur[i]["src0_ready"] & cur[i]["src1_ready"] for i in range(e)]

    issue_sel = [[u(1, 0) for _ in range(e)] for _ in range(n_issue)]
    issue_valid = [u(1, 0) for _ in range(n_issue)]

    remaining = list(entry_ready)
    for k in range(n_issue):
        oldest = []
        for i in range(e):
            older_exists = u(1, 0)
            for j in range(e):
                older_exists = older_exists | (remaining[j] & age_state[j][i].out())
            oldest_i = remaining[i] & _not1(m, older_exists)
            oldest.append(oldest_i)
        issue_sel[k] = oldest
        v = u(1, 0)
        for i in range(e):
            v = v | oldest[i]
        issue_valid[k] = v
        remaining = [remaining[i] & _not1(m, oldest[i]) for i in range(e)]

    issue_win = [u(1, 0) for _ in range(e)]
    for i in range(e):
        w = u(1, 0)
        for k in range(n_issue):
            w = w | issue_sel[k][i]
        issue_win[i] = w

    keep_valid = [cur[i]["valid"] & _not1(m, issue_win[i]) for i in range(e)]
    free_slot = [_not1(m, keep_valid[i]) for i in range(e)]

    alloc_lane = [[u(1, 0) for _ in range(e)] for _ in range(n_enq)]
    enq_ready = [u(1, 0) for _ in range(n_enq)]

    free_avail = list(free_slot)
    for k in range(n_enq):
        any_free = u(1, 0)
        for i in range(e):
            any_free = any_free | free_avail[i]
        enq_ready[k] = any_free

        first = [u(1, 0) for _ in range(e)]
        lower_seen = u(1, 0)
        for i in range(e):
            first_i = free_avail[i] & _not1(m, lower_seen)
            first[i] = first_i
            lower_seen = lower_seen | free_avail[i]

        accept_k = enq_valid[k] & any_free
        for i in range(e):
            alloc_lane[k][i] = first[i] & accept_k
            free_avail[i] = free_avail[i] & _not1(m, alloc_lane[k][i])

    new_alloc = [u(1, 0) for _ in range(e)]
    for i in range(e):
        n = u(1, 0)
        for k in range(n_enq):
            n = n | alloc_lane[k][i]
        new_alloc[i] = n

    next_valid = [keep_valid[i] | new_alloc[i] for i in range(e)]

    field_lists = {
        "src0.valid": [cur[i]["src0_valid"] for i in range(e)],
        "src0.ptag": [cur[i]["src0_ptag"] for i in range(e)],
        "src0.ready": [cur[i]["src0_ready"] for i in range(e)],
        "src1.valid": [cur[i]["src1_valid"] for i in range(e)],
        "src1.ptag": [cur[i]["src1_ptag"] for i in range(e)],
        "src1.ready": [cur[i]["src1_ready"] for i in range(e)],
        "dst.valid": [cur[i]["dst_valid"] for i in range(e)],
        "dst.ptag": [cur[i]["dst_ptag"] for i in range(e)],
        "dst.ready": [cur[i]["dst_ready"] for i in range(e)],
        "payload": [cur[i]["payload"] for i in range(e)],
    }

    issue_uops = []
    for k in range(n_issue):
        uop_vals = {
            "src0.valid": _onehot_mux(m, issue_sel[k], field_lists["src0.valid"], 1),
            "src0.ptag": _onehot_mux(m, issue_sel[k], field_lists["src0.ptag"], ptag_w),
            "src0.ready": _onehot_mux(m, issue_sel[k], field_lists["src0.ready"], 1),
            "src1.valid": _onehot_mux(m, issue_sel[k], field_lists["src1.valid"], 1),
            "src1.ptag": _onehot_mux(m, issue_sel[k], field_lists["src1.ptag"], ptag_w),
            "src1.ready": _onehot_mux(m, issue_sel[k], field_lists["src1.ready"], 1),
            "dst.valid": _onehot_mux(m, issue_sel[k], field_lists["dst.valid"], 1),
            "dst.ptag": _onehot_mux(m, issue_sel[k], field_lists["dst.ptag"], ptag_w),
            "dst.ready": _onehot_mux(m, issue_sel[k], field_lists["dst.ready"], 1),
            "payload": _onehot_mux(m, issue_sel[k], field_lists["payload"], payload_w),
        }
        issue_uops.append(uop_vals)
        m.output(f"iss{k}_valid", issue_valid[k])
        m.outputs(uop_spec, uop_vals, prefix=f"iss{k}_")

    wake_valid = [issue_valid[k] & issue_uops[k]["dst.valid"] for k in range(n_issue)]
    wake_ptag = [issue_uops[k]["dst.ptag"] for k in range(n_issue)]

    for i in range(e):
        new_src0_valid = _alloc_field(m, enq_uops, alloc_lane, i, "src0.valid", 1, n_enq)
        new_src0_ptag = _alloc_field(m, enq_uops, alloc_lane, i, "src0.ptag", ptag_w, n_enq)
        new_src0_ready_in = _alloc_field(m, enq_uops, alloc_lane, i, "src0.ready", 1, n_enq)

        new_src1_valid = _alloc_field(m, enq_uops, alloc_lane, i, "src1.valid", 1, n_enq)
        new_src1_ptag = _alloc_field(m, enq_uops, alloc_lane, i, "src1.ptag", ptag_w, n_enq)
        new_src1_ready_in = _alloc_field(m, enq_uops, alloc_lane, i, "src1.ready", 1, n_enq)

        new_dst_valid = _alloc_field(m, enq_uops, alloc_lane, i, "dst.valid", 1, n_enq)
        new_dst_ptag = _alloc_field(m, enq_uops, alloc_lane, i, "dst.ptag", ptag_w, n_enq)
        new_dst_ready = _alloc_field(m, enq_uops, alloc_lane, i, "dst.ready", 1, n_enq)
        new_payload = _alloc_field(m, enq_uops, alloc_lane, i, "payload", payload_w, n_enq)

        keep_src0_ready = (
            cur[i]["src0_ready"]
            | _not1(m, cur[i]["src0_valid"])
            | _ready_lookup(m, ready_state, cur[i]["src0_ptag"], ptag_w, p)
            | (cur[i]["src0_valid"] & _wake_hit(m, wake_valid, wake_ptag, cur[i]["src0_ptag"], n_issue))
        )
        keep_src1_ready = (
            cur[i]["src1_ready"]
            | _not1(m, cur[i]["src1_valid"])
            | _ready_lookup(m, ready_state, cur[i]["src1_ptag"], ptag_w, p)
            | (cur[i]["src1_valid"] & _wake_hit(m, wake_valid, wake_ptag, cur[i]["src1_ptag"], n_issue))
        )

        new_src0_ready = (
            _not1(m, new_src0_valid)
            | new_src0_ready_in
            | _ready_lookup(m, ready_state, new_src0_ptag, ptag_w, p)
            | (new_src0_valid & _wake_hit(m, wake_valid, wake_ptag, new_src0_ptag, n_issue))
        )
        new_src1_ready = (
            _not1(m, new_src1_valid)
            | new_src1_ready_in
            | _ready_lookup(m, ready_state, new_src1_ptag, ptag_w, p)
            | (new_src1_valid & _wake_hit(m, wake_valid, wake_ptag, new_src1_ptag, n_issue))
        )

        st = entry_state[i]
        st["valid"].set(next_valid[i])
        st["uop.src0.valid"].set(_slot_select(m, keep_valid[i], new_alloc[i], cur[i]["src0_valid"], new_src0_valid, 1))
        st["uop.src0.ptag"].set(_slot_select(m, keep_valid[i], new_alloc[i], cur[i]["src0_ptag"], new_src0_ptag, ptag_w))
        st["uop.src0.ready"].set(_slot_select(m, keep_valid[i], new_alloc[i], keep_src0_ready, new_src0_ready, 1))

        st["uop.src1.valid"].set(_slot_select(m, keep_valid[i], new_alloc[i], cur[i]["src1_valid"], new_src1_valid, 1))
        st["uop.src1.ptag"].set(_slot_select(m, keep_valid[i], new_alloc[i], cur[i]["src1_ptag"], new_src1_ptag, ptag_w))
        st["uop.src1.ready"].set(_slot_select(m, keep_valid[i], new_alloc[i], keep_src1_ready, new_src1_ready, 1))

        st["uop.dst.valid"].set(_slot_select(m, keep_valid[i], new_alloc[i], cur[i]["dst_valid"], new_dst_valid, 1))
        st["uop.dst.ptag"].set(_slot_select(m, keep_valid[i], new_alloc[i], cur[i]["dst_ptag"], new_dst_ptag, ptag_w))
        st["uop.dst.ready"].set(_slot_select(m, keep_valid[i], new_alloc[i], cur[i]["dst_ready"], new_dst_ready, 1))

        st["uop.payload"].set(_slot_select(m, keep_valid[i], new_alloc[i], cur[i]["payload"], new_payload, payload_w))

    for i in range(e):
        for j in range(e):
            if i == j:
                age_state[i][j].set(u(1, 0))
            else:
                keep_keep = keep_valid[i] & keep_valid[j] & age_state[i][j].out()
                keep_new = keep_valid[i] & new_alloc[j]
                new_new = new_alloc[i] & new_alloc[j] & _lane_lt(m, alloc_lane, i, j, n_enq)
                rel = keep_keep | keep_new | new_new
                age_state[i][j].set(next_valid[i] & next_valid[j] & rel)

    for t in range(p):
        wake_t = u(1, 0)
        t_const = u(ptag_w, t)
        for k in range(n_issue):
            wake_t = wake_t | (wake_valid[k] & (wake_ptag[k] == t_const))
        ready_state[t].set(ready_state[t].out() | wake_t)

    occupancy = _count_ones(m, [cur[i]["valid"] for i in range(e)], occ_w)
    issued_this = _count_ones(m, issue_valid, issue_cnt_w)
    issued_total_q.set((issued_total_q.out() + issued_this)[0:issued_total_w])

    for k in range(n_enq):
        m.output(f"enq{k}_ready", enq_ready[k])

    m.output("occupancy", occupancy)
    m.output("issued_total", issued_total_q.out())


if __name__ == "__main__":
    print(
        compile(
            build,
            name="issq",
            entries=16,
            ptag_count=64,
            payload_width=32,
            enq_ports=2,
            issue_ports=2,
            init_ready_mask=0,
        ).emit_mlir()
    )
