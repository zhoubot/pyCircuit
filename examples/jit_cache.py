# -*- coding: utf-8 -*-
"""JIT Cache example using Cycle-Aware API.

A simplified set-associative cache demonstrating:
- CycleAwareQueue for request/response queuing
- CycleAwareByteMem for backing memory
- CycleAwareReg for cache state (valid/tag/data)
- Conditional register updates with priority mux
"""
from __future__ import annotations

from pycircuit import (
    CycleAwareCircuit,
    CycleAwareDomain,
    CycleAwareReg,
    CycleAwareSignal,
    compile_cycle_aware,
    mux,
)


def clog2(x: int) -> int:
    """Ceil(log2(x)) for x>0, with a minimum width of 1."""
    if x <= 0:
        raise ValueError("clog2 expects x > 0")
    return max(1, (int(x) - 1).bit_length())


def bytes_of(bit_width: int) -> int:
    if bit_width <= 0 or (bit_width % 8) != 0:
        raise ValueError("bit_width must be a positive multiple of 8")
    return int(bit_width) // 8


def tag_width(addr_width: int, off_bits: int, set_bits: int) -> int:
    tag = int(addr_width) - int(off_bits) - int(set_bits)
    if tag <= 0:
        raise ValueError("invalid address split")
    return tag


def _mk_2d_regs(
    m: CycleAwareCircuit,
    name: str,
    domain: CycleAwareDomain,
    *,
    sets: int,
    ways: int,
    width: int,
    init: int = 0,
) -> list[list[CycleAwareReg]]:
    """Create a 2D array of registers."""
    out: list[list[CycleAwareReg]] = []
    for s in range(int(sets)):
        row: list[CycleAwareReg] = []
        for w in range(int(ways)):
            row.append(m.ca_reg(f"{name}_s{s}_w{w}", domain=domain, width=width, init=init))
        out.append(row)
    return out


def _mk_1d_regs(
    m: CycleAwareCircuit,
    name: str,
    domain: CycleAwareDomain,
    *,
    n: int,
    width: int,
    init: int = 0,
) -> list[CycleAwareReg]:
    """Create a 1D array of registers."""
    out: list[CycleAwareReg] = []
    for i in range(int(n)):
        out.append(m.ca_reg(f"{name}_s{i}", domain=domain, width=width, init=init))
    return out


def _mux_by_index(
    m: CycleAwareCircuit,
    idx: CycleAwareSignal,
    options: list[CycleAwareSignal],
    *,
    default: CycleAwareSignal,
) -> CycleAwareSignal:
    """Select from options by index."""
    if not options:
        raise ValueError("mux_by_index requires a non-empty options list")
    out = default
    for i, v in enumerate(options):
        cond = idx.eq(i)
        out = mux(cond, v, out)
    return out


def _select_set_way(
    m: CycleAwareCircuit,
    set_idx: CycleAwareSignal,
    table: list[list[CycleAwareReg]],
    *,
    way: int,
) -> CycleAwareSignal:
    """Select a specific way's value from all sets."""
    opts = [table[s][int(way)].out() for s in range(len(table))]
    zero = m.ca_const(0, width=opts[0].width)
    return _mux_by_index(m, set_idx, opts, default=zero)


def _cache_lookup(
    m: CycleAwareCircuit,
    set_idx: CycleAwareSignal,
    tag: CycleAwareSignal,
    *,
    valids: list[list[CycleAwareReg]],
    tags: list[list[CycleAwareReg]],
    datas: list[list[CycleAwareReg]],
) -> tuple[CycleAwareSignal, CycleAwareSignal]:
    """Lookup cache: returns (hit, rdata)."""
    ways = len(valids[0])
    
    hit = m.ca_const(0, width=1)
    rdata = m.ca_const(0, width=datas[0][0].width)
    
    for w in range(ways):
        v = _select_set_way(m, set_idx, valids, way=w)
        t = _select_set_way(m, set_idx, tags, way=w)
        d = _select_set_way(m, set_idx, datas, way=w)
        
        hw = v & t.eq(tag)
        hit = hit | hw
        rdata = mux(hw, d, rdata)
    
    return hit, rdata


def _cache_fill_on_miss(
    m: CycleAwareCircuit,
    set_idx: CycleAwareSignal,
    repl_way: CycleAwareSignal,
    *,
    miss: CycleAwareSignal,
    tag: CycleAwareSignal,
    fill_data: CycleAwareSignal,
    valids: list[list[CycleAwareReg]],
    tags: list[list[CycleAwareReg]],
    datas: list[list[CycleAwareReg]],
    rr_ptrs: list[CycleAwareReg],
) -> None:
    """Fill cache on miss."""
    sets = len(valids)
    ways = len(valids[0])
    
    one = m.ca_const(1, width=rr_ptrs[0].width)
    zero_way = m.ca_const(0, width=repl_way.width)
    
    for s in range(sets):
        set_sel = set_idx.eq(s)
        do_set = miss & set_sel
        
        # Round-robin pointer update for this set
        rr = rr_ptrs[s]
        rr_inc = rr.out() + one
        if ways > 1:
            last = rr.out().eq(ways - 1)
            rr_next = mux(last, zero_way, rr_inc)
        else:
            rr_next = zero_way
        rr.set(rr_next, when=do_set)
        
        # Fill the selected way
        for w in range(ways):
            way_sel = repl_way.eq(w) if ways > 1 else one_i1
            do = do_set & way_sel
            valids[s][w].set(1, when=do)
            tags[s][w].set(tag, when=do)
            datas[s][w].set(fill_data, when=do)


def _jit_cache_impl(
    m: CycleAwareCircuit,
    domain: CycleAwareDomain,
    SETS: int,
    WAYS: int,
) -> None:
    """A tiny set-associative cache.
    
    - Request in via a ready/valid queue.
    - Response out via a ready/valid queue.
    - Miss fill from a byte-addressed backing memory.
    """
    ADDR_W = 32
    DATA_W = 32
    DATA_BYTES = bytes_of(DATA_W)
    
    set_bits = clog2(SETS)
    off_bits = clog2(DATA_BYTES)
    tag_bits = tag_width(ADDR_W, off_bits, set_bits)
    
    # Input signals
    req_valid = domain.create_signal("req_valid", width=1)
    req_addr = domain.create_signal("req_addr", width=ADDR_W)
    rsp_ready = domain.create_signal("rsp_ready", width=1)
    
    # Backing memory (byte addressed)
    wvalid0 = m.ca_const(0, width=1)
    waddr0 = m.ca_const(0, width=ADDR_W)
    wdata0 = m.ca_const(0, width=DATA_W)
    wstrb0 = m.ca_const(0, width=DATA_BYTES)
    
    # Request/response queues
    req_q = m.ca_queue("req_q", domain=domain, width=ADDR_W, depth=2)
    req_q.push(req_addr, when=req_valid)
    
    rsp_q = m.ca_queue("rsp_q", domain=domain, width=1 + DATA_W, depth=2)
    rsp_pop = rsp_q.pop(when=rsp_ready)
    rsp_hit = rsp_pop.data[DATA_W]
    rsp_rdata = rsp_pop.data[0:DATA_W]
    
    # Cache state
    valids = _mk_2d_regs(m, "valid", domain, sets=SETS, ways=WAYS, width=1, init=0)
    tags = _mk_2d_regs(m, "tag", domain, sets=SETS, ways=WAYS, width=tag_bits, init=0)
    datas = _mk_2d_regs(m, "data", domain, sets=SETS, ways=WAYS, width=DATA_W, init=0)
    rr_ptrs = _mk_1d_regs(m, "rr", domain, n=SETS, width=clog2(WAYS), init=0)
    
    # Pop request when we can enqueue response
    req_pop = req_q.pop(when=rsp_q.in_ready)
    req_fire = req_pop.fire
    
    addr = req_pop.data
    set_idx = addr[off_bits : off_bits + set_bits]
    tag = addr[off_bits + set_bits : ADDR_W]
    
    # Lookup
    hit, hit_data = _cache_lookup(m, set_idx, tag, valids=valids, tags=tags, datas=datas)
    
    # Miss fill: read from backing memory
    mem = m.ca_byte_mem("main_mem", domain=domain, depth=4096, data_width=DATA_W)
    mem_rdata = mem.read(addr)
    mem.write(waddr0, wdata0, wstrb0, when=wvalid0)
    
    miss = req_fire & ~hit
    rr_outs = [r.out() for r in rr_ptrs]
    repl_way = _mux_by_index(m, set_idx, rr_outs, default=m.ca_const(0, width=rr_ptrs[0].width))
    
    _cache_fill_on_miss(
        m,
        set_idx,
        repl_way,
        miss=miss,
        tag=tag,
        fill_data=mem_rdata,
        valids=valids,
        tags=tags,
        datas=datas,
        rr_ptrs=rr_ptrs,
    )
    
    rdata = mux(hit, hit_data, mem_rdata)
    rsp_pkt = m.cat_signals(hit, rdata)
    rsp_q.push(rsp_pkt, when=req_fire)
    
    # Outputs
    m.output("req_ready", req_q.in_ready.sig)
    m.output("rsp_valid", rsp_pop.valid.sig)
    m.output("rsp_hit", rsp_hit.sig)
    m.output("rsp_rdata", rsp_rdata.sig)


def jit_cache(m: CycleAwareCircuit, domain: CycleAwareDomain) -> None:
    """Wrapper with default cache parameters."""
    _jit_cache_impl(m, domain, SETS=8, WAYS=2)


if __name__ == "__main__":
    circuit = compile_cycle_aware(jit_cache, name="jit_cache")
    print(circuit.emit_mlir())
