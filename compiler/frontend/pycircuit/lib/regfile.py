from __future__ import annotations

from ..connectors import Connector, ConnectorBundle, ConnectorError
from ..design import module
from ..dsl import Signal
from ..hw import Circuit
from ..literals import u


@module(structural=True)
def RegFile(
    m: Circuit,
    clk: Connector,
    rst: Connector,
    raddr_bus: Connector,
    wen_bus: Connector,
    waddr_bus: Connector,
    wdata_bus: Connector,
    *,
    ptag_count: int = 256,
    const_count: int = 128,
    nr: int = 10,
    nw: int = 5,
) -> ConnectorBundle:
    ptag_n = int(ptag_count)
    const_n = int(const_count)
    nr_n = int(nr)
    nw_n = int(nw)
    if ptag_n <= 0:
        raise ValueError("RegFile ptag_count must be > 0")
    if const_n < 0 or const_n > ptag_n:
        raise ValueError("RegFile const_count must satisfy 0 <= const_count <= ptag_count")
    if nr_n <= 0:
        raise ValueError("RegFile nr must be > 0")
    if nw_n <= 0:
        raise ValueError("RegFile nw must be > 0")
    ptag_w = max(1, (ptag_n - 1).bit_length())

    clk_v = clk.read() if isinstance(clk, Connector) else clk
    rst_v = rst.read() if isinstance(rst, Connector) else rst
    if not isinstance(clk_v, Signal) or clk_v.ty != "!pyc.clock":
        raise ConnectorError("RegFile.clk must be !pyc.clock")
    if not isinstance(rst_v, Signal) or rst_v.ty != "!pyc.reset":
        raise ConnectorError("RegFile.rst must be !pyc.reset")

    raddr_bus_v = raddr_bus.read() if isinstance(raddr_bus, Connector) else raddr_bus
    if isinstance(raddr_bus_v, Signal):
        raddr_bus_w = m.wire(raddr_bus_v)
    else:
        raddr_bus_w = raddr_bus_v

    wen_bus_v = wen_bus.read() if isinstance(wen_bus, Connector) else wen_bus
    if isinstance(wen_bus_v, Signal):
        wen_bus_w = m.wire(wen_bus_v)
    else:
        wen_bus_w = wen_bus_v

    waddr_bus_v = waddr_bus.read() if isinstance(waddr_bus, Connector) else waddr_bus
    if isinstance(waddr_bus_v, Signal):
        waddr_bus_w = m.wire(waddr_bus_v)
    else:
        waddr_bus_w = waddr_bus_v

    wdata_bus_v = wdata_bus.read() if isinstance(wdata_bus, Connector) else wdata_bus
    if isinstance(wdata_bus_v, Signal):
        wdata_bus_w = m.wire(wdata_bus_v)
    else:
        wdata_bus_w = wdata_bus_v

    exp_raddr_w = nr_n * ptag_w
    exp_wen_w = nw_n
    exp_waddr_w = nw_n * ptag_w
    exp_wdata_w = nw_n * 64

    if raddr_bus_w.width != exp_raddr_w:
        raise ConnectorError(f"RegFile.raddr_bus must be i{exp_raddr_w}")
    if wen_bus_w.width != exp_wen_w:
        raise ConnectorError(f"RegFile.wen_bus must be i{exp_wen_w}")
    if waddr_bus_w.width != exp_waddr_w:
        raise ConnectorError(f"RegFile.waddr_bus must be i{exp_waddr_w}")
    if wdata_bus_w.width != exp_wdata_w:
        raise ConnectorError(f"RegFile.wdata_bus must be i{exp_wdata_w}")

    storage_depth = ptag_n - const_n
    bank0 = [m.out(f"rf_bank0_{i}", clk=clk_v, rst=rst_v, width=32, init=u(32, 0)) for i in range(storage_depth)]
    bank1 = [m.out(f"rf_bank1_{i}", clk=clk_v, rst=rst_v, width=32, init=u(32, 0)) for i in range(storage_depth)]

    raddr_lanes = [raddr_bus_w[i * ptag_w : (i + 1) * ptag_w] for i in range(nr_n)]
    wen_lanes = [wen_bus_w[i] for i in range(nw_n)]
    waddr_lanes = [waddr_bus_w[i * ptag_w : (i + 1) * ptag_w] for i in range(nw_n)]
    wdata_lanes = [wdata_bus_w[i * 64 : (i + 1) * 64] for i in range(nw_n)]
    wdata_lo = [w[0:32] for w in wdata_lanes]
    wdata_hi = [w[32:64] for w in wdata_lanes]

    # Multiple writes to the same storage PTAG in one cycle are intentionally
    # left undefined by contract (strict no-conflict mode).
    for sidx in range(storage_depth):
        ptag = const_n + sidx
        we_any = u(1, 0)
        next_lo = bank0[sidx].out()
        next_hi = bank1[sidx].out()
        for lane in range(nw_n):
            hit = wen_lanes[lane] & (waddr_lanes[lane] == u(ptag_w, ptag))
            we_any = we_any | hit
            next_lo = wdata_lo[lane] if hit else next_lo
            next_hi = wdata_hi[lane] if hit else next_hi
        bank0[sidx].set(next_lo, when=we_any)
        bank1[sidx].set(next_hi, when=we_any)

    cmp_w = ptag_w + 1
    rdata_lanes = []
    for lane in range(nr_n):
        raddr_i = raddr_lanes[lane]
        raddr_ext = raddr_i + u(cmp_w, 0)
        is_valid = raddr_ext < u(cmp_w, ptag_n)
        is_const = raddr_ext < u(cmp_w, const_n)

        if raddr_i.width > 32:
            const32 = raddr_i[0:32]
        else:
            const32 = raddr_i + u(32, 0)
        const64 = m.cat(const32, const32)

        store_lo = u(32, 0)
        store_hi = u(32, 0)
        for sidx in range(storage_depth):
            ptag = const_n + sidx
            hit = raddr_i == u(ptag_w, ptag)
            store_lo = bank0[sidx].out() if hit else store_lo
            store_hi = bank1[sidx].out() if hit else store_hi
        store64 = m.cat(store_hi, store_lo)

        lane_data = const64 if is_const else store64
        lane_data = lane_data if is_valid else u(64, 0)
        rdata_lanes.append(lane_data)

    rdata_bus_out = rdata_lanes[0]
    for lane in range(1, nr_n):
        rdata_bus_out = m.cat(rdata_lanes[lane], rdata_bus_out)

    return m.bundle_connector(
        rdata_bus=rdata_bus_out,
    )
