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
    raddr0: Connector,
    raddr1: Connector,
    wen: Connector,
    waddr: Connector,
    wdata: Connector,
    *,
    regs: int = 32,
    data_width: int = 64,
) -> ConnectorBundle:
    clk_v = clk.read() if isinstance(clk, Connector) else clk
    rst_v = rst.read() if isinstance(rst, Connector) else rst
    if not isinstance(clk_v, Signal) or clk_v.ty != "!pyc.clock":
        raise ConnectorError("RegFile.clk must be !pyc.clock")
    if not isinstance(rst_v, Signal) or rst_v.ty != "!pyc.reset":
        raise ConnectorError("RegFile.rst must be !pyc.reset")

    def wire_of(v):
        vv = v.read() if isinstance(v, Connector) else v
        if isinstance(vv, Signal):
            return m.wire(vv)
        return vv

    raddr0_w = wire_of(raddr0)
    raddr1_w = wire_of(raddr1)
    wen_w = wire_of(wen)
    waddr_w = wire_of(waddr)
    wdata_w = wire_of(wdata)
    if wen_w.ty != "i1":
        raise ConnectorError("RegFile.wen must be i1")

    n = int(regs)
    if n <= 0:
        raise ValueError("RegFile regs must be > 0")

    arr = [m.out(f"rf_{i}", clk=clk_v, rst=rst_v, width=int(data_width), init=0) for i in range(n)]

    for i in range(n):
        hit = wen_w & (waddr_w == i)
        arr[i].set(wdata_w, when=hit)

    r0 = u(int(data_width), 0)
    r1 = u(int(data_width), 0)
    for i in range(n):
        r0 = arr[i].out() if (raddr0_w == i) else r0
        r1 = arr[i].out() if (raddr1_w == i) else r1

    return m.bundle_connector(
        rdata0=m.as_connector(r0, name="rdata0"),
        rdata1=m.as_connector(r1, name="rdata1"),
    )
