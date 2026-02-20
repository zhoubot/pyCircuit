from __future__ import annotations

from pycircuit import Circuit, compile, module
from pycircuit.lib import RegFile


@module
def build(
    m: Circuit,
    *,
    ptag_count: int = 256,
    const_count: int = 128,
    nr: int = 10,
    nw: int = 5,
) -> None:
    ptag_n = int(ptag_count)
    const_n = int(const_count)
    nr_n = int(nr)
    nw_n = int(nw)
    if ptag_n <= 0:
        raise ValueError("regfile ptag_count must be > 0")
    if const_n < 0 or const_n > ptag_n:
        raise ValueError("regfile const_count must satisfy 0 <= const_count <= ptag_count")
    if nr_n <= 0:
        raise ValueError("regfile nr must be > 0")
    if nw_n <= 0:
        raise ValueError("regfile nw must be > 0")

    ptag_w = max(1, (ptag_n - 1).bit_length())

    clk = m.clock("clk")
    rst = m.reset("rst")

    raddr = [m.input(f"raddr{i}", width=ptag_w) for i in range(nr_n)]
    wen = [m.input(f"wen{i}", width=1) for i in range(nw_n)]
    waddr = [m.input(f"waddr{i}", width=ptag_w) for i in range(nw_n)]
    wdata = [m.input(f"wdata{i}", width=64) for i in range(nw_n)]

    raddr_bus = raddr[0]
    for i in range(1, nr_n):
        raddr_bus = m.cat(raddr[i], raddr_bus)

    wen_bus = wen[0]
    for i in range(1, nw_n):
        wen_bus = m.cat(wen[i], wen_bus)

    waddr_bus = waddr[0]
    for i in range(1, nw_n):
        waddr_bus = m.cat(waddr[i], waddr_bus)

    wdata_bus = wdata[0]
    for i in range(1, nw_n):
        wdata_bus = m.cat(wdata[i], wdata_bus)

    rf = RegFile(
        m,
        clk=clk,
        rst=rst,
        raddr_bus=raddr_bus,
        wen_bus=wen_bus,
        waddr_bus=waddr_bus,
        wdata_bus=wdata_bus,
        ptag_count=ptag_n,
        const_count=const_n,
        nr=nr_n,
        nw=nw_n,
    )

    rdata_bus = rf["rdata_bus"].read()
    for i in range(nr_n):
        m.output(f"rdata{i}", rdata_bus[i * 64 : (i + 1) * 64])


build.__pycircuit_name__ = "regfile"


if __name__ == "__main__":
    print(compile(build, name="regfile").emit_mlir())
