from __future__ import annotations

from pycircuit import Circuit


def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")
    en = m.in_wire("en", width=1)

    count = m.out("count", clk=clk, rst=rst, width=8, init=0, en=en)
    count <<= count + 1

    m.output("count", count)
