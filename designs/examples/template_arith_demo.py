from __future__ import annotations

from pycircuit import Circuit, compile_design, ct, module, template, u


@template
def _acc_width(m: Circuit, *, lanes: int, lane_width: int) -> int:
    _ = m
    lanes_i = max(1, int(lanes))
    lane_w = max(1, int(lane_width))
    return lane_w + ct.clog2(lanes_i)


@template
def _lane_mask(m: Circuit, *, width: int) -> int:
    _ = m
    w = max(1, int(width))
    return ct.bitmask(w)


@module
def build(m: Circuit, lanes: int = 8, lane_width: int = 16) -> None:
    acc_w = _acc_width(m, lanes=lanes, lane_width=lane_width)
    lane_mask = _lane_mask(m, width=lane_width)

    a = m.input("a", width=acc_w)
    b = m.input("b", width=acc_w)

    m.output("sum", a + b)
    m.output("lane_mask", u(lane_width, lane_mask))
    m.output("acc_width", u(max(1, ct.clog2(256)), acc_w))


if __name__ == "__main__":
    print(compile_design(build, name="template_arith_demo", lanes=8, lane_width=16).emit_mlir())
