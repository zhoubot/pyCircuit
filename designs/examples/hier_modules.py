from __future__ import annotations

from pycircuit import Circuit, compile, module


@module
def _incrementer(m: Circuit, x, *, width: int = 8):
    m.output("y", (x + 1)[0:width])


@module
def build(m: Circuit, width: int = 8, stages: int = 3) -> None:
    x = m.input("x", width=width)
    v_conn = x
    for i in range(stages):
        v_conn = _incrementer(m, x=v_conn, width=width)
    m.output("y", v_conn)


if __name__ == "__main__":
    print(compile(build, name="hier_modules", width=8, stages=3).emit_mlir())
