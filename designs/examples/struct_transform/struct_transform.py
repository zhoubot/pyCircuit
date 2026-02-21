from __future__ import annotations

from pycircuit import Circuit, compile, const, module, spec, u


@const
def _base_struct(m: Circuit, *, width: int):
    _ = m
    return (
        spec.struct("packet")
        .field("hdr.op", width=4)
        .field("hdr.dst", width=6)
        .field("payload.data", width=width)
        .build()
    )


@const
def _pipe_struct(m: Circuit, *, width: int):
    _ = m
    spec = _base_struct(m, width=width)
    spec = spec.add_field("ctrl.valid", width=1)
    spec = spec.add_field("ctrl.ready", width=1)
    spec = spec.remove_field("ctrl.ready")
    spec = spec.rename_field("payload.data", "word")
    spec = spec.select_fields(["hdr.op", "hdr.dst", "payload.word", "ctrl.valid"])
    return spec.with_prefix("u_")


@module
def build(m: Circuit, *, width: int = 32):
    clk = m.clock("clk")
    rst = m.reset("rst")

    spec = _pipe_struct(m, width=width)
    ins = m.inputs(spec, prefix="in_")

    regs = m.state(spec, clk=clk, rst=rst, prefix="st_")
    m.connect(regs, ins)

    op = regs["u_hdr.op"].read()
    data = regs["u_payload.word"].read()
    nxt_word = (data + op + u(width, 1))[0:width]

    out_vals = regs.flatten()
    out_vals["u_payload.word"] = nxt_word
    m.outputs(spec, out_vals, prefix="out_")


build.__pycircuit_name__ = "struct_transform"
if __name__ == "__main__":
    print(compile(build, name="struct_transform", width=32).emit_mlir())
