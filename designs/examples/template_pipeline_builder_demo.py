from __future__ import annotations

from pycircuit import Circuit, compile_design, meta, module, template, u


@template
def _pipe_spec(m: Circuit, *, width: int):
    _ = m
    payload = meta.bundle("payload").field("data", width=width).build()
    return meta.stage_pipe("pipe", payload=payload, has_valid=True, has_ready=False)


@module
def build(m: Circuit, *, width: int = 32):
    clk = m.clock("clk")
    rst = m.reset("rst")
    clk_c = m.as_connector(clk, name="clk")
    rst_c = m.as_connector(rst, name="rst")

    s = _pipe_spec(m, width=width)
    in_b = m.io_in(s, prefix="in_")

    st0 = m.pipe_regs(s, in_b, clk=clk_c, rst=rst_c, prefix="st0_")
    st1_in = m.bundle_connector(
        data=m.as_connector((st0["data"].read() + u(width, 1))[0:width], name="data"),
        valid=m.as_connector(st0["valid"].read(), name="valid"),
    )
    st1 = m.pipe_regs(s, st1_in, clk=clk_c, rst=rst_c, prefix="st1_")

    m.io_out(s, st1, prefix="out_")


build.__pycircuit_name__ = "template_pipeline_builder_demo"


if __name__ == "__main__":
    print(compile_design(build, name="template_pipeline_builder_demo", width=32).emit_mlir())
