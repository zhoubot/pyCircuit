from __future__ import annotations

from ..connectors import Connector, ConnectorBundle, ConnectorError
from ..design import module
from ..dsl import Signal
from ..hw import Circuit, Wire


@module(structural=True)
def FIFO(
    m: Circuit,
    clk: Connector,
    rst: Connector,
    in_valid: Connector,
    in_data: Connector,
    out_ready: Connector,
    *,
    depth: int = 2,
) -> ConnectorBundle:
    clk_v = clk.read() if isinstance(clk, Connector) else clk
    rst_v = rst.read() if isinstance(rst, Connector) else rst
    in_valid_v = in_valid.read() if isinstance(in_valid, Connector) else in_valid
    in_data_v = in_data.read() if isinstance(in_data, Connector) else in_data
    out_ready_v = out_ready.read() if isinstance(out_ready, Connector) else out_ready

    if not isinstance(clk_v, Signal) or clk_v.ty != "!pyc.clock":
        raise ConnectorError("FIFO.clk must be !pyc.clock")
    if not isinstance(rst_v, Signal) or rst_v.ty != "!pyc.reset":
        raise ConnectorError("FIFO.rst must be !pyc.reset")

    if isinstance(in_valid_v, Signal):
        in_valid_w = Wire(m, in_valid_v)
    else:
        in_valid_w = in_valid_v
    if isinstance(in_data_v, Signal):
        in_data_w = Wire(m, in_data_v)
    else:
        in_data_w = in_data_v
    if isinstance(out_ready_v, Signal):
        out_ready_w = Wire(m, out_ready_v)
    else:
        out_ready_w = out_ready_v

    if not isinstance(in_valid_w, Wire) or in_valid_w.ty != "i1":
        raise ConnectorError("FIFO.in_valid must be i1")
    if not isinstance(in_data_w, Wire):
        raise ConnectorError("FIFO.in_data must be integer wire")
    if not isinstance(out_ready_w, Wire) or out_ready_w.ty != "i1":
        raise ConnectorError("FIFO.out_ready must be i1")

    in_ready, out_valid, out_data = m.fifo(
        clk_v,
        rst_v,
        in_valid=in_valid_w,
        in_data=in_data_w,
        out_ready=out_ready_w,
        depth=int(depth),
    )

    return m.bundle_connector(
        in_ready=in_ready,
        out_valid=out_valid,
        out_data=out_data,
    )
