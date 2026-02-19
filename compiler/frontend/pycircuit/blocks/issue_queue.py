from __future__ import annotations

from ..connectors import Connector, ConnectorBundle
from ..design import module
from ..hw import Circuit
from ..literals import u
from .fifo import FIFO


@module(structural=True)
def IssueQueue(
    m: Circuit,
    clk: Connector,
    rst: Connector,
    in_valid: Connector,
    in_data: Connector,
    out0_ready: Connector,
    out1_ready: Connector,
    *,
    depth: int = 4,
    data_width: int = 32,
) -> ConnectorBundle:
    fifo = FIFO(
        m,
        clk,
        rst,
        in_valid,
        in_data,
        out0_ready,
        depth=int(depth),
    )

    # Simple structural baseline: lane0 issues from FIFO, lane1 is idle.
    z1 = u(1, 0)
    zd = u(int(data_width), 0)
    _ = out1_ready

    return m.bundle_connector(
        in_ready=fifo["in_ready"],
        out0_valid=fifo["out_valid"],
        out0_data=fifo["out_data"],
        out1_valid=m.as_connector(z1, name="out1_valid"),
        out1_data=m.as_connector(zd, name="out1_data"),
    )
