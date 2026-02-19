from __future__ import annotations

from ..connectors import Connector, ConnectorBundle
from ..design import module
from ..hw import Circuit
from .fifo import FIFO


@module(structural=True)
def Queue(
    m: Circuit,
    clk: Connector,
    rst: Connector,
    in_valid: Connector,
    in_data: Connector,
    out_ready: Connector,
    *,
    depth: int = 2,
) -> ConnectorBundle:
    return FIFO(
        m,
        clk,
        rst,
        in_valid,
        in_data,
        out_ready,
        depth=int(depth),
    )
