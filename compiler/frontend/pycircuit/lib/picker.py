from __future__ import annotations

from ..connectors import Connector, ConnectorBundle, ConnectorError
from ..design import module
from ..dsl import Signal
from ..hw import Circuit
from ..literals import u


@module(structural=True)
def Picker(
    m: Circuit,
    req: Connector,
    *,
    width: int | None = None,
) -> ConnectorBundle:
    req_v = req.read() if isinstance(req, Connector) else req
    if isinstance(req_v, Signal):
        req_w = m.wire(req_v)
    else:
        req_w = req_v
    if not hasattr(req_w, "ty") or not str(req_w.ty).startswith("i"):
        raise ConnectorError("Picker.req must be an integer wire connector")
    w = int(width) if width is not None else int(req_w.width)
    if w <= 0:
        raise ValueError("Picker width must be > 0")

    idx_w = max(1, (w - 1).bit_length())
    grant = req_w & 0
    index = req_w[0:idx_w] & 0
    found = req_w[0] & 0

    for i in range(w):
        take = req_w[i] & ~found
        grant = u(w, 1 << i) if take else grant
        index = u(idx_w, i) if take else index
        found = found | req_w[i]

    return m.bundle_connector(
        valid=found,
        grant=grant,
        index=index,
    )
