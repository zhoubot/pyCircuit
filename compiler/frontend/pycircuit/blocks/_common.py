from __future__ import annotations

from ..connectors import Connector, ConnectorError
from ..dsl import Signal
from ..hw import Circuit, Wire


def _payload(m: Circuit, c: Connector, *, ctx: str):
    if not isinstance(c, Connector):
        raise ConnectorError(f"{ctx}: expected Connector, got {type(c).__name__}")
    if c.owner is not m:
        raise ConnectorError(f"{ctx}: connector belongs to a different Circuit")
    return c.read()


def as_clock(m: Circuit, c: Connector, *, ctx: str) -> Signal:
    v = _payload(m, c, ctx=ctx)
    if isinstance(v, Signal) and v.ty == "!pyc.clock":
        return v
    raise ConnectorError(f"{ctx}: expected !pyc.clock connector, got {getattr(v, 'ty', type(v).__name__)}")


def as_reset(m: Circuit, c: Connector, *, ctx: str) -> Signal:
    v = _payload(m, c, ctx=ctx)
    if isinstance(v, Signal) and v.ty == "!pyc.reset":
        return v
    raise ConnectorError(f"{ctx}: expected !pyc.reset connector, got {getattr(v, 'ty', type(v).__name__)}")


def as_wire(m: Circuit, c: Connector, *, ctx: str) -> Wire:
    v = _payload(m, c, ctx=ctx)
    if isinstance(v, Wire):
        return v
    if isinstance(v, Signal) and v.ty.startswith("i"):
        return Wire(m, v)
    raise ConnectorError(f"{ctx}: expected integer wire connector, got {getattr(v, 'ty', type(v).__name__)}")


def as_i1(m: Circuit, c: Connector, *, ctx: str) -> Wire:
    w = as_wire(m, c, ctx=ctx)
    if w.ty != "i1":
        raise ConnectorError(f"{ctx}: expected i1 connector, got {w.ty}")
    return w


def as_conn(m: Circuit, v, *, name: str) -> Connector:
    return m.as_connector(v, name=name)
