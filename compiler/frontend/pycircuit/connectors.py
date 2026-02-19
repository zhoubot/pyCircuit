from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, MutableMapping


class ConnectorError(TypeError):
    pass


class Connector:
    """Base class for inter-module connection objects."""

    owner: Any
    name: str

    @property
    def ty(self) -> str:
        raise NotImplementedError

    def read(self) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class WireConnector(Connector):
    owner: Any
    name: str
    wire: Any

    def __post_init__(self) -> None:
        maybe_owner = getattr(self.wire, "m", None)
        if maybe_owner is not None and maybe_owner is not self.owner:
            raise ConnectorError("wire connector must belong to the declaring Circuit")
        if not hasattr(self.wire, "ty"):
            raise ConnectorError("wire connector payload must expose a `ty` attribute")

    @property
    def ty(self) -> str:
        return str(self.wire.ty)

    @property
    def signed(self) -> bool:
        return bool(getattr(self.wire, "signed", False))

    def read(self) -> Any:
        return self.wire


@dataclass(frozen=True)
class RegConnector(Connector):
    owner: Any
    name: str
    reg: Any

    def __post_init__(self) -> None:
        q = getattr(self.reg, "q", None)
        if q is None or getattr(q, "m", None) is not self.owner:
            raise ConnectorError("reg connector must belong to the declaring Circuit")

    @property
    def ty(self) -> str:
        return str(self.reg.ty)

    def read(self) -> Any:
        return self.reg.q

    def set(self, value: Any, *, when: Any = 1) -> None:
        self.reg.set(value, when=when)


class ConnectorBundle:
    def __init__(self, fields: Mapping[str, Connector]) -> None:
        out: MutableMapping[str, Connector] = {}
        for k, v in fields.items():
            key = str(k)
            if not key:
                raise ConnectorError("bundle field name must be non-empty")
            if not isinstance(v, Connector):
                raise ConnectorError(f"bundle field {key!r}: expected Connector, got {type(v).__name__}")
            out[key] = v
        self.fields: dict[str, Connector] = dict(out)

    def __getitem__(self, key: str) -> Connector:
        return self.fields[str(key)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def items(self) -> Iterable[tuple[str, Connector]]:
        return self.fields.items()

    def values(self) -> Iterable[Connector]:
        return self.fields.values()

    def keys(self) -> Iterable[str]:
        return self.fields.keys()


@dataclass(frozen=True)
class ModuleInstanceHandle:
    name: str
    symbol: str
    inputs: Mapping[str, Connector]
    outputs: Connector | ConnectorBundle


ConnectorLike = Connector | ConnectorBundle


def is_connector(v: Any) -> bool:
    return isinstance(v, Connector)


def is_connector_bundle(v: Any) -> bool:
    return isinstance(v, ConnectorBundle)


def connector_owner(v: ConnectorLike) -> Any | None:
    if isinstance(v, Connector):
        return v.owner
    if isinstance(v, ConnectorBundle):
        owner: Any | None = None
        for c in v.values():
            if owner is None:
                owner = c.owner
                continue
            if c.owner is not owner:
                raise ConnectorError("connector bundle fields must belong to the same Circuit")
        return owner
    return None


def connector_to_wire(v: Connector, *, ctx: str) -> Any:
    if not isinstance(v, Connector):
        raise ConnectorError(f"{ctx}: expected Connector, got {type(v).__name__}")
    return v.read()
