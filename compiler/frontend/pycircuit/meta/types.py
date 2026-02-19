from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


_ALLOWED_PARAM_TYPES = (bool, int, str)


def _check_name(name: str, *, ctx: str) -> str:
    out = str(name).strip()
    if not out:
        raise ValueError(f"{ctx} name must be non-empty")
    return out


@dataclass(frozen=True)
class FieldSpec:
    name: str
    width: int
    signed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _check_name(self.name, ctx="field"))
        w = int(self.width)
        if w <= 0:
            raise ValueError("field width must be > 0")
        object.__setattr__(self, "width", w)
        object.__setattr__(self, "signed", bool(self.signed))

    def __pyc_template_value__(self) -> dict[str, Any]:
        return {"kind": "field", "name": self.name, "width": self.width, "signed": self.signed}


@dataclass(frozen=True)
class BundleSpec:
    name: str
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _check_name(self.name, ctx="bundle"))
        fs = tuple(self.fields)
        if not fs:
            raise ValueError("bundle must contain at least one field")
        seen: set[str] = set()
        for f in fs:
            if not isinstance(f, FieldSpec):
                raise TypeError(f"bundle field must be FieldSpec, got {type(f).__name__}")
            if f.name in seen:
                raise ValueError(f"duplicate field name in bundle {self.name!r}: {f.name!r}")
            seen.add(f.name)
        object.__setattr__(self, "fields", fs)

    def __pyc_template_value__(self) -> dict[str, Any]:
        return {
            "kind": "bundle",
            "name": self.name,
            "fields": [f.__pyc_template_value__() for f in self.fields],
        }


@dataclass(frozen=True)
class InterfaceSpec:
    name: str
    bundles: tuple[BundleSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _check_name(self.name, ctx="interface"))
        bs = tuple(self.bundles)
        if not bs:
            raise ValueError("interface must contain at least one bundle")
        seen: set[str] = set()
        for b in bs:
            if not isinstance(b, BundleSpec):
                raise TypeError(f"interface bundle must be BundleSpec, got {type(b).__name__}")
            if b.name in seen:
                raise ValueError(f"duplicate bundle name in interface {self.name!r}: {b.name!r}")
            seen.add(b.name)
        object.__setattr__(self, "bundles", bs)

    def __pyc_template_value__(self) -> dict[str, Any]:
        return {
            "kind": "interface",
            "name": self.name,
            "bundles": [b.__pyc_template_value__() for b in self.bundles],
        }


@dataclass(frozen=True)
class StagePipeSpec:
    name: str
    payload: BundleSpec
    has_valid: bool = True
    has_ready: bool = False
    valid_name: str = "valid"
    ready_name: str = "ready"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _check_name(self.name, ctx="stage pipe"))
        if not isinstance(self.payload, BundleSpec):
            raise TypeError("stage pipe payload must be a BundleSpec")
        object.__setattr__(self, "has_valid", bool(self.has_valid))
        object.__setattr__(self, "has_ready", bool(self.has_ready))
        object.__setattr__(self, "valid_name", _check_name(self.valid_name, ctx="valid field"))
        object.__setattr__(self, "ready_name", _check_name(self.ready_name, ctx="ready field"))

    def __pyc_template_value__(self) -> dict[str, Any]:
        return {
            "kind": "stage_pipe",
            "name": self.name,
            "payload": self.payload.__pyc_template_value__(),
            "has_valid": self.has_valid,
            "has_ready": self.has_ready,
            "valid_name": self.valid_name,
            "ready_name": self.ready_name,
        }


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: bool | int | str
    min_value: int | None = None
    max_value: int | None = None
    choices: tuple[bool | int | str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _check_name(self.name, ctx="param"))
        if not isinstance(self.default, _ALLOWED_PARAM_TYPES):
            raise TypeError(f"param default must be bool/int/str, got {type(self.default).__name__}")
        if self.min_value is not None:
            object.__setattr__(self, "min_value", int(self.min_value))
        if self.max_value is not None:
            object.__setattr__(self, "max_value", int(self.max_value))
        if self.min_value is not None and self.max_value is not None and int(self.max_value) < int(self.min_value):
            raise ValueError("param max_value must be >= min_value")
        cs = tuple(self.choices)
        for c in cs:
            if not isinstance(c, _ALLOWED_PARAM_TYPES):
                raise TypeError(f"param choice must be bool/int/str, got {type(c).__name__}")
        object.__setattr__(self, "choices", cs)

    def validate(self, value: bool | int | str) -> bool | int | str:
        if not isinstance(value, _ALLOWED_PARAM_TYPES):
            raise TypeError(f"param {self.name!r} expects bool/int/str, got {type(value).__name__}")
        if self.choices and value not in self.choices:
            raise ValueError(f"param {self.name!r} value {value!r} not in choices")
        if isinstance(value, int):
            if self.min_value is not None and value < int(self.min_value):
                raise ValueError(f"param {self.name!r} value {value} < min {int(self.min_value)}")
            if self.max_value is not None and value > int(self.max_value):
                raise ValueError(f"param {self.name!r} value {value} > max {int(self.max_value)}")
        return value

    def __pyc_template_value__(self) -> dict[str, Any]:
        return {
            "kind": "param_spec",
            "name": self.name,
            "default": self.default,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "choices": list(self.choices),
        }


@dataclass(frozen=True)
class ParamSet:
    values: tuple[tuple[str, bool | int | str], ...]
    name: str | None = None

    def __post_init__(self) -> None:
        vals = tuple((str(k), v) for k, v in self.values)
        if not vals:
            raise ValueError("ParamSet must contain at least one value")
        for k, v in vals:
            _check_name(k, ctx="param key")
            if not isinstance(v, _ALLOWED_PARAM_TYPES):
                raise TypeError(f"param value for {k!r} must be bool/int/str, got {type(v).__name__}")
        object.__setattr__(self, "values", tuple(sorted(vals, key=lambda kv: kv[0])))
        if self.name is not None:
            object.__setattr__(self, "name", _check_name(self.name, ctx="variant"))

    @classmethod
    def from_mapping(cls, values: Mapping[str, bool | int | str], *, name: str | None = None) -> "ParamSet":
        return cls(values=tuple((str(k), v) for k, v in values.items()), name=name)

    def as_dict(self) -> dict[str, bool | int | str]:
        return {k: v for k, v in self.values}

    def __getitem__(self, key: str) -> bool | int | str:
        k = str(key)
        for kk, vv in self.values:
            if kk == k:
                return vv
        raise KeyError(k)

    def __pyc_template_value__(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": "param_set",
            "values": [[k, v] for k, v in self.values],
        }
        if self.name is not None:
            out["name"] = self.name
        return out


@dataclass(frozen=True)
class ParamSpace:
    variants: tuple[ParamSet, ...]

    def __post_init__(self) -> None:
        vs = tuple(self.variants)
        if not vs:
            raise ValueError("ParamSpace must contain at least one variant")
        for v in vs:
            if not isinstance(v, ParamSet):
                raise TypeError(f"param space variants must be ParamSet, got {type(v).__name__}")
        object.__setattr__(self, "variants", vs)

    def __iter__(self):
        return iter(self.variants)

    def __len__(self) -> int:
        return len(self.variants)

    def __pyc_template_value__(self) -> dict[str, Any]:
        return {
            "kind": "param_space",
            "variants": [v.__pyc_template_value__() for v in self.variants],
        }


@dataclass(frozen=True)
class DecodeRule:
    name: str
    mask: int
    match: int
    updates: tuple[tuple[str, bool | int | str], ...]
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _check_name(self.name, ctx="decode rule"))
        object.__setattr__(self, "mask", int(self.mask))
        object.__setattr__(self, "match", int(self.match))
        object.__setattr__(self, "priority", int(self.priority))
        ups = tuple((str(k), v) for k, v in self.updates)
        if not ups:
            raise ValueError("decode rule must contain at least one update")
        for k, v in ups:
            _check_name(k, ctx="decode update key")
            if not isinstance(v, _ALLOWED_PARAM_TYPES):
                raise TypeError(f"decode update value for {k!r} must be bool/int/str, got {type(v).__name__}")
        object.__setattr__(self, "updates", tuple(sorted(ups, key=lambda kv: kv[0])))

    @classmethod
    def from_mapping(
        cls,
        *,
        name: str,
        mask: int,
        match: int,
        updates: Mapping[str, bool | int | str],
        priority: int = 0,
    ) -> "DecodeRule":
        return cls(name=name, mask=int(mask), match=int(match), updates=tuple(updates.items()), priority=priority)

    def __pyc_template_value__(self) -> dict[str, Any]:
        return {
            "kind": "decode_rule",
            "name": self.name,
            "mask": self.mask,
            "match": self.match,
            "updates": [[k, v] for k, v in self.updates],
            "priority": self.priority,
        }


def ensure_bundle_spec(spec: BundleSpec | InterfaceSpec | StagePipeSpec) -> BundleSpec:
    if isinstance(spec, BundleSpec):
        return spec
    if isinstance(spec, StagePipeSpec):
        return spec.payload
    if isinstance(spec, InterfaceSpec):
        if len(spec.bundles) != 1:
            raise ValueError(
                f"interface {spec.name!r} has {len(spec.bundles)} bundles; select one bundle explicitly"
            )
        return spec.bundles[0]
    raise TypeError(f"expected BundleSpec/InterfaceSpec/StagePipeSpec, got {type(spec).__name__}")


def map_param_specs(
    specs: Sequence[ParamSpec],
    values: Mapping[str, bool | int | str] | None = None,
) -> ParamSet:
    vals: dict[str, bool | int | str] = {}
    values_dict = dict(values or {})
    for s in specs:
        if not isinstance(s, ParamSpec):
            raise TypeError(f"param spec list must contain ParamSpec, got {type(s).__name__}")
        if s.name in values_dict:
            vals[s.name] = s.validate(values_dict[s.name])
        else:
            vals[s.name] = s.default
    extra = sorted(set(values_dict.keys()) - set(vals.keys()))
    if extra:
        raise ValueError(f"unknown param value keys: {', '.join(extra)}")
    return ParamSet.from_mapping(vals)


def filter_param_space(space: ParamSpace, pred: Callable[[ParamSet], bool]) -> ParamSpace:
    out = [v for v in space if bool(pred(v))]
    if not out:
        raise ValueError("filtered ParamSpace is empty")
    return ParamSpace(tuple(out))
