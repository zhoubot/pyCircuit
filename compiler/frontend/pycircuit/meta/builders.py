from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .types import (
    BundleSpec,
    DecodeRule,
    FieldSpec,
    InterfaceSpec,
    ParamSet,
    ParamSpace,
    ParamSpec,
    StagePipeSpec,
    filter_param_space,
    map_param_specs,
)


@dataclass
class _BundleBuilder:
    _name: str
    _fields: list[FieldSpec]

    def field(self, name: str, *, width: int, signed: bool = False) -> "_BundleBuilder":
        self._fields.append(FieldSpec(name=name, width=width, signed=signed))
        return self

    def build(self) -> BundleSpec:
        return BundleSpec(name=self._name, fields=tuple(self._fields))


@dataclass
class _InterfaceBuilder:
    _name: str
    _bundles: list[BundleSpec]

    def bundle(self, spec: BundleSpec) -> "_InterfaceBuilder":
        if not isinstance(spec, BundleSpec):
            raise TypeError(f"bundle() expects BundleSpec, got {type(spec).__name__}")
        self._bundles.append(spec)
        return self

    def build(self) -> InterfaceSpec:
        return InterfaceSpec(name=self._name, bundles=tuple(self._bundles))


@dataclass
class _ParamsBuilder:
    _specs: list[ParamSpec]

    def add(
        self,
        name: str,
        *,
        default: bool | int | str,
        min_value: int | None = None,
        max_value: int | None = None,
        choices: tuple[bool | int | str, ...] = (),
    ) -> "_ParamsBuilder":
        self._specs.append(
            ParamSpec(
                name=name,
                default=default,
                min_value=min_value,
                max_value=max_value,
                choices=choices,
            )
        )
        return self

    def build(self, values: Mapping[str, bool | int | str] | None = None) -> ParamSet:
        return map_param_specs(tuple(self._specs), values=values)


@dataclass
class _RulesetBuilder:
    _rules: list[DecodeRule]

    def rule(
        self,
        *,
        name: str,
        mask: int,
        match: int,
        updates: Mapping[str, bool | int | str],
        priority: int = 0,
    ) -> "_RulesetBuilder":
        self._rules.append(
            DecodeRule.from_mapping(
                name=name,
                mask=mask,
                match=match,
                updates=updates,
                priority=priority,
            )
        )
        return self

    def build(self) -> tuple[DecodeRule, ...]:
        return tuple(sorted(self._rules, key=lambda r: (int(r.priority), r.name), reverse=True))


def bundle(name: str) -> _BundleBuilder:
    return _BundleBuilder(_name=str(name), _fields=[])


def iface(name: str) -> _InterfaceBuilder:
    return _InterfaceBuilder(_name=str(name), _bundles=[])


def stage_pipe(
    name: str,
    *,
    payload: BundleSpec,
    has_valid: bool = True,
    has_ready: bool = False,
    valid_name: str = "valid",
    ready_name: str = "ready",
) -> StagePipeSpec:
    return StagePipeSpec(
        name=str(name),
        payload=payload,
        has_valid=bool(has_valid),
        has_ready=bool(has_ready),
        valid_name=str(valid_name),
        ready_name=str(ready_name),
    )


def params() -> _ParamsBuilder:
    return _ParamsBuilder(_specs=[])


def ruleset() -> _RulesetBuilder:
    return _RulesetBuilder(_rules=[])


def filtered(space: ParamSpace, pred) -> ParamSpace:
    return filter_param_space(space, pred)
