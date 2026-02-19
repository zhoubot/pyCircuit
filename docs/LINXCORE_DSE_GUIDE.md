# LinxCore DSE Guide (v3.2)

This guide defines the v3.2 migration direction for a parameterized LinxCore frontend.

## Objectives

- Move structural config derivation into `@template` code.
- Represent interface/stage contracts with `meta` specs.
- Reduce manual port/instance wiring with `instance_bind` + spec-driven helpers.
- Keep deterministic parameter-space expansion for reproducible sweeps.

## Recommended structure

1. `common/config.py`
- Provide template-derived normalized parameter sets.
- Validate constraints once at compile time.

2. `common/meta_specs.py`
- Build `BundleSpec`/`InterfaceSpec` from canonical stage contracts.
- Export helpers consumed by top/stage modules.

3. top-level composition
- Use spec declarations (`io_in/io_out`) for external ports.
- Use `instance_bind` for repeated interface wiring paths.

4. sweep scripts
- Use `meta.dse.product/grid/filter` to generate variant sets.
- Emit deterministic names and manifests for each variant.

## Migration checkpoints

- Wave 1: top composition + frontend shell modules use spec-driven wiring.
- Wave 2: decode tables authored from `DecodeRule` sets.
- Wave 3: backend internal generators parameterized for ROB/IQ/issue/store structures.

## Hygiene gates

- No `jit_inline` usage in migrated paths.
- No removed method-style frontend APIs in migrated paths.
- Deterministic compile outputs for identical parameter sets.
