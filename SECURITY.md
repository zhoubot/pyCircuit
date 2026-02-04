# Security policy

pyCircuit is a prototype compiler/toolchain. It is **not** intended for use in security-critical contexts yet.

## Reporting a vulnerability

If you believe you have found a security issue, please open a private report via your normal internal process,
or (if this repo is mirrored publicly) contact the maintainers to coordinate a fix before public disclosure.

## Scope

Potential security issues include, but are not limited to:

- Code execution via malicious `.pyc` / MLIR input
- Unsafe file writes when running `pycircuit` tooling
- Incorrect memory bounds in the C++ simulator templates

