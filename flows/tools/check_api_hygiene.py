#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("jit_inline", re.compile(r"\bjit_inline\b")),
    ("public compile import", re.compile(r"from\s+pycircuit\s+import[^\n]*\bcompile\b")),
    ("pycircuit.compile", re.compile(r"\bpycircuit\.compile\b")),
)

EXAMPLES_ONLY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("forbidden api .eq(", re.compile(r"\.eq\(")),
    ("forbidden api .lt(", re.compile(r"\.lt\(")),
    ("forbidden api .select(", re.compile(r"\.select\(")),
    ("forbidden api mux(", re.compile(r"\bmux\(")),
    ("forbidden api cond(", re.compile(r"\bcond\(")),
    ("forbidden api .trunc(", re.compile(r"\.trunc\(")),
    ("forbidden api .zext(", re.compile(r"\.zext\(")),
    ("forbidden api .sext(", re.compile(r"\.sext\(")),
    ("forbidden api m.const(", re.compile(r"\bm\.const\(")),
    ("forbidden api CycleAware", re.compile(r"\bCycleAware[A-Za-z_]*\b")),
    ("forbidden api compile_cycle_aware", re.compile(r"\bcompile_cycle_aware\b")),
    ("forbidden api .as_unsigned(", re.compile(r"\.as_unsigned\(")),
)

DEFAULT_TARGETS: tuple[str, ...] = (
    "compiler/frontend/pycircuit",
    "designs/examples",
    "docs",
    "flows/tools",
    "README.md",
)

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
}

SKIP_DIRS = {".git", ".pycircuit_out", "__pycache__", "build", "build-top", "compiler/mlir/build2"}


def iter_target_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    out: list[Path] = []
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        if any(part in SKIP_DIRS for part in f.parts):
            continue
        if f.name == "check_api_hygiene.py":
            continue
        if f.suffix and f.suffix.lower() not in TEXT_SUFFIXES:
            continue
        out.append(f)
    return out


def scan_file(path: Path, *, extra_patterns: tuple[tuple[str, re.Pattern[str]], ...] = ()) -> list[tuple[int, int, str]]:
    hits: list[tuple[int, int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return hits
    pats = (*FORBIDDEN_PATTERNS, *extra_patterns)
    for line_no, line in enumerate(text.splitlines(), start=1):
        for label, pattern in pats:
            for m in pattern.finditer(line):
                hits.append((line_no, m.start() + 1, label))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fail if stale/forbidden pyCircuit frontend API tokens are present."
    )
    ap.add_argument(
        "--scan-root",
        default=None,
        help="Optional root directory to scan. Defaults to this repo root.",
    )
    ap.add_argument(
        "targets",
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Files/directories to scan",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    scan_root = Path(args.scan_root).resolve() if args.scan_root else root
    if not scan_root.exists():
        raise SystemExit(f"--scan-root does not exist: {scan_root}")
    violations = 0

    for target in args.targets:
        tp = (scan_root / target).resolve() if not Path(target).is_absolute() else Path(target)
        for f in iter_target_files(tp):
            rel = f.relative_to(scan_root) if f.is_relative_to(scan_root) else f
            rel_posix = rel.as_posix() if isinstance(rel, Path) else str(rel)
            extra = (
                EXAMPLES_ONLY_PATTERNS
                if (
                    rel_posix.startswith("designs/examples/")
                    or rel_posix.startswith("docs/")
                    or rel_posix == "README.md"
                )
                else ()
            )
            for line_no, col, label in scan_file(f, extra_patterns=extra):
                print(f"{rel}:{line_no}:{col}: forbidden API token `{label}`")
                violations += 1

    if violations:
        print(f"error: found {violations} forbidden API token(s)")
        return 1
    print("ok: API hygiene check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
