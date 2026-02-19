from __future__ import annotations


def _as_int(name: str, value: int) -> int:
    try:
        out = int(value)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"{name} expects an integer value, got {type(value).__name__}") from e
    return out


def clog2(n: int) -> int:
    n_i = _as_int("clog2", n)
    if n_i <= 0:
        raise ValueError("clog2 expects n > 0")
    if n_i <= 1:
        return 0
    return int((n_i - 1).bit_length())


def flog2(n: int) -> int:
    n_i = _as_int("flog2", n)
    if n_i <= 0:
        raise ValueError("flog2 expects n > 0")
    return int(n_i.bit_length() - 1)


def div_ceil(a: int, b: int) -> int:
    a_i = _as_int("div_ceil(a)", a)
    b_i = _as_int("div_ceil(b)", b)
    if b_i == 0:
        raise ValueError("div_ceil expects b != 0")
    return -(-a_i // b_i)


def align_up(v: int, a: int) -> int:
    v_i = _as_int("align_up(v)", v)
    a_i = _as_int("align_up(a)", a)
    if a_i <= 0:
        raise ValueError("align_up expects a > 0")
    return ((v_i + a_i - 1) // a_i) * a_i


def pow2_ceil(n: int) -> int:
    n_i = _as_int("pow2_ceil", n)
    if n_i <= 0:
        raise ValueError("pow2_ceil expects n > 0")
    if n_i <= 1:
        return 1
    return 1 << int((n_i - 1).bit_length())


def bitmask(width: int) -> int:
    w_i = _as_int("bitmask", width)
    if w_i < 0:
        raise ValueError("bitmask expects width >= 0")
    if w_i == 0:
        return 0
    return (1 << w_i) - 1


__all__ = [
    "align_up",
    "bitmask",
    "clog2",
    "div_ceil",
    "flog2",
    "pow2_ceil",
]
