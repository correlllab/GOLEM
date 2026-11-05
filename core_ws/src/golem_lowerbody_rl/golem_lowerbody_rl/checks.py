"""
Check values read from this package's YAML configs.

Each function returns the value in the type the loaders keep, or raises ValueError
naming where the value came from (a dotted key path).
"""

from collections.abc import Iterable
import math

__all__ = ['check_keys', 'count', 'flag', 'name', 'names', 'number', 'numbers', 'positive']


def number(where: str, value) -> float:
    """Return value as a float if it is a finite int or float (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{where} must be a finite number, not {value!r}')
    return float(value)


def positive(where: str, value) -> float:
    """Return value as a float if it is a positive finite number."""
    result = number(where, value)
    if result <= 0.0:
        raise ValueError(f'{where} must be positive, not {value!r}')
    return result


def count(where: str, value) -> int:
    """Return value if it is a positive int (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{where} must be a positive integer, not {value!r}')
    return value


def flag(where: str, value) -> bool:
    """Return value if it is true or false."""
    if not isinstance(value, bool):
        raise ValueError(f'{where} must be true or false, not {value!r}')
    return value


def numbers(where: str, value, length: int) -> tuple[float, ...]:
    """Return value as a tuple of floats if it is a list of length finite numbers."""
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{where} must be a list of {length} numbers, not {value!r}')
    return tuple(number(where, item) for item in value)


def name(where: str, value) -> str:
    """Return value if it is a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f'{where} must be a name, not {value!r}')
    return value


def names(where: str, value) -> tuple[str, ...]:
    """Return value as a tuple if it is a non-empty list of distinct names."""
    if not isinstance(value, list) or not value:
        raise ValueError(f'{where} must be a non-empty list of names, not {value!r}')
    result = tuple(name(where, item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f'{where} must name each entry once, not {value!r}')
    return result


def check_keys(where: str, mapping, keys: Iterable[str]) -> None:
    """Raise ValueError naming the first missing or unknown key of mapping."""
    keys = tuple(keys)
    if not isinstance(mapping, dict):
        raise ValueError(f'{where or "the file"} must be a mapping')
    prefix = f'{where}.' if where else ''
    for key in mapping:
        if key not in keys:
            raise ValueError(f"unknown key '{prefix}{key}'")
    for key in keys:
        if key not in mapping:
            raise ValueError(f"missing key '{prefix}{key}'")
