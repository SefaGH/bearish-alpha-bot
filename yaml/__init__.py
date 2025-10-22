"""Minimal YAML loader for offline test environments.

This module implements a tiny subset of YAML sufficient for the
configuration files used in the bearish-alpha-bot repository.  It only
supports mappings, lists, and scalar values (strings, integers, floats,
booleans, and nulls).  The implementation is intentionally small so it
can live in the repository and be imported as a drop-in replacement for
``PyYAML`` when the dependency cannot be installed in the sandbox.

The parser is indentation-sensitive and expects two-space indents, which
matches the style of the configuration templates in the project.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

__all__ = ["safe_load", "load"]


@dataclass
class _Line:
    indent: int
    content: str


def _prepare_lines(text: str) -> List[_Line]:
    lines: List[_Line] = []
    for raw in text.splitlines():
        if "#" in raw:
            raw = raw.split("#", 1)[0]
        stripped = raw.rstrip()
        if not stripped:
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        content = stripped.lstrip()
        lines.append(_Line(indent=indent, content=content))
    return lines


def _split_key_value(text: str) -> Tuple[str, str]:
    in_single = False
    in_double = False
    for idx, char in enumerate(text):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == ':' and not in_single and not in_double:
            return text[:idx], text[idx + 1 :]
    raise ValueError(f"Could not parse key/value pair: {text!r}")


def _parse_block(lines: List[_Line], index: int, current_indent: int) -> Tuple[Any, int]:
    container: Any = None

    while index < len(lines):
        line = lines[index]
        if line.indent < current_indent:
            break
        if line.indent > current_indent:
            raise ValueError(f"Unexpected indentation at line {index + 1}: {line.content!r}")

        if container is None:
            container = [] if line.content.startswith("- ") else {}

        if isinstance(container, list):
            if not line.content.startswith("- "):
                raise ValueError(f"Mixed list/dict entries at line {index + 1}: {line.content!r}")
            value_text = line.content[2:].strip()
            index += 1
            if value_text:
                container.append(_parse_scalar(value_text))
            else:
                value, index = _parse_block(lines, index, current_indent + 2)
                container.append(value if value is not None else {})
        else:  # dict
            key_text, value_part = _split_key_value(line.content)
            key = key_text.strip().strip('"').strip("'")
            value_part = value_part.strip()
            index += 1
            if value_part:
                container[key] = _parse_scalar(value_part)
            else:
                value, index = _parse_block(lines, index, current_indent + 2)
                container[key] = value if value is not None else {}

    return container, index


def _parse_scalar(token: str) -> Any:
    lowered = token.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if token == "{}":
        return {}
    if token == "[]":
        return []
    try:
        if token.startswith("0") and token != "0" and not token.startswith("0."):
            raise ValueError
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def safe_load(stream: Any) -> Any:
    if hasattr(stream, "read"):
        text = stream.read()
    else:
        text = stream
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    prepared = _prepare_lines(text)
    if not prepared:
        return None
    result, index = _parse_block(prepared, 0, prepared[0].indent)
    if index != len(prepared):
        raise ValueError("Unexpected trailing lines while parsing YAML")
    return result


def load(stream: Any) -> Any:
    return safe_load(stream)
