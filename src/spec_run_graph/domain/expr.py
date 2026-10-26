"""Assertion check grammar:  <path> <op> [<literal>]

path     := response.status | response.headers.<name> | response.body(.<key> | [<index>])*
op       := == | != | > | >= | < | <= | in | contains | matches | exists | not exists
literal  := 'str' | "str" | number | true | false | null | [lit, ...] | {{ $ns.name }}

Examples:
  response.status == 201
  response.body.data.status == 'DRAFT'
  response.body.data.items[0].qty >= 1
  response.body.data.status in ['CONFIRMED', 'FAILED']
  response.headers.content-type contains 'json'
  response.body.data.id exists
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

OPS = ("==", "!=", ">=", "<=", ">", "<", "not exists", "exists", "contains", "matches", "in")
UNARY = {"exists", "not exists"}
PATH_RE = re.compile(r"^response\.(status|headers\.[A-Za-z0-9_\-]+|body((\.[A-Za-z_][A-Za-z0-9_\-]*)|(\[\d+\]))*)$")
TEMPLATE_RE = re.compile(r"^\{\{\s*\$[a-z]+\.[A-Za-z0-9_.]+\s*\}\}$")


class CheckSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Check:
    path: str
    op: str
    value: Any = None  # python value; templates kept as str
    is_template: bool = False


def _split_list(inner: str) -> list[str]:
    parts, buf, quote = [], [], None
    for ch in inner:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def parse_literal(text: str) -> tuple[Any, bool]:
    text = text.strip()
    if TEMPLATE_RE.match(text):
        return text, True
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1], False
    if text.startswith("["):
        if not text.endswith("]"):
            raise CheckSyntaxError(f"bad list literal: {text}")
        inner = text[1:-1].strip()
        return ([parse_literal(p)[0] for p in _split_list(inner)] if inner else []), False
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true", False
    if lowered == "null":
        return None, False
    try:
        return json.loads(text), False
    except json.JSONDecodeError as exc:
        raise CheckSyntaxError(f"cannot parse literal {text!r} (quote strings with '...')") from exc


def parse_check(text: str) -> Check:
    text = text.strip()
    m = re.match(r"^(\S+)\s+(.*)$", text)
    if not m:
        raise CheckSyntaxError(f"expected '<path> <op> <value>', got {text!r}")
    path, rest = m.group(1), m.group(2).strip()
    if not PATH_RE.match(path):
        raise CheckSyntaxError(
            f"bad path {path!r}: use response.status, response.headers.<h> or response.body.<json.path>"
        )
    for op in OPS:
        if rest == op or rest.startswith(op + " ") or (op not in UNARY and op[0] in "=!<>" and rest.startswith(op)):
            remainder = rest[len(op) :].strip()
            if op in UNARY:
                if remainder:
                    raise CheckSyntaxError(f"'{op}' takes no value: {text!r}")
                return Check(path, op)
            if not remainder:
                raise CheckSyntaxError(f"'{op}' needs a value: {text!r}")
            value, is_template = parse_literal(remainder)
            if op == "in" and not (isinstance(value, list) or is_template):
                raise CheckSyntaxError(f"'in' needs a list literal: {text!r}")
            if op in (">", ">=", "<", "<=") and not (isinstance(value, int | float) or is_template):
                raise CheckSyntaxError(f"'{op}' needs a number: {text!r}")
            return Check(path, op, value, is_template)
    raise CheckSyntaxError(f"unknown operator in {text!r}; allowed: {', '.join(OPS)}")
