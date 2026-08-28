#!/usr/bin/env python3
"""Recursive-descent parser for a compact record-filter language.

Grammar:
    expression  := or_expr
    or_expr     := and_expr ("or" and_expr)*
    and_expr    := unary_expr ("and" unary_expr)*
    unary_expr  := "not" unary_expr | primary
    primary     := "(" expression ")" | comparison
    comparison  := identifier operator value
    operator    := "=" | "!=" | "<" | "<=" | ">" | ">=" | "~" | "in"
    value       := STRING | NUMBER | BOOL | NULL | "[" value_list? "]"

The parser returns an AST and includes a small evaluator for dictionaries.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from enum import Enum, auto
from typing import Any, Iterable


class Kind(Enum):
    IDENT = auto()
    STRING = auto()
    NUMBER = auto()
    BOOL = auto()
    NULL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IN = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    MATCH = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    EOF = auto()


@dataclasses.dataclass(frozen=True)
class Token:
    kind: Kind
    text: str
    offset: int
    value: Any = None


class LexError(ValueError):
    pass


class ParseError(ValueError):
    pass


IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def decode_string(source: str, start: int) -> tuple[str, int]:
    quote = source[start]
    i = start + 1
    out: list[str] = []

    escapes = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "\\": "\\",
        "'": "'",
        '"': '"',
    }

    while i < len(source):
        ch = source[i]
        if ch == quote:
            return "".join(out), i + 1
        if ch == "\\":
            i += 1
            if i >= len(source):
                raise LexError(f"unterminated escape at offset {i}")
            esc = source[i]
            if esc == "u":
                if i + 4 >= len(source):
                    raise LexError(f"incomplete unicode escape at offset {i - 1}")
                digits = source[i + 1:i + 5]
                if not all(c in "0123456789abcdefABCDEF" for c in digits):
                    raise LexError(f"invalid unicode escape \\u{digits}")
                out.append(chr(int(digits, 16)))
                i += 5
                continue
            if esc not in escapes:
                raise LexError(f"unknown escape \\{esc} at offset {i - 1}")
            out.append(escapes[esc])
            i += 1
            continue
        out.append(ch)
        i += 1

    raise LexError(f"unterminated string starting at offset {start}")


def lex(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0

    while i < len(source):
        ch = source[i]

        if ch.isspace():
            i += 1
            continue

        if ch in "'\"":
            value, end = decode_string(source, i)
            tokens.append(Token(Kind.STRING, source[i:end], i, value))
            i = end
            continue

        two_char = source[i:i + 2]
        if two_char in {"!=", "<=", ">="}:
            kind = {"!=": Kind.NE, "<=": Kind.LE, ">=": Kind.GE}[two_char]
            tokens.append(Token(kind, two_char, i))
            i += 2
            continue

        punctuation = {
            "=": Kind.EQ,
            "<": Kind.LT,
            ">": Kind.GT,
            "~": Kind.MATCH,
            "(": Kind.LPAREN,
            ")": Kind.RPAREN,
            "[": Kind.LBRACKET,
            "]": Kind.RBRACKET,
            ",": Kind.COMMA,
        }
        if ch in punctuation:
            tokens.append(Token(punctuation[ch], ch, i))
            i += 1
            continue

        number = NUMBER_RE.match(source, i)
        if number:
            text = number.group(0)
            value = float(text) if any(c in text for c in ".eE") else int(text)
            tokens.append(Token(Kind.NUMBER, text, i, value))
            i = number.end()
            continue

        ident = IDENT_RE.match(source, i)
        if ident:
            text = ident.group(0)
            lower = text.lower()
            keywords = {
                "and": Kind.AND,
                "or": Kind.OR,
                "not": Kind.NOT,
                "in": Kind.IN,
                "true": Kind.BOOL,
                "false": Kind.BOOL,
                "null": Kind.NULL,
            }
            kind = keywords.get(lower, Kind.IDENT)
            value: Any = None
            if kind == Kind.BOOL:
                value = lower == "true"
            elif kind == Kind.NULL:
                value = None
            elif kind == Kind.IDENT:
                value = text
            tokens.append(Token(kind, text, i, value))
            i = ident.end()
            continue

        raise LexError(f"unexpected character {ch!r} at offset {i}")

    tokens.append(Token(Kind.EOF, "", len(source)))
    return tokens


@dataclasses.dataclass(frozen=True)
class Expr:
    pass


@dataclasses.dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr


@dataclasses.dataclass(frozen=True)
class Unary(Expr):
    op: str
    operand: Expr


@dataclasses.dataclass(frozen=True)
class Compare(Expr):
    field: str
    op: str
    value: Any


class Parser:
    def __init__(self, tokens: Iterable[Token]):
        self.tokens = list(tokens)
        self.pos = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    def take(self, kind: Kind) -> Token:
        token = self.current
        if token.kind != kind:
            raise ParseError(
                f"expected {kind.name} at offset {token.offset}, got {token.kind.name}"
            )
        self.pos += 1
        return token

    def match(self, kind: Kind) -> bool:
        if self.current.kind == kind:
            self.pos += 1
            return True
        return False

    def parse(self) -> Expr:
        expr = self.parse_or()
        self.take(Kind.EOF)
        return expr

    def parse_or(self) -> Expr:
        expr = self.parse_and()
        while self.match(Kind.OR):
            expr = Binary("or", expr, self.parse_and())
        return expr

    def parse_and(self) -> Expr:
        expr = self.parse_unary()
        while self.match(Kind.AND):
            expr = Binary("and", expr, self.parse_unary())
        return expr

    def parse_unary(self) -> Expr:
        if self.match(Kind.NOT):
            return Unary("not", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        if self.match(Kind.LPAREN):
            inner = self.parse_or()
            self.take(Kind.RPAREN)
            return inner
        return self.parse_comparison()

    def parse_comparison(self) -> Expr:
        field = self.take(Kind.IDENT).value
        token = self.current
        operators = {
            Kind.EQ: "=",
            Kind.NE: "!=",
            Kind.LT: "<",
            Kind.LE: "<=",
            Kind.GT: ">",
            Kind.GE: ">=",
            Kind.MATCH: "~",
            Kind.IN: "in",
        }
        if token.kind not in operators:
            raise ParseError(f"expected comparison operator at offset {token.offset}")
        self.pos += 1
        value = self.parse_value()
        return Compare(field, operators[token.kind], value)

    def parse_value(self) -> Any:
        token = self.current
        if token.kind in {Kind.STRING, Kind.NUMBER, Kind.BOOL, Kind.NULL}:
            self.pos += 1
            return token.value
        if token.kind == Kind.LBRACKET:
            return self.parse_list()
        raise ParseError(f"expected value at offset {token.offset}")

    def parse_list(self) -> list[Any]:
        self.take(Kind.LBRACKET)
        values: list[Any] = []
        if self.match(Kind.RBRACKET):
            return values

        while True:
            value = self.parse_value()
            if isinstance(value, list):
                raise ParseError("nested lists are not supported")
            values.append(value)
            if self.match(Kind.RBRACKET):
                return values
            self.take(Kind.COMMA)


def lookup(record: dict[str, Any], dotted: str) -> Any:
    current: Any = record
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "=":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "in":
        if not isinstance(expected, list):
            raise TypeError("right side of 'in' must be a list")
        return actual in expected
    if op == "~":
        if actual is None:
            return False
        return re.search(str(expected), str(actual), re.IGNORECASE) is not None

    if actual is None:
        return False

    try:
        if op == "<":
            return actual < expected
        if op == "<=":
            return actual <= expected
        if op == ">":
            return actual > expected
        if op == ">=":
            return actual >= expected
    except TypeError:
        return False

    raise ValueError(f"unknown operator {op}")


def evaluate(expr: Expr, record: dict[str, Any]) -> bool:
    if isinstance(expr, Binary):
        if expr.op == "and":
            return evaluate(expr.left, record) and evaluate(expr.right, record)
        if expr.op == "or":
            return evaluate(expr.left, record) or evaluate(expr.right, record)
        raise ValueError(expr.op)

    if isinstance(expr, Unary):
        if expr.op == "not":
            return not evaluate(expr.operand, record)
        raise ValueError(expr.op)

    if isinstance(expr, Compare):
        return compare(lookup(record, expr.field), expr.op, expr.value)

    raise TypeError(type(expr))


def ast_to_dict(expr: Expr) -> dict[str, Any]:
    if isinstance(expr, Binary):
        return {
            "type": "binary",
            "op": expr.op,
            "left": ast_to_dict(expr.left),
            "right": ast_to_dict(expr.right),
        }
    if isinstance(expr, Unary):
        return {"type": "unary", "op": expr.op, "operand": ast_to_dict(expr.operand)}
    if isinstance(expr, Compare):
        return {"type": "compare", "field": expr.field, "op": expr.op, "value": expr.value}
    raise TypeError(type(expr))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("expression")
    ap.add_argument("--records", help="JSON Lines file; stdin if omitted")
    ap.add_argument("--show-ast", action="store_true")
    args = ap.parse_args(argv)

    try:
        tree = Parser(lex(args.expression)).parse()
    except (LexError, ParseError) as exc:
        print(f"parse error: {exc}", file=sys.stderr)
        return 2

    if args.show_ast:
        print(json.dumps(ast_to_dict(tree), indent=2))

    source = open(args.records, "r", encoding="utf-8") if args.records else sys.stdin
    try:
        for line_no, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"line {line_no}: invalid JSON: {exc}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                print(f"line {line_no}: expected object", file=sys.stderr)
                continue
            if evaluate(tree, record):
                print(json.dumps(record, ensure_ascii=False))
    finally:
        if source is not sys.stdin:
            source.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
