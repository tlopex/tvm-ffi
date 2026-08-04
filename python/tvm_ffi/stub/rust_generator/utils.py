# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Rust generator helpers for ``tvm-ffi-stubgen``.

Import/use modelling (:class:`RustUse`, :class:`RustImports`) and stateless
rendering helpers; the stateful per-object orchestration lives in
``rust_generator.codegen``.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from typing import TYPE_CHECKING, Callable

from ..utils import UnsupportedTypeError
from . import consts as C
from .consts import RUST_KEYWORDS, RUST_NON_RAW_IDENTS, RUST_TYPE_ERASED_ORIGINS

if TYPE_CHECKING:
    from tvm_ffi.core import TypeSchema


@dataclasses.dataclass(frozen=True, eq=True)
class RustUse:
    """A single Rust ``use`` item: ``use <path>;``.

    Construction normalizes dotted FFI names into ``::`` paths, rewriting the
    leading module via :data:`~.consts.RUST_MOD_MAP` (``ffi.String ->
    tvm_ffi::String``). Rust paths and bare identifiers are validated and
    keywords are rendered as raw identifiers. Bare primitive/prelude names
    (``i64``, ``bool``) and the unit type stay bare and need no ``use``.
    """

    path: str
    alias: str | None

    def __init__(self, name: str, alias: str | None = None) -> None:
        """Normalize ``name`` into a Rust ``use`` path and store it."""
        object.__setattr__(self, "path", _normalize_rust_path_or_bare(name))
        object.__setattr__(
            self, "alias", _normalize_rust_ident(alias) if alias is not None else None
        )

    @property
    def leaf(self) -> str:
        """The final path segment (the in-scope name), e.g. ``Array`` for ``tvm_ffi::Array``."""
        return self.alias or self.path.rsplit("::", 1)[-1]

    def as_use_line(self) -> str:
        """Render the ``use`` statement, or ``""`` for a bare prelude/primitive type."""
        if "::" not in self.path:
            return ""
        suffix = f" as {self.alias}" if self.alias else ""
        return f"use {self.path}{suffix};"


@dataclasses.dataclass
class RustImports:
    """Collects the ``use`` items of one generated file (all via :meth:`record`).

    Two *different* paths wanting the same in-scope name raise
    :class:`UnsupportedTypeError` (the enclosing object is skipped with a
    warning): the backend declares such pathological type names unsupported
    rather than auto-aliasing.
    """

    items: list[RustUse] = dataclasses.field(default_factory=list)
    known_type_keys: set[str] = dataclasses.field(default_factory=set)
    local_type_keys: set[str] = dataclasses.field(default_factory=set)
    canonical_type_keys: set[str] = dataclasses.field(default_factory=set)
    module_segments: tuple[str, ...] | None = None
    local_items: set[str] = dataclasses.field(default_factory=set)

    def record(self, name: str, alias: str | None = None) -> str:
        """Record a ``use`` (deduped by path) and return the in-scope name (the leaf).

        Bare prelude/primitive names record no ``use``.
        """
        probe = RustUse(name, alias)
        if not probe.as_use_line():
            return probe.leaf
        # `items` stays small (a handful of `use`s per file): linear scans.
        for item in self.items:
            if item.path == probe.path:
                if alias is not None and item.alias != probe.alias:
                    raise UnsupportedTypeError(
                        name, f"import path already has alias {item.alias!r}, not {probe.alias!r}"
                    )
                return item.leaf
        if probe.leaf in self.local_items:
            raise UnsupportedTypeError(name, f"import shadows local Rust item {probe.leaf!r}")
        if any(item.leaf == probe.leaf for item in self.items):
            raise UnsupportedTypeError(
                name, f"`use` name {probe.leaf!r} collides with an existing import"
            )
        self.items.append(probe)
        return probe.leaf

    def has_import(self, name: str) -> bool:
        """Return whether an explicit directive already resolves ``name``."""
        path = RustUse(name).path
        return any(item.path == path for item in self.items)

    def generated_type_path(self, type_key: str, suffix: str = "") -> str:
        """Resolve one generated object key from this file's actual module."""
        head, _, _ = type_key.partition(".")
        if head in C.RUST_MOD_MAP:
            return type_key
        split_rust_type_key(type_key)
        if type_key in self.local_type_keys:
            return generated_rust_type_path(type_key, suffix).rsplit("::", 1)[-1]
        if type_key not in self.canonical_type_keys or self.module_segments is None:
            raise UnsupportedTypeError(
                type_key,
                "referenced object is not in this file or a canonical generated module; "
                "add an import-object/ty-map directive or generate it under its type-key prefix",
            )
        supers = "super::" * len(self.module_segments)
        return f"{supers or 'self::'}{generated_rust_type_path(type_key, suffix)}"

    def reserve_local(self, *names: str) -> None:
        """Reserve generated items in Rust's shared type/value namespace."""
        duplicates = self.local_items.intersection(names)
        imported = {item.leaf for item in self.items}.intersection(names)
        forbidden = _RUST_BUILTIN_TYPE_NAMES.intersection(names)
        conflicts = duplicates | imported | forbidden
        if conflicts:
            joined = ", ".join(sorted(conflicts))
            raise UnsupportedTypeError(joined, f"generated Rust item name collision: {joined}")
        self.local_items.update(names)


def _escape_ident(name: str) -> str:
    """Escape a reflected field/method name into a valid Rust identifier.

    A Rust-keyword name (``impl``, ``type``, ``match``, ...) becomes the raw
    identifier ``r#<name>``, valid in every code position the generator emits
    (struct fields, builder fields, setters, ``let`` bindings, literals, ``fn``
    names). The few names rustc rejects even raw (``crate``/``self``/``Self``/
    ``super``/``_``) have no rendering: raise, skipping the object loudly.
    Message strings and FFI lookup names keep the original spelling -- escape
    only at code positions.
    """
    if name in RUST_NON_RAW_IDENTS:
        raise UnsupportedTypeError(
            name, f"name {name!r} cannot be a Rust identifier (not even raw: `r#{name}`)"
        )
    if name in RUST_KEYWORDS:
        return f"r#{name}"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise UnsupportedTypeError(name, f"name {name!r} is not a valid Rust identifier")
    return name


def _normalize_rust_ident(name: str) -> str:
    """Validate one ordinary or raw Rust identifier and normalize keywords."""
    if name.startswith("r#"):
        raw_name = name[2:]
        if raw_name in RUST_NON_RAW_IDENTS:
            raise UnsupportedTypeError(
                name, f"name {name!r} cannot be used as a raw Rust identifier"
            )
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw_name) is None:
            raise UnsupportedTypeError(name, f"name {name!r} is not a valid Rust identifier")
        return name
    return _escape_ident(name)


def _normalize_rust_path(name: str, segments: list[str]) -> str:
    """Validate and render already-split Rust path segments."""
    if len(segments) < 2 or any(not segment for segment in segments):
        raise UnsupportedTypeError(name, "Rust paths require non-empty segments")

    normalized: list[str] = []
    leading_super = True
    for index, segment in enumerate(segments):
        if segment in {"crate", "self"}:
            if index != 0:
                raise UnsupportedTypeError(
                    name, f"path keyword {segment!r} is only valid as the first segment"
                )
            normalized.append(segment)
            leading_super = False
        elif segment == "super":
            if not leading_super:
                raise UnsupportedTypeError(
                    name, "`super` is only valid in a consecutive leading path prefix"
                )
            normalized.append(segment)
        else:
            normalized.append(_normalize_rust_ident(segment))
            leading_super = False
    return "::".join(normalized)


def _normalize_rust_path_or_bare(name: str) -> str:
    """Validate a type-map value and normalize it to a Rust path or bare type.

    This intentionally accepts only one identifier/path, rather than arbitrary
    Rust source. Generic composition is owned by the schema renderer, so an
    unchecked value such as ``Option<T>`` must not be injected through a
    reflected type map.
    """
    if name == "()":
        return name
    if not name:
        raise UnsupportedTypeError(name, "Rust type path cannot be empty")
    if "." in name and "::" in name:
        raise UnsupportedTypeError(name, "Rust type paths cannot mix `.` and `::` separators")

    if "." in name:
        segments = name.split(".")
        segments[0] = C.RUST_MOD_MAP.get(segments[0], segments[0])
    elif "::" in name:
        segments = name.split("::")
    else:
        return _normalize_rust_ident(name)
    return _normalize_rust_path(name, segments)


_RUST_BUILTIN_TYPE_NAMES = frozenset(
    {
        "bool",
        "char",
        "f32",
        "f64",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "str",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
    }
)


def split_rust_module_prefix(prefix: str) -> tuple[str, ...]:
    """Validate and split a non-empty dotted Rust generation prefix."""
    parts = tuple(prefix.split("."))
    if not prefix or any(not part for part in parts):
        raise UnsupportedTypeError(prefix, "Rust module prefixes require non-empty dotted parts")
    for part in parts:
        _escape_ident(part)
    return parts


def split_rust_type_key(type_key: str) -> tuple[tuple[str, ...], str]:
    """Validate an FFI type key and split its module segments from its leaf."""
    parts = type_key.split(".")
    if len(parts) < 2 or any(not part for part in parts):
        raise UnsupportedTypeError(type_key, "Rust object type keys require non-empty dotted parts")
    for part in parts:
        _escape_ident(part)
    return tuple(parts[:-1]), parts[-1]


def generated_rust_type_path(type_key: str, suffix: str = "") -> str:
    """Render a validated generated type key as a Rust path."""
    modules, leaf = split_rust_type_key(type_key)
    parts = [*(_escape_ident(module) for module in modules), _escape_ident(leaf + suffix)]
    return "::".join(parts)


def _rust_string_literal(value: str) -> str:
    """Render arbitrary reflected text as one valid UTF-8 Rust string literal."""
    escaped: list[str] = ['"']
    replacements = {
        "\\": "\\\\",
        '"': '\\"',
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\0": "\\0",
    }
    for character in value:
        replacement = replacements.get(character)
        if replacement is not None:
            escaped.append(replacement)
        elif ord(character) < 0x20 or ord(character) == 0x7F:
            escaped.append(f"\\u{{{ord(character):x}}}")
        else:
            escaped.append(character)
    escaped.append('"')
    return "".join(escaped)


def allocate_rust_names(
    base_names: list[str], reserved: set[str] | None = None, *, collision_suffix: str
) -> list[str]:
    """Allocate deterministic, globally unique Rust names for one namespace.

    Exact non-overloaded reflected names have priority.  This means a real
    ``run_overload_1`` keeps its spelling while overloads of ``run`` receive a
    further suffix instead of creating duplicate Rust items.
    """
    counts = Counter(base_names)
    used = set(reserved or ())
    result: list[str | None] = [None] * len(base_names)

    def reserve(candidate: str) -> str:
        if candidate not in used:
            used.add(candidate)
            return candidate
        index = 2
        while f"{candidate}_{index}" in used:
            index += 1
        unique = f"{candidate}_{index}"
        used.add(unique)
        return unique

    # Preserve every unique name that does not collide with a helper.
    for index, name in enumerate(base_names):
        if counts[name] == 1 and name not in used:
            result[index] = reserve(name)

    # Helper collisions are explicit in the generated API.
    for index, name in enumerate(base_names):
        if result[index] is None and counts[name] == 1:
            result[index] = reserve(f"{name}_{collision_suffix}")

    seen: dict[str, int] = {}
    for index, name in enumerate(base_names):
        if result[index] is not None:
            continue
        seen[name] = seen.get(name, 0) + 1
        result[index] = reserve(f"{name}_overload_{seen[name]}")

    final: list[str] = []
    for name in result:
        assert name is not None
        final.append(_escape_ident(name))
    return final


def is_type_erased(schema: TypeSchema) -> bool:
    """Return whether a top-level schema needs an ``Any``/``AnyView`` boundary."""
    return schema.origin in RUST_TYPE_ERASED_ORIGINS


def _validate_typed_expr_operand(
    schema: TypeSchema, role: str, is_object_ref: Callable[[str], bool] | None
) -> None:
    """Require positive proof that a schema's Rust carrier is ObjectRefCore."""
    if is_object_ref is None or not is_object_ref(schema.origin):
        raise UnsupportedTypeError(
            schema.origin,
            f"TypedExpr {role} must be an object-reference type, got {schema.origin!r}",
        )


def _element_rust_type(
    elem: TypeSchema,
    ty_render: Callable[[str], str],
    is_object_ref: Callable[[str], bool] | None,
) -> str:
    """Render a container element / ``Optional`` payload type.

    ``Any`` uses the runtime's transparent ``AnyValue`` carrier, so containers
    preserve scalar and object elements alike. Every other origin recurses
    through :func:`render_rust_type`, which rejects the unrepresentable
    (``Dict``/``List``/``Union``/``tuple`` up front, and ``void*`` / unmapped
    leaves at ``ty_render``).
    """
    if elem.origin == "Any" or is_type_erased(elem):
        return ty_render("AnyValue")
    return render_rust_type(elem, ty_render, is_object_ref)


def render_rust_type(
    schema: TypeSchema,
    ty_render: Callable[[str], str],
    is_object_ref: Callable[[str], bool] | None = None,
) -> str:
    """Render a :class:`TypeSchema` into a Rust type expression.

    ``ty_render`` maps a leaf origin name to its Rust leaf name, recording the
    ``use`` it needs via :meth:`RustImports.record`. Raises
    :class:`UnsupportedTypeError` for origins the crate cannot represent.
    """
    origin = schema.origin
    args = schema.args

    if is_type_erased(schema):
        return ty_render("AnyValue")

    if origin == "Array":
        assert args  # TypeSchema's post_init fills a missing element type.
        elem = _element_rust_type(args[0], ty_render, is_object_ref)
        return f"{ty_render('Array')}<{elem}>"

    if origin == "Map":
        assert len(args) == 2  # TypeSchema's post_init fills a bare Map to (Any, Any).
        key = _element_rust_type(args[0], ty_render, is_object_ref)
        value = _element_rust_type(args[1], ty_render, is_object_ref)
        return f"{ty_render('Map')}<{key}, {value}>"

    if origin == "Optional":
        # Value position only (`None` <-> kTVMFFINone via Any); FIELD position
        # is layout-sensitive and routes through `render_struct_field`.
        (payload,) = args  # TypeSchema's post_init enforces exactly one argument.
        # Nullable C++ ObjectRefs already describe themselves as Optional.  An
        # explicit Optional<NullableRef> therefore produces nested Optional
        # schemas even though the packed ABI has only one None value and cannot
        # distinguish the two absent states.  Expose that wire semantics
        # honestly as a single Rust Option rather than a fictitious third state.
        while payload.origin == "Optional":
            (payload,) = payload.args
        return f"{ty_render('Optional')}<{_element_rust_type(payload, ty_render, is_object_ref)}>"

    if origin == "TypedExpr":
        if len(args) != 2:
            raise UnsupportedTypeError(origin, "TypedExpr requires exactly two arguments")
        base, expected = args
        _validate_typed_expr_operand(base, "base", is_object_ref)
        _validate_typed_expr_operand(expected, "expected type", is_object_ref)
        return (
            f"{ty_render('TypedExpr')}<"
            f"{render_rust_type(base, ty_render, is_object_ref)}, "
            f"{render_rust_type(expected, ty_render, is_object_ref)}>"
        )

    # Callable maps to the crate's type-erased Function like any other leaf.
    return ty_render(origin)  # leaf / object type


def _packed_args_expr(params: list[tuple[str, str, bool]], is_member: bool) -> str:
    """Build the ``&[AnyView]`` element list for a packed call.

    The third tuple element records whether the parameter is already an
    ``AnyView``.  Keep this semantic fact separate from its rendered spelling:
    aliases and fully-qualified paths must not change how values are packed.
    """
    parts = ["AnyView::from(&*self)"] if is_member else []
    for name, _ty, is_any_view in params:
        parts.append(name if is_any_view else f"AnyView::from(&{name})")
    return ", ".join(parts)


def _packed_call_lines(
    fvar: str, getter: list[str], packed: str, ret: str, *, returns_any: bool
) -> list[str]:
    """Build the body lines for a reflected call via ``Function::call_packed``.

    ``getter`` is the (multi-line) binding of ``fvar`` to the reflected method.
    """
    if returns_any:
        return [*getter, f"    {fvar}.call_packed(&[{packed}])"]
    return [
        *getter,
        f"    {fvar}.call_packed(&[{packed}])?.try_into_strict::<{ret}>()",
    ]
