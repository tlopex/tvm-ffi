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
"""Rust code generation for the ``tvm-ffi-stubgen`` tool.

Codegen orchestration lives here; low-level rendering helpers live in
``rust_generator.utils``.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from .. import consts as C
from ..file_utils import write_text_atomic
from . import consts as C_RUST
from .utils import (
    RustImports,
    UnsupportedTypeError,
    _escape_ident,
    _packed_args_expr,
    _packed_call_lines,
    _rust_string_literal,
    allocate_rust_names,
    is_type_erased,
    render_rust_type,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tvm_ffi.core import TypeSchema

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, NamedTypeSchema, ObjectInfo, Options


RUST_LICENSE_HEADER = """/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * \"License\"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
"""

_MODULES_BEGIN = "// @tvm-ffi-stubgen-rust-modules(begin)"
_MODULES_END = "// @tvm-ffi-stubgen-rust-modules(end)"
_MODULE_DECL_PATTERN = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*;\s*(?://.*)?$"
)


def _callable_schema_args(func: FuncInfo) -> tuple[TypeSchema, ...]:
    """Return a reflected callable signature, rejecting non-callable metadata."""
    if func.schema.origin != "Callable":
        raise UnsupportedTypeError(
            func.schema.origin,
            f"function {func.schema.name!r} has non-callable schema {func.schema.origin!r}",
        )
    return func.schema.args or ()


@dataclasses.dataclass
class _ObjectRenderer:
    """Renders one ``object/<key>`` block into Rust source lines.

    Holds the per-object rendering context (imports, ``ty_map``, resolved
    names) so helper methods don't have to thread it through.
    """

    info: ObjectInfo
    leaf: str
    obj_struct: str
    base_type: str
    is_root: bool
    imports: RustImports
    ty_map: dict[str, str]
    #: Module segments of the file this object lands in (its type key minus the
    #: leaf; ``tirx.transform.X`` -> ``("tirx", "transform")``): one file per
    #: prefix, mounted at ``<out>/<seg>/.../mod.rs`` (see ``cli`` and
    #: :func:`finalize_rust_module_tree`).
    mod_segments: tuple[str, ...]
    #: In-scope name of the parent's REF type (set by :meth:`_resolve_parent`
    #: for derived types; unused for roots, whose ``base`` is the bare
    #: ``tvm_ffi::Object`` data struct with no generated ref).
    parent_ref: str = ""

    def _ty_render(self, origin: str) -> str:
        """Resolve a leaf origin to its Rust name and record its ``use``.

        Unmapped dotted names (object type keys) resolve against the generated
        module tree via :meth:`_generated_type_path`. An unmapped bare origin
        (e.g. ``const char*``) or a ``ctypes.*`` sentinel (``ctypes.c_void_p``
        -- ``void*`` -- is dotted but is not an object key and has no Rust
        rendering) raises, skipping the enclosing object. Rejecting here covers
        every position uniformly (field, container element, method arg/return),
        so no separate element blocklist is needed.
        """
        mapped = self.ty_map.get(origin)
        if mapped is None:
            if "." not in origin or origin.startswith("ctypes."):
                raise UnsupportedTypeError(origin)
            mapped = self._generated_type_path(origin)
        return self.imports.record(mapped)

    def _generated_type_path(self, type_key: str) -> str:
        """Resolve a generated-tree type key to a path valid from this file.

        A bare ``use ir::Expr;`` is broken in edition 2021 (it resolves to an
        extern crate ``ir``, or silently captures an equally-named *submodule*),
        so cross-module references must anchor at the shared generated root:
        ``super::`` once per segment of this file's own module path, then the
        referenced key's full path (``super::ir::Expr`` from ``tirx/mod.rs``,
        ``super::super::ir::Expr`` from ``tirx/transform/mod.rs``). A key in
        *this* file's module is a local item: bare leaf, no ``use``. A head
        with a :data:`~.consts.RUST_MOD_MAP` rewrite (builtin ``ffi.*`` keys)
        lives in the crate, not the generated tree, and passes through for
        :class:`~.utils.RustUse` to rewrite.
        """
        head, _, _ = type_key.partition(".")
        if head in C_RUST.RUST_MOD_MAP:
            return type_key
        mod, _, type_leaf = type_key.rpartition(".")
        if tuple(mod.split(".")) == self.mod_segments:
            return type_leaf
        supers = "super::" * len(self.mod_segments)
        return f"{supers or 'self::'}{type_key.replace('.', '::')}"

    def render_struct_field(self, schema: NamedTypeSchema) -> str:
        """Render a reflected getter's owned output type.

        Getters cross the FFI Any boundary, so C++ integer/float widths are
        normalized and Optional values use ordinary Rust ``Option``. No native
        field layout is exposed here.
        """
        if schema.origin == "Any" or is_type_erased(schema):
            return self.imports.record("tvm_ffi::Any")
        return render_rust_type(schema, self._ty_render)

    def render_param(self, schema: TypeSchema) -> str:
        """Render an argument type (a top-level ``Any`` is the non-owning ``AnyView``)."""
        if schema.origin == "Any" or is_type_erased(schema):
            return self.imports.record("tvm_ffi::AnyView")
        return render_rust_type(schema, self._ty_render)

    def _resolve_parent(self) -> None:
        """Bring BOTH parent names into scope: ``<Parent>Obj`` and ``<Parent>``.

        The embedded ``base`` field and the ``Deref`` target need the parent's
        data struct; the upcast ``From`` target and the builder's default-
        construction fallback need its ref type. Both are items of the
        parent's OWN module, so each resolves through the same generated-tree
        path rule as any cross-module type reference (``use super::ir::Attrs;``
        + ``use super::ir::AttrsObj;``); a same-module parent stays a bare
        local name with no ``use``.
        """
        parent_key = self.info.parent_type_key
        assert isinstance(parent_key, str)  # non-root implies a parent key
        mod, dot, parent_leaf = parent_key.rpartition(".")
        self.parent_ref = self.imports.record(self._generated_type_path(parent_key))
        self.base_type = self.imports.record(
            self._generated_type_path(f"{mod}{dot}{parent_leaf}Obj")
        )

    def body(self) -> list[str]:
        """Build an opaque object marker plus safe reflection-backed accessors."""
        self.imports.record("tvm_ffi::ObjectCore")
        self.imports.record("tvm_ffi::ObjectArc")
        self.imports.record("std::marker::PhantomData")
        self.imports.record("std::rc::Rc")
        if self.is_root:
            self.base_type = self.imports.record("tvm_ffi::Object")
        else:
            self.imports.record("std::ops::Deref")
            self._resolve_parent()

        leaf, obj_struct, base_type = self.leaf, self.obj_struct, self.base_type
        lines: list[str] = [
            "#[repr(C)]",
            "#[derive(tvm_ffi::derive::Object)]",
            f"#[type_key = {_rust_string_literal(self.info.type_key or '')}]",
            f"pub struct {obj_struct} {{",
            f"    base: {base_type},",
            "    // Reflection does not prove C++ thread safety.",
            "    _not_send_sync: PhantomData<Rc<()>>,",
            "}",
            "",
            "#[repr(transparent)]",
            "#[derive(tvm_ffi::derive::ObjectRef, Clone)]",
            f"pub struct {leaf} {{",
            f"    data: ObjectArc<{obj_struct}>,",
            "}",
            "",
        ]

        if not self.is_root:
            lines += self._parent_deref_lines()
            lines += self._upcast_lines()
        lines += self._impl_block()

        lines.pop()  # every section above ends with a `""` separator
        return lines

    def _impl_block(self) -> list[str]:
        """Emit typed field getters, explicit constructors, and methods."""
        explicit = self._explicit_init_methods()
        if explicit and not self.info.has_init:
            raise UnsupportedTypeError(
                self.info.type_key or self.leaf,
                "reflected __ffi_init__ exists but the object has no init metadata",
            )
        methods = [
            m for m in self.info.methods if m.schema.name.rsplit(".", 1)[-1] != "__ffi_init__"
        ]

        sections: list[list[str]] = []
        if self.info.has_init:
            if len(explicit) == 1:
                sections.append(self._new_fn_explicit(explicit[0]))
            elif explicit:
                sections += [
                    self._new_fn_explicit(method, f"ffi_new_overload_{index}")
                    for index, method in enumerate(explicit, start=1)
                ]

        # Reflected members must not shadow ObjectRefCore/ObjectRefCast methods.
        reserved = {"is_defined", "is_null", "same_as", "downcast", "try_cast", "ffi_new"}
        reserved.update(f"ffi_new_overload_{index}" for index in range(1, len(explicit) + 1))
        method_names = self._method_names(methods, reserved)
        reserved.update(method_names)
        field_names = allocate_rust_names(
            [field.name for field in self.info.fields], reserved, collision_suffix="field"
        )
        sections += [
            self._field_fn(field, rust_name)
            for field, rust_name in zip(self.info.fields, field_names)
        ]
        sections += [
            self._method_fn(method, rust_name) for method, rust_name in zip(methods, method_names)
        ]

        if not sections:
            return []

        inner: list[str] = []
        for i, section in enumerate(sections):
            if i:
                inner.append("")
            inner += section

        return [
            f"impl {self.leaf} {{",
            *[f"    {line}" if line else "" for line in inner],
            "}",
            "",
        ]

    def _parent_deref_lines(self) -> list[str]:
        """Borrow the one-pointer child handle as its one-pointer parent handle."""
        return [
            f"impl Deref for {self.leaf} {{",
            f"    type Target = {self.parent_ref};",
            f"    fn deref(&self) -> &{self.parent_ref} {{",
            "        // Both refs transparently wrap the same object pointer.",
            f"        unsafe {{ &*(self as *const {self.leaf} as *const {self.parent_ref}) }}",
            "    }",
            "}",
            "",
        ]

    def _explicit_init_methods(self) -> list[FuncInfo]:
        """Validate and return canonical static `__ffi_init__` factories."""
        methods = [
            method
            for method in self.info.methods
            if method.schema.name.rsplit(".", 1)[-1] == "__ffi_init__"
        ]
        for method in methods:
            args = _callable_schema_args(method)
            if method.is_member or not args:
                raise UnsupportedTypeError(
                    self.info.type_key or self.leaf,
                    "__ffi_init__ must be a typed static factory",
                )
            result = args[0]
            while result.origin == "Optional":
                (result,) = result.args
            if result.origin != self.info.type_key:
                raise UnsupportedTypeError(
                    result.origin,
                    f"__ffi_init__ for {self.info.type_key!r} returns {result.origin!r}",
                )
        return methods

    def _method_names(self, methods: list[FuncInfo], reserved: set[str]) -> list[str]:
        """Give overloads and helper collisions deterministic Rust names."""
        base_names = [
            method.schema.name.rsplit(".", 1)[-1] + ("_packed" if not method.schema.args else "")
            for method in methods
        ]
        return allocate_rust_names(base_names, reserved, collision_suffix="method")

    def _field_fn(self, field: NamedTypeSchema, rust_name: str) -> list[str]:
        """Emit one owning field getter backed by `TVMFFIFieldInfo.getter`."""
        ret = self.render_struct_field(field)
        self.imports.record("tvm_ffi::Result")
        erased = field.origin == "Any" or is_type_erased(field)
        getter = "get_object_field_any" if erased else "get_object_field"
        turbofish = "" if erased else f"::<{ret}, _>"
        return [
            f"pub fn {rust_name}(&self) -> Result<{ret}> {{",
            f"    tvm_ffi::object::{getter}{turbofish}(",
            f"        self, {self.obj_struct}::type_index(), {_rust_string_literal(field.name)},",
            "    )",
            "}",
        ]

    def _upcast_lines(self) -> list[str]:
        """`impl From<Leaf> for <ParentRef>` -- offset-0 prefix retype (upcast).

        Sound because `<Leaf>Obj` embeds the parent as its offset-0 `base`, so
        the object pointer is unchanged; only the arc's static type moves
        (ownership transfers, no refcount change). Emitted for derived types
        only -- the parent's ref is the generated `<ParentLeaf>`; a root object
        has no ref-typed parent (its `base` is the bare `Object` data struct).
        """
        self.imports.record("tvm_ffi::ObjectRefCore")
        parent_ref = self.parent_ref  # in scope via `_resolve_parent`
        parent_obj = self.base_type
        return [
            f"impl From<{self.leaf}> for {parent_ref} {{",
            f"    fn from(x: {self.leaf}) -> {parent_ref} {{",
            f"        let arc = <{self.leaf} as tvm_ffi::ObjectRefCore>::into_data(x);",
            "        let up = unsafe {",
            f"            ObjectArc::from_raw(ObjectArc::into_raw(arc) as *const {parent_obj})",
            "        };",
            f"        <{parent_ref} as tvm_ffi::ObjectRefCore>::from_data(up)",
            "    }",
            "}",
            "",
        ]

    def _new_fn_explicit(self, method: FuncInfo, name: str = "ffi_new") -> list[str]:
        """Call an explicitly reflected constructor; never allocate in Rust."""
        args = _callable_schema_args(method)
        rest = args[2:] if method.is_member else args[1:]
        params = [(f"_{index}", self.render_param(schema)) for index, schema in enumerate(rest)]
        self.imports.record("tvm_ffi::Result")
        if params:
            self.imports.record("tvm_ffi::AnyView")
        packed = _packed_args_expr(params, False)
        signature = ", ".join(f"{param}: {ty}" for param, ty in params)
        getter = self._cached_getter_lines("f", "__ffi_init__")
        return [
            f"pub fn {_escape_ident(name)}({signature}) -> Result<{self.leaf}> {{",
            *_packed_call_lines("f", getter, packed, self.leaf),
            "}",
        ]

    def _cached_getter_lines(self, fvar: str, ffi_name: str) -> list[str]:
        """Body lines binding ``fvar`` to the reflected method, cached per call site.

        A ``thread_local!`` ``OnceCell`` makes the crate's method-table scan run
        once per thread (``Function`` is not ``Sync``, ruling out a ``OnceLock``).
        """
        cell = fvar.upper()
        return [
            f"    thread_local!(static {cell}: std::cell::OnceCell<tvm_ffi::Function> = "
            "const { std::cell::OnceCell::new() });",
            f"    let {fvar} = tvm_ffi::Function::from_type_method_cached(&{cell}, "
            f"{self.obj_struct}::type_index(), {_rust_string_literal(ffi_name)})?;",
        ]

    def _method_fn(self, method: FuncInfo, rust_name: str) -> list[str]:
        """Emit one reflected method (instance or static) on `impl <T>`."""
        ffi_name = method.schema.name.rsplit(".", 1)[-1]
        args = _callable_schema_args(method)
        if not args:
            self.imports.record("tvm_ffi::Any")
            self.imports.record("tvm_ffi::AnyView")
            self.imports.record("tvm_ffi::Result")
            getter = self._cached_getter_lines("f", ffi_name)
            if method.is_member:
                return [
                    f"pub fn {rust_name}(&self, args: &[AnyView<'_>]) -> Result<Any> {{",
                    *getter,
                    "    let mut packed_args = Vec::with_capacity(args.len() + 1);",
                    "    packed_args.push(AnyView::from(&*self));",
                    "    packed_args.extend_from_slice(args);",
                    "    f.call_packed(&packed_args)",
                    "}",
                ]
            return [
                f"pub fn {rust_name}(args: &[AnyView<'_>]) -> Result<Any> {{",
                *getter,
                "    f.call_packed(args)",
                "}",
            ]
        # The return type stays owning (a top-level `Any` is `Any`, not `AnyView`).
        ret_schema = args[0] if args else None
        ret = (
            self._ty_render("Any")
            if ret_schema is None or ret_schema.origin == "Any" or is_type_erased(ret_schema)
            else render_rust_type(ret_schema, self._ty_render)
        )
        rest = args[2:] if method.is_member else args[1:]
        params = [(f"_{i}", self.render_param(p)) for i, p in enumerate(rest)]

        if method.is_member:
            sig_parts = ["&self", *[f"{n}: {t}" for n, t in params]]
        else:
            sig_parts = [f"{n}: {t}" for n, t in params]
        self.imports.record("tvm_ffi::Result")
        if method.is_member or params:
            self.imports.record("tvm_ffi::AnyView")
        packed = _packed_args_expr(params, method.is_member)
        # The FFI lookup string keeps the reflected name; only the Rust `fn`
        # identifier is keyword-escaped.
        getter = self._cached_getter_lines("f", ffi_name)
        header = f"pub fn {rust_name}({', '.join(sig_parts)}) -> Result<{ret}> {{"
        return [header, *_packed_call_lines("f", getter, packed, ret), "}"]


def generate_rust_object(
    code: CodeBlock,
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
    obj_info: ObjectInfo,
) -> None:
    """Emit a Rust ``struct``/``impl`` binding for an ``object/<key>`` block.

    Emits an opaque ``<T>Obj`` prefix marker, the owning ``<T>`` ref wrapper,
    safe reflection-backed field getters, reflected methods, and explicit C++
    constructors. Generic reflected initialization is intentionally omitted:
    field metadata does not prove constructor invariants.
    Raises :class:`UnsupportedTypeError` for types the crate cannot represent;
    generation is fail-closed, so no output is committed in that case.
    """
    assert len(code.lines) >= 2
    type_key = obj_info.type_key
    assert isinstance(type_key, str)
    leaf = type_key.rsplit(".", 1)[-1]
    obj_struct = f"{leaf}Obj"
    renderer = _ObjectRenderer(
        info=obj_info,
        leaf=leaf,
        obj_struct=obj_struct,
        base_type="",  # resolved by `body()` (crate `Object` / `_resolve_parent`)
        is_root=obj_info.parent_type_key in (None, "ffi.Object"),
        imports=imports,
        ty_map=ty_map,
        mod_segments=tuple(type_key.split(".")[:-1]),
    )

    body = renderer.body()

    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in body],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust object layout needs no `opt`


def _snake_case(name: str) -> str:
    """Convert a reflected global leaf to an idiomatic Rust identifier."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_").lower()


def _render_rust_global_func(
    func: FuncInfo,
    rust_name: str,
    ty_render: Callable[[str], str],
    imports: RustImports,
) -> list[str]:
    """Render one global, using an honest packed wrapper when its schema is absent."""
    schema_args = _callable_schema_args(func)
    global_name = _rust_string_literal(func.schema.name)
    getter = [
        "    thread_local!(static F: std::cell::OnceCell<tvm_ffi::Function> = "
        "const { std::cell::OnceCell::new() });",
        f"    let f = tvm_ffi::Function::get_global_cached(&F, {global_name})?;",
    ]
    imports.record("tvm_ffi::Result")
    if not schema_args:
        imports.record("tvm_ffi::Any")
        imports.record("tvm_ffi::AnyView")
        return [
            f"pub fn {rust_name}(args: &[AnyView<'_>]) -> Result<Any> {{",
            *getter,
            "    f.call_packed(args)",
            "}",
        ]

    ret_schema = schema_args[0]
    ret = (
        ty_render("Any")
        if ret_schema.origin == "Any" or is_type_erased(ret_schema)
        else render_rust_type(ret_schema, ty_render)
    )
    params: list[tuple[str, str]] = []
    for index, schema in enumerate(schema_args[1:]):
        ty = (
            imports.record("tvm_ffi::AnyView")
            if schema.origin == "Any" or is_type_erased(schema)
            else render_rust_type(schema, ty_render)
        )
        params.append((f"_{index}", ty))
    if params:
        imports.record("tvm_ffi::AnyView")
    signature = ", ".join(f"{name}: {ty}" for name, ty in params)
    packed = _packed_args_expr(params, False)
    return [
        f"pub fn {rust_name}({signature}) -> Result<{ret}> {{",
        *_packed_call_lines("f", getter, packed, ret),
        "}",
    ]


def generate_rust_global_funcs(
    code: CodeBlock,
    global_funcs: list[FuncInfo],
    ty_map: dict[str, str],
    imports: RustImports,
    opt: Options,
) -> None:
    """Emit typed wrappers for reflected global packed functions."""
    assert len(code.lines) >= 2
    prefix = code.param[0] if isinstance(code.param, tuple) else str(code.param)
    mod_segments = tuple(segment for segment in prefix.split(".") if segment)

    def generated_path(type_key: str) -> str:
        head, _, _ = type_key.partition(".")
        if head in C_RUST.RUST_MOD_MAP:
            return type_key
        module, _, leaf = type_key.rpartition(".")
        if tuple(module.split(".")) == mod_segments:
            return leaf
        return f"{'super::' * len(mod_segments) or 'self::'}{type_key.replace('.', '::')}"

    def ty_render(origin: str) -> str:
        mapped = ty_map.get(origin)
        if mapped is None:
            if "." not in origin or origin.startswith("ctypes."):
                raise UnsupportedTypeError(origin)
            mapped = generated_path(origin)
        return imports.record(mapped)

    base_names = [
        _snake_case(func.schema.name.rsplit(".", 1)[-1])
        + ("_packed" if not func.schema.args else "")
        for func in global_funcs
    ]
    rust_names = allocate_rust_names(base_names, collision_suffix="global")
    rendered: list[str] = []
    for func, rust_name in zip(global_funcs, rust_names):
        section = _render_rust_global_func(func, rust_name, ty_render, imports)
        if rendered:
            rendered.append("")
        rendered += section

    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[(indent + line) if line else "" for line in rendered],
        code.lines[-1],
    ]


# --- import section (`use` statements) --------------------------------------


def generate_rust_import_section(
    code: CodeBlock,
    imports: RustImports,
    opt: Options,
    defined_types: set[str],
) -> None:
    """Render the collected ``use`` statements into an ``import-section`` block.

    Imports for types defined in this same file are dropped; the rest are
    deduped and sorted.
    """
    assert len(code.lines) >= 2
    # `record` never admits bare types, so every `as_use_line()` is non-empty.
    use_lines = sorted(
        {item.as_use_line() for item in imports.items if item.path not in defined_types}
    )
    indent = " " * code.indent
    code.lines = [
        code.lines[0],
        *[indent + line for line in use_lines],
        code.lines[-1],
    ]
    _ = opt  # accepted for protocol parity; Rust needs no indent/TYPE_CHECKING handling


# --- whole-file scaffolding (`--init` mode) ---------------------------------


def generate_rust_api_file(
    code_blocks: list[CodeBlock],
    ty_map: dict[str, str],
    module_name: str,
    object_infos: list[ObjectInfo],
    init_cfg: InitConfig,
    is_root: bool,
    syntax: C.MarkerSyntax,
) -> str:
    """Scaffold a single Rust binding file (one file per module prefix)."""
    append = ""
    if not code_blocks:
        append += RUST_LICENSE_HEADER
        # Generated identifiers preserve schema positions (for example `_0`),
        # and deliberately explicit conversion code favors uniform rendering
        # over handwritten-style simplification. Keep downstream Clippy focused
        # on consumer code while rustc warnings remain enforced independently.
        append += "\n#![allow(clippy::all, dead_code, unused_imports)]\n"
        append += f"\n//! FFI bindings for `{module_name}` (generated by tvm-ffi-stubgen).\n\n"
    if not any(c.kind == "import-section" for c in code_blocks):
        append += f"{syntax.begin} import-section\n{syntax.end}\n\n"
    if not any(c.kind == "global" for c in code_blocks):
        append += f"{syntax.begin} global/{module_name}\n{syntax.end}\n\n"
    defined = {c.param for c in code_blocks if c.kind == "object"}
    for info in object_infos:
        type_key = info.type_key
        if type_key is None or type_key in defined:
            continue
        append += f"{syntax.begin} object/{type_key}\n{syntax.end}\n\n"
    _ = (ty_map, init_cfg, is_root)  # unused for the Rust single-file layout
    return append


# --- module-tree stitching (auto-form `pub mod` declarations) ----------------


def _remove_managed_module_block(mod_rs: Path, lines: list[str]) -> tuple[list[str], set[str]]:
    """Validate/remove one managed block and return its declared child names."""
    begin = [i for i, line in enumerate(lines) if line.strip() == _MODULES_BEGIN]
    end = [i for i, line in enumerate(lines) if line.strip() == _MODULES_END]
    if len(begin) != len(end) or len(begin) > 1:
        raise ValueError(
            f"Malformed Rust stubgen module markers in {mod_rs}: "
            f"found {len(begin)} begin and {len(end)} end markers"
        )
    if not begin:
        return list(lines), set()
    if begin[0] >= end[0]:
        raise ValueError(f"Malformed Rust stubgen module markers in {mod_rs}: end precedes begin")

    previous_names: set[str] = set()
    for line in lines[begin[0] + 1 : end[0]]:
        if not line.strip():
            continue
        match = _MODULE_DECL_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed Rust stubgen module block in {mod_rs}: {line!r}")
        previous_names.add(match.group(1))
    return lines[: begin[0]] + lines[end[0] + 1 :], previous_names


def _plan_rust_module_tree(init_path: Path, prefixes: set[str]) -> dict[Path, str]:
    """Return every module-tree replacement after validating all marker blocks."""
    children: dict[Path, set[str]] = {}
    for prefix in sorted(prefixes):
        segs = [s for s in prefix.split(".") if s]
        for i, seg in enumerate(segs):
            parent = init_path.joinpath(*segs[:i])
            children.setdefault(parent, set()).add(seg)

    planned: dict[Path, str] = {}
    for parent in sorted(children, key=str):
        names = children[parent]
        mod_rs = parent / "mod.rs"
        existing = mod_rs.read_text(encoding="utf-8") if mod_rs.exists() else ""
        lines = existing.splitlines()
        output, previous_managed_names = _remove_managed_module_block(mod_rs, lines)

        external_names = {
            match.group(1)
            for line in output
            if (match := _MODULE_DECL_PATTERN.fullmatch(line)) is not None
        }
        # One CLI invocation may initialize only one namespace prefix. Preserve
        # siblings owned by earlier invocations; fresh staging directories still
        # start empty and therefore remain a deterministic full-regeneration path.
        managed_names = (names | previous_managed_names) - external_names

        while output and not output[-1].strip():
            output.pop()
        if not output:
            output.extend(RUST_LICENSE_HEADER.rstrip().splitlines())
        output += [
            "",
            _MODULES_BEGIN,
            *[f"pub mod {name};" for name in sorted(managed_names)],
            _MODULES_END,
        ]
        source = "\n".join(output) + "\n"
        if source != existing:
            planned[mod_rs] = source
    return planned


def validate_rust_module_tree(init_path: Path, prefixes: set[str]) -> None:
    """Validate module markers and renderability without touching the filesystem."""
    _plan_rust_module_tree(init_path, prefixes)


def finalize_rust_module_tree(init_path: Path, prefixes: set[str]) -> None:
    """Stitch the generated tree under ``init_path`` into a valid Rust module tree.

    Ensures every generated prefix is declared via ``pub mod`` in its parent's
    ``mod.rs``, creating intermediate files as needed. Declarations live in a
    deterministic managed block. User-authored declarations outside that block
    remain untouched; an external declaration of the same child suppresses the
    generated duplicate. The user still mounts ``init_path`` with one ``mod``
    line at the crate root (stubgen does not edit ``lib.rs``/``main.rs``).

    Every target is parsed and rendered before the first write, so malformed
    or duplicate managed markers fail closed without partially finalizing the
    tree. Individual replacements are then same-directory atomic writes.
    """
    for mod_rs, source in _plan_rust_module_tree(init_path, prefixes).items():
        write_text_atomic(mod_rs, source)
