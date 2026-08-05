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
"""Safety and API-shape tests for the Rust stub generator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
import tvm_ffi.stub.rust_generator.codegen as rust_codegen
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo
from tvm_ffi.stub.generator import get_generator
from tvm_ffi.stub.rust_generator.codegen import (
    RUST_LICENSE_HEADER,
    finalize_rust_module_tree,
    generate_rust_api_file,
    generate_rust_global_funcs,
    generate_rust_object,
)
from tvm_ffi.stub.rust_generator.consts import RUST_TY_MAP_DEFAULTS
from tvm_ffi.stub.rust_generator.utils import (
    RustImports,
    RustUse,
    _rust_string_literal,
    render_rust_type,
)
from tvm_ffi.stub.utils import (
    FuncInfo,
    InitConfig,
    NamedTypeSchema,
    ObjectInfo,
    Options,
    UnsupportedTypeError,
    _parse_func_type_schema,
)


def _block(kind: str, param: str | tuple[str, str]) -> CodeBlock:
    marker_param = param[0] if isinstance(param, tuple) else param
    return CodeBlock(
        kind=kind,  # type: ignore[arg-type]
        param=param,
        lineno_start=1,
        lineno_end=2,
        lines=[
            f"// tvm-ffi-stubgen(begin): {kind}/{marker_param}",
            "// tvm-ffi-stubgen(end)",
        ],
    )


def _generate_object(info: ObjectInfo, known_type_keys: set[str] | None = None) -> str:
    block = _block("object", info.type_key or "test.Missing")
    known_type_keys = known_type_keys or {info.type_key or "test.Missing"}
    generate_rust_object(
        block,
        RUST_TY_MAP_DEFAULTS.copy(),
        RustImports(
            known_type_keys=known_type_keys,
            local_type_keys=known_type_keys,
            canonical_type_keys=known_type_keys,
            module_segments=("test",),
        ),
        Options(target="rust"),
        info,
    )
    return "\n".join(block.lines)


def _generate_globals(functions: list[FuncInfo], known_type_keys: set[str] | None = None) -> str:
    block = _block("global", ("test", ""))
    generate_rust_global_funcs(
        block,
        functions,
        RUST_TY_MAP_DEFAULTS.copy(),
        RustImports(
            known_type_keys=known_type_keys or set(),
            canonical_type_keys=known_type_keys or set(),
            module_segments=("test",),
        ),
        Options(target="rust"),
    )
    return "\n".join(block.lines)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("i64", "i64"),
        ("()", "()"),
        ("ffi.String", "tvm_ffi::String"),
        ("crate::match::type", "crate::r#match::r#type"),
        ("super::super::test::Node", "super::super::test::Node"),
    ],
)
def test_rust_use_normalizes_only_valid_paths(source: str, expected: str) -> None:
    assert RustUse(source).path == expected


@pytest.mark.parametrize(
    "source",
    [
        "bad-name",
        "crate::bad-name::T",
        "foo::::T",
        "foo..T",
        "foo::self::T",
        "Option<i64>",
    ],
)
def test_rust_use_rejects_unchecked_or_malformed_source(source: str) -> None:
    with pytest.raises(UnsupportedTypeError):
        RustUse(source)


def test_any_container_uses_type_erased_value() -> None:
    imports = RustImports()

    def render(origin: str) -> str:
        return imports.record(RUST_TY_MAP_DEFAULTS[origin])

    result = render_rust_type(TypeSchema("Array", (TypeSchema("Any"),)), render)
    assert result == "Array<AnyValue>"


def test_nested_optional_schema_uses_single_wire_option() -> None:
    imports = RustImports()

    def render(origin: str) -> str:
        return imports.record(RUST_TY_MAP_DEFAULTS.get(origin, origin.replace(".", "::")))

    result = render_rust_type(
        TypeSchema(
            "Optional",
            (TypeSchema("Optional", (TypeSchema("test.Node"),)),),
        ),
        render,
    )

    assert result == "Option<Node>"


def test_typed_expr_schema_preserves_result_type_refinement() -> None:
    imports = RustImports()

    def render(origin: str) -> str:
        path = RUST_TY_MAP_DEFAULTS.get(origin, origin.replace(".", "::"))
        return imports.record(path)

    schema = TypeSchema(
        "Optional",
        (
            TypeSchema(
                "TypedExpr",
                (TypeSchema("test.Expr"), TypeSchema("test.PrimType")),
            ),
        ),
    )

    assert (
        render_rust_type(schema, render, lambda origin: origin.startswith("test."))
        == "Option<TypedExpr<Expr, PrimType>>"
    )
    assert {item.path for item in imports.items} == {
        "std::option::Option",
        "tvm_ffi::TypedExpr",
        "test::Expr",
        "test::PrimType",
    }


@pytest.mark.parametrize(
    ("base", "expected", "role"),
    [
        (TypeSchema("int"), TypeSchema("test.Type"), "base"),
        (
            TypeSchema("test.Expr"),
            TypeSchema("Optional", (TypeSchema("test.Type"),)),
            "expected type",
        ),
        (
            TypeSchema("TypedExpr", (TypeSchema("test.Expr"), TypeSchema("test.Type"))),
            TypeSchema("test.Type"),
            "base",
        ),
    ],
)
def test_typed_expr_rejects_non_object_ref_operands(
    base: TypeSchema, expected: TypeSchema, role: str
) -> None:
    with pytest.raises(UnsupportedTypeError, match=rf"TypedExpr {role} must be"):
        render_rust_type(
            TypeSchema("TypedExpr", (base, expected)),
            RustImports().record,
            lambda origin: origin.startswith("test."),
        )


def test_nullable_bare_function_metadata_remains_packed() -> None:
    schema = _parse_func_type_schema({"type": "Optional", "args": [{"type": "ffi.Function"}]})
    source = _generate_globals([FuncInfo.from_schema("test.DynamicCall", schema)])

    assert "pub fn dynamic_call_packed(args: &[AnyView<'_>]) -> Result<Any>" in source
    assert "pub fn dynamic_call()" not in source


def test_object_api_exposes_only_proven_safe_operations() -> None:
    info = ObjectInfo(
        fields=[NamedTypeSchema("value", TypeSchema("int"), size=4, alignment=4, offset=24)],
        methods=[],
        type_key="test.Node",
        parent_type_key="ffi.Object",
    )
    source = _generate_object(info)

    assert "pub struct NodeObj {\n    base: Object," in source
    assert "_not_send_sync: PhantomData<Rc<()>>" in source
    assert "pub value:" not in source
    assert "get_object_field::<i64, _>" in source
    assert "ObjectArc::new" not in source
    assert "DerefMut" not in source
    assert "Builder" not in source
    assert "build_unchecked" not in source
    assert "pub fn same_as" not in source
    assert "pub fn downcast" not in source

    # Keep the marker on derived objects too: a standalone generation request
    # may inherit from an external parent whose auto-trait contract is unknown.
    derived = _generate_object(
        ObjectInfo(
            fields=[],
            methods=[],
            type_key="test.Child",
            parent_type_key="test.Node",
        ),
        {"test.Node", "test.Child"},
    )
    assert "base: NodeObj" in derived
    assert "_not_send_sync: PhantomData<Rc<()>>" in derived


def test_constructor_requires_an_explicit_reflected_initializer() -> None:
    metadata_only = ObjectInfo(
        fields=[],
        methods=[],
        type_key="test.Node",
        parent_type_key="ffi.Object",
        has_init=True,
    )
    assert "pub fn ffi_new(" not in _generate_object(metadata_only)

    explicit = ObjectInfo(
        fields=[],
        methods=[
            FuncInfo.from_schema(
                "__ffi_init__",
                TypeSchema("Callable", (TypeSchema("Object"), TypeSchema("int"))),
                is_member=False,
            )
        ],
        type_key="test.Node",
        parent_type_key="ffi.Object",
        has_init=True,
    )
    source = _generate_object(explicit)
    assert "pub fn ffi_new(_0: i64) -> Result<Node>" in source
    assert 'from_type_method_cached(&F, NodeObj::type_index(), "__ffi_init__")' in source


@pytest.mark.parametrize(
    "constructor",
    [
        FuncInfo.from_schema("__ffi_init__", TypeSchema("Callable"), is_member=False),
        FuncInfo.from_schema(
            "__ffi_init__",
            TypeSchema("Callable", (TypeSchema("test.Node"),)),
            is_member=True,
        ),
    ],
)
def test_invalid_reflected_constructor_fails_closed(constructor: FuncInfo) -> None:
    info = ObjectInfo(
        fields=[],
        methods=[constructor],
        type_key="test.Node",
        parent_type_key="ffi.Object",
        has_init=True,
    )

    with pytest.raises(UnsupportedTypeError, match="typed static factory"):
        _generate_object(info)


def test_global_wrapper_is_typed_and_cached() -> None:
    block = _block("global", ("test.transform", ""))
    function = FuncInfo.from_schema(
        "test.transform.SkipAssert",
        TypeSchema("Callable", (TypeSchema("test.Node"), TypeSchema("test.Node"))),
    )
    generate_rust_global_funcs(
        block,
        [function],
        RUST_TY_MAP_DEFAULTS.copy(),
        RustImports(
            known_type_keys={"test.Node"},
            canonical_type_keys={"test.Node"},
            module_segments=("test", "transform"),
        ),
        Options(target="rust"),
    )
    source = "\n".join(block.lines)
    assert (
        "pub fn skip_assert(_0: super::super::test::Node) -> Result<super::super::test::Node>"
    ) in source
    assert 'get_global_cached(&F, "test.transform.SkipAssert")' in source


def test_global_generation_rejects_malformed_prefix_before_mutating_block() -> None:
    block = _block("global", ("test..transform", ""))
    original = block.lines.copy()

    with pytest.raises(UnsupportedTypeError, match="module prefixes"):
        generate_rust_global_funcs(
            block,
            [],
            RUST_TY_MAP_DEFAULTS.copy(),
            RustImports(),
            Options(target="rust"),
        )

    assert block.lines == original


def test_unsupported_global_schema_aborts_generation() -> None:
    function = FuncInfo.from_schema(
        "test.Unsupported",
        TypeSchema("Callable", (TypeSchema("mystery"),)),
    )

    with pytest.raises(UnsupportedTypeError, match="unsupported FFI type 'mystery'"):
        _generate_globals([function])


def test_union_and_tuple_globals_use_type_erased_boundary() -> None:
    function = FuncInfo.from_schema(
        "test.convert",
        TypeSchema(
            "Callable",
            (
                TypeSchema("Union", (TypeSchema("int"), TypeSchema("float"))),
                TypeSchema("tuple", (TypeSchema("int"), TypeSchema("float"))),
            ),
        ),
    )
    source = _generate_globals([function])

    assert "pub fn convert(_0: AnyView<'_>) -> Result<Any>" in source
    assert "f.call_packed(&[_0])" in source
    assert "Union" not in source
    assert "tuple" not in source


def test_global_without_schema_is_explicit_packed_wrapper() -> None:
    source = _generate_globals([FuncInfo.from_schema("test.DynamicCall", TypeSchema("Callable"))])

    assert "pub fn dynamic_call_packed(args: &[AnyView<'_>]) -> Result<Any>" in source
    assert "f.call_packed(args)" in source
    assert "pub fn dynamic_call()" not in source


def test_method_name_collisions_are_unique_and_stable() -> None:
    callable_schema = TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int")))
    methods = [
        FuncInfo.from_schema("run", callable_schema),
        FuncInfo.from_schema("run", callable_schema),
        FuncInfo.from_schema("run_overload_1", callable_schema),
        FuncInfo.from_schema("same_as", callable_schema),
        FuncInfo.from_schema("same_as_method", callable_schema),
    ]
    source = _generate_object(
        ObjectInfo(
            fields=[],
            methods=methods,
            type_key="test.Runner",
            parent_type_key="ffi.Object",
        )
    )

    assert source.count("pub fn run_overload_1(") == 1
    assert source.count("pub fn run_overload_1_2(") == 1
    assert source.count("pub fn run_overload_2(") == 1
    assert source.count("pub fn same_as_method(") == 1
    assert source.count("pub fn same_as_method_2(") == 1


def test_global_name_collisions_are_unique_and_stable() -> None:
    callable_schema = TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int")))
    source = _generate_globals(
        [
            FuncInfo.from_schema("test.Run", callable_schema),
            FuncInfo.from_schema("test.Run", callable_schema),
            FuncInfo.from_schema("test.run_overload_1", callable_schema),
        ]
    )

    assert source.count("pub fn run_overload_1(") == 1
    assert source.count("pub fn run_overload_1_2(") == 1
    assert source.count("pub fn run_overload_2(") == 1


def test_rust_string_literal_escapes_reflected_text() -> None:
    reflected = 'quote" slash\\\n\r\t\0\x01\x7f snowman ☃'
    assert (
        _rust_string_literal(reflected) == '"quote\\" slash\\\\\\n\\r\\t\\0\\u{1}\\u{7f} snowman ☃"'
    )


@pytest.mark.parametrize("type_key", ['test"slash\\.Node', "test..Node", "test.Self"])
def test_invalid_generated_type_key_fails_closed(type_key: str) -> None:
    with pytest.raises(UnsupportedTypeError):
        _generate_object(
            ObjectInfo(
                fields=[],
                methods=[],
                type_key=type_key,
                parent_type_key="ffi.Object",
            )
        )


def test_rust_api_scaffold_has_license() -> None:
    source = generate_rust_api_file(
        [],
        RUST_TY_MAP_DEFAULTS.copy(),
        "test",
        [],
        InitConfig(pkg="", shared_target="", prefixes=("test",)),
        is_root=True,
        syntax=C.RUST_SYNTAX,
    )

    assert source.startswith(RUST_LICENSE_HEADER)
    assert source.count("Licensed to the Apache Software Foundation") == 1
    assert "#![allow(dead_code, non_camel_case_types, non_snake_case)]" in source
    assert "clippy::all" not in source
    assert "unused_imports" not in source
    assert f"{C.RUST_SYNTAX.begin} import-section" in source
    assert f"{C.RUST_SYNTAX.begin} global/test" in source


def test_rust_cli_parsing_needs_only_reflection_prefixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tvm-ffi-stubgen",
            str(tmp_path),
            "--target",
            "rust",
            "--init-prefix",
            "ir.",
            "--init-prefix",
            "tirx.",
        ],
    )

    options = stub_cli._parse_args()

    assert options.init is not None
    assert options.init.normalized_prefixes() == ("ir", "tirx")


def test_mocked_registry_multi_prefix_generation_is_cross_referenced_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    infos = {
        "alpha.Node": ObjectInfo(
            fields=[NamedTypeSchema("peer", TypeSchema("beta.Node"))],
            methods=[],
            type_key="alpha.Node",
            parent_type_key="ffi.Object",
        ),
        "beta.Node": ObjectInfo(
            fields=[NamedTypeSchema("peer", TypeSchema("alpha.Node"))],
            methods=[],
            type_key="beta.Node",
            parent_type_key="ffi.Object",
        ),
    }
    options = Options(
        target="rust",
        files=[str(tmp_path)],
        init=InitConfig(pkg="", shared_target="", prefixes=("beta", "alpha")),
    )

    def collect_types(_exact: tuple[str, ...], recursive: set[str]) -> dict[str, list[str]]:
        assert recursive == {"alpha", "beta"}
        return {"beta": ["beta.Node"], "alpha": ["alpha.Node"]}

    monkeypatch.setattr(stub_cli, "_parse_args", lambda: options)
    monkeypatch.setattr(stub_cli, "collect_global_funcs", lambda *_args: {})
    monkeypatch.setattr(stub_cli, "collect_type_keys", collect_types)
    monkeypatch.setattr(stub_cli, "toposort_objects", lambda keys: [infos[key] for key in keys])
    monkeypatch.setattr(stub_cli, "object_info_from_type_key", infos.__getitem__)

    assert stub_cli.__main__() == 0
    generated_paths = [
        tmp_path / "mod.rs",
        tmp_path / "alpha" / "mod.rs",
        tmp_path / "beta" / "mod.rs",
    ]
    first = {path: path.read_bytes() for path in generated_paths}
    root_source = first[tmp_path / "mod.rs"].decode()
    assert (
        "// @tvm-ffi-stubgen-rust-modules(begin)\n"
        "pub mod alpha;\n"
        "pub mod beta;\n"
        "// @tvm-ffi-stubgen-rust-modules(end)"
    ) in root_source
    assert "Result<super::beta::Node>" in first[tmp_path / "alpha" / "mod.rs"].decode()
    assert "Result<super::alpha::Node>" in first[tmp_path / "beta" / "mod.rs"].decode()

    assert stub_cli.__main__() == 0
    assert {path: path.read_bytes() for path in generated_paths} == first


def test_module_finalization_is_licensed_sorted_and_idempotent(tmp_path: Path) -> None:
    finalize_rust_module_tree(tmp_path, {"test.zeta", "test.alpha"})
    root_mod = tmp_path / "mod.rs"
    test_mod = tmp_path / "test" / "mod.rs"
    first_root = root_mod.read_text(encoding="utf-8")
    first_test = test_mod.read_text(encoding="utf-8")

    finalize_rust_module_tree(tmp_path, {"test.alpha", "test.zeta"})

    assert root_mod.read_text(encoding="utf-8") == first_root
    assert test_mod.read_text(encoding="utf-8") == first_test
    assert first_root.count("Licensed to the Apache Software Foundation") == 1
    assert first_test.count("Licensed to the Apache Software Foundation") == 1
    assert "pub mod test;" in first_root
    assert first_test.index("pub mod alpha;") < first_test.index("pub mod zeta;")


def test_module_finalization_preserves_external_module_declarations(tmp_path: Path) -> None:
    root_mod = tmp_path / "mod.rs"
    test_mod = tmp_path / "test" / "mod.rs"
    test_mod.parent.mkdir()
    root_mod.write_text("// user root\npub mod test;\n", encoding="utf-8")
    test_mod.write_text("// user child\npub mod alpha;\n", encoding="utf-8")

    finalize_rust_module_tree(tmp_path, {"test.alpha", "test.zeta"})

    root_source = root_mod.read_text(encoding="utf-8")
    test_source = test_mod.read_text(encoding="utf-8")
    assert root_source.startswith("// user root\npub mod test;\n")
    assert root_source.count("pub mod test;") == 1
    assert test_source.startswith("// user child\npub mod alpha;\n")
    assert test_source.count("pub mod alpha;") == 1
    assert test_source.count("pub mod zeta;") == 1


@pytest.mark.parametrize(("child", "local"), [("Child", "Child"), ("match", "r#match")])
def test_module_planning_rejects_type_child_namespace_collision(
    tmp_path: Path, child: str, local: str
) -> None:
    module = tmp_path / "test" / "mod.rs"
    with pytest.raises(UnsupportedTypeError, match="module/type namespace collision"):
        rust_codegen.plan_rust_module_tree(
            tmp_path,
            {f"test.{child}"},
            overlay={module: "// generated object module\n"},
            generated_items={module: {local}},
        )


def test_module_finalization_manages_exactly_the_current_children(tmp_path: Path) -> None:
    finalize_rust_module_tree(tmp_path, {"old", "tirx.stale"})
    finalize_rust_module_tree(tmp_path, {"ir"})

    root_source = (tmp_path / "mod.rs").read_text(encoding="utf-8")
    managed = root_source.split("// @tvm-ffi-stubgen-rust-modules(begin)\n", 1)[1]
    managed = managed.split("// @tvm-ffi-stubgen-rust-modules(end)", 1)[0]
    assert managed.splitlines() == ["pub mod ir;"]


@pytest.mark.parametrize(
    "malformed",
    [
        "// @tvm-ffi-stubgen-rust-modules(begin)\n",
        "// @tvm-ffi-stubgen-rust-modules(end)\n// @tvm-ffi-stubgen-rust-modules(begin)\n",
        "// @tvm-ffi-stubgen-rust-modules(begin)\n"
        "// @tvm-ffi-stubgen-rust-modules(end)\n"
        "// @tvm-ffi-stubgen-rust-modules(begin)\n"
        "// @tvm-ffi-stubgen-rust-modules(end)\n",
    ],
)
def test_module_finalization_rejects_malformed_markers_without_writes(
    tmp_path: Path, malformed: str
) -> None:
    root_mod = tmp_path / "mod.rs"
    root_original = "// root must remain byte-for-byte unchanged\n"
    root_mod.write_text(root_original, encoding="utf-8")
    test_mod = tmp_path / "test" / "mod.rs"
    test_mod.parent.mkdir()
    test_mod.write_text(malformed, encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed Rust stubgen module markers"):
        finalize_rust_module_tree(tmp_path, {"test.child"})

    assert root_mod.read_text(encoding="utf-8") == root_original
    assert test_mod.read_text(encoding="utf-8") == malformed


def test_conflicting_type_maps_report_both_sources_and_do_not_partially_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_path = tmp_path / "first.rs"
    second_path = tmp_path / "second.rs"
    first_source = f"{C.RUST_SYNTAX.ty_map} A -> crate::A\n"
    second_source = (
        f"{C.RUST_SYNTAX.ty_map} B -> crate::B\n{C.RUST_SYNTAX.ty_map} A -> crate::DifferentA\n"
    )
    first_path.write_text(first_source, encoding="utf-8")
    second_path.write_text(second_source, encoding="utf-8")
    files = [FileInfo.from_file(first_path), FileInfo.from_file(second_path)]
    assert all(file is not None for file in files)
    parsed_files = [file for file in files if file is not None]

    ty_map, failed = stub_cli._collect_type_map(parsed_files, get_generator("rust"))

    assert failed
    expected = get_generator("rust").default_ty_map()
    expected["A"] = "crate::A"
    assert ty_map == expected
    diagnostic = capsys.readouterr().out
    assert f"{first_path}:1" in diagnostic
    assert f"{second_path}:2" in diagnostic


@pytest.mark.parametrize("mapping", [" -> crate::X", "X -> "])
def test_type_map_rejects_an_empty_side(tmp_path: Path, mapping: str) -> None:
    file = FileInfo.from_text(
        tmp_path / "invalid.rs",
        f"{C.RUST_SYNTAX.ty_map} {mapping}\n",
        include_empty=True,
    )
    assert file is not None

    ty_map, failed = stub_cli._collect_type_map([file], get_generator("rust"))

    assert failed
    assert ty_map == get_generator("rust").default_ty_map()
