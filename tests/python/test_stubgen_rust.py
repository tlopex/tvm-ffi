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

import os
import stat
from pathlib import Path

import pytest
import tvm_ffi.stub.cli as stub_cli
import tvm_ffi.stub.rust_generator.codegen as rust_codegen
from tvm_ffi.core import TypeSchema
from tvm_ffi.stub import consts as C
from tvm_ffi.stub.cli import _commit_files, _stage_2, _validate_init
from tvm_ffi.stub.file_utils import CodeBlock, FileInfo, write_text_atomic
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
    _rust_string_literal,
    render_rust_type,
)
from tvm_ffi.stub.utils import (
    FuncInfo,
    InitConfig,
    InitFieldInfo,
    NamedTypeSchema,
    ObjectInfo,
    Options,
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


def _generate_object(info: ObjectInfo) -> str:
    block = _block("object", info.type_key or "test.Missing")
    generate_rust_object(
        block,
        RUST_TY_MAP_DEFAULTS.copy(),
        RustImports(),
        Options(target="rust"),
        info,
    )
    return "\n".join(block.lines)


def _generate_globals(functions: list[FuncInfo]) -> str:
    block = _block("global", ("test", ""))
    generate_rust_global_funcs(
        block,
        functions,
        RUST_TY_MAP_DEFAULTS.copy(),
        RustImports(),
        Options(target="rust"),
    )
    return "\n".join(block.lines)


def test_any_container_uses_type_erased_value() -> None:
    imports = RustImports()

    def render(origin: str) -> str:
        return imports.record(RUST_TY_MAP_DEFAULTS[origin])

    result = render_rust_type(TypeSchema("Array", (TypeSchema("Any"),)), render)
    assert result == "Array<AnyValue>"


def test_nested_optional_schema_uses_single_wire_option() -> None:
    imports = RustImports()

    result = render_rust_type(
        TypeSchema(
            "Optional",
            (TypeSchema("Optional", (TypeSchema("test.Node"),)),),
        ),
        imports.record,
    )

    assert result == "Option<Node>"


def test_nullable_bare_function_metadata_remains_packed() -> None:
    schema = _parse_func_type_schema({"type": "Optional", "args": [{"type": "ffi.Function"}]})
    source = _generate_globals([FuncInfo.from_schema("test.DynamicCall", schema)])

    assert "pub fn dynamic_call_packed(args: &[AnyView<'_>]) -> Result<Any>" in source
    assert "pub fn dynamic_call()" not in source


def test_generated_object_is_opaque_and_reflection_backed() -> None:
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
    assert "build_obj" not in source


def test_auto_constructor_is_named_kwargs_builder() -> None:
    optional = NamedTypeSchema("span", TypeSchema("Optional", (TypeSchema("test.Span"),)))
    required = NamedTypeSchema("value", TypeSchema("int"))
    info = ObjectInfo(
        fields=[optional, required],
        methods=[],
        type_key="test.Node",
        parent_type_key="ffi.Object",
        init_fields=[
            InitFieldInfo("span", optional, kw_only=True, has_default=True),
            InitFieldInfo("value", required, kw_only=False, has_default=False),
        ],
        has_init=True,
    )
    source = _generate_object(info)

    assert "span: Option<Option<Span>>" in source
    assert "pub fn ffi_new_unchecked() -> NodeBuilder" in source
    assert "pub fn with_span(mut self, value: Option<Span>) -> Self" in source
    assert "pub fn with_value(mut self, value: i64) -> Self" in source
    assert "pub unsafe fn build_unchecked(self) -> Result<Node>" in source
    assert "get_kwargs_object()?" in source
    assert 'String::from("span")' in source
    assert "field `value` is not set" in source
    assert "pub fn ffi_new(" not in source
    assert "pub fn build(" not in source
    assert "ObjectArc::new" not in source


def test_builder_setter_normalizes_dunder_field_name() -> None:
    field = NamedTypeSchema("__dict__", TypeSchema("int"))
    source = _generate_object(
        ObjectInfo(
            fields=[field],
            methods=[],
            type_key="test.Node",
            parent_type_key="ffi.Object",
            init_fields=[InitFieldInfo("__dict__", field, kw_only=True, has_default=False)],
            has_init=True,
        )
    )

    assert "pub fn with_dict(" in source
    assert "with___dict__" not in source


def test_overloaded_methods_receive_stable_names() -> None:
    methods = [
        FuncInfo.from_schema("run", TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int")))),
        FuncInfo.from_schema(
            "run", TypeSchema("Callable", (TypeSchema("int"), TypeSchema("float")))
        ),
    ]
    source = _generate_object(
        ObjectInfo(
            fields=[],
            methods=methods,
            type_key="test.Runner",
            parent_type_key="ffi.Object",
        )
    )
    assert "pub fn run_overload_1" in source
    assert "pub fn run_overload_2" in source


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
        RustImports(),
        Options(target="rust"),
    )
    source = "\n".join(block.lines)
    assert "pub fn skip_assert(_0: Node) -> Result<Node>" in source
    assert 'get_global_cached(&F, "test.transform.SkipAssert")' in source


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

    assert "pub fn convert(_0: AnyView) -> Result<Any>" in source
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


def test_generated_type_key_uses_escaped_rust_string_literal() -> None:
    source = _generate_object(
        ObjectInfo(
            fields=[],
            methods=[],
            type_key='test"slash\\.Node',
            parent_type_key="ffi.Object",
        )
    )

    assert '#[type_key = "test\\"slash\\\\.Node"]' in source


def test_rust_api_scaffold_has_license() -> None:
    source = generate_rust_api_file(
        [],
        RUST_TY_MAP_DEFAULTS.copy(),
        "test",
        [],
        InitConfig(pkg="test", shared_target="test", prefix="test"),
        is_root=True,
        syntax=C.RUST_SYNTAX,
    )

    assert source.startswith(RUST_LICENSE_HEADER)
    assert source.count("Licensed to the Apache Software Foundation") == 1
    assert "#![allow(clippy::all, dead_code, unused_imports)]" in source
    assert f"{C.RUST_SYNTAX.begin} import-section" in source
    assert f"{C.RUST_SYNTAX.begin} global/test" in source


def test_atomic_writer_uses_source_file_modes(tmp_path: Path) -> None:
    if os.name != "posix":
        return
    path = tmp_path / "generated.rs"

    write_text_atomic(path, "first\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o644

    path.chmod(0o600)
    write_text_atomic(path, "second\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_rust_init_scaffolding_stays_in_memory_and_deduplicates_mod_rs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(stub_cli, "collect_type_keys", lambda: {"test": []})
    monkeypatch.setattr(stub_cli, "toposort_objects", lambda objects: [])
    function = FuncInfo.from_schema(
        "test.Identity",
        TypeSchema("Callable", (TypeSchema("int"), TypeSchema("int"))),
    )
    files: list[FileInfo] = []

    prefixes = _stage_2(
        files,
        RUST_TY_MAP_DEFAULTS.copy(),
        InitConfig(pkg="test", shared_target="test", prefix="test"),
        tmp_path,
        {"test": [function]},
        get_generator("rust"),
    )

    assert prefixes == {"test"}
    assert len(files) == 1
    assert files[0].path == tmp_path / "test" / "mod.rs"
    assert not files[0].path.exists()
    source = "\n".join(line for block in files[0].code_blocks for line in block.lines)
    assert source.startswith(RUST_LICENSE_HEADER)
    assert f"{C.RUST_SYNTAX.begin} global/test" in source


def test_module_finalization_is_licensed_sorted_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finalize_rust_module_tree(tmp_path, {"test.zeta", "test.alpha"})
    root_mod = tmp_path / "mod.rs"
    test_mod = tmp_path / "test" / "mod.rs"
    first_root = root_mod.read_text(encoding="utf-8")
    first_test = test_mod.read_text(encoding="utf-8")

    writes: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        rust_codegen,
        "write_text_atomic",
        lambda path, source: writes.append((path, source)),
    )
    finalize_rust_module_tree(tmp_path, {"test.alpha", "test.zeta"})

    assert root_mod.read_text(encoding="utf-8") == first_root
    assert test_mod.read_text(encoding="utf-8") == first_test
    assert writes == []
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


def test_module_finalization_preserves_managed_siblings_across_prefix_runs(
    tmp_path: Path,
) -> None:
    finalize_rust_module_tree(tmp_path, {"ir"})
    finalize_rust_module_tree(tmp_path, {"tirx.transform"})
    finalize_rust_module_tree(tmp_path, {"arith"})

    root_source = (tmp_path / "mod.rs").read_text(encoding="utf-8")
    assert "pub mod arith;" in root_source
    assert "pub mod ir;" in root_source
    assert "pub mod tirx;" in root_source
    assert "pub mod transform;" in (tmp_path / "tirx" / "mod.rs").read_text(encoding="utf-8")


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


@pytest.mark.parametrize(
    ("strict", "dry_run", "failed", "should_write"),
    [
        (True, False, True, False),
        (True, True, False, False),
        (False, False, True, True),
    ],
)
def test_commit_policy_preserves_strict_and_dry_run_transactionality(
    tmp_path: Path,
    strict: bool,
    dry_run: bool,
    failed: bool,
    should_write: bool,
) -> None:
    path = tmp_path / "generated.rs"
    original = f"{C.RUST_SYNTAX.begin} global/test\n{C.RUST_SYNTAX.end}\n"
    path.write_text(original, encoding="utf-8")
    file = FileInfo.from_file(path)
    assert file is not None
    file.code_blocks[0].lines.insert(1, "pub fn generated() {}")

    committed = _commit_files(
        [file],
        Options(target="rust", strict=strict, dry_run=dry_run),
        failed=failed,
    )

    assert committed is should_write
    if should_write:
        assert "pub fn generated() {}" in path.read_text(encoding="utf-8")
    else:
        assert path.read_text(encoding="utf-8") == original


def test_strict_failure_and_dry_run_do_not_create_scaffold_files(tmp_path: Path) -> None:
    for options, failed in [
        (Options(target="rust", strict=True), True),
        (Options(target="rust", strict=True, dry_run=True), False),
    ]:
        path = tmp_path / ("strict.rs" if failed else "dry-run.rs")
        file = FileInfo.from_text(
            path,
            f"{C.RUST_SYNTAX.begin} global/test\n{C.RUST_SYNTAX.end}\n",
            include_empty=True,
        )
        assert file is not None

        assert not _commit_files([file], options, failed)
        assert not path.exists()


def test_finalization_preflight_blocks_allow_partial_writes(tmp_path: Path) -> None:
    path = tmp_path / "generated.rs"
    original = f"{C.RUST_SYNTAX.begin} global/test\n{C.RUST_SYNTAX.end}\n"
    path.write_text(original, encoding="utf-8")
    file = FileInfo.from_file(path)
    assert file is not None
    file.code_blocks[0].lines.insert(1, "pub fn generated() {}")

    committed = _commit_files(
        [file],
        Options(target="rust", strict=False),
        failed=True,
        block_writes=True,
    )

    assert not committed
    assert path.read_text(encoding="utf-8") == original


def test_strict_finalization_preflight_fails_before_generated_file_commit(
    tmp_path: Path,
) -> None:
    module_file = tmp_path / "mod.rs"
    module_file.write_text("// @tvm-ffi-stubgen-rust-modules(begin)\n", encoding="utf-8")
    generated_file = tmp_path / "bindings.rs"
    original = f"{C.RUST_SYNTAX.begin} global/test\n{C.RUST_SYNTAX.end}\n"
    generated_file.write_text(original, encoding="utf-8")
    file = FileInfo.from_file(generated_file)
    assert file is not None
    file.code_blocks[0].lines.insert(1, "pub fn generated() {}")
    options = Options(
        target="rust",
        strict=True,
        init=InitConfig(pkg="test", shared_target="test", prefix="test"),
    )

    failed = _validate_init(True, options, tmp_path, {"test"}, get_generator("rust"))
    committed = _commit_files([file], options, failed, block_writes=failed)

    assert failed
    assert not committed
    assert generated_file.read_text(encoding="utf-8") == original
