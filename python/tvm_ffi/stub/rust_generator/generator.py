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
"""The Rust code generator for ``tvm-ffi-stubgen``.

:class:`RustGenerator` implements the :class:`tvm_ffi.stub.generator.Generator`
protocol, delegating the actual rendering to ``rust_generator.codegen``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import consts as C
from .codegen import (
    generate_rust_api_file,
    generate_rust_global_funcs,
    generate_rust_import_section,
    generate_rust_object,
    plan_rust_module_tree,
)
from .consts import RUST_TY_MAP_DEFAULTS
from .utils import RustImports, RustUse, generated_rust_type_path

if TYPE_CHECKING:
    from pathlib import Path

    from ..file_utils import CodeBlock
    from ..utils import FuncInfo, InitConfig, ObjectInfo, Options


class RustGenerator:
    """Generator that emits Rust binding stubs.

    Generated object nodes are opaque prefix markers. Field reads, reflected
    methods, and explicitly reflected constructors delegate to the loaded runtime,
    so Rust never mirrors or allocates a complete C++ object layout. Schemas
    without a native Rust sum/container representation remain available through
    type-erased ``Any`` carriers rather than being silently omitted.
    """

    name = "rust"
    syntax = C.RUST_SYNTAX

    def default_ty_map(self) -> dict[str, str]:
        """Return the default FFI-origin -> Rust-type name map."""
        return RUST_TY_MAP_DEFAULTS.copy()

    def new_imports(
        self,
        known_type_keys: set[str],
        *,
        local_type_keys: set[str],
        canonical_type_keys: set[str],
        module_segments: tuple[str, ...] | None,
    ) -> RustImports:
        """Create an empty Rust ``use`` collector."""
        return RustImports(
            known_type_keys=known_type_keys,
            local_type_keys=local_type_keys,
            canonical_type_keys=canonical_type_keys,
            module_segments=module_segments,
        )

    def add_imported_object(
        self, imports: RustImports, name: str, type_checking_only: str, alias: str
    ) -> None:
        """Record an ``import-object`` directive as a ``use``.

        Rust has no ``TYPE_CHECKING`` split, but preserves an explicit alias as
        ``use <path> as <alias>``.
        """
        _ = type_checking_only
        imports.record(name, alias or None)

    def canonical_type_name(self, type_key: str) -> str:
        """Return the Rust path for a defined type key (matches :attr:`RustUse.path`)."""
        return RustUse(generated_rust_type_path(type_key)).path

    def extra_export_names(self, imports: RustImports) -> set[str]:
        """No extra export names for Rust."""
        return set()

    def generated_item_names(self, imports: RustImports) -> set[str]:
        """Return items occupying Rust's module type namespace."""
        return set(imports.local_items)

    def generate_global_funcs_block(
        self,
        code: CodeBlock,
        global_funcs: list[FuncInfo],
        ty_map: dict[str, str],
        imports: RustImports,
        opt: Options,
    ) -> None:
        """Emit typed wrappers around dynamically registered global functions."""
        generate_rust_global_funcs(code, global_funcs, ty_map, imports, opt)

    def generate_object_block(
        self,
        code: CodeBlock,
        ty_map: dict[str, str],
        imports: RustImports,
        opt: Options,
        obj_info: ObjectInfo,
    ) -> None:
        """Emit a Rust ``struct``/``impl`` binding for an ``object/<key>`` block."""
        generate_rust_object(code, ty_map, imports, opt, obj_info)

    def generate_import_section_block(
        self, code: CodeBlock, imports: RustImports, opt: Options, defined_types: set[str]
    ) -> None:
        """Emit Rust ``use`` statements for the collected imports."""
        generate_rust_import_section(code, imports, opt, defined_types)

    def generate_all_block(self, code: CodeBlock, names: set[str], opt: Options) -> None:
        """No-op for now: Rust re-exports are not generated."""

    def generate_export_block(self, code: CodeBlock) -> None:
        """No-op for now: submodule re-export is not generated."""

    def api_filename(self) -> str:
        """One Rust file per module prefix."""
        return "mod.rs"

    def init_filename(self) -> str:
        """No separate entry file for Rust; reuse the API file."""
        return "mod.rs"

    def generate_api_file(
        self,
        code_blocks: list[CodeBlock],
        ty_map: dict[str, str],
        module_name: str,
        object_infos: list[ObjectInfo],
        init_cfg: InitConfig,
        is_root: bool,
    ) -> str:
        """Scaffold a Rust binding file: header + object/import markers."""
        return generate_rust_api_file(
            code_blocks,
            ty_map,
            module_name,
            object_infos,
            init_cfg,
            is_root,
            self.syntax,
        )

    def generate_init_file(
        self, code_blocks: list[CodeBlock], module_name: str, submodule: str
    ) -> str:
        """No-op: Rust has no separate package-entry file (the API file IS the module)."""
        return ""

    def plan_init(
        self,
        init_path: Path,
        generated_prefixes: set[str],
        overlay: dict[Path, str],
        generated_items: dict[Path, set[str]],
    ) -> dict[Path, str]:
        """Plan ``pub mod <child>;`` wiring against staged generated files."""
        return plan_rust_module_tree(
            init_path,
            generated_prefixes,
            overlay,
            generated_items,
        )
