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
"""TVM-FFI Stub Generator (``tvm-ffi-stubgen``)."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from . import consts as C
from .file_utils import (
    FileInfo,
    collect_files,
    print_text_diff,
    syntax_for,
    write_text_atomic,
)
from .generator import get_generator
from .lib_state import (
    collect_global_funcs,
    collect_type_keys,
    object_info_from_type_key,
    toposort_objects,
)
from .rust_generator.utils import split_rust_module_prefix
from .utils import FuncInfo, InitConfig, Options

if TYPE_CHECKING:
    from .generator import Generator


def __main__() -> int:
    """Command line entry point for ``tvm-ffi-stubgen``.

    This generates in-place type stubs inside special ``tvm-ffi-stubgen`` blocks
    in the given files or directories. See the module docstring for an
    overview and examples of the block syntax.
    """
    opt = _parse_args()
    generator = get_generator(opt.target)
    dlls = _load_extensions(opt)
    files, init_path = _collect_inputs(opt)
    marker_prefixes = {
        code.param[0]
        for file in files
        for code in file.code_blocks
        if code.kind == "global" and isinstance(code.param, tuple)
    }
    recursive_prefixes = set(opt.init.normalized_prefixes()) if opt.init is not None else set()
    if opt.target == "rust":
        for prefix in recursive_prefixes:
            split_rust_module_prefix(prefix)
    # Init-generated files contain global markers too. A marker already covered
    # by a recursive root is redundant on an idempotent rerun.
    exact_prefixes = {
        prefix
        for prefix in marker_prefixes
        if not any(prefix == root or prefix.startswith(f"{root}.") for root in recursive_prefixes)
    }
    global_funcs = collect_global_funcs(exact_prefixes, recursive_prefixes)

    # All generation happens against in-memory FileInfo objects. No file is
    # committed if any collection, scaffolding, or rendering step fails.
    ty_map, collect_failed = _collect_type_map(files, generator)
    generated_prefixes, scaffold_failed = _generate_scaffolds(
        files, opt, init_path, ty_map, global_funcs, generator
    )
    render_failed = _render_files(files, opt, ty_map, global_funcs, generator, init_path)
    failed = collect_failed or scaffold_failed or render_failed
    try:
        _commit_files(
            files,
            opt,
            failed,
            init_path=init_path,
            generated_prefixes=generated_prefixes,
            generator=generator,
        )
    except Exception:
        failed = True
        print(f"{C.TERM_RED}[Failed] Commit: {traceback.format_exc()}{C.TERM_RESET}")

    # Keep preloaded libraries alive until every registry-backed stage finishes.
    del dlls
    return int(failed)


def _load_extensions(opt: Options) -> list[ctypes.CDLL]:
    """Load requested Python modules and shared libraries."""
    for imp in opt.imports:
        importlib.import_module(imp)
    return [ctypes.CDLL(lib) for lib in opt.dlls]


def _collect_inputs(opt: Options) -> tuple[list[FileInfo], Path]:
    """Validate target extensions, parse inputs, and resolve the init root."""
    expected_exts = {"python": {".py", ".pyi"}, "rust": {".rs"}}[opt.target]
    input_paths = [Path(path) for path in opt.files]
    mismatched = [
        str(path) for path in input_paths if path.is_file() and path.suffix not in expected_exts
    ]
    if mismatched:
        expected = ", ".join(sorted(expected_exts))
        raise ValueError(
            f"--target {opt.target} only accepts {expected} sources; got: " + ", ".join(mismatched)
        )
    files = [file for file in collect_files(input_paths) if file.path.suffix in expected_exts]
    init_path = input_paths[0].resolve()
    if init_path.is_file():
        init_path = init_path.parent
    return files, init_path


def _print_file_failure(file: FileInfo) -> None:
    """Report the active exception for one input file."""
    print(f'{C.TERM_RED}[Failed] File "{file.path}": {traceback.format_exc()}{C.TERM_RESET}')


def _collect_type_map(files: list[FileInfo], generator: Generator) -> tuple[dict[str, str], bool]:
    """Run stage 1 for every file, retaining independent successes."""
    ty_map = generator.default_ty_map()
    definitions: dict[str, tuple[str, Path, int]] = {}
    failed = False
    for file in files:
        try:
            _stage_1(file, ty_map, definitions)
        except Exception:
            failed = True
            _print_file_failure(file)
    return ty_map, failed


def _generate_scaffolds(
    files: list[FileInfo],
    opt: Options,
    init_path: Path,
    ty_map: dict[str, str],
    global_funcs: dict[str, list[FuncInfo]],
    generator: Generator,
) -> tuple[set[str], bool]:
    """Run optional init scaffolding entirely in memory."""
    if opt.init is None:
        return set(), False
    try:
        prefixes = _stage_2(
            files,
            ty_map,
            init_cfg=opt.init,
            init_path=init_path,
            global_funcs=global_funcs,
            generator=generator,
        )
        return prefixes, False
    except Exception:
        print(f"{C.TERM_RED}[Failed] Scaffolding: {traceback.format_exc()}{C.TERM_RESET}")
        return set(), True


def _render_files(
    files: list[FileInfo],
    opt: Options,
    ty_map: dict[str, str],
    global_funcs: dict[str, list[FuncInfo]],
    generator: Generator,
    init_path: Path | None = None,
) -> bool:
    """Run stage 3 for every file, retaining independent successes."""
    failed = False
    type_locations: dict[str, Path] = {}
    for file in files:
        for code in file.code_blocks:
            if code.kind != "object" or not isinstance(code.param, str):
                continue
            location = file.path.resolve()
            previous = type_locations.setdefault(code.param, location)
            if previous != location:
                print(
                    f"{C.TERM_RED}[Failed] Object {code.param!r} is declared in both "
                    f"{previous} and {location}{C.TERM_RESET}"
                )
                return True

    known_type_keys = set(type_locations)
    module_by_file: dict[Path, tuple[str, ...] | None] = {}
    root = init_path.resolve() if init_path is not None else None
    for file in files:
        module_segments: tuple[str, ...] | None = None
        if opt.target == "rust" and root is not None and file.path.name == "mod.rs":
            try:
                relative = file.path.resolve().relative_to(root)
                module_segments = tuple(relative.parts[:-1])
            except ValueError:
                pass
        module_by_file[file.path.resolve()] = module_segments

    canonical_type_keys = {
        type_key
        for type_key, path in type_locations.items()
        if module_by_file.get(path) == tuple(type_key.split(".")[:-1])
    }
    for file in files:
        local_type_keys = {
            code.param
            for code in file.code_blocks
            if code.kind == "object" and isinstance(code.param, str)
        }
        try:
            _stage_3(
                file,
                opt,
                ty_map,
                global_funcs,
                generator=generator,
                known_type_keys=known_type_keys,
                local_type_keys=local_type_keys,
                canonical_type_keys=canonical_type_keys,
                module_segments=module_by_file[file.path.resolve()],
            )
        except Exception:
            failed = True
            _print_file_failure(file)
    return failed


def _plan_outputs(
    files: list[FileInfo],
    opt: Options,
    init_path: Path | None,
    generated_prefixes: set[str] | None,
    generator: Generator | None,
) -> dict[Path, str]:
    """Compose generated source and module-tree outputs without writing."""
    outputs = {file.path.resolve(): file.rendered_text() for file in files}
    prefixes = generated_prefixes or set()
    if opt.init is not None and prefixes:
        if init_path is None or generator is None:
            raise ValueError("init generation requires an init path and language generator")
        generated_items = {file.path.resolve(): file.generated_items for file in files}
        module_outputs = generator.plan_init(
            init_path,
            prefixes,
            outputs,
            generated_items,
        )
        module_outputs = {path.resolve(): source for path, source in module_outputs.items()}
        outputs.update(module_outputs)
    return outputs


def _changed_outputs(
    outputs: dict[Path, str],
) -> dict[Path, str]:
    """Filter a fully rendered plan to files whose bytes would change."""
    return {
        path: source
        for path, source in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != source
    }


def _preview_outputs(outputs: dict[Path, str]) -> None:
    """Print final planned source and module-tree outputs."""
    for path, source in sorted(outputs.items(), key=lambda item: str(item[0])):
        before = tuple(path.read_text(encoding="utf-8").splitlines()) if path.exists() else ()
        print_text_diff(path, before, tuple(source.splitlines()))


def _commit_files(
    files: list[FileInfo],
    opt: Options,
    failed: bool,
    *,
    init_path: Path | None = None,
    generated_prefixes: set[str] | None = None,
    generator: Generator | None = None,
) -> bool:
    """Plan every output, then replace each changed file atomically."""
    if failed:
        # A failed generation may still show its successfully rendered files,
        # but no module-tree plan is valid and nothing is committed.
        if opt.verbose or opt.dry_run:
            for file in files:
                print_text_diff(file.path, file.lines, file.rendered_lines())
        return False

    outputs = _plan_outputs(files, opt, init_path, generated_prefixes, generator)
    changes = _changed_outputs(outputs)
    if opt.verbose or opt.dry_run:
        _preview_outputs(changes)
    if opt.dry_run:
        return False

    for path, source in sorted(changes.items(), key=lambda item: str(item[0])):
        write_text_atomic(path, source)
    for file in files:
        path = file.path.resolve()
        if path in outputs:
            file.lines = tuple(outputs[path].splitlines())
    return True


def _stage_1(
    file: FileInfo,
    ty_map: dict[str, str],
    definitions: dict[str, tuple[str, Path, int]] | None = None,
) -> None:
    """Parse one file's type maps atomically and reject ambiguous definitions."""
    if definitions is None:
        definitions = {}
    pending: dict[str, tuple[str, Path, int]] = {}
    for code in file.code_blocks:
        if code.kind == "ty-map":
            try:
                assert isinstance(code.param, str)
                lhs, rhs = code.param.split("->")
            except ValueError as e:
                raise ValueError(
                    f"Invalid ty-map at {file.path}:{code.lineno_start}; "
                    "expected exactly `A.B -> C.D`"
                ) from e
            lhs = lhs.strip()
            rhs = rhs.strip()
            if not lhs or not rhs:
                raise ValueError(
                    f"Invalid ty-map at {file.path}:{code.lineno_start}; "
                    "both sides of `->` must be non-empty"
                )

            previous = pending.get(lhs) or definitions.get(lhs)
            if previous is not None and previous[0] != rhs:
                previous_rhs, previous_path, previous_line = previous
                raise ValueError(
                    f"Conflicting ty-map for {lhs!r} at {file.path}:{code.lineno_start}: "
                    f"{rhs!r}; first defined as {previous_rhs!r} at "
                    f"{previous_path}:{previous_line}"
                )
            pending.setdefault(lhs, (rhs, file.path, code.lineno_start))

    for lhs, definition in pending.items():
        rhs, _, _ = definition
        ty_map[lhs] = rhs
        definitions.setdefault(lhs, definition)


def _stage_2(
    files: list[FileInfo],
    ty_map: dict[str, str],
    init_cfg: InitConfig,
    init_path: Path,
    global_funcs: dict[str, list[FuncInfo]],
    generator: Generator,
) -> set[str]:
    def _find_or_insert_file(path: Path) -> FileInfo:
        # Search the in-memory render plan first. In particular, Rust uses the
        # same ``mod.rs`` as both API and init file, and previously staged files
        # do not exist on disk yet.
        resolved = path.resolve()
        for file in files:
            if resolved == file.path.resolve():
                return file
        if not path.exists():
            ret: FileInfo | None = FileInfo(
                path=path, lines=(), code_blocks=[], syntax=syntax_for(path)
            )
        else:
            ret = FileInfo.from_file(file=path, include_empty=True)
            assert ret is not None, f"Failed to read file: {path}"
        files.append(ret)
        return ret

    def _append_in_memory(file: FileInfo, append: str) -> None:
        if not append:
            return
        current = "\n".join(line for block in file.code_blocks for line in block.lines)
        if current and not current.endswith("\n"):
            current += "\n"
        parsed = FileInfo.from_text(
            file.path,
            current + append,
            include_empty=True,
            syntax=file.syntax,
        )
        assert parsed is not None
        file.code_blocks = parsed.code_blocks

    # Step 0. Find out functions and classes already defined on files.
    defined_func_prefixes: set[str] = {
        code.param[0] for file in files for code in file.code_blocks if code.kind == "global"
    }
    defined_objs: set[str] = {  # ty: ignore[invalid-assignment]
        code.param for file in files for code in file.code_blocks if code.kind == "object"
    } | C.BUILTIN_TYPE_KEYS

    # Step 0. Generate missing `_ffi_api.py` and `__init__.py` under each prefix.
    roots = set(init_cfg.normalized_prefixes())
    prefixes: dict[str, list[str]] = collect_type_keys((), roots)
    for prefix in global_funcs:
        prefixes.setdefault(prefix, [])
    generated_prefixes: set[str] = set()
    for prefix, obj_names in prefixes.items():
        if not any(prefix == root or prefix.startswith(f"{root}.") for root in roots):
            continue
        funcs = sorted(
            [] if prefix in defined_func_prefixes else global_funcs.get(prefix, []),
            key=lambda f: f.schema.name,
        )
        objs = sorted(set(obj_names) - defined_objs)
        object_infos = toposort_objects(objs)
        if not funcs and not object_infos:
            if obj_names or global_funcs.get(prefix):
                generated_prefixes.add(prefix)
            continue
        generated_prefixes.add(prefix)
        # Step 1. Compute the target path. Directories/files are committed only
        # after every input has generated successfully.
        directory = init_path / prefix.replace(".", "/")
        # Step 2. Generate the API file.
        api_filename = generator.api_filename()
        target_path = directory / api_filename
        target_file = _find_or_insert_file(target_path)
        api_append = generator.generate_api_file(
            target_file.code_blocks,
            ty_map,
            prefix,
            object_infos,
            init_cfg,
            is_root=prefix in roots,
        )
        _append_in_memory(target_file, api_append)
        # Step 3. Generate the package entry (Python `__init__.py`; re-exports the
        # API submodule). `submodule` is the API file's stem.
        submodule = api_filename.rsplit(".", 1)[0]
        target_path = directory / generator.init_filename()
        target_file = _find_or_insert_file(target_path)
        init_append = generator.generate_init_file(target_file.code_blocks, prefix, submodule)
        _append_in_memory(target_file, init_append)
    return generated_prefixes


def _stage_3(  # noqa: PLR0912
    file: FileInfo,
    opt: Options,
    ty_map: dict[str, str],
    global_funcs: dict[str, list[FuncInfo]],
    generator: Generator,
    known_type_keys: set[str] | None = None,
    local_type_keys: set[str] | None = None,
    canonical_type_keys: set[str] | None = None,
    module_segments: tuple[str, ...] | None = None,
) -> None:
    defined_funcs: set[str] = set()
    defined_types: set[str] = set()
    if known_type_keys is None:
        known_type_keys = {
            code.param
            for code in file.code_blocks
            if code.kind == "object" and isinstance(code.param, str)
        }
    if local_type_keys is None:
        local_type_keys = {
            code.param
            for code in file.code_blocks
            if code.kind == "object" and isinstance(code.param, str)
        }
    if canonical_type_keys is None:
        canonical_type_keys = local_type_keys
    imports = generator.new_imports(
        known_type_keys,
        local_type_keys=local_type_keys,
        canonical_type_keys=canonical_type_keys,
        module_segments=module_segments,
    )
    # Stage 1. Collect `tvm-ffi-stubgen(import-object): ...`
    for code in file.code_blocks:
        if code.kind == "import-object":
            name, type_checking_only, alias = code.param
            generator.add_imported_object(imports, name, type_checking_only, alias)
    # Stage 2. Process `tvm-ffi-stubgen(begin): global/...`
    for code in file.code_blocks:
        if code.kind == "global":
            funcs = global_funcs.get(code.param[0], [])
            for func in funcs:
                defined_funcs.add(func.schema.name)
            generator.generate_global_funcs_block(code, funcs, ty_map, imports, opt)
    # Stage 3. Process `tvm-ffi-stubgen(begin): object/...`
    for code in file.code_blocks:
        if code.kind == "object":
            type_key = code.param
            assert isinstance(type_key, str)
            obj_info = object_info_from_type_key(type_key)
            type_key = ty_map.get(type_key, type_key)
            generator.generate_object_block(code, ty_map, imports, opt, obj_info)
            defined_types.add(generator.canonical_type_name(type_key))
    # Stage 4. Add imports for used types.
    for code in file.code_blocks:
        if code.kind == "import-section":
            generator.generate_import_section_block(code, imports, opt, defined_types)
            break  # Only one import block per file is supported for now.
    # Stage 5. Add `__all__` for defined classes and functions.
    for code in file.code_blocks:
        if code.kind == "__all__":
            export_names = defined_funcs | defined_types | generator.extra_export_names(imports)
            generator.generate_all_block(code, export_names, opt)
            break  # Only one __all__ block per file is supported for now.
    # Stage 6. Process `tvm-ffi-stubgen(begin): export/...`
    for code in file.code_blocks:
        if code.kind == "export":
            generator.generate_export_block(code)
    file.generated_items = generator.generated_item_names(imports)


def _parse_args() -> Options:
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    def _split_list_arg(arg: str | None) -> list[str]:
        if not arg:
            return []
        return [item.strip() for item in arg.split(";") if item.strip()]

    parser = argparse.ArgumentParser(
        prog="tvm-ffi-stubgen",
        description=(
            "Generate type stubs for TVM FFI extensions. It supports two modes\n"
            "- With `--init-prefix`, it scaffolds missing target-language modules from the "
            "registered global functions and object types in the loaded libraries.\n"
            "- In normal mode, it processes the given files/directories in-place, generating "
            "type stubs inside special `tvm-ffi-stubgen` directive blocks.\n\n"
            f"Documentation: {C.TERM_CYAN}{C.DOC_URL}{C.TERM_RESET}."
        ),
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--imports",
        type=str,
        default="",
        metavar="IMPORTS",
        help=(
            "Additional imports to load before generation, separated by ';' "
            "(e.g. 'pkgA;pkgB.submodule')."
        ),
    )
    parser.add_argument(
        "--dlls",
        type=str,
        default="",
        metavar="LIBS",
        help=(
            "Shared libraries to preload before generation (e.g. TVM runtime or "
            "your extension), separated by ';'. This ensures global function and "
            "object metadata is available. Platform-specific suffixes like "
            ".so/.dylib/.dll are supported."
        ),
    )
    parser.add_argument(
        "--init-pypkg",
        type=str,
        default="",
        help=(
            "Python package name to generate stubs for (e.g. apache-tvm-ffi). "
            "Python target only; required together with --init-lib and --init-prefix."
        ),
    )
    parser.add_argument(
        "--init-lib",
        type=str,
        default="",
        help=(
            "CMake target that produces the shared library to load for stub generation "
            "(e.g. tvm_ffi_shared). Python target only; required together with "
            "--init-pypkg and --init-prefix."
        ),
    )
    parser.add_argument(
        "--init-prefix",
        type=str,
        action="append",
        default=[],
        help=(
            "Global function/object prefix to include when generating stubs "
            "(e.g. tvm_ffi.). Repeat this flag to generate multiple roots in one invocation. "
            "For Rust this is the only init option required."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help=(
            "Extra spaces added inside each generated block, relative to the "
            "indentation of the corresponding stub 'begin' marker line."
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="PATH",
        help=(
            "Files or directories to process. Directories are scanned recursively; "
            "only .py, .pyi (Python), and .rs (Rust) files are modified. Use "
            "tvm-ffi-stubgen directives to select where stubs are generated."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default="python",
        choices=["python", "rust"],
        help="Code generator target: 'python' (default) or 'rust'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print a unified diff of changes to each file. This is useful for "
            "debugging or previewing changes before applying them."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Don't write changes to files. This is useful for previewing changes "
            "without modifying any files."
        ),
    )
    args = parser.parse_args()

    init_cfg: InitConfig | None = None
    if args.target == "rust":
        if args.init_pypkg or args.init_lib:
            parser.error("--init-pypkg and --init-lib are only valid with --target python")
        if args.init_prefix:
            init_cfg = InitConfig(pkg="", shared_target="", prefixes=tuple(args.init_prefix))
    elif args.init_pypkg or args.init_lib or args.init_prefix:
        if not args.init_pypkg or not args.init_lib or not args.init_prefix:
            parser.error("--init-pypkg, --init-lib, and --init-prefix must be provided together")
        init_cfg = InitConfig(
            pkg=args.init_pypkg,
            shared_target=args.init_lib,
            prefixes=tuple(args.init_prefix),
        )

    if not args.files:
        parser.print_help()
        sys.exit(1)

    return Options(
        imports=_split_list_arg(args.imports),
        dlls=_split_list_arg(args.dlls),
        init=init_cfg,
        indent=args.indent,
        files=args.files,
        verbose=args.verbose,
        dry_run=args.dry_run,
        target=args.target,
    )


if __name__ == "__main__":
    sys.exit(__main__())
