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
from .file_utils import FileInfo, collect_files, syntax_for
from .generator import get_generator
from .lib_state import (
    collect_global_funcs,
    collect_type_keys,
    object_info_from_type_key,
    toposort_objects,
)
from .utils import FuncInfo, InitConfig, Options, UnsupportedTypeError

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
    global_funcs = collect_global_funcs()

    # All generation happens against in-memory FileInfo objects.  In strict
    # mode no file is committed if any collection, scaffolding, or rendering
    # step fails; allow-partial mode commits the successful portions.
    ty_map, collect_failed = _collect_type_map(files, generator)
    generated_prefixes, scaffold_failed = _generate_scaffolds(
        files, opt, init_path, ty_map, global_funcs, generator
    )
    render_failed = _render_files(files, opt, ty_map, global_funcs, generator)
    failed = collect_failed or scaffold_failed or render_failed
    preflight_failed = _validate_init(
        not failed or not opt.strict, opt, init_path, generated_prefixes, generator
    )
    failed |= preflight_failed
    committed = _commit_files(files, opt, failed, block_writes=preflight_failed)
    failed |= _finalize_init(
        committed and not preflight_failed, opt, init_path, generated_prefixes, generator
    )

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
    failed = False
    for file in files:
        try:
            _stage_1(file, ty_map)
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
) -> bool:
    """Run stage 3 for every file, retaining independent successes."""
    failed = False
    for file in files:
        if opt.verbose:
            print(f"{C.TERM_CYAN}[File] {file.path}{C.TERM_RESET}")
        try:
            _stage_3(file, opt, ty_map, global_funcs, generator=generator)
        except Exception:
            failed = True
            _print_file_failure(file)
    return failed


def _commit_files(
    files: list[FileInfo], opt: Options, failed: bool, *, block_writes: bool = False
) -> bool:
    """Atomically write generated files when the selected policy permits it."""
    # A malformed module tree cannot be made coherent by a partial commit, so
    # finalization preflight failures block writes in both generation modes.
    commit_outputs = not block_writes and (not failed or not opt.strict)
    if commit_outputs:
        for file in files:
            file.update(verbose=opt.verbose, dry_run=opt.dry_run)
    elif opt.verbose:
        # A strict failure may still show the complete would-be diff.
        for file in files:
            file.update(verbose=True, dry_run=True)
    return commit_outputs and not opt.dry_run


def _validate_init(
    would_commit: bool,
    opt: Options,
    init_path: Path,
    generated_prefixes: set[str],
    generator: Generator,
) -> bool:
    """Preflight tree finalization so a strict failure precedes all writes."""
    if not would_commit or opt.init is None or not generated_prefixes:
        return False
    try:
        generator.validate_init(init_path, generated_prefixes)
        return False
    except Exception:
        print(
            f"{C.TERM_RED}[Failed] Finalization preflight: {traceback.format_exc()}{C.TERM_RESET}"
        )
        return True


def _finalize_init(
    committed: bool,
    opt: Options,
    init_path: Path,
    generated_prefixes: set[str],
    generator: Generator,
) -> bool:
    """Stitch a committed init tree and report whether finalization failed."""
    if not committed or opt.init is None or not generated_prefixes:
        return False
    try:
        generator.finalize_init(init_path, generated_prefixes)
        return False
    except Exception:
        print(f"{C.TERM_RED}[Failed] Finalization: {traceback.format_exc()}{C.TERM_RESET}")
        return True


def _stage_1(
    file: FileInfo,
    ty_map: dict[str, str],
) -> None:
    for code in file.code_blocks:
        if code.kind == "ty-map":
            try:
                assert isinstance(code.param, str)
                lhs, rhs = code.param.split("->")
            except ValueError as e:
                raise ValueError(
                    f"Invalid ty_map format at line {code.lineno_start}. Example: `A.B -> C.D`"
                ) from e
            ty_map[lhs.strip()] = rhs.strip()


def _stage_2(
    files: list[FileInfo],
    ty_map: dict[str, str],
    init_cfg: InitConfig,
    init_path: Path,
    global_funcs: dict[str, list[FuncInfo]],
    generator: Generator,
) -> set[str]:
    def _find_or_insert_file(path: Path) -> FileInfo:
        # Search the in-memory transaction first. In particular, Rust uses the
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
    prefix_filter = init_cfg.prefix.strip()
    if prefix_filter and not prefix_filter.endswith("."):
        prefix_filter += "."
    root_prefix = prefix_filter.rstrip(".")
    prefixes: dict[str, list[str]] = collect_type_keys()
    for prefix in global_funcs:
        prefixes.setdefault(prefix, [])
    generated_prefixes: set[str] = set()
    for prefix, obj_names in prefixes.items():
        if not (prefix == root_prefix or prefix.startswith(prefix_filter)):
            continue
        funcs = sorted(
            [] if prefix in defined_func_prefixes else global_funcs.get(prefix, []),
            key=lambda f: f.schema.name,
        )
        objs = sorted(set(obj_names) - defined_objs)
        object_infos = toposort_objects(objs)
        if not funcs and not object_infos:
            continue
        generated_prefixes.add(prefix)
        # Step 1. Compute the target path. Directories/files are committed only
        # after every strict-mode input has generated successfully.
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
            is_root=prefix == root_prefix,
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
) -> None:
    defined_funcs: set[str] = set()
    defined_types: set[str] = set()
    imports = generator.new_imports()
    skipped_objects: list[tuple[str, str]] = []
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
            try:
                generator.generate_object_block(code, ty_map, imports, opt, obj_info)
            except UnsupportedTypeError as err:
                # Fail closed for this object.  Leaving only its markers keeps
                # the file deterministic and lets a later run regenerate it
                # after the schema/runtime gains support.
                code.lines = [code.lines[0], code.lines[-1]]
                print(f"{C.TERM_YELLOW}[Skipped] object {type_key}: {err}{C.TERM_RESET}")
                skipped_objects.append((type_key, str(err)))
                continue
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
    if skipped_objects and opt.strict:
        details = "; ".join(f"{key}: {reason}" for key, reason in skipped_objects)
        raise RuntimeError(f"strict generation rejected unsupported objects: {details}")


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
            "- In `--init-*` mode, it generates missing `_ffi_api.py` and `__init__.py` files, "
            "based on the registered global functions and object types in the loaded libraries.\n"
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
            "Required together with --init-lib and --init-prefix."
        ),
    )
    parser.add_argument(
        "--init-lib",
        type=str,
        default="",
        help=(
            "CMake target that produces the shared library to load for stub generation "
            "(e.g. tvm_ffi_shared). Required together with --init-pypkg and "
            "--init-prefix."
        ),
    )
    parser.add_argument(
        "--init-prefix",
        type=str,
        default="",
        help=(
            "Global function/object prefix to include when generating stubs "
            "(e.g. tvm_ffi.). Required together with --init-pypkg and --init-lib."
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
        "--allow-partial",
        action="store_true",
        help=(
            "Keep successfully generated bindings and return success when some reflected "
            "constructs are unsupported. Strict failure is the default."
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

    init_flags = [args.init_pypkg, args.init_lib, args.init_prefix]
    init_cfg: InitConfig | None = None
    if any(init_flags):
        if not all(init_flags):
            parser.error("--init-pypkg, --init-lib, and --init-prefix must be provided together")
        init_cfg = InitConfig(
            pkg=args.init_pypkg,
            shared_target=args.init_lib,
            prefix=args.init_prefix,
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
        strict=not args.allow_partial,
    )


if __name__ == "__main__":
    sys.exit(__main__())
