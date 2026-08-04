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
"""Rust-specific constants for the ``tvm-ffi-stubgen`` Rust backend."""

from __future__ import annotations

#: Default FFI-origin -> Rust-type map. Values are fully qualified paths so
#: ``RustUse``/``RustImports`` can derive both the leaf name and the ``use``
#: import; values without ``::`` (primitives) need no import.
RUST_TY_MAP_DEFAULTS = {
    "int": "i64",
    "float": "f64",
    "bool": "bool",
    "None": "()",
    "str": "tvm_ffi::String",
    "bytes": "tvm_ffi::Bytes",
    "Any": "tvm_ffi::Any",
    "AnyValue": "tvm_ffi::AnyValue",
    "Callable": "tvm_ffi::Function",
    "Array": "tvm_ffi::Array",  # the crate's own Array<T>, NOT Vec
    "Map": "tvm_ffi::Map",  # the crate's own Map<K, V>, NOT HashMap
    "TypedExpr": "tvm_ffi::TypedExpr",
    "Optional": "std::option::Option",
    # A generic/opaque object VALUE is the single-pointer `ObjectRef` handle
    # (AnyCompatible, niche-optimizable), NOT the 24-byte `Object` data struct
    # (which is only ever the embedded struct `base`, spelled literally by codegen).
    "Object": "tvm_ffi::object::ObjectRef",
    "Tensor": "tvm_ffi::Tensor",
    "Shape": "tvm_ffi::Shape",
    "Device": "tvm_ffi::DLDevice",
    "dtype": "tvm_ffi::DLDataType",
    "DataType": "tvm_ffi::DLDataType",
    # --- builtin object type keys (ffi.*) ---
    "ffi.String": "tvm_ffi::String",
    "ffi.Bytes": "tvm_ffi::Bytes",
    "ffi.Module": "tvm_ffi::Module",
    "ffi.Error": "tvm_ffi::Error",
    "ffi.Object": "tvm_ffi::object::ObjectRef",
    "ffi.Tensor": "tvm_ffi::Tensor",
    "ffi.Shape": "tvm_ffi::Shape",
    "ffi.Function": "tvm_ffi::Function",
}

#: Schema forms that have no single native Rust representation.  They remain
#: fully usable through the runtime's owning/borrowed type-erased carriers:
#: ``Any``/``AnyView`` at a top-level FFI boundary and ``AnyValue`` inside a
#: typed container.  This is fail-safe and preserves every value without
#: pretending that a Python ``Union`` or FFI tuple has Rust's native layout.
RUST_TYPE_ERASED_ORIGINS = frozenset({"Dict", "List", "Union", "tuple"})

# Origins whose default Rust carrier implements ObjectRefCore. Generated
# object keys are handled separately from this explicit runtime list.
RUST_OBJECT_REF_ORIGINS = frozenset(
    {
        "Array",
        "Callable",
        "Map",
        "Object",
        "Shape",
        "Tensor",
        "ffi.Error",
        "ffi.Function",
        "ffi.Module",
        "ffi.Object",
        "ffi.Shape",
        "ffi.Tensor",
    }
)

#: Module-prefix rewrites for ``use`` paths: builtin ``ffi.*`` type keys live at
#: the crate root.
RUST_MOD_MAP = {
    "ffi": "tvm_ffi",
}

#: Rust keywords (strict + reserved, all editions): a reflected field/method
#: named after one must be emitted as the raw identifier ``r#<name>`` in every
#: code position (struct field, builder, setter, `let`, literals, `fn` names);
#: message/FFI-name strings keep the original spelling.
RUST_KEYWORDS = frozenset(
    # strict
    "as break const continue crate dyn else enum extern false fn for if impl in let loop "
    "match mod move mut pub ref return self Self static struct super trait true type unsafe "
    "use where while async await "
    # reserved
    "abstract become box do final gen macro override priv try typeof unsized virtual yield".split()
)

#: Names that cannot be identifiers at all -- raw or otherwise (``r#self`` etc.
#: are rejected by rustc): a field so named has no rendering, skip loudly.
RUST_NON_RAW_IDENTS = frozenset({"crate", "self", "Self", "super", "_"})
