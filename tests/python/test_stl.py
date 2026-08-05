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
import pathlib

import pytest
import tvm_ffi.cpp
from tvm_ffi.core import TypeSchema
from tvm_ffi.module import Module


def test_stl() -> None:
    cpp_path = pathlib.Path(__file__).parent.resolve() / "cpp_src" / "test_stl.cc"
    output_lib_path = tvm_ffi.cpp.build(
        name="test_stl",
        sources=[str(cpp_path)],
        extra_cflags=["-DTVM_FFI_DLL_EXPORT_INCLUDE_METADATA=1"],
    )

    mod: Module = tvm_ffi.load_module(output_lib_path)

    assert list(mod.test_tuple([1, 2.5])) == [2.5, 1]
    assert mod.test_vector(None) is None
    assert list(mod.test_vector([[1, 2], [3, 4]])) == [3, 7]
    assert mod.test_variant(1) == "int"
    assert mod.test_variant(1.0) == "float"
    assert list(mod.test_variant([1, 1.0])) == ["int", "float"]
    assert dict(mod.test_map({"a": 1, "b": 2})) == {1: "a", 2: "b"}
    assert dict(mod.test_map_2({"a": 1, "b": 2})) == {1: "a", 2: "b"}
    assert mod.test_function(lambda value: value)(10) == 11

    metadata = mod.get_function_metadata("test_function")
    assert metadata is not None
    schema = TypeSchema.from_json_str(metadata["type_schema"])
    assert schema.repr() == "Callable[[Callable[[int], int]], Callable[[int], int]]"

    vector_metadata = mod.get_function_metadata("test_vector")
    assert vector_metadata is not None
    vector_schema = TypeSchema.from_json_str(vector_metadata["type_schema"])
    assert vector_schema.repr() == "Callable[[Array[Array[int]] | None], Array[int] | None]"

    with pytest.raises(TypeError):
        mod.test_tuple([1.5, 2.5])
    with pytest.raises(TypeError):
        mod.test_vector([[1]])
    with pytest.raises(TypeError):
        mod.test_function(lambda value: value)()


if __name__ == "__main__":
    pytest.main([__file__])
