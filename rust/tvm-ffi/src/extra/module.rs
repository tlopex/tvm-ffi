/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
use crate::derive::{Object, ObjectRef};
use crate::error::Result;
use crate::function::Function;
use crate::object::{Object, ObjectArc};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

//-----------------------------------------------------
// Module
//-----------------------------------------------------

/// A TVM FFI Module for loading dynamic libraries and retrieving functions.
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Module"]
#[type_index(TypeIndex::kTVMFFIModule)]
#[type_final]
pub struct ModuleObj {
    object: Object,
}

/// ABI-stable owned Module for FFI operations.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Module {
    data: ObjectArc<ModuleObj>,
}

impl Module {
    /// Load a module from a dynamic library file.
    ///
    /// # Arguments
    /// * `file_name` - Path to the dynamic library file to load
    ///
    /// # Returns
    /// * `Result<Module>` - A `Module` instance on success
    pub fn load_from_file<Str: AsRef<str>>(file_name: Str) -> Result<Module> {
        let file_name = crate::string::String::from(file_name);
        crate::cached_global_func!("ffi.ModuleLoadFromFile")
            .call_tuple_with_len::<1, _>((file_name,))?
            .try_into()
    }

    /// Get a function from the module by name.
    ///
    /// # Arguments
    /// * `name` - The name of the function to retrieve
    ///
    /// # Returns
    /// * `Result<Function>` - A `Function` instance on success
    pub fn get_function<Str: AsRef<str>>(&self, name: Str) -> Result<Function> {
        let name = crate::string::String::from(name);
        crate::cached_global_func!("ffi.ModuleGetFunction")
            .call_tuple_with_len::<3, _>((self, name, true))?
            .try_into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::collections::array::ArrayObj;
    use crate::collections::map::MapObj;
    use crate::string::{BytesObj, StringObj};
    use crate::{match_any, Any, AnyCompatible, AnyView, Array, ObjectRefCore, Tensor};

    // Test-only object handles for distinct final built-in container types.
    #[repr(C)]
    #[derive(ObjectRef, Clone)]
    struct RawArray {
        data: ObjectArc<ArrayObj>,
    }

    #[repr(C)]
    #[derive(ObjectRef, Clone)]
    struct RawMap {
        data: ObjectArc<MapObj>,
    }

    #[repr(C)]
    #[derive(ObjectRef, Clone)]
    struct RawBytes {
        data: ObjectArc<BytesObj>,
    }

    #[repr(C)]
    #[derive(ObjectRef, Clone)]
    struct RawString {
        data: ObjectArc<StringObj>,
    }

    #[test]
    fn match_any_preserves_arm_body_across_generic_dispatch_paths() {
        fn classify<T>(value: Any) -> usize
        where
            T: AnyCompatible + ObjectRefCore + 'static,
            for<'a> T: TryFrom<AnyView<'a>>,
        {
            match_any! {
                value {
                    T(ref mut object) => {
                        let _ = object;
                        static CALLS: AtomicUsize = AtomicUsize::new(0);
                        return CALLS.fetch_add(1, Ordering::SeqCst) + 1;
                    },
                    RawMap(_) => (),
                    RawBytes(_) => (),
                    RawString(_) => (),
                    _ => (),
                }
            }
            0
        }

        let module = Module {
            data: ObjectArc::new(ModuleObj {
                object: Object::new(),
            }),
        };
        assert_eq!(classify::<Module>(Any::from(module)), 1);

        let tensor = Tensor::from_slice(&[0_f32; 1], &[1]).unwrap();
        assert_eq!(classify::<Module>(Any::from(tensor)), 0);

        let array = [1_i64, 2].into_iter().collect::<Array<i64>>();
        assert_eq!(classify::<RawArray>(Any::from(array)), 2);
    }
}
