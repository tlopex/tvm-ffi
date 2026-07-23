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

use tvm_ffi::object::ObjectRef;
use tvm_ffi::{Object, ObjectArc, ObjectRefCore, Shape, Tensor};

fn erase<T: ObjectRefCore>(value: T) -> ObjectRef {
    // SAFETY: ObjectRefCore guarantees a leading TVMFFIObject header; Object
    // is its repr(C) wrapper.
    let ptr = unsafe { ObjectArc::into_raw(T::into_data(value)) };
    let data = unsafe { ObjectArc::<Object>::from_raw(ptr.cast()) };
    ObjectRef::from_data(data)
}

#[test]
fn casts_object_refs_using_the_runtime_type() {
    let tensor = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    // SAFETY: tensor owns this allocation until erase transfers it.
    let tensor_ptr = unsafe { ObjectArc::as_raw(Tensor::data(&tensor)) };
    let object = erase(tensor);

    let tensor = object.try_cast::<Tensor>().unwrap();

    // SAFETY: The successful cast transferred the same allocation to tensor.
    assert_eq!(
        unsafe { ObjectArc::as_raw(Tensor::data(&tensor)) },
        tensor_ptr
    );
    assert!(tensor.try_cast::<Shape>().is_err());
}
