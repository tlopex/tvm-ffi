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
use crate::object::{Object, ObjectArc, ObjectCoreWithExtraItems, RustAllocatableObject};
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::ops::Deref;
use tvm_ffi_sys::TVMFFIShapeCell;
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

//-----------------------------------------------------
// Shape
//-----------------------------------------------------
// ShapeObj for heap-allocated shape
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Shape"]
#[type_index(TypeIndex::kTVMFFIShape)]
pub struct ShapeObj {
    object: Object,
    data: TVMFFIShapeCell,
}

unsafe impl RustAllocatableObject for ShapeObj {
    unsafe fn drop_payload(_this: *mut Self) {}
}

/// ABI stable owned Shape for ffi
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Shape {
    data: ObjectArc<ShapeObj>,
}

impl Shape {
    /// Create a new empty Shape
    pub fn new() -> Self {
        let shape_obj = ShapeObj {
            object: Object::new(),
            data: TVMFFIShapeCell {
                data: std::ptr::null(),
                size: 0,
            },
        };
        Self {
            data: ObjectArc::new(shape_obj),
        }
    }

    /// Get the shape as a slice
    pub fn as_slice(&self) -> &[i64] {
        if self.data.data.size == 0 {
            &[]
        } else {
            assert!(
                !self.data.data.data.is_null(),
                "non-empty Shape has null data"
            );
            unsafe { std::slice::from_raw_parts(self.data.data.data, self.data.data.size) }
        }
    }

    /// Fill the strides from the shape
    pub fn fill_strides_from_shape<T>(shape: T, strides: &mut [i64])
    where
        T: AsRef<[i64]>,
    {
        let shape = shape.as_ref();
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride = stride
                .checked_mul(shape[i])
                .expect("shape strides exceed i64::MAX");
        }
    }
}

unsafe impl ObjectCoreWithExtraItems for ShapeObj {
    type ExtraItem = i64;
    fn extra_items_count(this: &Self) -> usize {
        this.data.size
    }
}

impl<T> From<T> for Shape
where
    T: AsRef<[i64]>,
{
    fn from(value: T) -> Self {
        unsafe {
            let value_slice: &[i64] = value.as_ref();
            let mut obj_arc = ObjectArc::new_with_extra_items(ShapeObj {
                object: Object::new(),
                data: TVMFFIShapeCell {
                    data: std::ptr::null(),
                    size: value_slice.len(),
                },
            });
            // reset the data ptr correctly after Arc is created
            let obj = &mut *ObjectArc::as_raw_mut(&mut obj_arc);
            let extra_items = ShapeObj::extra_items_uninit_mut(obj);
            let data = extra_items.as_mut_ptr().cast::<i64>();
            for (slot, value) in extra_items.iter_mut().zip(value_slice) {
                slot.write(*value);
            }
            obj.data.data = data;
            Self { data: obj_arc }
        }
    }
}

impl Deref for Shape {
    type Target = [i64];
    #[inline]
    fn deref(&self) -> &[i64] {
        self.as_slice()
    }
}

impl PartialEq for Shape {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Shape {}

impl PartialOrd for Shape {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl Ord for Shape {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
