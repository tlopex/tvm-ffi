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
use tvm_ffi::collections::array::ArrayObj;
use tvm_ffi::*;

#[derive(Clone)]
struct PanicOnMove {
    value: AnyValue,
    panic_on_move: bool,
}

unsafe impl AnyCompatible for PanicOnMove {
    fn type_str() -> std::string::String {
        AnyValue::type_str()
    }

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        unsafe { AnyValue::copy_to_any_view(&src.value, data) };
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        if src.panic_on_move {
            panic!("injected Array element-conversion failure");
        }
        unsafe { AnyValue::move_to_any(src.value, data) };
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        unsafe { AnyValue::check_any_strict(data) }
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        Self {
            value: unsafe { AnyValue::copy_from_any_view_after_check(data) },
            panic_on_move: false,
        }
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self {
            value: unsafe { AnyValue::move_from_any_after_check(data) },
            panic_on_move: false,
        }
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        Ok(Self {
            value: unsafe { AnyValue::try_cast_from_any_view(data)? },
            panic_on_move: false,
        })
    }
}

/// Helper to create a Tensor with a specific float value and shape
fn create_tensor(val: f32, shape: &[i64]) -> Tensor {
    let dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
    let mut tensor = Tensor::from_nd_alloc(CPUNDAlloc::default(), shape, dtype, device);
    // The freshly allocated test Tensor has no other data owner or view.
    unsafe { tensor.data_as_slice_mut_unchecked::<f32>() }.unwrap()[0] = val;
    tensor
}

/// Helper to extract the first float value from a Tensor
fn get_val(tensor: &Tensor) -> f32 {
    // Test fixtures do not expose their CPU buffers to another owner.
    unsafe { tensor.data_as_slice_unchecked::<f32>() }.expect("Type mismatch or null")[0]
}

#[test]
fn test_array_core_and_iteration() {
    let t1 = create_tensor(10.0, &[1, 2]);
    let t2 = create_tensor(20.0, &[3, 4, 5]);

    let array = Array::new(vec![t1.clone(), t2.clone()]);

    // Core Accessors
    assert_eq!(array.len(), 2);
    assert!(!array.is_empty());

    // Value Integrity
    assert_eq!(get_val(&array.get(0).unwrap()), 10.0);
    assert_eq!(array.get(0).unwrap().ndim(), 2);
    assert_eq!(array.get(1).unwrap().ndim(), 3);
    assert_eq!(
        get_val(&Tensor::try_from(array.get_any(0).unwrap()).unwrap()),
        10.0
    );

    // Iteration
    let vals: Vec<f32> = array.iter().map(|t| get_val(&t)).collect();
    assert_eq!(vals, vec![10.0, 20.0]);
}

#[test]
fn test_array_drop_releases_object_elements_once() {
    let tensor = create_tensor(10.0, &[1]);
    let base = AnyView::from(&tensor)
        .debug_strong_count()
        .expect("tensor is reference counted");

    let array = Array::new(vec![tensor.clone(), tensor.clone()]);
    assert_eq!(AnyView::from(&tensor).debug_strong_count(), Some(base + 2));

    // Array clones share one container, so a non-final drop must not release
    // its elements. The final drop releases each of the two owning slots once.
    let array_clone = array.clone();
    drop(array);
    assert_eq!(AnyView::from(&tensor).debug_strong_count(), Some(base + 2));
    drop(array_clone);
    assert_eq!(AnyView::from(&tensor).debug_strong_count(), Some(base));
}

#[test]
fn test_array_partial_initialization_releases_completed_elements() {
    let tensor = create_tensor(20.0, &[1]);
    let base = AnyView::from(&tensor)
        .debug_strong_count()
        .expect("tensor is reference counted");
    let items = vec![
        PanicOnMove {
            value: AnyValue::from_value(tensor.clone()),
            panic_on_move: false,
        },
        PanicOnMove {
            value: AnyValue::from_value(tensor.clone()),
            panic_on_move: true,
        },
        PanicOnMove {
            value: AnyValue::from_value(tensor.clone()),
            panic_on_move: false,
        },
    ];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = Array::new(items);
    }));
    assert!(result.is_err());

    // The first item was already moved into the trailing buffer. The failing
    // item and untouched suffix unwind from the input Vec; all three references
    // must be gone after construction fails.
    assert_eq!(AnyView::from(&tensor).debug_strong_count(), Some(base));
}

#[test]
fn test_array_any_conversions() {
    let array = Array::new(vec![
        create_tensor(1.0, &[1]),
        create_tensor(2.0, &[1]),
        create_tensor(3.0, &[1]),
    ]);

    // Test Any/AnyView Roundtrip (Verifies AnyCompatible and Trait Bounds)
    let any = Any::from(array);
    assert_eq!(any.type_index(), TypeIndex::kTVMFFIArray as i32);

    let back: Array<Tensor> = Array::try_from(any).expect("Any -> Array failed");
    assert_eq!(back.len(), 3);
    assert_eq!(get_val(&back.get(2).unwrap()), 3.0);

    let view = AnyView::from(&back);
    let back_from_view: Array<Tensor> = Array::try_from(view).expect("AnyView -> Array failed");
    assert_eq!(back_from_view.len(), 3);
}

#[test]
fn test_null_array_encodes_as_ffi_none() {
    let array = <Array<i64> as ObjectRefCore>::from_data(unsafe {
        ObjectArc::<ArrayObj>::from_raw(std::ptr::null())
    });
    assert_eq!(array.len(), 0);
    assert!(array.is_empty());
    assert!(array.get(0).is_err());
    assert!(array.get_any(0).is_err());
    assert_eq!(
        Any::from(array.clone()).type_index(),
        TypeIndex::kTVMFFINone as i32
    );
    assert_eq!(
        AnyView::from(&array).type_index(),
        TypeIndex::kTVMFFINone as i32
    );
}

#[test]
fn test_array_rejects_incompatible_element_type() {
    // 1. Create an Array of Shapes
    let shape_array = Array::new(vec![Shape::from(vec![1, 2]), Shape::from(vec![3])]);

    // 2. Wrap it in Any
    let any_val = Any::from(shape_array);

    // 3. Try to convert Any (containing Shapes) into Array<Tensor>
    // This should FAIL because T::check_any_strict (Tensor) will fail on Shape elements
    let tensor_cast = Array::<Tensor>::try_from(any_val.clone());
    assert!(
        tensor_cast.is_err(),
        "Should not be able to cast Array<Shape> to Array<Tensor>"
    );

    // 4. Verify valid cast works
    let shape_cast = Array::<Shape>::try_from(any_val);
    assert!(
        shape_cast.is_ok(),
        "Should be able to cast back to correct type"
    );
}

#[test]
fn test_array_supports_distinct_homogeneous_object_types() {
    let shape_array = Array::new(vec![Shape::from(vec![1, 2, 3]), Shape::from(vec![10])]);
    assert_eq!(shape_array.get(0).unwrap().as_slice(), &[1, 2, 3]);
    assert_eq!(shape_array.get(1).unwrap().as_slice(), &[10]);

    let function_array = Array::new(vec![
        Function::get_global("ffi.String").unwrap(),
        Function::get_global("ffi.Bytes").unwrap(),
    ]);
    assert_eq!(
        into_typed_fn!(
            function_array.get(0).unwrap(),
            Fn(String) -> Result<String>
        )("hello".into())
        .unwrap(),
        "hello"
    );
    assert_eq!(
        into_typed_fn!(
            function_array.get(1).unwrap(),
            Fn(Bytes) -> Result<Bytes>
        )([1, 2, 3].into())
        .unwrap(),
        &[1, 2, 3]
    );
}
