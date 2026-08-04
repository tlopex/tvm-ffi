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
use tvm_ffi::*;

#[test]
fn test_function_dummpy_c_api() {
    let ret = unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() };
    assert_eq!(ret, 0);
}

#[test]
fn test_function_get_global_required() {
    let fecho = Function::get_global("testing.echo").unwrap();
    let a = 1;
    let args = [AnyView::from(&a)];
    let result = fecho.call_packed(&args).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 1);
}

#[test]
fn test_function_from_type_method_uses_checked_method_and_attr_tables() {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let type_key = unsafe { tvm_ffi_sys::TVMFFIByteArray::from_str("testing.TestIntPair") };
    let mut type_index = -1;
    assert_eq!(
        unsafe { tvm_ffi_sys::TVMFFITypeKeyToIndex(&type_key, &mut type_index) },
        0
    );

    // Constructors are mirrored into a TypeAttrColumn; ordinary reflected
    // methods live in the TypeInfo method table.
    let init = Function::from_type_method(type_index, "__ffi_init__").unwrap();
    let left = 3_i64;
    let right = 4_i64;
    let pair = init
        .call_packed(&[AnyView::from(&left), AnyView::from(&right)])
        .unwrap();
    let sum = Function::from_type_method(type_index, "sum").unwrap();
    let result = sum.call_packed(&[AnyView::from(&pair)]).unwrap();
    assert_eq!(i64::try_from(result).unwrap(), 7);
}

#[test]
fn test_register_global_rejects_undefined_function() {
    let function = <Function as ObjectRefCore>::from_data(unsafe {
        ObjectArc::<tvm_ffi::function::FunctionObj>::from_raw(std::ptr::null())
    });
    let error = Function::register_global("testing.undefined_function", function)
        .expect_err("an undefined Function must not enter the C++ global registry");
    assert_eq!(error.kind(), VALUE_ERROR);
}

#[test]
fn test_function_from_packed() {
    let value = 2;
    let v2 = 4;
    let check_and_add_value = Function::from_packed(move |args: &[AnyView]| -> Result<Any> {
        ensure!(
            args.len() == 1,
            VALUE_ERROR,
            "Expected 1 argument, got {}",
            args.len()
        );
        let v0 = i32::try_from(args[0])?;
        ensure!(v0 == value, VALUE_ERROR, "Expected {}, got {}", value, v0);
        Ok(Any::from(v0 + v2))
    });
    let args = [AnyView::from(&value)];
    let result = check_and_add_value.call_packed(&args).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 6);
}

#[test]
fn test_function_from_packed_accepts_zero_args() {
    let function = Function::from_packed(|args| {
        assert!(args.is_empty());
        Ok(Any::from(7_i32))
    });
    assert_eq!(
        i32::try_from(function.call_packed(&[]).unwrap()).unwrap(),
        7
    );
}

#[test]
fn test_function_call_packed_with_kwargs_encodes_protocol() {
    let function = Function::from_packed(|args| {
        assert_eq!(args.len(), 6);
        assert_eq!(i32::try_from(args[0])?, 7);
        let kwargs = tvm_ffi::object::ObjectRef::try_from(args[1])?;
        let expected_kwargs = get_kwargs_object()?;
        assert!(kwargs.same_as(&expected_kwargs));
        assert_eq!(String::try_from(args[2])?, "left");
        assert_eq!(i32::try_from(args[3])?, 11);
        assert_eq!(String::try_from(args[4])?, "name");
        assert_eq!(String::try_from(args[5])?, "right");
        Ok(Any::from(23_i32))
    });

    let positional = 7_i32;
    let left = 11_i32;
    let right = String::from("right");
    let result = function
        .call_packed_with_kwargs(
            &[AnyView::from(&positional)],
            &[
                ("left", AnyView::from(&left)),
                ("name", AnyView::from(&right)),
            ],
        )
        .unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 23);
}

#[test]
fn test_function_from_packed_contains_panics() {
    let function = Function::from_packed(|_| -> Result<Any> { panic!("callback failed") });
    let error = match function.call_packed(&[]) {
        Ok(_) => panic!("panicking callback unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), RUNTIME_ERROR);
    assert_eq!(error.message(), "callback failed");
}

#[test]
fn test_function_from_packed_contains_panicking_payload_destructors() {
    struct PanicsOnDrop;

    impl Drop for PanicsOnDrop {
        fn drop(&mut self) {
            // The nested panic also owns this payload type. The trampoline
            // must not drop it outside a catch boundary.
            std::panic::panic_any(PanicsOnDrop);
        }
    }

    let function = Function::from_packed(|_| -> Result<Any> {
        std::panic::panic_any(PanicsOnDrop);
    });
    let error = match function.call_packed(&[]) {
        Ok(_) => panic!("panicking callback unexpectedly succeeded"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), RUNTIME_ERROR);
    assert_eq!(error.message(), "Rust callback panicked");
}

#[test]
fn test_scoped_function_borrows_thread_bound_state_and_rejects_late_calls() {
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let mut callback = |args: &[AnyView<'_>]| {
        assert!(args.is_empty());
        calls.set(calls.get() + 1);
        Ok(Any::from(19_i32))
    };

    let retained = Function::with_scoped_packed(&mut callback, |function| {
        let value = function.call_packed(&[])?;
        assert_eq!(i32::try_from(value)?, 19);
        Ok::<_, Error>(function.clone())
    })
    .unwrap();
    assert_eq!(calls.get(), 1);

    let error = match retained.call_packed(&[]) {
        Ok(_) => panic!("a callback retained past its lexical scope must be inactive"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), RUNTIME_ERROR);
    assert!(error.message().contains("scoped callback is inactive"));
    assert_eq!(calls.get(), 1);
}

#[test]
fn test_scoped_function_rejects_recursion_and_resets_busy_state() {
    let calls = std::rc::Rc::new(std::cell::Cell::new(0));
    let retained = std::rc::Rc::new(std::cell::RefCell::new(None::<Function>));
    let callback_calls = calls.clone();
    let callback_function = retained.clone();
    let mut callback = move |args: &[AnyView<'_>]| {
        assert!(args.is_empty());
        let call = callback_calls.get();
        callback_calls.set(call + 1);
        if call == 0 {
            let error = match callback_function
                .borrow()
                .as_ref()
                .unwrap()
                .call_packed(&[])
            {
                Ok(_) => panic!("a scoped callback must reject recursive invocation"),
                Err(error) => error,
            };
            assert_eq!(error.kind(), RUNTIME_ERROR);
            assert!(error.message().contains("cannot be invoked recursively"));
        }
        Ok(Any::from(31_i32))
    };

    Function::with_scoped_packed(&mut callback, |function| {
        retained.replace(Some(function.clone()));
        assert_eq!(i32::try_from(function.call_packed(&[])?)?, 31);
        // The recursive error above must not leave the callback marked busy.
        assert_eq!(i32::try_from(function.call_packed(&[])?)?, 31);
        Ok::<_, Error>(())
    })
    .unwrap();
    assert_eq!(calls.get(), 2);
}

#[test]
fn test_scoped_function_resets_busy_state_after_callback_panic() {
    let calls = std::cell::Cell::new(0);
    let mut callback = |_args: &[AnyView<'_>]| {
        let call = calls.get();
        calls.set(call + 1);
        if call == 0 {
            panic!("first scoped call failed");
        }
        Ok(Any::from(43_i32))
    };

    Function::with_scoped_packed(&mut callback, |function| {
        let error = match function.call_packed(&[]) {
            Ok(_) => panic!("the first scoped call unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), RUNTIME_ERROR);
        assert_eq!(error.message(), "first scoped call failed");
        assert_eq!(i32::try_from(function.call_packed(&[])?)?, 43);
        Ok::<_, Error>(())
    })
    .unwrap();
    assert_eq!(calls.get(), 2);
}

#[test]
fn test_scoped_function_is_inactive_after_body_panics() {
    let retained = std::cell::RefCell::new(None::<Function>);
    let mut callback = |_args: &[AnyView<'_>]| Ok(Any::from(1_i32));

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = Function::with_scoped_packed(&mut callback, |function| -> Result<()> {
            retained.replace(Some(function));
            panic!("scoped body failed");
        });
    }));
    assert!(panic.is_err());

    let error = match retained.borrow().as_ref().unwrap().call_packed(&[]) {
        Ok(_) => panic!("unwinding the body must deactivate the callback"),
        Err(error) => error,
    };
    assert_eq!(error.kind(), RUNTIME_ERROR);
    assert!(error.message().contains("scoped callback is inactive"));
}

#[test]
fn test_scoped_functions_nest_in_lifo_order() {
    let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let outer_events = events.clone();
    let mut outer_callback = move |_args: &[AnyView<'_>]| {
        outer_events.borrow_mut().push("outer");
        Ok(Any::from(7_i32))
    };

    Function::with_scoped_packed(&mut outer_callback, |outer| {
        let inner_events = events.clone();
        let mut inner_callback = move |_args: &[AnyView<'_>]| {
            inner_events.borrow_mut().push("inner");
            Ok(Any::from(11_i32))
        };
        let retained_inner = Function::with_scoped_packed(&mut inner_callback, |inner| {
            assert_eq!(i32::try_from(inner.call_packed(&[])?)?, 11);
            // Looking up the outer token must walk past the active inner scope.
            assert_eq!(i32::try_from(outer.call_packed(&[])?)?, 7);
            Ok::<_, Error>(inner.clone())
        })?;
        assert!(retained_inner.call_packed(&[]).is_err());
        assert_eq!(i32::try_from(outer.call_packed(&[])?)?, 7);
        Ok::<_, Error>(())
    })
    .unwrap();

    assert_eq!(&*events.borrow(), &["inner", "outer", "outer"]);
}

#[test]
fn test_function_from_typed() {
    let offset = 2;
    // test one argument
    let sum1 = Function::from_typed(move |x: i32| -> Result<i32> { Ok(x + offset) });
    let result = sum1.call_packed(&[AnyView::from(&1)]).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 1 + offset);
    // test two arguments
    let sum2 = Function::from_typed(move |x: i32, y: i32| -> Result<i32> { Ok(x + y) });
    let result = sum2
        .call_packed(&[AnyView::from(&1), AnyView::from(&2)])
        .unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 3);
    // test three arguments
    let sum3f = Function::from_typed(move |x: i32, y: i32, z: f32| -> Result<f32> {
        Ok((x + y) as f32 + z)
    });
    let result = sum3f
        .call_packed(&[AnyView::from(&1), AnyView::from(&2), AnyView::from(&3)])
        .unwrap();
    assert_eq!(f32::try_from(result).unwrap(), 6.0);
}

#[test]
fn test_function_call_tuple() {
    let offset = 2;
    // test one argument
    let sum1 = Function::from_typed(move |x: i32| -> Result<i32> { Ok(x + offset) });
    let result = sum1.call_tuple((1,)).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 1 + offset);
    // test pass by reference
    let result = sum1.call_tuple_with_len::<1, _>((&1,)).unwrap();
    assert_eq!(i32::try_from(result).unwrap(), 1 + offset);
    let typed_fn = |x: &i32| -> Result<i32> { Ok(sum1.call_tuple((x,))?.try_into()?) };
    let result = typed_fn(&1);
    assert_eq!(result.unwrap(), 1 + offset);
}

#[test]
fn test_function_into_typed_fn() {
    let offset = 2;
    let typed_sum1 = into_typed_fn!(
        Function::from_typed(move |x: i32| -> Result<i32> { Ok(x + offset) }),
        Fn(&i32) -> Result<i32>);
    assert_eq!(typed_sum1(&1).unwrap(), 1 + offset);
    // try to box the resulting function
    let sum2 = Function::from_typed(move |x: i32, y: i32| -> Result<i32> { Ok(x + y) });
    let typed_sum2 = Box::new(into_typed_fn!(sum2, Fn(&i32, i32) -> Result<i32>));
    assert_eq!(typed_sum2(&1, 2).unwrap(), 3);

    // test three arguments
    let sum3 = Function::from_typed(move |x: i32, y: i32, z: f32| -> Result<f32> {
        Ok((x + y) as f32 + z)
    });
    let typed_sum3 = Box::new(into_typed_fn!(sum3, Fn(&i32, i32, f32) -> Result<f32>));
    assert_eq!(typed_sum3(&1, 2, 3.0).unwrap(), 6.0);
}

#[test]
fn test_function_echo_tensor_typed() {
    let echo = into_typed_fn!(
        Function::get_global("testing.echo").unwrap(),
        Fn(&Tensor) -> Result<Tensor>
    );
    let data: &[f32] = &[1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_slice(data, &[1, 2, 2]).unwrap();
    // write tensor content here
    let result = echo(&tensor).unwrap();
    assert_eq!(result.data_ptr(), tensor.data_ptr());
    assert_eq!(AnyView::from(&result).debug_strong_count(), Some(2));
    // The echo call has completed and this test performs no concurrent writes.
    let result_data = unsafe { result.data_as_slice_unchecked::<f32>() }.unwrap();
    assert_eq!(result_data.len(), 4);
    assert_eq!(result_data[0], 1.0);
    assert_eq!(result_data[1], 2.0);
    assert_eq!(result_data[2], 3.0);
    assert_eq!(result_data[3], 4.0);
}

fn testing_add_one(x: i32) -> Result<i32> {
    Ok(x + 1)
}
tvm_ffi_dll_export_typed_func!(testing_add_one, testing_add_one);

fn testing_no_args() -> Result<i32> {
    Ok(17)
}
tvm_ffi_dll_export_typed_func!(testing_no_args, testing_no_args);

fn testing_panics() -> Result<i32> {
    panic!("exported callback failed")
}
tvm_ffi_dll_export_typed_func!(testing_panics, testing_panics);

#[test]
fn test_exported_c_trampoline_validates_raw_inputs_and_contains_panics() {
    unsafe {
        let mut result = tvm_ffi_sys::TVMFFIAny::new();
        assert_eq!(
            __tvm_ffi_testing_no_args(std::ptr::null_mut(), std::ptr::null(), 0, &mut result),
            0
        );
        assert_eq!(i32::try_from(Any::from_raw_ffi_any(result)).unwrap(), 17);

        let mut result = tvm_ffi_sys::TVMFFIAny::new();
        assert_eq!(
            __tvm_ffi_testing_no_args(std::ptr::null_mut(), std::ptr::null(), -1, &mut result),
            -1
        );
        assert_eq!(Error::from_raised().kind(), VALUE_ERROR);

        assert_eq!(
            __tvm_ffi_testing_no_args(std::ptr::null_mut(), std::ptr::null(), 1, &mut result),
            -1
        );
        assert_eq!(Error::from_raised().kind(), VALUE_ERROR);

        assert_eq!(
            __tvm_ffi_testing_no_args(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            ),
            -1
        );
        assert_eq!(Error::from_raised().kind(), VALUE_ERROR);

        assert_eq!(
            __tvm_ffi_testing_panics(std::ptr::null_mut(), std::ptr::null(), 0, &mut result),
            -1
        );
        let error = Error::from_raised();
        assert_eq!(error.kind(), RUNTIME_ERROR);
        assert_eq!(error.message(), "exported callback failed");
    }
}

#[test]
fn test_function_from_extern_c() {
    // SAFETY: null handle is valid for testing_add_one which doesn't use the handle.
    let add_one =
        unsafe { Function::from_extern_c(std::ptr::null_mut(), __tvm_ffi_testing_add_one, None) };
    let typed_add_one = into_typed_fn!(add_one, Fn(i32) -> Result<i32>);
    assert_eq!(typed_add_one(1).unwrap(), 2);
}

#[test]
fn test_function_echo_string_bytes() {
    let echo = Function::get_global("testing.echo").unwrap();
    let echo_str = into_typed_fn!(
        echo.clone(),
        Fn(&str) -> Result<String>
    );
    let result = echo_str("hello").unwrap();
    assert_eq!(result, "hello");
    let echo_bytes = into_typed_fn!(
        echo.clone(),
        Fn(&[u8]) -> Result<Bytes>
    );
    let result = echo_bytes(b"hello").unwrap();
    assert_eq!(result, b"hello");
}

#[test]
fn test_function_apply() {
    let add_one = Function::from_typed(|x: i32| -> Result<i32> { Ok(x + 1) });
    let fapply = into_typed_fn!(
        Function::get_global("testing.apply").unwrap(),
        Fn(Function, i32) -> Result<i32>
    );
    let result = fapply(add_one, 3).unwrap();
    assert_eq!(result, 4);
}

fn test_add_one_tensor(x: tvm_ffi::Tensor, mut y: tvm_ffi::Tensor) -> Result<()> {
    // The packed-function contract supplies a read-only input allocation.
    let x_data = unsafe { x.data_as_slice_unchecked::<f32>()? };
    // The packed-function contract supplies a distinct output allocation.
    let y_data = unsafe { y.data_as_slice_mut_unchecked::<f32>()? };
    for i in 0..x_data.len() {
        y_data[i] = x_data[i] + 1.0;
    }
    Ok(())
}

tvm_ffi_dll_export_typed_func!(test_add_one_tensor, test_add_one_tensor);

#[test]
fn test_function_call_tensor_fn() {
    // SAFETY: null handle is valid for test_add_one_tensor which doesn't use the handle.
    let add_one = unsafe {
        Function::from_extern_c(std::ptr::null_mut(), __tvm_ffi_test_add_one_tensor, None)
    };
    let typed_add_one = into_typed_fn!(add_one, Fn(&Tensor, &Tensor) -> Result<()>);
    let x_data: &[f32] = &[0.0, 1.0, 2.0, 3.0];
    let x = Tensor::from_slice(x_data, &[2, 2]).unwrap();
    let y_data: &[f32] = &[0.0, 0.0, 0.0, 0.0];
    let y = Tensor::from_slice(y_data, &[2, 2]).unwrap();
    typed_add_one(&x, &y).unwrap();
    // The mutating packed call has completed before this read.
    let y_data = unsafe { y.data_as_slice_unchecked::<f32>() }.unwrap();
    assert_eq!(y_data[0], 1.0);
    assert_eq!(y_data[1], 2.0);
    assert_eq!(y_data[2], 3.0);
    assert_eq!(y_data[3], 4.0);
}
