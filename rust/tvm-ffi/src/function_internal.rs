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
use crate::any::{Any, AnyView, ArgTryFromAnyView};
use crate::error::{Error, Result};
use crate::string::{Bytes, String};
use crate::type_traits::AnyCompatible;

//------------------------------------------------------------------------
// PackedCallable
//------------------------------------------------------------------------
pub trait AsPackedCallable<I, O> {
    // Call the function in packed convention
    fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any>;
}

#[inline]
pub fn call_packed_callable<Fun, I, O>(func: Fun, packed_args: &[AnyView]) -> Result<Any>
where
    Fun: AsPackedCallable<I, O>,
{
    func.call_packed(packed_args)
}

/// Validate and execute one Rust implementation of the packed C ABI.
///
/// # Safety
///
/// A non-null `args` must point to `num_args` initialized `TVMFFIAny` values,
/// and a non-null `result` must point to writable `TVMFFIAny` storage.
#[doc(hidden)]
pub unsafe fn invoke_packed_c_abi<F>(
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
    callback: F,
) -> i32
where
    F: FnOnce(&[AnyView]) -> Result<Any>,
{
    // Keep an outermost unwind barrier around validation, callback handling,
    // error-object creation, result publication, and every destructor. The
    // normal inner barrier below preserves the original panic message. This
    // final barrier is for failures while reporting that panic (for example,
    // TVMFFIErrorCreate failing and Error::new asserting); such a failure may
    // leave no raised Error object, but it must never unwind across C.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        invoke_packed_c_abi_inner(args, num_args, result, callback)
    })) {
        Ok(status) => status,
        Err(payload) => {
            std::mem::forget(payload);
            -1
        }
    }
}

unsafe fn invoke_packed_c_abi_inner<F>(
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
    callback: F,
) -> i32
where
    F: FnOnce(&[AnyView]) -> Result<Any>,
{
    let returned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::ensure!(
            !result.is_null(),
            crate::error::VALUE_ERROR,
            "callback result pointer is null"
        );
        let num_args = usize::try_from(num_args).map_err(|_| {
            Error::new(
                crate::error::VALUE_ERROR,
                "callback argument count is negative",
                "",
            )
        })?;
        crate::ensure!(
            num_args <= isize::MAX as usize / std::mem::size_of::<AnyView<'_>>(),
            crate::error::VALUE_ERROR,
            "callback argument span exceeds Rust slice limits"
        );
        let packed_args = if num_args == 0 {
            &[]
        } else {
            crate::ensure!(
                !args.is_null(),
                crate::error::VALUE_ERROR,
                "callback arguments are null"
            );
            std::slice::from_raw_parts(args.cast::<AnyView>(), num_args)
        };
        callback(packed_args)
    }));
    match returned {
        Ok(Ok(value)) => {
            std::ptr::write(result, Any::into_raw_ffi_any(value));
            0
        }
        Ok(Err(error)) => {
            Error::set_raised(&error);
            -1
        }
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<std::string::String>() {
                message.clone()
            } else {
                "Rust callback panicked".to_owned()
            };
            // A panic payload is arbitrary safe Rust and its destructor may
            // itself panic. Keep that second unwind inside Rust as well; if it
            // carries another panicking payload, leak only that exceptional
            // payload rather than allowing unwind across the C ABI.
            if let Err(nested_payload) =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload)))
            {
                std::mem::forget(nested_payload);
            }
            Error::set_raised(&Error::new(crate::error::RUNTIME_ERROR, &message, ""));
            -1
        }
    }
}

macro_rules! impl_as_packed_callable {
    ($len:literal; $($t:ident),*) => {
        impl<Fun, $($t,)* Out> AsPackedCallable<($($t,)*), Out> for Fun
        where
            Fun: Fn($($t,)*) -> Result<Out> + 'static,
            Any: From<Out>,
            $($t: ArgTryFromAnyView),*
        {
            fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any>
            {
                crate::ensure!(
                    packed_args.len() == $len, crate::error::VALUE_ERROR,
                    "Expected {} arguments, got {}", $len, packed_args.len()
                );
                // Expand the function call, consuming the iterator.
                let mut _arg_iter = packed_args.iter().enumerate();
                let ret_value = self(
                    $({
                        // unwrap is safe due to the length check above
                        let (i, view) = _arg_iter.next().unwrap();
                        $t::try_from_any_view(view, i)?
                    }),*
                )?;
                Ok(Any::from(ret_value))
            }
        }
    }
}

impl_as_packed_callable!(0;);
impl_as_packed_callable!(1; T0);
impl_as_packed_callable!(2; T0, T1);
impl_as_packed_callable!(3; T0, T1, T2);
impl_as_packed_callable!(4; T0, T1, T2, T3);
impl_as_packed_callable!(5; T0, T1, T2, T3, T4);
impl_as_packed_callable!(6; T0, T1, T2, T3, T4, T5);
impl_as_packed_callable!(7; T0, T1, T2, T3, T4, T5, T6);
impl_as_packed_callable!(8; T0, T1, T2, T3, T4, T5, T6, T7);

//--------------------------------------------------------------
// IntoArgHolder, helper to convert to canonical holding type
//
// This is needed sometimes for reference types that may need to
// be converted to value types.
//--------------------------------------------------------------
pub trait IntoArgHolder {
    type Target;
    fn into_arg_holder(self) -> Self::Target;
}

crate::impl_into_arg_holder_default!(
    bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64, String, Bytes
);

// string will be converted to String for argument passing
impl IntoArgHolder for &str {
    type Target = String;
    fn into_arg_holder(self) -> Self::Target {
        String::from(self)
    }
}

// string will be converted to String for argument passing
impl IntoArgHolder for &[u8] {
    type Target = Bytes;
    fn into_arg_holder(self) -> Self::Target {
        Bytes::from(self)
    }
}

// helper trait to implement IntoArgHolderTuple to apply into_arg_holder to each element
pub trait IntoArgHolderTuple {
    type Target;
    fn into_arg_holder_tuple(self) -> Self::Target;
}

macro_rules! impl_into_arg_holder_tuple {
    ( $($T:ident),* ; $($idx:tt),* ) => {
        impl<$($T),*> $crate::function_internal::IntoArgHolderTuple for ($($T,)*)
        where
            $($T: IntoArgHolder),* {
            type Target = ($($T::Target,)*);

            fn into_arg_holder_tuple(self) -> Self::Target {
                ($(self.$idx.into_arg_holder(),)*)
            }
        }
    };
}

impl_into_arg_holder_tuple!(;);
impl_into_arg_holder_tuple!(T0; 0);
impl_into_arg_holder_tuple!(T0, T1; 0, 1);
impl_into_arg_holder_tuple!(T0, T1, T2; 0, 1, 2);
impl_into_arg_holder_tuple!(T0, T1, T2, T3; 0, 1, 2, 3);
impl_into_arg_holder_tuple!(T0, T1, T2, T3, T4; 0, 1, 2, 3, 4);
impl_into_arg_holder_tuple!(T0, T1, T2, T3, T4, T5; 0, 1, 2, 3, 4, 5);
impl_into_arg_holder_tuple!(T0, T1, T2, T3, T4, T5, T6; 0, 1, 2, 3, 4, 5, 6);
impl_into_arg_holder_tuple!(T0, T1, T2, T3, T4, T5, T6, T7; 0, 1, 2, 3, 4, 5, 6, 7);

//------------------------------------------------------------
// ArgIntoRef
//
// Helper to turn argument type to reference type
// This is effectively AsRef<T> but removes the need of T
//-----------------------------------------------------------
pub trait ArgIntoRef {
    type Target;
    fn to_ref(&self) -> &Self::Target;
}

crate::impl_arg_into_ref!(
    bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64, String, Bytes
);

// `Map<K, V>` passes by value/reference like the scalars above, but its type
// parameters keep it out of the `impl_*!` macros, so the impls are spelled out.
impl<K: AnyCompatible, V: AnyCompatible> IntoArgHolder for crate::Map<K, V> {
    type Target = crate::Map<K, V>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}
impl<'a, K: AnyCompatible, V: AnyCompatible> IntoArgHolder for &'a crate::Map<K, V> {
    type Target = &'a crate::Map<K, V>;
    fn into_arg_holder(self) -> Self::Target {
        self
    }
}
impl<K: AnyCompatible, V: AnyCompatible> ArgIntoRef for crate::Map<K, V> {
    type Target = crate::Map<K, V>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}
impl<K: AnyCompatible, V: AnyCompatible> ArgIntoRef for &crate::Map<K, V> {
    type Target = crate::Map<K, V>;
    fn to_ref(&self) -> &Self::Target {
        self
    }
}

//-----------------------------------------------------------
// TupleAsPackedArgs
//
// Helper to turn tuple type to packed arguments
//-----------------------------------------------------------
pub trait TupleAsPackedArgs {
    const LEN: usize;
    fn fill_any_view<'a>(&'a self, any_view: &mut [AnyView<'a>]);
}

macro_rules! impl_tuple_as_packed_args {
    ( $len:expr; $($T:ident),* ; $($idx:tt),* ) => {
        impl<$($T),*> TupleAsPackedArgs for ($($T,)*)
        where
            $(
                $T: ArgIntoRef,
                $T::Target: AnyCompatible,
            )*
        {
            const LEN: usize = $len;

            fn fill_any_view<'a>(&'a self, _any_view: &mut [AnyView<'a>]) {
                $(
                    _any_view[$idx] = AnyView::from(self.$idx.to_ref());
                )*
            }
        }
    };
}

impl_tuple_as_packed_args!(0;;);
impl_tuple_as_packed_args!(1; T0; 0);
impl_tuple_as_packed_args!(2; T0, T1; 0, 1);
impl_tuple_as_packed_args!(3; T0, T1, T2; 0, 1, 2);
impl_tuple_as_packed_args!(4; T0, T1, T2, T3; 0, 1, 2, 3);
impl_tuple_as_packed_args!(5; T0, T1, T2, T3, T4; 0, 1, 2, 3, 4);
impl_tuple_as_packed_args!(6; T0, T1, T2, T3, T4, T5; 0, 1, 2, 3, 4, 5);
impl_tuple_as_packed_args!(7; T0, T1, T2, T3, T4, T5, T6; 0, 1, 2, 3, 4, 5, 6);
impl_tuple_as_packed_args!(8; T0, T1, T2, T3, T4, T5, T6, T7; 0, 1, 2, 3, 4, 5, 6, 7);
