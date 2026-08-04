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
use crate::any::{Any, AnyView};
use crate::derive::{Object, ObjectRef};
use crate::error::{Error, Result};
use crate::function_internal::{AsPackedCallable, TupleAsPackedArgs};
use crate::object::{
    checked_metadata_str, checked_object_cell, checked_type_attr, checked_type_info,
    checked_type_methods, Object, ObjectArc, ObjectCore, ObjectRef,
};
use crate::type_traits::AnyCompatible;
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFunctionCell, TVMFFIFunctionCreate, TVMFFIFunctionGetGlobal,
    TVMFFIFunctionSetGlobal, TVMFFIGetTypeAttrColumn, TVMFFIObjectHandle, TVMFFISafeCallType,
    TVMFFITypeIndex,
};

/// function object
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Function"]
#[type_index(TVMFFITypeIndex::kTVMFFIFunction)]
pub struct FunctionObj {
    object: Object,
    cell: TVMFFIFunctionCell,
}

/// Error reference class
#[derive(Clone, ObjectRef)]
pub struct Function {
    data: ObjectArc<FunctionObj>,
}

//------------------------------------------------------------------------
// CallbackFunctionObjImpl
//------------------------------------------------------------------------
/// Special helper class to hold a generic callback state as Object
/// Logically this Impl can be viewed as a FunctionObj
/// We can create an ObjectArc<CallbackFunctionObjImpl<F>> so the deleter
/// can correctly delete the entire object including callback part
/// then we will convert to ObjectArc<FunctionObj> to be used as function
#[repr(C)]
struct CallbackFunctionObjImpl<F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static> {
    function: FunctionObj,
    callback: F,
}

impl<F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static> CallbackFunctionObjImpl<F> {
    pub fn from_callback(callback: F) -> Self {
        Self {
            function: FunctionObj {
                object: Object::new(),
                cell: TVMFFIFunctionCell {
                    // specfic callback for F
                    safe_call: Self::invoke_callback,
                    cxx_call: std::ptr::null_mut(),
                },
            },
            callback,
        }
    }

    unsafe extern "C" fn invoke_callback(
        handle: *mut std::ffi::c_void,
        args: *const TVMFFIAny,
        num_args: i32,
        result: *mut TVMFFIAny,
    ) -> i32 {
        crate::function_internal::invoke_packed_c_abi(args, num_args, result, |packed_args| {
            crate::ensure!(
                !handle.is_null(),
                crate::error::VALUE_ERROR,
                "callback handle is null"
            );
            let this = &*handle.cast::<Self>();
            (this.callback)(packed_args)
        })
    }
}

unsafe impl<F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static> ObjectCore
    for CallbackFunctionObjImpl<F>
{
    const TYPE_KEY: &'static str = FunctionObj::TYPE_KEY;
    const TYPE_DEPTH: i32 = FunctionObj::TYPE_DEPTH;
    fn type_index() -> i32 {
        FunctionObj::type_index()
    }
    unsafe fn object_header_mut(this: &mut Self) -> &mut tvm_ffi_sys::TVMFFIObject {
        FunctionObj::object_header_mut(&mut this.function)
    }
}

unsafe impl<F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static>
    crate::object::RustAllocatableObject for CallbackFunctionObjImpl<F>
{
    unsafe fn drop_payload(this: *mut Self) {
        std::ptr::drop_in_place(std::ptr::addr_of_mut!((*this).callback));
    }
}

type ScopedPackedCall = unsafe fn(*mut std::ffi::c_void, &[AnyView<'_>]) -> Result<Any>;

struct ScopedPackedEntry {
    handle: *mut std::ffi::c_void,
    callback: *mut std::ffi::c_void,
    call: ScopedPackedCall,
    active: std::cell::Cell<bool>,
    previous: *const ScopedPackedEntry,
}

thread_local! {
    static SCOPED_PACKED_HEAD: std::cell::Cell<*const ScopedPackedEntry> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

struct ScopedPackedRegistration<'a> {
    entry: &'a ScopedPackedEntry,
}

impl Drop for ScopedPackedRegistration<'_> {
    fn drop(&mut self) {
        SCOPED_PACKED_HEAD.with(|head| {
            debug_assert_eq!(head.get(), self.entry as *const ScopedPackedEntry);
            head.set(self.entry.previous);
        });
    }
}

struct ScopedCallReset<'a>(&'a std::cell::Cell<bool>);

impl Drop for ScopedCallReset<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

unsafe fn invoke_scoped_callback<F>(
    callback: *mut std::ffi::c_void,
    args: &[AnyView<'_>],
) -> Result<Any>
where
    F: FnMut(&[AnyView<'_>]) -> Result<Any>,
{
    unsafe { (&mut *callback.cast::<F>())(args) }
}

unsafe extern "C" fn call_scoped_packed(
    handle: *mut std::ffi::c_void,
    args: *const TVMFFIAny,
    num_args: i32,
    result: *mut TVMFFIAny,
) -> i32 {
    unsafe {
        crate::function_internal::invoke_packed_c_abi(args, num_args, result, |args| {
            let entry = SCOPED_PACKED_HEAD.with(|head| {
                let mut entry = head.get();
                while !entry.is_null() {
                    if (*entry).handle == handle {
                        return entry;
                    }
                    entry = (*entry).previous;
                }
                std::ptr::null()
            });
            if entry.is_null() {
                return Err(Error::new(
                    crate::error::RUNTIME_ERROR,
                    "scoped callback is inactive or was invoked from another thread",
                    "",
                ));
            }

            let entry = &*entry;
            if entry.active.replace(true) {
                return Err(Error::new(
                    crate::error::RUNTIME_ERROR,
                    "scoped callback cannot be invoked recursively",
                    "",
                ));
            }
            let _reset = ScopedCallReset(&entry.active);
            (entry.call)(entry.callback, args)
        })
    }
}

unsafe extern "C" fn drop_scoped_handle(handle: *mut std::ffi::c_void) {
    unsafe { drop(Box::from_raw(handle.cast::<u8>())) }
}

impl Function {
    /// Call the function in packed format.
    pub fn call_packed(&self, packed_args: &[AnyView]) -> Result<Any> {
        let num_args = i32::try_from(packed_args.len()).map_err(|_| {
            Error::new(
                crate::error::VALUE_ERROR,
                "packed argument count exceeds i32::MAX",
                "",
            )
        })?;
        unsafe {
            let packed_args_ptr = packed_args.as_ptr() as *const TVMFFIAny;
            let mut result = Any::new();
            let ret_code = (self.data.cell.safe_call)(
                ObjectArc::as_raw(&self.data) as *mut FunctionObj as *mut std::ffi::c_void,
                packed_args_ptr,
                num_args,
                Any::as_data_ptr(&mut result),
            );
            if ret_code == 0 {
                Ok(result)
            } else {
                Err(Error::from_raised())
            }
        }
    }

    /// Use a packed callback that may borrow local, thread-bound state.
    ///
    /// The callback is active only while `body` runs and only on the current
    /// thread. A cloned function invoked later, recursively, or from another
    /// thread returns a normal TVM error without touching the borrowed state.
    /// This makes the helper suitable for native APIs whose documented
    /// contract invokes a callback synchronously without retaining it.
    #[doc(hidden)]
    pub fn with_scoped_packed<F, T>(
        callback: &mut F,
        body: impl FnOnce(Function) -> Result<T>,
    ) -> Result<T>
    where
        F: FnMut(&[AnyView<'_>]) -> Result<Any>,
    {
        // A non-zero-sized token remains owned by the native Function object,
        // including any clones retained beyond this lexical scope.
        let mut token = Box::new(0_u8);
        let handle = (&mut *token as *mut u8).cast::<std::ffi::c_void>();
        // SAFETY: `handle` is valid until `drop_scoped_handle`; the C callback
        // treats it only as an identity key. Borrowed state lives separately in
        // the thread-local entry below and is never dereferenced when inactive.
        let function = unsafe {
            Function::try_from_extern_c(handle, call_scoped_packed, Some(drop_scoped_handle))?
        };
        std::mem::forget(token);
        let previous = SCOPED_PACKED_HEAD.with(|head| head.get());
        let entry = ScopedPackedEntry {
            handle,
            callback: (callback as *mut F).cast(),
            call: invoke_scoped_callback::<F>,
            active: std::cell::Cell::new(false),
            previous,
        };
        SCOPED_PACKED_HEAD.with(|head| head.set(&entry));
        let _registration = ScopedPackedRegistration { entry: &entry };
        // Keep the original Function (and therefore its token) alive until the
        // registration is removed. `body` may drop or retain this clone.
        body(function.clone())
    }

    /// Call this function using the packed KWARGS protocol.
    ///
    /// `positional` is emitted before the process-wide KWARGS sentinel.  Each
    /// named entry is then emitted as a string key followed by its value.  All
    /// temporary strings and the sentinel remain alive for the synchronous
    /// native call.
    pub fn call_packed_with_kwargs(
        &self,
        positional: &[AnyView<'_>],
        named: &[(&str, AnyView<'_>)],
    ) -> Result<Any> {
        let total_args = named
            .len()
            .checked_mul(2)
            .and_then(|slots| positional.len().checked_add(slots))
            .and_then(|slots| slots.checked_add(1))
            .ok_or_else(|| {
                Error::new(
                    crate::error::VALUE_ERROR,
                    "packed argument count overflow",
                    "",
                )
            })?;
        i32::try_from(total_args).map_err(|_| {
            Error::new(
                crate::error::VALUE_ERROR,
                "packed argument count exceeds i32::MAX",
                "",
            )
        })?;

        let kwargs = get_kwargs_object()?;
        let keys = named
            .iter()
            .map(|(key, _)| crate::String::from(*key))
            .collect::<Vec<_>>();
        let mut packed_args = Vec::with_capacity(total_args);
        packed_args.extend_from_slice(positional);
        packed_args.push(AnyView::from(&kwargs));
        for ((_, value), key) in named.iter().zip(&keys) {
            packed_args.push(AnyView::from(key));
            packed_args.push(*value);
        }
        self.call_packed(&packed_args)
    }

    pub fn call_tuple<TupleType>(&self, tuple_args: TupleType) -> Result<Any>
    where
        TupleType: TupleAsPackedArgs,
    {
        // This is a workaround for Rust's requirement that stack allocation size
        // must be known at compile time for generic types.
        // While we know args_len is a constant, Rust doesn't allow us to directly
        // declare [AnyView::new(); args_len] in generic contexts.
        //
        // We use a small vector optimization pattern:
        // 1. First allocate a small stack buffer (stack_args)
        // 2. If args_len exceeds STACK_LEN, allocate a heap buffer (heap_args)
        // 3. Use the appropriate buffer based on size
        //
        // Since args_len is a compile-time constant, the compiler should optimize
        // away the unused branch, making this approach efficient.
        const STACK_LEN: usize = 4;
        let mut stack_args = [AnyView::new(); STACK_LEN];
        let mut heap_args = Vec::<AnyView>::new();
        let args_len = <TupleType as TupleAsPackedArgs>::LEN;
        // get packed arguments
        let packed_args: &mut [AnyView] = if args_len <= STACK_LEN {
            &mut stack_args[..args_len]
        } else {
            heap_args.resize(args_len, AnyView::new());
            &mut heap_args[..args_len]
        };
        (&tuple_args).fill_any_view(packed_args);
        self.call_packed(packed_args)
    }
    /// Call function with compile-time known argument count
    /// This is an optimized version of call_tuple for when the argument count
    /// is known at compile time, avoiding the small vector optimization overhead.
    ///
    /// # Arguments
    /// * `tuple_args` - The tuple arguments
    ///
    /// # Returns
    /// * `Any` - The result
    pub fn call_tuple_with_len<const LEN: usize, TupleType>(
        &self,
        tuple_args: TupleType,
    ) -> Result<Any>
    where
        TupleType: TupleAsPackedArgs,
    {
        let mut packed_args = [AnyView::new(); LEN];
        (&tuple_args).fill_any_view(&mut packed_args);
        self.call_packed(&packed_args)
    }
    /// Get global function by name
    /// This function will throw an error if the function is not found.
    ///
    /// # Arguments
    /// * `name` - The name of the function
    ///
    /// # Returns
    /// * `Function` - The global function
    pub fn get_global(name: &str) -> Result<Function> {
        unsafe {
            let name_arg = TVMFFIByteArray::from_str(name);
            let mut result: TVMFFIObjectHandle = ::std::ptr::null_mut();
            crate::check_safe_call!(TVMFFIFunctionGetGlobal(&name_arg, &mut result))?;
            if result.is_null() {
                crate::bail!(crate::error::RUNTIME_ERROR, "Function {} not found", name);
            }
            Ok(Self {
                data: ObjectArc::<FunctionObj>::from_raw(result as *mut FunctionObj),
            })
        }
    }

    /// Cached front of [`Function::get_global`] for generated bindings.
    pub fn get_global_cached(
        cell: &'static std::thread::LocalKey<std::cell::OnceCell<Function>>,
        name: &str,
    ) -> Result<Function> {
        cell.with(|cell| {
            if let Some(function) = cell.get() {
                return Ok(function.clone());
            }
            let function = Self::get_global(name)?;
            let _ = cell.set(function.clone());
            Ok(function)
        })
    }

    /// Look up a reflected method or type attribute as an FFI function.
    ///
    /// Explicit `TypeMethod` entries take precedence. Auto-generated hooks
    /// such as `__ffi_init__` live in a `TypeAttrColumn`, so the column is the
    /// required fallback. This matches the Python binding's lookup order.
    pub fn from_type_method(type_index: i32, method_name: &str) -> Result<Function> {
        unsafe {
            let info = checked_type_info(type_index)?;
            for method in checked_type_methods(info)? {
                let registered_name =
                    checked_metadata_str(type_index, "method name", &method.name)?;
                if registered_name == method_name {
                    if !<Function as AnyCompatible>::check_any_strict(&method.method) {
                        crate::bail!(
                            crate::error::TYPE_ERROR,
                            "method `{}` on type_index `{}` is not a Function",
                            method_name,
                            type_index
                        );
                    }
                    checked_object_cell(&method.method, "reflected method Function cell")?;
                    return Ok(<Function as AnyCompatible>::copy_from_any_view_after_check(
                        &method.method,
                    ));
                }
            }

            let attr_name = TVMFFIByteArray::from_str(method_name);
            let column = TVMFFIGetTypeAttrColumn(&attr_name);
            if let Some(attr) = checked_type_attr(column, type_index, method_name)? {
                if attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
                    if !<Function as AnyCompatible>::check_any_strict(&attr) {
                        crate::bail!(
                            crate::error::TYPE_ERROR,
                            "type attribute `{}` on type_index `{}` is not a Function",
                            method_name,
                            type_index
                        );
                    }
                    checked_object_cell(&attr, "reflected type-attribute Function cell")?;
                    return Ok(<Function as AnyCompatible>::copy_from_any_view_after_check(
                        &attr,
                    ));
                }
            }
        }
        crate::bail!(
            crate::error::TYPE_ERROR,
            "method `{}` not found on type_index `{}`",
            method_name,
            type_index
        )
    }

    /// Cached front of [`Function::from_type_method`] for generated bindings.
    pub fn from_type_method_cached(
        cell: &'static std::thread::LocalKey<std::cell::OnceCell<Function>>,
        type_index: i32,
        method_name: &str,
    ) -> Result<Function> {
        cell.with(|cell| {
            if let Some(function) = cell.get() {
                return Ok(function.clone());
            }
            let function = Self::from_type_method(type_index, method_name)?;
            let _ = cell.set(function.clone());
            Ok(function)
        })
    }

    /// Register a function as a global function
    /// # Arguments
    /// * `name` - The name of the function
    /// * `func` - The function to register
    ///
    /// # Returns
    /// * `Result<()>` - The result of the registration
    pub fn register_global(name: &str, func: Function) -> Result<()> {
        if ObjectArc::is_null(&func.data) {
            crate::bail!(
                crate::error::VALUE_ERROR,
                "Cannot register undefined Function {}",
                name
            );
        }
        unsafe {
            let name_arg = TVMFFIByteArray::from_str(name);
            let can_override = 0;
            crate::check_safe_call!(TVMFFIFunctionSetGlobal(
                &name_arg,
                ObjectArc::as_raw(&func.data) as *mut FunctionObj as TVMFFIObjectHandle,
                can_override
            ))?;
            Ok(())
        }
    }
    /// Construct a function from a packed function.
    ///
    /// Callbacks may be retained by the C++ registry and invoked from another
    /// thread, so captured state must be both `Send` and `Sync`.
    ///
    /// ```compile_fail
    /// use std::rc::Rc;
    /// use tvm_ffi::{Any, Function, Result};
    ///
    /// let local = Rc::new(());
    /// let _ = Function::from_packed(move |_| -> Result<Any> {
    ///     let _ = local.clone();
    ///     Ok(Any::new())
    /// });
    /// ```
    /// # Arguments
    /// * `func` - The packed function in signature of `Fn(&[AnyView]) -> Result<Any>`
    ///
    /// # Returns
    /// * `Function` - The function
    pub fn from_packed<F>(func: F) -> Self
    where
        F: Fn(&[AnyView]) -> Result<Any> + Send + Sync + 'static,
    {
        unsafe {
            let callback_arc = ObjectArc::new(CallbackFunctionObjImpl::from_callback(func));
            let func_arc = ObjectArc::<FunctionObj>::from_raw(
                ObjectArc::into_raw(callback_arc) as *mut FunctionObj
            );
            Self { data: func_arc }
        }
    }

    /// Construct a function from a typed function
    /// # Arguments
    /// * `func` - The typed function with function signature of `F(T0, T1, ...) -> Result<O>`
    ///
    /// # Returns
    /// * `Function` - The function
    pub fn from_typed<F, I, O>(func: F) -> Self
    where
        F: AsPackedCallable<I, O> + Send + Sync + 'static,
    {
        let closure = move |packed_args: &[AnyView]| -> Result<Any> {
            let ret_value = func.call_packed(packed_args)?;
            Ok(ret_value)
        };
        Self::from_packed(closure)
    }

    /// # Safety
    ///
    /// `handle` must be a valid pointer (or null) that is compatible with
    /// `safe_call` and `deleter`. The caller must ensure the handle outlives
    /// the returned `Function` (or that `deleter` properly frees it).
    pub unsafe fn from_extern_c(
        handle: *mut std::ffi::c_void,
        safe_call: TVMFFISafeCallType,
        deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    ) -> Self {
        unsafe { Self::try_from_extern_c(handle, safe_call, deleter) }.unwrap()
    }

    unsafe fn try_from_extern_c(
        handle: *mut std::ffi::c_void,
        safe_call: TVMFFISafeCallType,
        deleter: Option<unsafe extern "C" fn(*mut std::ffi::c_void)>,
    ) -> Result<Self> {
        unsafe {
            let mut out_handle: TVMFFIObjectHandle = std::ptr::null_mut();
            crate::check_safe_call!(TVMFFIFunctionCreate(
                handle,
                safe_call,
                deleter,
                &mut out_handle
            ))?;
            Ok(Self {
                data: ObjectArc::<FunctionObj>::from_raw(out_handle as *mut FunctionObj),
            })
        }
    }
}

/// Return the process-wide sentinel used by the packed KWARGS protocol.
///
/// The object is cached per thread because FFI object handles are not `Sync`.
pub fn get_kwargs_object() -> Result<ObjectRef> {
    thread_local! {
        static KWARGS: std::cell::OnceCell<ObjectRef> = const { std::cell::OnceCell::new() };
    }
    KWARGS.with(|cell| {
        if let Some(value) = cell.get() {
            return Ok(value.clone());
        }
        let value: ObjectRef = Function::get_global("ffi.GetKwargsObject")?
            .call_packed(&[])?
            .try_into_strict()?;
        let _ = cell.set(value.clone());
        Ok(value)
    })
}
