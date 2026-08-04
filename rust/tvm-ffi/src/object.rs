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
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::AtomicU64;

use crate::any::{Any, TryFromTemp};
use crate::derive::ObjectRef;
use crate::error::Result;
use crate::type_traits::AnyCompatible;
pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
/// Object related ABI handling
use tvm_ffi_sys::{TVMFFIAny, TVMFFIGetTypeInfo, TVMFFIObject, COMBINED_REF_COUNT_BOTH_ONE};

/// Object type is by default the TVMFFIObject
#[repr(C)]
pub struct Object {
    /// example implementation of the object
    header: TVMFFIObject,
    // Type erasure must not manufacture thread safety for an arbitrary C++
    // object. Concrete types may opt in only after their contract proves it.
    _not_send_sync: PhantomData<Rc<()>>,
}

/// Arc-like wrapper for Object that allows shared ownership.
///
/// Its single pointer slot may be null to mirror a nullable C++ `ObjectRef`
/// field in place. Cloning and dropping null are no-ops; dereferencing null
/// panics before a Rust reference is formed.
///
/// \tparam T The type of the object to be wrapped
///
/// A fully type-erased object remains thread-bound:
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<tvm_ffi::ObjectArc<tvm_ffi::Object>>();
/// ```
#[repr(C)]
pub struct ObjectArc<T: ObjectCore> {
    // C++ ObjectRef uses a nullable pointer slot.  Top-level handles returned
    // as concrete values are normally defined, but the same Rust wrapper also
    // appears in generated in-place object layouts where null is a valid field
    // value (most notably Span and Attrs).  A raw pointer preserves that ABI
    // without asserting NonNull before the field is actually dereferenced.
    ptr: *mut T,
    _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Send + Sync + ObjectCore> Send for ObjectArc<T> {}
unsafe impl<T: Send + Sync + ObjectCore> Sync for ObjectArc<T> {}

/// Traits that can be used to check if a type is an object
///
/// This trait is unsafe because it is used to access the object header
/// and the object header is unsafe to access
pub unsafe trait ObjectCore: Sized + 'static {
    /// the type key of the object
    const TYPE_KEY: &'static str;
    /// Depth of this type in the object inheritance tree.
    ///
    /// The root [`Object`] has depth zero, and every registered subtype has
    /// depth one greater than its parent. This value must be non-negative and
    /// agree with the runtime type table entry for `Self`.
    const TYPE_DEPTH: i32;
    /// Whether every instance of this type has exactly `Self::type_index()`.
    ///
    /// A final type has no separately registered object-system subtype.
    #[doc(hidden)]
    const TYPE_FINAL: bool = false;
    // return the type index of the object
    fn type_index() -> i32;
    /// Return the object header
    /// This function is implemented as a static function so
    ///
    /// # Arguments
    /// * `this` - The object to get the header
    ///
    /// # Returns
    /// * `&mut TVMFFIObject` - The object header
    /// \return The object header
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject;
}

/// Traits for objects with extra items that follows the object
///
/// This extra trait can be helpful to implement array types and string types
pub unsafe trait ObjectCoreWithExtraItems: ObjectCore {
    /// type of extra items storage that follows the object
    type ExtraItem;
    /// Return the number of extra items
    fn extra_items_count(this: &Self) -> usize;
    /// Return the extra items data pointer
    unsafe fn extra_items(this: &Self) -> &[Self::ExtraItem] {
        let extra_items_ptr = (this as *const Self as *const u8).add(std::mem::size_of::<Self>());
        std::slice::from_raw_parts(
            extra_items_ptr as *const Self::ExtraItem,
            Self::extra_items_count(this),
        )
    }
    /// Return the extra items data pointer
    unsafe fn extra_items_mut(this: &mut Self) -> &mut [Self::ExtraItem] {
        let extra_items_ptr = (this as *mut Self as *mut u8).add(std::mem::size_of::<Self>());
        std::slice::from_raw_parts_mut(
            extra_items_ptr as *mut Self::ExtraItem,
            Self::extra_items_count(this),
        )
    }
}

/// Traits to specify core operations of ObjectRef
///
/// used by the ffi Any system and not user facing
///
/// We mark as unsafe since it moves out the internal of the ObjectRef
///
/// # Safety
///
/// `data`, `into_data`, and `from_data` must preserve the same object pointer
/// and form an ownership-preserving round trip. A non-null allocation must start
/// with a valid `TVMFFIObject` header whose registered object-range runtime type
/// index correctly describes its layout and inheritance. A null pointer is the
/// C++ nullable-ObjectRef representation and owns no reference.
///
/// When `Self` also implements [`AnyCompatible`], `copy_to_any_view` must
/// produce a non-owning view, while `move_to_any` must transfer ownership of
/// the same object pointer and dynamic type index (or FFI None for null).
/// `move_from_any_after_check` must be able to reclaim a defined owned
/// representation exactly once, and a true `check_any_strict` result must
/// guarantee that both after-check constructors are valid for it.
pub unsafe trait ObjectRefCore: Sized + Clone {
    type ContainerType: ObjectCore;
    fn data(this: &Self) -> &ObjectArc<Self::ContainerType>;
    fn into_data(this: Self) -> ObjectArc<Self::ContainerType>;
    fn from_data(data: ObjectArc<Self::ContainerType>) -> Self;

    /// Return whether this handle contains a C++ object.
    ///
    /// This is the Rust equivalent of C++ `ObjectRef::defined()`.  Generated
    /// object fields such as `Span`, `Attrs`, and `IterVar::dom` may legally
    /// contain a null object-reference slot even though their Rust field type
    /// is not wrapped in [`Option`].
    #[inline]
    fn is_defined(&self) -> bool {
        !ObjectArc::is_null(Self::data(self))
    }

    /// Return whether this handle is C++'s null `ObjectRef` representation.
    #[inline]
    fn is_null(&self) -> bool {
        ObjectArc::is_null(Self::data(self))
    }

    /// Return whether two object references point to the same C++ object.
    ///
    /// This is C++ `ObjectRef::same_as`. Defining it on the core trait keeps
    /// generated bindings free of one identical inherent method per type.
    #[inline]
    fn same_as<O: ObjectRefCore>(&self, other: &O) -> bool {
        ObjectArc::ptr_eq(Self::data(self), O::data(other))
    }
}

/// Check whether a runtime type index refers to `Target` or one of its
/// subtypes.
///
/// The subtype relation lives in the process-wide type table maintained by the
/// tvm-ffi library: every registered type records its depth in the single
/// inheritance tree together with the chain of its ancestors. The check is
/// O(1) — if `target` really is an ancestor, it must appear in the candidate's
/// ancestor array exactly at `target`'s depth.
///
/// This is a hidden support function for derive-generated object checks. Object
/// indices in the registered range must refer to entries in the runtime type
/// table.
#[doc(hidden)]
#[inline(always)]
pub fn is_instance_of<Target: ObjectCore>(object_type_index: i32) -> bool {
    let target_type_index = Target::type_index();
    is_instance_of_index(object_type_index, target_type_index)
}

/// Runtime-index form of [`is_instance_of`].
#[doc(hidden)]
pub fn is_instance_of_index(object_type_index: i32, target_type_index: i32) -> bool {
    if object_type_index == target_type_index {
        return true;
    }
    let object_begin = TypeIndex::kTVMFFIStaticObjectBegin as i32;
    // Only object types participate in the type hierarchy.
    if object_type_index < object_begin || target_type_index < object_begin {
        return false;
    }
    // Parent indices are always smaller than their descendants.
    if object_type_index < target_type_index {
        return false;
    }
    unsafe {
        let object_info = TVMFFIGetTypeInfo(object_type_index);
        let target_info = TVMFFIGetTypeInfo(target_type_index);
        if object_info.is_null() || target_info.is_null() {
            return false;
        }
        let target_depth = (*target_info).type_depth;
        if (*object_info).type_depth <= target_depth {
            return false;
        }
        let ancestor = *(*object_info).type_acenstors.add(target_depth as usize);
        !ancestor.is_null() && (*ancestor).type_index == target_type_index
    }
}

/// Read one reflected object field through the runtime's owning getter.
///
/// Generated bindings use this path instead of constructing a Rust reference
/// to the complete C++ node layout. The loaded runtime supplies the field
/// offset and getter. The result is an owning [`Any`].
#[doc(hidden)]
pub fn get_object_field_any<O>(object: &O, owner_type_index: i32, field_name: &str) -> Result<Any>
where
    O: ObjectRefCore,
{
    unsafe {
        let object_ptr = ObjectArc::as_raw(O::data(object));
        if object_ptr.is_null() {
            crate::bail!(
                crate::error::VALUE_ERROR,
                "cannot read field `{}` from an undefined object",
                field_name
            );
        }
        let header = object_ptr as *mut TVMFFIObject;
        if !is_instance_of_index((*header).type_index, owner_type_index) {
            crate::bail!(
                crate::error::TYPE_ERROR,
                "runtime object type_index `{}` is not an instance of field owner `{}`",
                (*header).type_index,
                owner_type_index
            );
        }

        let owner_info = TVMFFIGetTypeInfo(owner_type_index);
        if owner_info.is_null() {
            crate::bail!(
                crate::error::TYPE_ERROR,
                "no type info for field owner type_index `{}`",
                owner_type_index
            );
        }
        let owner_info = &*owner_info;
        for index in 0..owner_info.num_fields {
            let field = &*owner_info.fields.add(index as usize);
            if field.name.as_str() != field_name {
                continue;
            }
            let getter = match field.getter {
                Some(getter) => getter,
                None => {
                    crate::bail!(
                        crate::error::RUNTIME_ERROR,
                        "reflected field `{}` has no getter",
                        field_name
                    )
                }
            };
            if field.offset < 0
                || field.size < 0
                || field.alignment <= 0
                || field.offset % field.alignment != 0
            {
                crate::bail!(
                    crate::error::RUNTIME_ERROR,
                    "reflected field `{}` has an invalid layout",
                    field_name
                );
            }
            if !owner_info.metadata.is_null() {
                let total_size = (*owner_info.metadata).total_size;
                let field_end = field.offset.checked_add(field.size);
                if total_size > 0
                    && match field_end {
                        Some(end) => end > i64::from(total_size),
                        None => true,
                    }
                {
                    crate::bail!(
                        crate::error::RUNTIME_ERROR,
                        "reflected field `{}` extends beyond its owner object",
                        field_name
                    );
                }
            }
            let field_ptr = (header as *mut u8).add(field.offset as usize);
            if field_ptr as usize % field.alignment as usize != 0 {
                crate::bail!(
                    crate::error::RUNTIME_ERROR,
                    "reflected field `{}` is not correctly aligned",
                    field_name
                );
            }
            let mut result = TVMFFIAny::new();
            crate::check_safe_call!(getter(field_ptr.cast(), &mut result))?;
            return Ok(Any::from_raw_ffi_any(result));
        }
        crate::bail!(
            crate::error::ATTRIBUTE_ERROR,
            "field `{}` is not registered on type `{}`",
            field_name,
            owner_info.type_key.as_str()
        )
    }
}

/// Read and type-check one reflected object field.
#[doc(hidden)]
pub fn get_object_field<T, O>(object: &O, owner_type_index: i32, field_name: &str) -> Result<T>
where
    T: AnyCompatible,
    O: ObjectRefCore,
{
    let owned = get_object_field_any(object, owner_type_index, field_name)?;
    let converted: TryFromTemp<T> = owned.try_into()?;
    Ok(TryFromTemp::into_value(converted))
}

/// Runtime-checked casting between arbitrary `ObjectRef` types.
///
/// The cast uses the target's [`AnyCompatible::check_any_strict`] implementation,
/// mirroring the semantics of `ObjectRef::as<T>` in C++. This supports both
/// object hierarchies and parameterized object containers.
///
/// This trait is blanket-implemented for every [`ObjectRefCore`] type that is
/// also [`AnyCompatible`].
pub trait ObjectRefCast: ObjectRefCore + AnyCompatible {
    /// Borrow and clone `self`, then cast it to another object-ref type.
    #[inline(always)]
    fn downcast<B>(&self) -> crate::error::Result<B>
    where
        B: ObjectRefCore + AnyCompatible,
    {
        self.clone().try_cast()
    }

    /// Consume `self` and rewrap the underlying object as `B` without copying.
    #[inline(always)]
    fn try_cast<B>(self) -> crate::error::Result<B>
    where
        B: ObjectRefCore + AnyCompatible,
    {
        let mut any_data = TVMFFIAny::new();
        unsafe {
            // Keep ownership in `self` while the target check runs. This makes
            // the failure and panic paths unwind normally instead of stranding
            // an owned object inside a raw TVMFFIAny.
            Self::copy_to_any_view(&self, &mut any_data);

            if B::check_any_strict(&any_data) {
                // Transfer ownership only after the borrowed representation has
                // passed the target's complete hierarchy/container check.
                Self::move_to_any(self, &mut any_data);
                Ok(B::move_from_any_after_check(&mut any_data))
            } else {
                let msg = format!(
                    "Cannot convert from type `{}` to `{}`",
                    B::get_mismatch_type_info(&any_data),
                    B::type_str()
                );
                Err(crate::error::Error::new(crate::error::TYPE_ERROR, &msg, ""))
            }
        }
    }
}

impl<T: ObjectRefCore + AnyCompatible> ObjectRefCast for T {}

/// Base class for ObjectRef.
///
/// This class is used to store the data of the ObjectRef. Erasing a concrete
/// object's type must not make it transferable to another thread:
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<tvm_ffi::object::ObjectRef>();
/// ```
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct ObjectRef {
    data: ObjectArc<Object>,
}

/// Unsafe operations on object
#[doc(hidden)]
pub mod unsafe_ {
    use tvm_ffi_sys::{
        COMBINED_REF_COUNT_BOTH_ONE, COMBINED_REF_COUNT_MASK_U32, COMBINED_REF_COUNT_STRONG_ONE,
        COMBINED_REF_COUNT_WEAK_ONE,
    };

    use std::ffi::c_void;
    use std::sync::atomic::{fence, Ordering};
    use tvm_ffi_sys::TVMFFIObject;
    use tvm_ffi_sys::TVMFFIObjectDeleterFlagBitMask::{
        kTVMFFIObjectDeleterFlagBitMaskBoth, kTVMFFIObjectDeleterFlagBitMaskStrong,
        kTVMFFIObjectDeleterFlagBitMaskWeak,
    };

    /// Increase the strong reference count of the object
    ///
    /// This function is same as TVMFFIObjectIncRef but implemented natively in Rust
    ///
    /// # Arguments
    /// * `obj` - The object to increase the reference count
    #[inline]
    pub unsafe fn inc_ref(handle: *mut TVMFFIObject) {
        let obj = &mut *handle;
        obj.combined_ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrease the strong reference count of the object
    ///
    /// This function is same as TVMFFIObjectDecRef but implemented natively in Rust
    ///
    /// # Arguments
    /// * `obj` - The object to decrease the reference count
    #[inline]
    pub(crate) unsafe fn dec_ref(handle: *mut TVMFFIObject) {
        let obj = &mut *handle;
        let old_combined_count = obj
            .combined_ref_count
            .fetch_sub(COMBINED_REF_COUNT_STRONG_ONE, Ordering::Relaxed);
        if old_combined_count == COMBINED_REF_COUNT_BOTH_ONE {
            if let Some(deleter) = obj.deleter {
                fence(Ordering::Acquire);
                deleter(
                    obj as *mut TVMFFIObject as *mut c_void,
                    kTVMFFIObjectDeleterFlagBitMaskBoth as i32,
                );
            }
        } else if (old_combined_count & COMBINED_REF_COUNT_MASK_U32)
            == COMBINED_REF_COUNT_STRONG_ONE
        {
            // slow path, there is still a weak reference left
            // need to run two phase decrement
            fence(Ordering::Acquire);
            if let Some(deleter) = obj.deleter {
                deleter(
                    obj as *mut TVMFFIObject as *mut c_void,
                    kTVMFFIObjectDeleterFlagBitMaskStrong as i32,
                );
            }
            let old_weak_count = obj
                .combined_ref_count
                .fetch_sub(COMBINED_REF_COUNT_WEAK_ONE, Ordering::Release);
            if old_weak_count == COMBINED_REF_COUNT_WEAK_ONE {
                fence(Ordering::Acquire);
                if let Some(deleter) = obj.deleter {
                    deleter(
                        obj as *mut TVMFFIObject as *mut c_void,
                        kTVMFFIObjectDeleterFlagBitMaskWeak as i32,
                    );
                }
            }
        }
    }

    #[inline]
    pub(crate) unsafe fn strong_count(handle: *mut TVMFFIObject) -> usize {
        let obj = &mut *handle;
        (obj.combined_ref_count.load(Ordering::Relaxed) & COMBINED_REF_COUNT_MASK_U32) as usize
    }

    #[inline]
    pub(crate) unsafe fn weak_count(handle: *mut TVMFFIObject) -> usize {
        let obj = &mut *handle;
        (obj.combined_ref_count.load(Ordering::Relaxed) >> 32) as usize
    }

    /// Generic object deleter for objects allocated through Rust's global allocator.
    pub(crate) unsafe extern "C" fn object_deleter_for_new<T>(ptr: *mut c_void, flags: i32)
    where
        T: super::ObjectCore,
    {
        let obj = ptr as *mut T;
        if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
            std::ptr::drop_in_place(obj);
        }
        if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
            std::alloc::dealloc(ptr as *mut u8, std::alloc::Layout::new::<T>());
        }
    }

    pub(crate) unsafe extern "C" fn object_deleter_for_new_with_extra_items<T, U>(
        ptr: *mut c_void,
        flags: i32,
    ) where
        T: super::ObjectCoreWithExtraItems<ExtraItem = U>,
    {
        let obj = ptr as *mut T;
        if flags == kTVMFFIObjectDeleterFlagBitMaskBoth as i32 {
            let extra_items_count = T::extra_items_count(&(*obj));
            std::ptr::drop_in_place(obj);
            let layout = std::alloc::Layout::from_size_align(
                std::mem::size_of::<T>() + extra_items_count * std::mem::size_of::<U>(),
                std::mem::align_of::<T>(),
            )
            .unwrap();
            std::alloc::dealloc(ptr as *mut u8, layout);
        } else {
            assert_eq!(std::mem::size_of::<T>() % std::mem::size_of::<u64>(), 0);
            if flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0 {
                let extra_items_count = T::extra_items_count(&(*obj));
                std::ptr::drop_in_place(obj);
                std::ptr::write(obj as *mut u64, extra_items_count as u64);
            }
            if flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0 {
                let extra_items_count = std::ptr::read(obj as *mut u64) as usize;
                let layout = std::alloc::Layout::from_size_align(
                    std::mem::size_of::<T>() + extra_items_count * std::mem::size_of::<U>(),
                    std::mem::align_of::<T>(),
                )
                .unwrap();
                std::alloc::dealloc(ptr as *mut u8, layout);
            }
        }
    }
}

//---------------------
// Object
//---------------------

impl Object {
    pub fn new() -> Self {
        Self {
            header: TVMFFIObject::new(),
            _not_send_sync: PhantomData,
        }
    }
}

unsafe impl ObjectCore for Object {
    const TYPE_KEY: &'static str = "ffi.Object";
    const TYPE_DEPTH: i32 = 0;
    #[inline]
    fn type_index() -> i32 {
        TypeIndex::kTVMFFIStaticObjectBegin as i32
    }
    #[inline]
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        &mut this.header
    }
}

//---------------------
// ObjectArc
//---------------------

impl<T: ObjectCore> ObjectArc<T> {
    /// Return whether two handles point to the same object allocation.
    #[inline]
    pub fn ptr_eq<U: ObjectCore>(this: &Self, other: &ObjectArc<U>) -> bool {
        this.ptr.cast::<()>() == other.ptr.cast::<()>()
    }

    pub fn new(data: T) -> Self {
        unsafe {
            let layout = std::alloc::Layout::new::<T>();
            let raw_data_ptr = std::alloc::alloc(layout);
            if raw_data_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let ptr = raw_data_ptr as *mut T;
            std::ptr::write(ptr, data);
            // now override the header directly
            std::ptr::write(
                ptr as *mut TVMFFIObject,
                TVMFFIObject {
                    combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                    type_index: T::type_index(),
                    __padding: 0,
                    deleter: Some(unsafe_::object_deleter_for_new::<T>),
                },
            );
            // move into the object arc ptr
            Self {
                ptr,
                _phantom: std::marker::PhantomData,
            }
        }
    }
    pub fn new_with_extra_items<U>(data: T) -> Self
    where
        T: ObjectCoreWithExtraItems<ExtraItem = U>,
    {
        unsafe {
            // ensure strict alignment requirements
            // so we can have { T, U*extra_items } layout
            assert_eq!(std::mem::align_of::<T>() % std::mem::align_of::<U>(), 0);
            assert_eq!(std::mem::size_of::<T>() % std::mem::align_of::<U>(), 0);
            let extra_items_count = T::extra_items_count(&data);
            let layout = std::alloc::Layout::from_size_align(
                std::mem::size_of::<T>() + extra_items_count * std::mem::size_of::<U>(),
                std::mem::align_of::<T>(),
            )
            .unwrap();
            let raw_data_ptr = std::alloc::alloc(layout);
            if raw_data_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            let ptr = raw_data_ptr as *mut T;
            std::ptr::write(ptr, data);
            // now override the header directly
            std::ptr::write(
                ptr as *mut TVMFFIObject,
                TVMFFIObject {
                    combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                    type_index: T::type_index(),
                    __padding: 0,
                    deleter: Some(unsafe_::object_deleter_for_new_with_extra_items::<T, U>),
                },
            );
            // move into the object arc ptr
            Self {
                ptr,
                _phantom: std::marker::PhantomData,
            }
        }
    }

    /// Move a previously allocated object into the ObjectArc
    ///
    /// # Arguments
    /// * `ptr` - The raw pointer to move into the ObjectArc
    ///
    /// # Returns
    /// * `ObjectArc<T>` - The ObjectArc
    /// \return The ObjectArc
    #[inline]
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        Self {
            ptr: ptr as *mut T,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Move the ObjectArc into a raw pointer
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to move into a raw pointer
    ///
    /// # Returns
    /// * `*const T` - The raw pointer
    #[inline]
    pub unsafe fn into_raw(this: Self) -> *const T {
        let droped_this = std::mem::ManuallyDrop::new(this);
        droped_this.ptr as *const T
    }

    /// Get the raw pointer from the ObjectArc
    ///
    /// Caller should view this as a non-owning reference
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the raw pointer
    ///
    /// # Returns
    /// * `*const T` - The raw pointer
    /// \return The raw pointer
    #[inline]
    pub unsafe fn as_raw(this: &Self) -> *const T {
        this.ptr as *const T
    }

    /// Get the raw mutable pointer from the ObjectArc
    ///
    /// Caller should view this as a non-owning reference
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the raw pointer
    ///
    /// # Returns
    /// * `*mut T` - The raw pointer
    #[inline]
    pub unsafe fn as_raw_mut(this: &mut Self) -> *mut T {
        this.ptr
    }

    /// Return whether this is C++'s null ObjectRef representation.
    #[inline]
    pub fn is_null(this: &Self) -> bool {
        this.ptr.is_null()
    }

    /// Get the strong reference count of the ObjectArc
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the strong reference count
    ///
    /// # Returns
    /// * `usize` - The strong reference count
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        if this.ptr.is_null() {
            0
        } else {
            unsafe { unsafe_::strong_count(this.ptr.cast::<TVMFFIObject>()) }
        }
    }

    /// Get the weak reference count of the ObjectArc
    ///
    /// # Arguments
    /// * `this` - The ObjectArc to get the weak reference count
    ///
    /// # Returns
    /// * `usize` - The weak reference count
    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        if this.ptr.is_null() {
            0
        } else {
            unsafe { unsafe_::weak_count(this.ptr.cast::<TVMFFIObject>()) }
        }
    }
}

// implement Deref for ObjectArc
impl<T: ObjectCore> Deref for ObjectArc<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        assert!(
            !self.ptr.is_null(),
            "attempted to dereference a null TVM ObjectRef"
        );
        unsafe { &*self.ptr }
    }
}

// implement Drop for ObjectArc
impl<T: ObjectCore> Drop for ObjectArc<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { unsafe_::dec_ref(self.ptr.cast::<TVMFFIObject>()) }
        }
    }
}

// implement Clone for ObjectArc
impl<T: ObjectCore> Clone for ObjectArc<T> {
    #[inline]
    fn clone(&self) -> Self {
        if !self.ptr.is_null() {
            unsafe { unsafe_::inc_ref(self.ptr.cast::<TVMFFIObject>()) }
        }
        Self {
            ptr: self.ptr,
            _phantom: std::marker::PhantomData,
        }
    }
}
