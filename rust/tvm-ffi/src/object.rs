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
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::AtomicU64;

use crate::any::Any;
use crate::derive::ObjectRef;
use crate::error::Result;
use crate::type_traits::AnyCompatible;
pub use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;
/// Object related ABI handling
use tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIFieldInfo, TVMFFIGetTypeInfo, TVMFFIMethodInfo, TVMFFIObject,
    TVMFFITypeAttrColumn, TVMFFITypeInfo, TVMFFITypeMetadata, COMBINED_REF_COUNT_BOTH_ONE,
};

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
///
/// ```compile_fail
/// fn require_sync<T: Sync>() {}
/// require_sync::<tvm_ffi::ObjectArc<tvm_ffi::Object>>();
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
    /// Whether `type_index()` is a process-lifetime constant that needs no
    /// dynamic registry lookup.
    #[doc(hidden)]
    const TYPE_INDEX_STATIC: bool = false;
    // return the type index of the object
    fn type_index() -> i32;
    /// Fallible type-index lookup for dynamically registered object types.
    ///
    /// Static builtins use this default. Derive-generated dynamic types
    /// override it so a missing registry entry is a normal mismatch rather
    /// than a poisoned lazy-initializer panic.
    fn try_type_index() -> Result<i32> {
        Ok(Self::type_index())
    }
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

/// Marker for a complete object layout that Rust may allocate and destroy.
///
/// [`ObjectCore`] alone only describes the object header and runtime type. It
/// is intentionally sufficient for a Rust view over an object allocated by
/// C++, including a partial layout mirror. Such a view must not be passed to
/// [`ObjectArc::new`].
///
/// # Safety
///
/// `Self` must be a complete `#[repr(C)]` Rust layout whose object header is at
/// offset zero. [`drop_payload`](RustAllocatableObject::drop_payload) must end
/// the lifetime of every resource-owning field after that header without
/// reading, writing, or ending the header itself. The header remains live while
/// concurrent C++ weak references inspect its atomic counts and until the last
/// weak reference releases the allocation.
///
/// Once an object is exposed to C++, its final strong reference may be released
/// on any thread. Every payload resource and its `drop_payload` implementation
/// must therefore permit destruction on an arbitrary thread, even though the
/// Rust handle itself is deliberately not `Send`.
///
/// After `drop_payload` returns, it must be valid to deallocate the storage
/// without running `Self`'s ordinary destructor. Implementations therefore
/// manually drop each payload field that needs it; they must never call
/// `drop_in_place` on the complete `Self`.
/// This marker intentionally has no safe derive: matching the complete native
/// layout and its invariants requires an explicit `unsafe impl` audit.
///
/// Deriving [`ObjectCore`] without opting into this marker leaves a foreign
/// layout non-allocatable:
///
/// ```compile_fail
/// #[repr(C)]
/// #[derive(tvm_ffi::derive::Object)]
/// #[type_key = "example.ForeignMirror"]
/// struct ForeignMirror {
///     base: tvm_ffi::Object,
/// }
///
/// let value = ForeignMirror { base: tvm_ffi::Object::new() };
/// let _ = tvm_ffi::ObjectArc::new(value);
/// ```
///
/// Object layouts must use the C ABI:
///
/// ```compile_fail
/// #[derive(tvm_ffi::derive::Object)]
/// #[type_key = "example.MissingReprC"]
/// struct MissingReprC {
///     base: tvm_ffi::Object,
/// }
/// ```
pub unsafe trait RustAllocatableObject: ObjectCore {
    /// Destroy all Rust-owned payload while leaving `TVMFFIObject` untouched.
    unsafe fn drop_payload(this: *mut Self);
}

/// Traits for objects with extra items that follows the object
///
/// This extra trait can be helpful to implement array types and string types
///
/// # Safety
///
/// Every instance must be followed immediately by storage for at least
/// `extra_items_count` properly aligned `ExtraItem` slots, and callers must
/// initialize a slot before reading it. The count describes allocated slots,
/// not necessarily the number currently initialized, and must never exceed the
/// count used when the Rust allocation was created. If initialized extra items
/// own resources but are stored in a raw ABI representation without [`Drop`],
/// the containing object's destructor must release exactly the initialized
/// items.
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

    /// Return the extra storage without claiming that its elements are initialized.
    ///
    /// Constructors must use this view until every returned slot has been
    /// initialized. Creating `&mut [ExtraItem]` over freshly allocated bytes is
    /// undefined behavior for types whose bit patterns are not all valid.
    unsafe fn extra_items_uninit_mut(this: &mut Self) -> &mut [MaybeUninit<Self::ExtraItem>] {
        let extra_items_ptr = (this as *mut Self as *mut u8).add(std::mem::size_of::<Self>());
        std::slice::from_raw_parts_mut(
            extra_items_ptr as *mut MaybeUninit<Self::ExtraItem>,
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

    /// Return the object's dynamic runtime type index, or `None` for a null handle.
    #[inline]
    fn runtime_type_index(&self) -> Option<i32> {
        let object = unsafe { ObjectArc::as_raw(Self::data(self)) };
        if object.is_null() {
            None
        } else {
            // ObjectRefCore requires every non-null allocation to begin with a
            // valid TVMFFIObject header. The dynamic index lives in that prefix.
            Some(unsafe { (*(object.cast::<TVMFFIObject>())).type_index })
        }
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
    Target::try_type_index()
        .is_ok_and(|target_type_index| is_instance_of_index(object_type_index, target_type_index))
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
    let (object_info, target_info) = unsafe {
        let Ok(object_info) = checked_type_info(object_type_index) else {
            return false;
        };
        let Ok(target_info) = checked_type_info(target_type_index) else {
            return false;
        };
        (object_info, target_info)
    };
    if object_info.type_depth <= target_info.type_depth {
        return false;
    }
    let Ok(ancestor) =
        (unsafe { checked_type_ancestor(object_info, target_info.type_depth as usize) })
    else {
        return false;
    };
    ancestor.type_index == target_type_index
}

fn invalid_reflection_metadata(type_index: i32, detail: &str) -> crate::error::Error {
    crate::error::Error::new(
        crate::error::RUNTIME_ERROR,
        &format!("invalid reflection metadata for type_index `{type_index}`: {detail}"),
        "",
    )
}

/// Whether the ABI permits a registry entry at this index. This catches the
/// fixed gaps before calling `TVMFFIGetTypeInfo`, whose native contract aborts
/// the process when its precondition (a registered index) is violated.
fn can_have_registered_type_info(type_index: i32) -> bool {
    (TypeIndex::kTVMFFINone as i32..=TypeIndex::kTVMFFISmallBytes as i32).contains(&type_index)
        || (TypeIndex::kTVMFFIStaticObjectBegin as i32..TypeIndex::kTVMFFIStaticObjectEnd as i32)
            .contains(&type_index)
        || type_index >= TypeIndex::kTVMFFIDynObjectBegin as i32
}

/// Validate the observable shape of one process-lifetime native metadata
/// table before creating a Rust slice for it.
///
/// This rejects malformed counts, null or misaligned non-empty tables, and
/// extents that exceed Rust's maximum slice size. As with every C ABI, Rust
/// cannot prove that an otherwise plausible foreign pointer still designates
/// an allocation of the advertised extent; that remains a native registry
/// invariant.
unsafe fn checked_metadata_slice<T>(
    type_index: i32,
    label: &str,
    data: *const T,
    count: i32,
) -> Result<&'static [T]> {
    if count < 0 {
        return Err(invalid_reflection_metadata(
            type_index,
            &format!("{label} count is negative"),
        ));
    }
    let count = count as usize;
    if count == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(invalid_reflection_metadata(
            type_index,
            &format!("{label} table is null"),
        ));
    }
    if data as usize % std::mem::align_of::<T>() != 0 {
        return Err(invalid_reflection_metadata(
            type_index,
            &format!("{label} table is misaligned"),
        ));
    }
    let table_bytes = count.checked_mul(std::mem::size_of::<T>());
    if !matches!(table_bytes, Some(bytes) if bytes <= isize::MAX as usize) {
        return Err(invalid_reflection_metadata(
            type_index,
            &format!("{label} table is too large"),
        ));
    }
    Ok(std::slice::from_raw_parts(data, count))
}

/// Validate and decode a UTF-8 string embedded in reflection metadata.
pub(crate) unsafe fn checked_metadata_str<'a>(
    type_index: i32,
    label: &str,
    value: &'a TVMFFIByteArray,
) -> Result<&'a str> {
    if value.size == 0 {
        return Ok("");
    }
    if value.data.is_null() {
        return Err(invalid_reflection_metadata(
            type_index,
            &format!("{label} has a null data pointer"),
        ));
    }
    if value.size > isize::MAX as usize {
        return Err(invalid_reflection_metadata(
            type_index,
            &format!("{label} is too large"),
        ));
    }
    let bytes = std::slice::from_raw_parts(value.data, value.size);
    std::str::from_utf8(bytes).map_err(|_| {
        invalid_reflection_metadata(type_index, &format!("{label} is not valid UTF-8"))
    })
}

/// Fetch a registry entry and validate its identity and hierarchy table shape.
///
/// The type table owns its entries for the process lifetime. This validation is
/// deliberately shared by generated field access and structural traversal so
/// neither path turns malformed native metadata into a Rust reference.
pub(crate) unsafe fn checked_type_info(type_index: i32) -> Result<&'static TVMFFITypeInfo> {
    if !can_have_registered_type_info(type_index) {
        return Err(invalid_reflection_metadata(
            type_index,
            "type index lies in an unregistered ABI range",
        ));
    }
    let info = TVMFFIGetTypeInfo(type_index);
    if info.is_null() {
        return Err(invalid_reflection_metadata(type_index, "type info is null"));
    }
    if info as usize % std::mem::align_of::<TVMFFITypeInfo>() != 0 {
        return Err(invalid_reflection_metadata(
            type_index,
            "type info is misaligned",
        ));
    }
    let info = &*info;
    if info.type_index != type_index {
        return Err(invalid_reflection_metadata(
            type_index,
            "registry entry reports a different type index",
        ));
    }
    if info.type_depth < 0 {
        return Err(invalid_reflection_metadata(
            type_index,
            "type depth is negative",
        ));
    }
    if info.type_depth != 0 && info.type_acenstors.is_null() {
        return Err(invalid_reflection_metadata(
            type_index,
            "ancestor table is null",
        ));
    }
    Ok(&*(info as *const TVMFFITypeInfo))
}

/// Return a diagnostic type key without trusting malformed registry strings.
pub(crate) fn type_key_or_index(type_index: i32) -> String {
    unsafe {
        let Ok(info) = checked_type_info(type_index) else {
            return format!("<type_index {type_index}>");
        };
        checked_metadata_str(type_index, "type key", &info.type_key)
            .map(str::to_owned)
            .unwrap_or_else(|_| format!("<type_index {type_index}>"))
    }
}

unsafe fn checked_type_ancestor_entry(
    info: &'static TVMFFITypeInfo,
    ancestors: &'static [*const TVMFFITypeInfo],
    depth: usize,
) -> Result<&'static TVMFFITypeInfo> {
    let Some(&ancestor) = ancestors.get(depth) else {
        return Err(invalid_reflection_metadata(
            info.type_index,
            "ancestor depth is out of range",
        ));
    };
    if ancestor.is_null() {
        return Err(invalid_reflection_metadata(
            info.type_index,
            "ancestor table contains a null entry",
        ));
    }
    if ancestor as usize % std::mem::align_of::<TVMFFITypeInfo>() != 0 {
        return Err(invalid_reflection_metadata(
            info.type_index,
            "ancestor table contains a misaligned entry",
        ));
    }
    let ancestor = &*ancestor;
    if ancestor.type_depth != depth as i32 {
        return Err(invalid_reflection_metadata(
            info.type_index,
            "ancestor table has an inconsistent depth",
        ));
    }
    if depth == 0 && ancestor.type_index != TypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return Err(invalid_reflection_metadata(
            info.type_index,
            "ancestor table does not start at ffi.Object",
        ));
    }
    if ancestor.type_index >= info.type_index {
        return Err(invalid_reflection_metadata(
            info.type_index,
            "ancestor type index is not older than its descendant",
        ));
    }
    Ok(ancestor)
}

/// Validate and borrow one entry from a type's ancestor table in O(1).
pub(crate) unsafe fn checked_type_ancestor(
    info: &'static TVMFFITypeInfo,
    depth: usize,
) -> Result<&'static TVMFFITypeInfo> {
    let ancestors = checked_metadata_slice(
        info.type_index,
        "ancestor",
        info.type_acenstors,
        info.type_depth,
    )?;
    checked_type_ancestor_entry(info, ancestors, depth)
}

/// Validate and borrow a type's complete ancestor table.
pub(crate) unsafe fn checked_type_ancestors(
    info: &'static TVMFFITypeInfo,
) -> Result<&'static [*const TVMFFITypeInfo]> {
    let ancestors = checked_metadata_slice(
        info.type_index,
        "ancestor",
        info.type_acenstors,
        info.type_depth,
    )?;
    for expected_depth in 0..ancestors.len() {
        checked_type_ancestor_entry(info, ancestors, expected_depth)?;
    }
    Ok(ancestors)
}

/// Validate and borrow one registry entry's field table.
pub(crate) unsafe fn checked_type_fields(
    info: &'static TVMFFITypeInfo,
) -> Result<&'static [TVMFFIFieldInfo]> {
    checked_metadata_slice(info.type_index, "field", info.fields, info.num_fields)
}

/// Validate and borrow one registry entry's method table.
pub(crate) unsafe fn checked_type_methods(
    info: &'static TVMFFITypeInfo,
) -> Result<&'static [TVMFFIMethodInfo]> {
    checked_metadata_slice(info.type_index, "method", info.methods, info.num_methods)
}

/// Validate and copy one cell from a type-attribute column.
pub(crate) unsafe fn checked_type_attr(
    column: *const TVMFFITypeAttrColumn,
    type_index: i32,
    attr_name: &str,
) -> Result<Option<TVMFFIAny>> {
    if column.is_null() {
        return Ok(None);
    }
    if column as usize % std::mem::align_of::<TVMFFITypeAttrColumn>() != 0 {
        return Err(invalid_reflection_metadata(
            type_index,
            &format!("type attribute `{attr_name}` column is misaligned"),
        ));
    }
    let column = &*column;
    let cells = checked_metadata_slice(
        type_index,
        &format!("type attribute `{attr_name}` column"),
        column.data,
        column.size,
    )?;
    let offset = i64::from(type_index) - i64::from(column.begin_index);
    if offset < 0 || offset >= i64::from(column.size) {
        return Ok(None);
    }
    Ok(Some(cells[offset as usize]))
}

/// Validate and borrow the optional fixed-layout metadata for a type.
pub(crate) unsafe fn checked_type_metadata(
    info: &'static TVMFFITypeInfo,
) -> Result<Option<&'static TVMFFITypeMetadata>> {
    if info.metadata.is_null() {
        return Ok(None);
    }
    if info.metadata as usize % std::mem::align_of::<TVMFFITypeMetadata>() != 0 {
        return Err(invalid_reflection_metadata(
            info.type_index,
            "object-size metadata is misaligned",
        ));
    }
    Ok(Some(&*info.metadata))
}

/// Check an object-valued reflection cell before an owning conversion touches
/// the object's reference count.
pub(crate) unsafe fn checked_object_cell(data: &TVMFFIAny, label: &str) -> Result<()> {
    if data.type_index < TypeIndex::kTVMFFIStaticObjectBegin as i32 {
        return Err(invalid_reflection_metadata(
            data.type_index,
            &format!("{label} is not object-valued"),
        ));
    }
    let object = data.data_union.v_obj;
    if object.is_null() {
        return Err(invalid_reflection_metadata(
            data.type_index,
            &format!("{label} has a null object pointer"),
        ));
    }
    if object as usize % std::mem::align_of::<TVMFFIObject>() != 0 {
        return Err(invalid_reflection_metadata(
            data.type_index,
            &format!("{label} has a misaligned object pointer"),
        ));
    }
    if (*object).type_index != data.type_index {
        return Err(invalid_reflection_metadata(
            data.type_index,
            &format!("{label} disagrees with its object header type"),
        ));
    }
    Ok(())
}

/// Validate one reflected field and compute its address within an object.
pub(crate) unsafe fn checked_field_address(
    object: *mut TVMFFIObject,
    owner_info: &'static TVMFFITypeInfo,
    field: &TVMFFIFieldInfo,
) -> Result<*mut std::ffi::c_void> {
    let metadata = checked_type_metadata(owner_info)?.ok_or_else(|| {
        invalid_reflection_metadata(owner_info.type_index, "object size metadata is missing")
    })?;
    let total_size = i64::from(metadata.total_size);
    let header_size = std::mem::size_of::<TVMFFIObject>() as i64;
    if total_size < header_size {
        return Err(invalid_reflection_metadata(
            owner_info.type_index,
            "object size is smaller than its header",
        ));
    }
    if field.offset < header_size || field.size < 0 || field.alignment <= 0 {
        return Err(invalid_reflection_metadata(
            owner_info.type_index,
            "field has a negative or header-overlapping layout",
        ));
    }
    let alignment = usize::try_from(field.alignment).map_err(|_| {
        invalid_reflection_metadata(owner_info.type_index, "field alignment does not fit usize")
    })?;
    if !alignment.is_power_of_two() || field.offset % field.alignment != 0 {
        return Err(invalid_reflection_metadata(
            owner_info.type_index,
            "field alignment is invalid",
        ));
    }
    let field_end = field.offset.checked_add(field.size).ok_or_else(|| {
        invalid_reflection_metadata(owner_info.type_index, "field extent overflows")
    })?;
    if field_end > total_size {
        return Err(invalid_reflection_metadata(
            owner_info.type_index,
            "field extends beyond its owner object",
        ));
    }
    let field_ptr = object.cast::<u8>().add(field.offset as usize);
    if field_ptr as usize % alignment != 0 {
        return Err(invalid_reflection_metadata(
            owner_info.type_index,
            "field address is not correctly aligned",
        ));
    }
    Ok(field_ptr.cast())
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

        let owner_info = checked_type_info(owner_type_index)?;
        for field in checked_type_fields(owner_info)? {
            let registered_name =
                checked_metadata_str(owner_info.type_index, "field name", &field.name)?;
            if registered_name != field_name {
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
            let field_ptr = checked_field_address(header, owner_info, field)?;
            let mut result = TVMFFIAny::new();
            crate::check_safe_call!(getter(field_ptr, &mut result))?;
            return Ok(Any::from_raw_ffi_any(result));
        }
        crate::bail!(
            crate::error::ATTRIBUTE_ERROR,
            "field `{}` is not registered on type `{}`",
            field_name,
            checked_metadata_str(owner_info.type_index, "type key", &owner_info.type_key)?
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
    owned.try_as::<T>().ok_or_else(|| {
        crate::error::Error::new(
            crate::error::TYPE_ERROR,
            &format!(
                "reflected field `{field_name}` returned type_index `{}`; expected exact `{}`",
                owned.type_index(),
                T::type_str(),
            ),
            "",
        )
    })
}

/// Runtime-checked casting from an `ObjectRef` into an exact compatible value.
///
/// The cast uses the target's [`AnyCompatible::check_any_strict`] implementation,
/// mirroring the semantics of `ObjectRef::as<T>` in C++. This supports object
/// hierarchies, parameterized object containers, and checked transparent
/// refinements that intentionally do not implement [`ObjectRefCore`].
///
/// This trait is blanket-implemented for every [`ObjectRefCore`] type that is
/// also [`AnyCompatible`].
pub trait ObjectRefCast: ObjectRefCore + AnyCompatible {
    /// Borrow and clone `self`, then cast it to another object-ref type.
    #[inline(always)]
    fn downcast<B>(&self) -> crate::error::Result<B>
    where
        B: AnyCompatible,
    {
        self.clone().try_cast()
    }

    /// Consume `self` and rewrap the underlying object as `B` without copying.
    #[inline(always)]
    fn try_cast<B>(self) -> crate::error::Result<B>
    where
        B: AnyCompatible,
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
///
/// ```compile_fail
/// fn require_sync<T: Sync>() {}
/// require_sync::<tvm_ffi::object::ObjectRef>();
/// ```
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct ObjectRef {
    data: ObjectArc<Object>,
}

#[repr(C)]
struct ExtraItemsAllocationHeader {
    count: usize,
}

fn extra_items_layout<T, U>(count: usize) -> (std::alloc::Layout, usize) {
    let (header_and_object, object_offset) =
        std::alloc::Layout::new::<ExtraItemsAllocationHeader>()
            .extend(std::alloc::Layout::new::<T>())
            .expect("allocation header plus object exceeds the maximum allocation size");
    let items = std::alloc::Layout::array::<U>(count)
        .expect("extra-item count exceeds the maximum allocation size");
    let (layout, items_offset) = header_and_object
        .extend(items)
        .expect("object plus extra items exceeds the maximum allocation size");
    assert_eq!(
        items_offset,
        object_offset + std::mem::size_of::<T>(),
        "extra items must be aligned immediately after the object"
    );
    (layout.pad_to_align(), object_offset)
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
        let count = &(*handle).combined_ref_count;
        let mut current = count.load(Ordering::Relaxed);
        loop {
            assert_ne!(
                current & COMBINED_REF_COUNT_MASK_U32,
                COMBINED_REF_COUNT_MASK_U32,
                "TVM object strong-reference count overflow"
            );
            match count.compare_exchange_weak(
                current,
                current + COMBINED_REF_COUNT_STRONG_ONE,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    /// Decrease the strong reference count of the object
    ///
    /// This function is same as TVMFFIObjectDecRef but implemented natively in Rust
    ///
    /// # Arguments
    /// * `obj` - The object to decrease the reference count
    #[inline]
    pub(crate) unsafe fn dec_ref(handle: *mut TVMFFIObject) {
        let old_combined_count = (*handle)
            .combined_ref_count
            .fetch_sub(COMBINED_REF_COUNT_STRONG_ONE, Ordering::Release);
        if old_combined_count == COMBINED_REF_COUNT_BOTH_ONE {
            if let Some(deleter) = (*handle).deleter {
                fence(Ordering::Acquire);
                deleter(
                    handle.cast::<c_void>(),
                    kTVMFFIObjectDeleterFlagBitMaskBoth as i32,
                );
            }
        } else if (old_combined_count & COMBINED_REF_COUNT_MASK_U32)
            == COMBINED_REF_COUNT_STRONG_ONE
        {
            // slow path, there is still a weak reference left
            // need to run two phase decrement
            fence(Ordering::Acquire);
            if let Some(deleter) = (*handle).deleter {
                deleter(
                    handle.cast::<c_void>(),
                    kTVMFFIObjectDeleterFlagBitMaskStrong as i32,
                );
            }
            let old_weak_count = (*handle)
                .combined_ref_count
                .fetch_sub(COMBINED_REF_COUNT_WEAK_ONE, Ordering::Release);
            if old_weak_count == COMBINED_REF_COUNT_WEAK_ONE {
                fence(Ordering::Acquire);
                if let Some(deleter) = (*handle).deleter {
                    deleter(
                        handle.cast::<c_void>(),
                        kTVMFFIObjectDeleterFlagBitMaskWeak as i32,
                    );
                }
            }
        }
    }

    #[inline]
    pub(crate) unsafe fn strong_count(handle: *mut TVMFFIObject) -> usize {
        ((*handle).combined_ref_count.load(Ordering::Relaxed) & COMBINED_REF_COUNT_MASK_U32)
            as usize
    }

    #[inline]
    pub(crate) unsafe fn weak_count(handle: *mut TVMFFIObject) -> usize {
        ((*handle).combined_ref_count.load(Ordering::Relaxed) >> 32) as usize
    }

    /// Generic object deleter for objects allocated through Rust's global allocator.
    pub(crate) unsafe extern "C" fn object_deleter_for_new<T>(ptr: *mut c_void, flags: i32)
    where
        T: super::RustAllocatableObject,
    {
        let obj = ptr as *mut T;
        let strong = flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0;
        let weak = flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0;
        if strong && weak {
            T::drop_payload(obj);
            std::alloc::dealloc(ptr as *mut u8, std::alloc::Layout::new::<T>());
        } else if strong {
            T::drop_payload(obj);
        } else if weak {
            std::alloc::dealloc(ptr as *mut u8, std::alloc::Layout::new::<T>());
        }
    }

    pub(crate) unsafe extern "C" fn object_deleter_for_new_with_extra_items<T, U>(
        ptr: *mut c_void,
        flags: i32,
    ) where
        T: super::ObjectCoreWithExtraItems<ExtraItem = U> + super::RustAllocatableObject,
    {
        let obj = ptr as *mut T;
        let strong = flags & kTVMFFIObjectDeleterFlagBitMaskStrong as i32 != 0;
        let weak = flags & kTVMFFIObjectDeleterFlagBitMaskWeak as i32 != 0;
        let (_, object_offset) = super::extra_items_layout::<T, U>(0);
        let allocation = (obj as *mut u8).sub(object_offset);
        let extra_items_count = (*(allocation.cast::<super::ExtraItemsAllocationHeader>())).count;
        let (layout, expected_offset) = super::extra_items_layout::<T, U>(extra_items_count);
        debug_assert_eq!(object_offset, expected_offset);
        if strong && weak {
            T::drop_payload(obj);
            std::alloc::dealloc(allocation, layout);
        } else if strong {
            T::drop_payload(obj);
        } else if weak {
            std::alloc::dealloc(allocation, layout);
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
    const TYPE_INDEX_STATIC: bool = true;
    #[inline]
    fn type_index() -> i32 {
        TypeIndex::kTVMFFIStaticObjectBegin as i32
    }
    #[inline]
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        &mut this.header
    }
}

unsafe impl RustAllocatableObject for Object {
    unsafe fn drop_payload(_this: *mut Self) {}
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

    pub fn new(data: T) -> Self
    where
        T: RustAllocatableObject,
    {
        let type_index = T::type_index();
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
                    type_index,
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
        T: ObjectCoreWithExtraItems<ExtraItem = U> + RustAllocatableObject,
    {
        let type_index = T::type_index();
        unsafe {
            let extra_items_count = T::extra_items_count(&data);
            let (layout, object_offset) = extra_items_layout::<T, U>(extra_items_count);
            let raw_data_ptr = std::alloc::alloc(layout);
            if raw_data_ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            std::ptr::write(
                raw_data_ptr.cast::<ExtraItemsAllocationHeader>(),
                ExtraItemsAllocationHeader {
                    count: extra_items_count,
                },
            );
            let ptr = raw_data_ptr.add(object_offset).cast::<T>();
            std::ptr::write(ptr, data);
            // now override the header directly
            std::ptr::write(
                ptr as *mut TVMFFIObject,
                TVMFFIObject {
                    combined_ref_count: AtomicU64::new(COMBINED_REF_COUNT_BOTH_ONE),
                    type_index,
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

#[cfg(test)]
mod metadata_tests {
    use super::*;

    fn byte_array(value: &'static str) -> TVMFFIByteArray {
        unsafe { TVMFFIByteArray::from_str(value) }
    }

    fn type_info() -> TVMFFITypeInfo {
        TVMFFITypeInfo {
            type_index: 1234,
            type_depth: 0,
            type_key: byte_array("testing.MetadataFixture"),
            type_acenstors: std::ptr::null(),
            type_key_hash: 0,
            num_fields: 0,
            num_methods: 0,
            fields: std::ptr::null(),
            methods: std::ptr::null(),
            metadata: std::ptr::null(),
        }
    }

    #[test]
    fn metadata_strings_reject_null_and_invalid_utf8() {
        let null_data = TVMFFIByteArray::new(std::ptr::null(), 1);
        assert!(unsafe { checked_metadata_str(1234, "method name", &null_data) }.is_err());

        let invalid_utf8 = [0xff];
        let invalid = TVMFFIByteArray::new(invalid_utf8.as_ptr(), invalid_utf8.len());
        assert!(unsafe { checked_metadata_str(1234, "method name", &invalid) }.is_err());

        let empty = TVMFFIByteArray::new(std::ptr::null(), 0);
        assert_eq!(
            unsafe { checked_metadata_str(1234, "method name", &empty) }.unwrap(),
            ""
        );
    }

    #[test]
    fn method_and_metadata_tables_reject_invalid_observable_shapes() {
        let info = Box::leak(Box::new(type_info()));
        info.num_methods = -1;
        assert!(unsafe { checked_type_methods(info) }.is_err());

        let info = Box::leak(Box::new(type_info()));
        info.num_methods = 1;
        assert!(unsafe { checked_type_methods(info) }.is_err());

        let info = Box::leak(Box::new(type_info()));
        info.num_methods = 1;
        info.methods = 1usize as *const TVMFFIMethodInfo;
        assert!(unsafe { checked_type_methods(info) }.is_err());

        let info = Box::leak(Box::new(type_info()));
        info.metadata = 1usize as *const TVMFFITypeMetadata;
        assert!(unsafe { checked_type_metadata(info) }.is_err());
    }

    #[test]
    fn type_attribute_lookup_handles_extreme_indices_and_invalid_tables() {
        let cells = Box::leak(Box::new([TVMFFIAny::new()]));
        let mut column = TVMFFITypeAttrColumn {
            data: cells.as_ptr(),
            size: 1,
            begin_index: i32::MIN,
        };

        // The i32 subtraction in the old lookup overflowed for this pair.
        assert!(
            unsafe { checked_type_attr(&column, i32::MAX, "__ffi_init__") }
                .unwrap()
                .is_none()
        );

        column.size = -1;
        assert!(unsafe { checked_type_attr(&column, 0, "__ffi_init__") }.is_err());

        column.size = 1;
        column.data = std::ptr::null();
        assert!(unsafe { checked_type_attr(&column, 0, "__ffi_init__") }.is_err());

        assert!(unsafe {
            checked_type_attr(1usize as *const TVMFFITypeAttrColumn, 0, "__ffi_init__")
        }
        .is_err());
    }

    #[test]
    fn object_cells_are_checked_before_refcount_access() {
        let mut cell = TVMFFIAny::new();
        cell.type_index = TypeIndex::kTVMFFIFunction as i32;
        cell.data_union.v_obj = std::ptr::null_mut();
        assert!(unsafe { checked_object_cell(&cell, "Function cell") }.is_err());

        let mut header = TVMFFIObject::new();
        header.type_index = TypeIndex::kTVMFFIStaticObjectBegin as i32;
        cell.data_union.v_obj = &mut header;
        assert!(unsafe { checked_object_cell(&cell, "Function cell") }.is_err());
    }
}
