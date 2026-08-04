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
use crate::dtype::AsDLDataType;
use crate::error::Result;
use crate::object::{Object, ObjectArc, ObjectCore, ObjectCoreWithExtraItems};
use tvm_ffi_sys::dlpack::{DLDataType, DLDevice, DLDeviceType, DLTensor};
use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

//-----------------------------------------------------
// NDAllocator Trait
//-----------------------------------------------------
/// Trait for n-dimensional array allocators.
///
/// # Safety
///
/// Implementations must return storage satisfying `MIN_ALIGN` and the DLPack
/// prototype, release each successful allocation exactly once, and keep any
/// bookkeeping valid even if native code mutates the public `DLTensor`
/// metadata. `free_data` and the allocator's destructor must not unwind: they
/// may run from an object deleter called through the native C ABI.
pub unsafe trait NDAllocator: Send + 'static {
    /// The minimum alignment of the data allocated by the allocator
    const MIN_ALIGN: usize;
    /// Allocate data for the given DLTensor
    ///
    /// # Arguments
    /// * `tensor` - The DLTensor to allocate data for
    ///
    /// The returned allocation must be suitably aligned and initialized for
    /// every element described by `prototype.dtype`.  A null pointer is only
    /// permitted when the allocation has zero bytes.
    ///
    /// # Safety
    ///
    /// The allocation must remain valid until the matching `free_data` call.
    /// Its initialized contents must satisfy the validity requirements of the
    /// Rust type used to access this dtype through [`AsDLDataType`].
    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut core::ffi::c_void;

    /// Free data for the given DLTensor
    ///
    /// # Arguments
    /// * `tensor` - The DLTensor to free data for
    ///
    /// This method should free the data pointer of the DLTensor.
    /// It must not unwind.
    unsafe fn free_data(&mut self, tensor: &DLTensor);
}

/// DLTensorExt trait
/// This trait provides methods to get the number of elements and the item size of a DLTensor
pub trait DLTensorExt {
    /// Compute the element count from the raw shape pointer.
    ///
    /// # Safety
    ///
    /// For non-scalar tensors, `shape` must point to `ndim` readable `i64`
    /// values for the duration of this call.
    unsafe fn numel(&self) -> usize;
    fn item_size(&self) -> usize;
}

impl DLTensorExt for DLTensor {
    unsafe fn numel(&self) -> usize {
        let ndim = usize::try_from(self.ndim).expect("Tensor ndim must be non-negative");
        if ndim == 0 {
            return 1;
        }
        assert!(!self.shape.is_null(), "non-scalar Tensor has null shape");
        let mut numel = 1_usize;
        for index in 0..ndim {
            let dimension = unsafe { *self.shape.add(index) };
            let dimension = usize::try_from(dimension)
                .expect("Tensor dimensions must be non-negative and fit usize");
            numel = numel
                .checked_mul(dimension)
                .expect("Tensor element count exceeds usize::MAX");
        }
        numel
    }

    fn item_size(&self) -> usize {
        assert!(self.dtype.bits > 0, "Tensor dtype bits must be positive");
        assert!(self.dtype.lanes > 0, "Tensor dtype lanes must be positive");
        usize::from(self.dtype.bits)
            .checked_mul(usize::from(self.dtype.lanes))
            .and_then(|bits| bits.checked_add(7))
            .expect("Tensor item size exceeds usize::MAX")
            / 8
    }
}

//-----------------------------------------------------
// Shape
//-----------------------------------------------------
// ShapeObj for heap-allocated shape
#[repr(C)]
#[derive(Object)]
#[type_key = "ffi.Tensor"]
#[type_index(TypeIndex::kTVMFFITensor)]
pub struct TensorObj {
    object: Object,
    dltensor: DLTensor,
}

/// ABI stable owned Shape for ffi
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Tensor {
    data: ObjectArc<TensorObj>,
}

impl Tensor {
    /// Get the data pointer of the Tensor
    ///
    /// # Returns
    /// * `*mut core::ffi::c_void` - The data pointer of the Tensor
    pub fn data_ptr(&self) -> *const core::ffi::c_void {
        self.data.dltensor.data
    }
    /// Get the data pointer of the Tensor
    ///
    /// # Returns
    /// * `*mut core::ffi::c_void` - The data pointer of the Tensor
    pub fn data_ptr_mut(&mut self) -> *mut core::ffi::c_void {
        self.data.dltensor.data
    }
    /// Check if the Tensor is contiguous
    ///
    /// # Returns
    /// * `bool` - True if the Tensor is contiguous, false otherwise
    pub fn is_contiguous(&self) -> bool {
        self.checked_contiguous().unwrap_or(false)
    }

    fn checked_ndim(&self) -> Result<usize> {
        let ndim = usize::try_from(self.data.dltensor.ndim).map_err(|_| {
            crate::Error::new(
                crate::error::VALUE_ERROR,
                "Tensor ndim must be non-negative",
                "",
            )
        })?;
        crate::ensure!(
            ndim <= isize::MAX as usize / std::mem::size_of::<i64>(),
            crate::error::VALUE_ERROR,
            "Tensor metadata exceeds Rust slice limits"
        );
        Ok(ndim)
    }

    fn checked_shape(&self) -> Result<&[i64]> {
        let ndim = self.checked_ndim()?;
        if ndim == 0 {
            return Ok(&[]);
        }
        crate::ensure!(
            !self.data.dltensor.shape.is_null(),
            crate::error::VALUE_ERROR,
            "non-scalar Tensor has null shape"
        );
        Ok(unsafe { std::slice::from_raw_parts(self.data.dltensor.shape, ndim) })
    }

    fn checked_strides(&self) -> Result<Option<&[i64]>> {
        let ndim = self.checked_ndim()?;
        if ndim == 0 || self.data.dltensor.strides.is_null() {
            return Ok(None);
        }
        Ok(Some(unsafe {
            std::slice::from_raw_parts(self.data.dltensor.strides, ndim)
        }))
    }

    fn checked_numel(&self) -> Result<usize> {
        let shape = self.checked_shape()?;
        let mut has_zero_dimension = false;
        for &dimension in shape {
            let dimension = usize::try_from(dimension).map_err(|_| {
                crate::Error::new(
                    crate::error::VALUE_ERROR,
                    "Tensor dimensions must be non-negative and fit usize",
                    "",
                )
            })?;
            has_zero_dimension |= dimension == 0;
        }
        if has_zero_dimension {
            return Ok(0);
        }

        let mut numel = 1_usize;
        for &dimension in shape {
            let dimension = usize::try_from(dimension).expect("dimensions were validated above");
            numel = numel.checked_mul(dimension).ok_or_else(|| {
                crate::Error::new(
                    crate::error::VALUE_ERROR,
                    "Tensor element count exceeds usize::MAX",
                    "",
                )
            })?;
        }
        Ok(numel)
    }

    fn checked_contiguous(&self) -> Result<bool> {
        let Some(strides) = self.checked_strides()? else {
            // A null DLPack strides pointer denotes compact row-major storage.
            self.checked_numel()?;
            return Ok(true);
        };
        let shape = self.checked_shape()?;
        let mut has_zero_dimension = false;
        for &dimension in shape {
            if dimension < 0 {
                crate::bail!(
                    crate::error::VALUE_ERROR,
                    "Tensor dimensions must be non-negative"
                );
            }
            // DLPack considers an empty tensor contiguous regardless of its
            // explicit strides because no element can be addressed.
            has_zero_dimension |= dimension == 0;
        }
        if has_zero_dimension {
            return Ok(true);
        }

        let mut expected_stride = 1_i64;
        for (&dimension, &stride) in shape.iter().zip(strides).rev() {
            // A singleton dimension cannot distinguish one stride from
            // another and therefore does not constrain contiguity.
            if dimension == 1 {
                continue;
            }
            if stride != expected_stride {
                return Ok(false);
            }
            expected_stride = expected_stride.checked_mul(dimension).ok_or_else(|| {
                crate::Error::new(
                    crate::error::VALUE_ERROR,
                    "Tensor strides exceed i64::MAX",
                    "",
                )
            })?;
        }
        Ok(true)
    }

    fn checked_data<T: AsDLDataType>(&self) -> Result<(*mut T, usize)> {
        let expected = T::DL_DATA_TYPE;
        let actual = self.dtype();
        if actual != expected {
            crate::bail!(
                crate::error::TYPE_ERROR,
                "Data type mismatch: expected code={}, bits={}, lanes={}; got code={}, bits={}, lanes={}",
                expected.code,
                expected.bits,
                expected.lanes,
                actual.code,
                actual.bits,
                actual.lanes
            );
        }
        if self.device().device_type != DLDeviceType::kDLCPU {
            crate::bail!(crate::error::RUNTIME_ERROR, "Tensor is not on CPU");
        }
        crate::ensure!(
            self.checked_contiguous()?,
            crate::error::RUNTIME_ERROR,
            "Tensor is not contiguous"
        );
        crate::ensure!(
            std::mem::size_of::<T>() > 0
                && std::mem::size_of::<T>() == self.data.dltensor.item_size(),
            crate::error::TYPE_ERROR,
            "Rust type size does not match the Tensor dtype"
        );

        let numel = self.checked_numel()?;
        let Some(byte_len) = numel.checked_mul(std::mem::size_of::<T>()) else {
            crate::bail!(
                crate::error::VALUE_ERROR,
                "Tensor byte size exceeds usize::MAX"
            );
        };
        if byte_len > isize::MAX as usize {
            crate::bail!(
                crate::error::VALUE_ERROR,
                "Tensor byte size exceeds Rust slice limits"
            );
        }
        if byte_len == 0 {
            return Ok((std::ptr::NonNull::<T>::dangling().as_ptr(), 0));
        }
        crate::ensure!(
            !self.data.dltensor.data.is_null(),
            crate::error::RUNTIME_ERROR,
            "non-empty Tensor has null data"
        );
        let byte_offset = usize::try_from(self.data.dltensor.byte_offset).map_err(|_| {
            crate::Error::new(
                crate::error::VALUE_ERROR,
                "Tensor byte offset does not fit usize",
                "",
            )
        })?;
        let allocation_span = byte_offset.checked_add(byte_len).ok_or_else(|| {
            crate::Error::new(
                crate::error::VALUE_ERROR,
                "Tensor byte offset plus size exceeds usize::MAX",
                "",
            )
        })?;
        crate::ensure!(
            allocation_span <= isize::MAX as usize,
            crate::error::VALUE_ERROR,
            "Tensor byte offset plus size exceeds Rust allocation limits"
        );
        let address = (self.data.dltensor.data as usize)
            .checked_add(byte_offset)
            .ok_or_else(|| {
                crate::Error::new(
                    crate::error::VALUE_ERROR,
                    "Tensor data address overflows usize",
                    "",
                )
            })?;
        let data = self
            .data
            .dltensor
            .data
            .cast::<u8>()
            .wrapping_add(byte_offset)
            .cast::<T>();
        crate::ensure!(
            address % std::mem::align_of::<T>() == 0,
            crate::error::RUNTIME_ERROR,
            "Tensor data is not aligned for the requested Rust type"
        );
        Ok((data, numel))
    }

    /// Borrow CPU data without proving that native code will not mutate it.
    ///
    /// # Safety
    ///
    /// The foreign DLTensor metadata must obey DLPack. Its shape and non-null
    /// strides pointers must each reference `ndim` initialized `i64` entries
    /// and must not be concurrently modified while this method validates them.
    /// `data + byte_offset` must retain provenance within the same live
    /// allocation as `data`, and that allocation must cover at least
    /// `numel * size_of::<T>()` bytes from the offset. The data region must not
    /// overlap the Tensor metadata or any incompatible live reference. Every
    /// element must be initialized and satisfy `T`'s validity requirements.
    /// For the returned lifetime, no native code or aliased Tensor/view may
    /// write the same allocation.
    ///
    /// The checked validation can reject observable null, integer-overflow,
    /// dtype, device, contiguity, and alignment errors. It cannot prove a
    /// foreign pointer's provenance, allocation extent, or aliasing guarantees;
    /// those remain requirements on the caller.
    pub unsafe fn data_as_slice_unchecked<T: AsDLDataType>(&self) -> Result<&[T]> {
        let (data, len) = self.checked_data::<T>()?;
        Ok(std::slice::from_raw_parts(data.cast_const(), len))
    }

    /// Borrow mutable CPU data.
    ///
    /// # Safety
    ///
    /// The foreign metadata, allocation provenance and extent, non-overlap,
    /// lifetime, and initialized-element requirements are the same as
    /// [`Tensor::data_as_slice_unchecked`]. For the returned lifetime, the
    /// caller must have exclusive read/write access: no native code, Tensor,
    /// ObjectRef, pointer, or other live reference may read or write the data
    /// region.
    pub unsafe fn data_as_slice_mut_unchecked<T: AsDLDataType>(&mut self) -> Result<&mut [T]> {
        let (data, len) = self.checked_data::<T>()?;
        Ok(std::slice::from_raw_parts_mut(data, len))
    }

    pub fn shape(&self) -> &[i64] {
        self.checked_shape()
            .expect("Tensor has invalid shape metadata")
    }

    pub fn ndim(&self) -> usize {
        self.checked_ndim()
            .expect("Tensor ndim must be non-negative")
    }

    pub fn numel(&self) -> usize {
        self.checked_numel()
            .expect("Tensor has invalid shape metadata")
    }

    pub fn strides(&self) -> &[i64] {
        self.checked_strides()
            .expect("Tensor has invalid stride metadata")
            .unwrap_or(&[])
    }

    pub fn dtype(&self) -> DLDataType {
        self.data.dltensor.dtype
    }

    pub fn device(&self) -> DLDevice {
        self.data.dltensor.device
    }
}

#[repr(C)]
struct TensorObjFromNDAlloc<TNDAlloc>
where
    TNDAlloc: NDAllocator,
{
    base: TensorObj,
    alloc: TNDAlloc,
}

unsafe impl<TNDAlloc: NDAllocator> ObjectCore for TensorObjFromNDAlloc<TNDAlloc> {
    const TYPE_KEY: &'static str = TensorObj::TYPE_KEY;
    const TYPE_DEPTH: i32 = TensorObj::TYPE_DEPTH;
    fn type_index() -> i32 {
        TensorObj::type_index()
    }
    unsafe fn object_header_mut(this: &mut Self) -> &mut tvm_ffi_sys::TVMFFIObject {
        TensorObj::object_header_mut(&mut this.base)
    }
}

unsafe impl<TNDAlloc: NDAllocator> crate::object::RustAllocatableObject
    for TensorObjFromNDAlloc<TNDAlloc>
{
    unsafe fn drop_payload(this: *mut Self) {
        let tensor = std::ptr::addr_of!((*this).base.dltensor);
        let allocator = std::ptr::addr_of_mut!((*this).alloc);
        if !(*tensor).data.is_null() {
            (&mut *allocator).free_data(&*tensor);
        }
        std::ptr::drop_in_place(allocator);
    }
}

unsafe impl<TNDAlloc: NDAllocator> ObjectCoreWithExtraItems for TensorObjFromNDAlloc<TNDAlloc> {
    type ExtraItem = i64;
    fn extra_items_count(this: &Self) -> usize {
        usize::try_from(this.base.dltensor.ndim)
            .expect("Tensor ndim must be non-negative")
            .checked_mul(2)
            .expect("Tensor metadata size exceeds usize::MAX")
    }
}

impl Tensor {
    // Create a Tensor from a NDAllocator
    ///
    /// # Arguments
    /// * `alloc` - The NDAllocator
    /// * `shape` - The shape of the Tensor
    /// * `dtype` - The data type of the Tensor
    /// * `device` - The device of the Tensor
    ///
    /// # Returns
    /// * `Tensor` - The created Tensor
    pub fn from_nd_alloc<TNDAlloc>(
        alloc: TNDAlloc,
        shape: &[i64],
        dtype: DLDataType,
        device: DLDevice,
    ) -> Self
    where
        TNDAlloc: NDAllocator,
    {
        let ndim = i32::try_from(shape.len()).expect("Tensor rank exceeds i32::MAX");
        assert!(dtype.bits > 0, "Tensor dtype bits must be positive");
        assert!(dtype.lanes > 0, "Tensor dtype lanes must be positive");

        let mut has_zero_dimension = false;
        for &dimension in shape {
            let dimension = usize::try_from(dimension)
                .expect("Tensor dimensions must be non-negative and fit usize");
            has_zero_dimension |= dimension == 0;
        }

        let mut strides = vec![0_i64; shape.len()];
        let mut stride = 1_i64;
        for index in (0..shape.len()).rev() {
            strides[index] = stride;
            stride = match stride.checked_mul(shape[index]) {
                Some(next) => next,
                // Empty tensors have no addressable elements, so DLPack does
                // not constrain their explicit strides. Use zero after an
                // otherwise unrepresentable suffix product.
                None if has_zero_dimension => 0,
                None => panic!("Tensor strides exceed i64::MAX"),
            };
        }
        let numel = if has_zero_dimension {
            0
        } else {
            shape.iter().fold(1_usize, |numel, &dimension| {
                numel
                    .checked_mul(
                        usize::try_from(dimension).expect("dimensions were validated above"),
                    )
                    .expect("Tensor element count exceeds usize::MAX")
            })
        };
        let item_size = usize::from(dtype.bits)
            .checked_mul(usize::from(dtype.lanes))
            .and_then(|bits| bits.checked_add(7))
            .expect("Tensor item size exceeds usize::MAX")
            / 8;
        let byte_len = numel
            .checked_mul(item_size)
            .expect("Tensor allocation size exceeds usize::MAX");

        let tensor_obj = TensorObjFromNDAlloc {
            base: TensorObj {
                object: Object::new(),
                dltensor: DLTensor {
                    data: std::ptr::null_mut(),
                    device: device,
                    ndim,
                    dtype: dtype,
                    shape: std::ptr::null_mut(),
                    strides: std::ptr::null_mut(),
                    byte_offset: 0,
                },
            },
            alloc: alloc,
        };
        unsafe {
            let mut obj_arc = ObjectArc::new_with_extra_items(tensor_obj);
            let obj = &mut *ObjectArc::as_raw_mut(&mut obj_arc);
            let extra_items = TensorObjFromNDAlloc::extra_items_uninit_mut(obj);
            let data = extra_items.as_mut_ptr().cast::<i64>();
            for (slot, value) in extra_items[..shape.len()].iter_mut().zip(shape) {
                slot.write(*value);
            }
            for (slot, value) in extra_items[shape.len()..].iter_mut().zip(strides) {
                slot.write(value);
            }
            obj.base.dltensor.shape = data;
            obj.base.dltensor.strides = data.add(shape.len());
            let dltensor_ptr = &obj.base.dltensor as *const DLTensor;
            let data = obj.alloc.alloc_data(&*dltensor_ptr);
            assert!(
                byte_len == 0 || !data.is_null(),
                "NDAllocator returned null"
            );
            obj.base.dltensor.data = data;
            Self {
                data: ObjectArc::from_raw(ObjectArc::into_raw(obj_arc) as *mut TensorObj),
            }
        }
    }
    /// Create a Tensor from a slice
    ///
    /// # Arguments
    /// * `slice` - The slice to create the Tensor from
    /// * `shape` - The shape of the Tensor
    ///
    /// # Returns
    /// * `Tensor` - The created Tensor
    pub fn from_slice<T: AsDLDataType>(slice: &[T], shape: &[i64]) -> Result<Self> {
        let dtype = T::DL_DATA_TYPE;
        let device = DLDevice::new(DLDeviceType::kDLCPU, 0);
        let mut tensor = Tensor::from_nd_alloc(CPUNDAlloc::default(), shape, dtype, device);
        if tensor.numel() != slice.len() {
            crate::bail!(crate::error::VALUE_ERROR, "Slice length mismatch");
        }
        // This Tensor and its CPUNDAlloc buffer were just created here, so no
        // other object or external owner can alias the allocation yet.
        unsafe { tensor.data_as_slice_mut_unchecked::<T>()? }.copy_from_slice(slice);
        Ok(tensor)
    }
}

/// Example CPU NDAllocator
/// This allocator allocates data on the CPU
#[derive(Debug, Default)]
pub struct CPUNDAlloc {
    allocation: Option<CPUAllocation>,
}

#[derive(Debug)]
struct CPUAllocation {
    address: usize,
    layout: std::alloc::Layout,
}

impl CPUNDAlloc {
    fn release(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            unsafe {
                std::alloc::dealloc(allocation.address as *mut u8, allocation.layout);
            }
        }
    }
}

impl Drop for CPUNDAlloc {
    fn drop(&mut self) {
        self.release();
    }
}

unsafe impl NDAllocator for CPUNDAlloc {
    const MIN_ALIGN: usize = 64;

    unsafe fn alloc_data(&mut self, prototype: &DLTensor) -> *mut core::ffi::c_void {
        assert!(
            self.allocation.is_none(),
            "CPUNDAlloc cannot own two live allocations"
        );
        let numel = unsafe { prototype.numel() };
        let item_size = prototype.item_size();
        let size = numel
            .checked_mul(item_size)
            .expect("Tensor allocation size exceeds usize::MAX");
        if size == 0 {
            return std::ptr::NonNull::<u8>::dangling().as_ptr().cast();
        }
        let layout = std::alloc::Layout::from_size_align(size, Self::MIN_ALIGN)
            .expect("invalid CPU Tensor allocation layout");
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        self.allocation = Some(CPUAllocation {
            address: ptr as usize,
            layout,
        });
        ptr.cast()
    }

    unsafe fn free_data(&mut self, _tensor: &DLTensor) {
        // Native code may legally obtain a mutable DLTensor pointer.  Release
        // the exact pointer and Layout recorded at allocation time instead of
        // trusting metadata that may since have changed.
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::{CPUNDAlloc, DLTensor, NDAllocator, Tensor, TensorObj, TensorObjFromNDAlloc};
    use crate::object::ObjectArc;

    #[repr(align(64))]
    struct AlignedAllocator {
        _payload: u8,
    }

    unsafe impl NDAllocator for AlignedAllocator {
        const MIN_ALIGN: usize = 64;

        unsafe fn alloc_data(&mut self, _prototype: &DLTensor) -> *mut core::ffi::c_void {
            unreachable!()
        }

        unsafe fn free_data(&mut self, _tensor: &DLTensor) {
            unreachable!()
        }
    }

    #[test]
    fn rust_owned_tensor_keeps_object_header_at_offset_zero() {
        assert_eq!(
            std::mem::offset_of!(TensorObjFromNDAlloc<CPUNDAlloc>, base),
            0
        );
        assert_eq!(
            std::mem::offset_of!(TensorObjFromNDAlloc<AlignedAllocator>, base),
            0
        );
    }

    #[test]
    fn slice_validation_rejects_malformed_dlpack_metadata_before_pointer_use() {
        let tensor = Tensor::from_slice(&[1_f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let object = unsafe { ObjectArc::as_raw(&tensor.data) }.cast_mut();
        let raw = unsafe { std::ptr::addr_of_mut!((*object.cast::<TensorObj>()).dltensor) };

        unsafe {
            let original_ndim = (*raw).ndim;
            (*raw).ndim = -1;
            let result = tensor.checked_data::<f32>();
            (*raw).ndim = original_ndim;
            assert!(result.is_err());

            let original_shape = (*raw).shape;
            (*raw).shape = std::ptr::null_mut();
            let result = tensor.checked_data::<f32>();
            (*raw).shape = original_shape;
            assert!(result.is_err());

            let first_dimension = *original_shape;
            *original_shape = -1;
            let result = tensor.checked_data::<f32>();
            *original_shape = first_dimension;
            assert!(result.is_err());

            let original_stride = *(*raw).strides;
            *(*raw).strides = original_stride + 1;
            let result = tensor.checked_data::<f32>();
            *(*raw).strides = original_stride;
            assert!(result.is_err());

            let original_device = (*raw).device;
            (*raw).device.device_type = tvm_ffi_sys::dlpack::DLDeviceType::kDLCUDA;
            let result = tensor.checked_data::<f32>();
            (*raw).device = original_device;
            assert!(result.is_err());

            let original_data = (*raw).data;
            (*raw).data = std::ptr::null_mut();
            let result = tensor.checked_data::<f32>();
            (*raw).data = original_data;
            assert!(result.is_err());

            (*raw).data = original_data.cast::<u8>().add(1).cast();
            let result = tensor.checked_data::<f32>();
            (*raw).data = original_data;
            assert!(result.is_err());

            let original_offset = (*raw).byte_offset;
            (*raw).byte_offset = u64::MAX;
            let result = tensor.checked_data::<f32>();
            (*raw).byte_offset = original_offset;
            assert!(result.is_err());

            let original_dtype = (*raw).dtype;
            (*raw).dtype.code = u8::MAX;
            let result = tensor.checked_data::<f32>();
            (*raw).dtype = original_dtype;
            assert!(result.is_err());
        }
    }

    #[test]
    fn contiguity_ignores_singleton_strides_and_empty_tensor_strides() {
        let singleton = Tensor::from_slice(&[1_f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let singleton_object = unsafe { ObjectArc::as_raw(&singleton.data) }.cast_mut();
        let singleton_raw =
            unsafe { std::ptr::addr_of_mut!((*singleton_object.cast::<TensorObj>()).dltensor) };
        unsafe {
            let original_stride = *(*singleton_raw).strides;
            *(*singleton_raw).strides = 99;
            assert!(singleton.checked_data::<f32>().is_ok());
            *(*singleton_raw).strides = original_stride;
        }

        let empty = Tensor::from_slice::<f32>(&[], &[0, 4]).unwrap();
        let empty_object = unsafe { ObjectArc::as_raw(&empty.data) }.cast_mut();
        let empty_raw =
            unsafe { std::ptr::addr_of_mut!((*empty_object.cast::<TensorObj>()).dltensor) };
        unsafe {
            let original_strides = [*(*empty_raw).strides, *(*empty_raw).strides.add(1)];
            *(*empty_raw).strides = 123;
            *(*empty_raw).strides.add(1) = -456;
            assert!(empty.checked_data::<f32>().is_ok());
            let original_dimension = *(*empty_raw).shape.add(1);
            *(*empty_raw).shape.add(1) = -1;
            assert!(!empty.is_contiguous());
            assert!(empty.checked_data::<f32>().is_err());
            *(*empty_raw).shape.add(1) = original_dimension;
            *(*empty_raw).strides = original_strides[0];
            *(*empty_raw).strides.add(1) = original_strides[1];
        }
    }

    #[test]
    fn cpu_allocator_releases_the_recorded_layout_after_metadata_changes() {
        let mut shape = [2_i64];
        let mut tensor = DLTensor {
            data: std::ptr::null_mut(),
            device: tvm_ffi_sys::dlpack::DLDevice::new(
                tvm_ffi_sys::dlpack::DLDeviceType::kDLCPU,
                0,
            ),
            ndim: 1,
            dtype: tvm_ffi_sys::dlpack::DLDataType::new(
                tvm_ffi_sys::dlpack::DLDataTypeCode::kDLFloat,
                32,
                1,
            ),
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let mut allocator = CPUNDAlloc::default();
        tensor.data = unsafe { allocator.alloc_data(&tensor) };
        assert!(allocator.allocation.is_some());

        tensor.data = std::ptr::null_mut();
        tensor.ndim = -1;
        tensor.dtype.bits = 0;
        tensor.shape = std::ptr::null_mut();
        unsafe { allocator.free_data(&tensor) };
        assert!(allocator.allocation.is_none());
    }

    #[test]
    fn empty_tensor_construction_does_not_overflow_before_zero_dimension() {
        let tensor = Tensor::from_slice::<f32>(&[], &[0, i64::MAX, 3]).unwrap();
        assert_eq!(tensor.numel(), 0);
        assert!(tensor.is_contiguous());
        assert!(unsafe { tensor.data_as_slice_unchecked::<f32>() }
            .unwrap()
            .is_empty());
    }
}
