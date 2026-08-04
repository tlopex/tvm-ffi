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
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tvm_ffi::object::ObjectRef;
use tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex;
use tvm_ffi::*;

struct DropCounter(Arc<AtomicU32>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

// must have repr(C) for the object header stays in the same position
#[repr(C)]
struct TestIntObj {
    object: Object,
    pub value: i64,
    // counter for recording the number of times the object is deleted
    delete_counter: DropCounter,
    pub extra_item_count: u64,
}

impl TestIntObj {
    pub fn new(value: i64, delete_counter: Arc<AtomicU32>, extra_item_count: u64) -> Self {
        Self {
            object: Object::new(),
            value,
            delete_counter: DropCounter(delete_counter),
            extra_item_count,
        }
    }
}

unsafe impl ObjectCore for TestIntObj {
    const TYPE_KEY: &'static str = Object::TYPE_KEY;
    const TYPE_DEPTH: i32 = Object::TYPE_DEPTH;
    #[inline]
    fn type_index() -> i32 {
        Object::type_index()
    }
    #[inline]
    unsafe fn object_header_mut(this: &mut Self) -> &mut TVMFFIObject {
        Object::object_header_mut(&mut this.object)
    }
}

unsafe impl RustAllocatableObject for TestIntObj {
    unsafe fn drop_payload(this: *mut Self) {
        std::ptr::drop_in_place(std::ptr::addr_of_mut!((*this).delete_counter));
    }
}

unsafe impl ObjectCoreWithExtraItems for TestIntObj {
    type ExtraItem = u64;
    #[inline]
    fn extra_items_count(this: &Self) -> usize {
        this.extra_item_count as usize
    }
}

#[test]
fn test_object_arc() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let obj_arc = ObjectArc::new(TestIntObj::new(11, delete_counter.clone(), 0));
    assert_eq!(obj_arc.value, 11);
    assert_eq!(ObjectArc::strong_count(&obj_arc), 1);
    assert_eq!(ObjectArc::weak_count(&obj_arc), 1);

    let ref1 = obj_arc.clone();
    assert_eq!(ObjectArc::strong_count(&obj_arc), 2);
    assert_eq!(ObjectArc::weak_count(&obj_arc), 1);

    let ref2 = obj_arc.clone();
    assert_eq!(ObjectArc::strong_count(&obj_arc), 3);
    assert_eq!(ObjectArc::weak_count(&obj_arc), 1);
    assert_eq!(ref1.value, 11);
    // drop obj_arc
    drop(obj_arc);
    assert_eq!(ObjectArc::strong_count(&ref1), 2);
    assert_eq!(ObjectArc::weak_count(&ref1), 1);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    // drop ref1
    drop(ref1);
    assert_eq!(ObjectArc::strong_count(&ref2), 1);
    assert_eq!(ObjectArc::weak_count(&ref2), 1);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    // drop ref2
    drop(ref2);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_object_arc_with_extra_items() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let mut obj_arc =
        ObjectArc::new_with_extra_items(TestIntObj::new(12, delete_counter.clone(), 10));
    assert_eq!(obj_arc.value, 12);
    assert_eq!(ObjectArc::strong_count(&obj_arc), 1);
    assert_eq!(ObjectArc::weak_count(&obj_arc), 1);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
    unsafe {
        let object = &mut *ObjectArc::as_raw_mut(&mut obj_arc);
        // layout check of extra items
        assert_eq!(TestIntObj::extra_items_count(object), 10);
        let expected =
            (object as *mut TestIntObj as *mut u8).add(std::mem::size_of::<TestIntObj>());
        let extra_items = TestIntObj::extra_items_uninit_mut(object);
        assert_eq!(extra_items.len(), 10);
        assert_eq!(extra_items.as_mut_ptr() as *mut u8, expected);
        for (index, slot) in extra_items.iter_mut().enumerate() {
            slot.write(index as u64);
        }
        assert_eq!(
            TestIntObj::extra_items(object),
            (0_u64..10).collect::<Vec<_>>()
        );
    }
    drop(obj_arc);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_extra_item_allocation_survives_two_phase_weak_destruction() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let mut object =
        ObjectArc::new_with_extra_items(TestIntObj::new(12, delete_counter.clone(), 2));
    unsafe {
        let allocation = &mut *ObjectArc::as_raw_mut(&mut object);
        for (index, slot) in TestIntObj::extra_items_uninit_mut(allocation)
            .iter_mut()
            .enumerate()
        {
            slot.write(index as u64);
        }

        let weak = tvm_ffi_sys::TVMFFITestingWeakObjectCreate(
            ObjectArc::as_raw(&object).cast_mut().cast(),
        );
        assert!(!weak.is_null());
        assert_eq!(ObjectArc::strong_count(&object), 1);
        assert_eq!(ObjectArc::weak_count(&object), 2);

        drop(object);
        assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
        assert_eq!(tvm_ffi_sys::TVMFFITestingWeakObjectExpired(weak), 1);
        assert_eq!(tvm_ffi_sys::TVMFFITestingWeakObjectLock(weak), 0);
        tvm_ffi_sys::TVMFFITestingWeakObjectDelete(weak);
        assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn test_extra_item_layout_overflow_is_rejected_before_allocation() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let data = TestIntObj::new(12, delete_counter.clone(), usize::MAX as u64);

    let result = std::panic::catch_unwind(|| ObjectArc::new_with_extra_items(data));

    assert!(result.is_err());
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}

#[test]
fn test_object_arc_from_raw() {
    unsafe {
        let delete_counter = Arc::new(AtomicU32::new(0));
        let obj_arc = ObjectArc::new(TestIntObj::new(11, delete_counter.clone(), 0));
        let raw_ptr = ObjectArc::into_raw(obj_arc);
        let obj_arc2 = ObjectArc::from_raw(raw_ptr);
        assert_eq!(obj_arc2.value, 11);
        assert_eq!(ObjectArc::strong_count(&obj_arc2), 1);
        assert_eq!(ObjectArc::weak_count(&obj_arc2), 1);
        assert_eq!(delete_counter.load(Ordering::Relaxed), 0);
        // drop obj_arc2
        drop(obj_arc2);
        assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn test_object_arc_nullable_representation() {
    // The thread-safety marker on Object is zero-sized and must not change the
    // C object-header prefix used by every Rust-owned object allocation.
    assert_eq!(
        std::mem::size_of::<Object>(),
        std::mem::size_of::<TVMFFIObject>()
    );
    assert_eq!(
        std::mem::align_of::<Object>(),
        std::mem::align_of::<TVMFFIObject>()
    );
    assert_eq!(
        std::mem::size_of::<ObjectArc<TestIntObj>>(),
        std::mem::size_of::<*const TestIntObj>()
    );

    let null_arc = unsafe { ObjectArc::<TestIntObj>::from_raw(std::ptr::null()) };
    assert!(ObjectArc::is_null(&null_arc));
    assert_eq!(ObjectArc::strong_count(&null_arc), 0);
    assert_eq!(ObjectArc::weak_count(&null_arc), 0);
    let null_clone = null_arc.clone();
    assert!(ObjectArc::is_null(&null_clone));
    assert!(std::panic::catch_unwind(|| std::hint::black_box(&*null_clone)).is_err());
    drop(null_arc);
    drop(null_clone);

    let null_ref = <ObjectRef as ObjectRefCore>::from_data(unsafe {
        ObjectArc::<Object>::from_raw(std::ptr::null())
    });
    assert!(null_ref.is_null());
    assert!(!null_ref.is_defined());
    let null_any = Any::from(null_ref.clone());
    assert_eq!(null_any.type_index(), TVMFFITypeIndex::kTVMFFINone as i32);
    assert!(AnyView::from(&null_ref).try_as::<ObjectRef>().is_none());

    let defined_ref = <ObjectRef as ObjectRefCore>::from_data(ObjectArc::new(Object::new()));
    assert!(!defined_ref.is_null());
    assert!(defined_ref.is_defined());
    assert!(defined_ref.same_as(&defined_ref.clone()));
    let other_ref = <ObjectRef as ObjectRefCore>::from_data(ObjectArc::new(Object::new()));
    assert!(!defined_ref.same_as(&other_ref));
}

#[test]
fn test_object_arc_clone_rejects_strong_count_overflow_without_corrupting_weak_count() {
    let delete_counter = Arc::new(AtomicU32::new(0));
    let object = ObjectArc::new(TestIntObj::new(11, delete_counter.clone(), 0));
    let header = unsafe { ObjectArc::as_raw(&object) }
        .cast_mut()
        .cast::<TVMFFIObject>();
    unsafe {
        (*header)
            .combined_ref_count
            .store(u32::MAX as u64 | (1_u64 << 32), Ordering::Relaxed);
    }

    let result = std::panic::catch_unwind(|| object.clone());

    assert!(result.is_err());
    assert_eq!(
        unsafe { (*header).combined_ref_count.load(Ordering::Relaxed) },
        u32::MAX as u64 | (1_u64 << 32)
    );
    // Restore the real ownership count so dropping the sole handle remains valid.
    unsafe {
        (*header)
            .combined_ref_count
            .store(1 | (1_u64 << 32), Ordering::Relaxed);
    }
    drop(object);
    assert_eq!(delete_counter.load(Ordering::Relaxed), 1);
}
