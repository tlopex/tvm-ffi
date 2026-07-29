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

use std::any::TypeId;
use std::env;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use tvm_ffi::any::TryFromTemp;
use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::match_any_internal::{ArmId, LeafLookupTable};
use tvm_ffi::object::{Object, ObjectArc, ObjectCore, ObjectRef, ObjectRefCore};
use tvm_ffi::{match_any, AnyView, ObjectRefCast, Shape};

const MISS: usize = usize::MAX;
const SAMPLE_COUNT: usize = 31;
const COLD_SAMPLE_COUNT: usize = 101;
const TARGET_SAMPLE_TIME: Duration = Duration::from_millis(20);
const COLD_CHILD_ENV: &str = "TVM_FFI_MATCH_ANY_COLD_CHILD";
const METADATA_COLD_CHILD_ENV: &str = "TVM_FFI_MATCH_ANY_METADATA_COLD_CHILD";

// These final type keys are registered by the linked TVM-FFI libraries. The
// benchmark only needs their object headers, so the Rust-side bodies stay empty.
macro_rules! define_leaf {
    ($object:ident, $reference:ident, $type_key:literal) => {
        #[repr(C)]
        #[derive(Object)]
        #[type_key = $type_key]
        #[type_final]
        struct $object {
            base: Object,
        }

        #[repr(C)]
        #[derive(ObjectRef, Clone)]
        struct $reference {
            data: ObjectArc<$object>,
        }

        impl $reference {
            fn new() -> Self {
                Self {
                    data: ObjectArc::new($object {
                        base: Object::new(),
                    }),
                }
            }
        }
    };
}

define_leaf!(Leaf0Obj, Leaf0, "testing.TestIntPair");
define_leaf!(Leaf1Obj, Leaf1, "testing.TestCompare");
define_leaf!(Leaf2Obj, Leaf2, "testing.TestHash");
define_leaf!(Leaf3Obj, Leaf3, "testing.TestCustomHash");
define_leaf!(Leaf4Obj, Leaf4, "testing.TestCustomCompare");
define_leaf!(Leaf5Obj, Leaf5, "testing.TestEqWithoutHash");
define_leaf!(Leaf6Obj, Leaf6, "testing.TestFrozenCxx");
define_leaf!(Leaf7Obj, Leaf7, "ffi.StructuralKey");
define_leaf!(Leaf8Obj, Leaf8, "ffi.VisitErrorContext");
define_leaf!(Leaf9Obj, Leaf9, "ffi.EnumState");
define_leaf!(Leaf10Obj, Leaf10, "ffi.reflection.AccessStep");
define_leaf!(Leaf11Obj, Leaf11, "ffi.reflection.AccessPath");
define_leaf!(Leaf12Obj, Leaf12, "testing.TestEnumVariant");
define_leaf!(Leaf13Obj, Leaf13, "testing.TestCxxIntEnum");
define_leaf!(Leaf14Obj, Leaf14, "testing.TestCxxStrEnum");
define_leaf!(Leaf15Obj, Leaf15, "testing.TestObjectDerived");
define_leaf!(Leaf16Obj, Leaf16, "testing.TestCxxAutoInitChild");

type Runner = fn(&[ObjectRef], u64) -> usize;

#[repr(C, align(4096))]
struct ColdTablePair<T> {
    warmup: T,
    target: T,
}

macro_rules! define_arm_handlers {
    ($($name:ident => $arm_id:expr),+ $(,)?) => {
        $(
            #[inline(never)]
            fn $name() -> usize {
                black_box($arm_id)
            }
        )+
    };
}

define_arm_handlers!(
    handle_arm_0 => 0,
    handle_arm_1 => 1,
    handle_arm_2 => 2,
    handle_arm_3 => 3,
    handle_arm_4 => 4,
    handle_arm_5 => 5,
    handle_arm_6 => 6,
    handle_arm_7 => 7,
    handle_arm_8 => 8,
    handle_arm_9 => 9,
    handle_arm_10 => 10,
    handle_arm_11 => 11,
    handle_arm_12 => 12,
    handle_arm_13 => 13,
    handle_arm_14 => 14,
    handle_arm_15 => 15,
    handle_arm_16 => 16,
    handle_fallback => MISS,
);

#[inline(always)]
fn dispatch_arm(arm_id: usize) -> usize {
    match arm_id {
        0 => handle_arm_0(),
        1 => handle_arm_1(),
        2 => handle_arm_2(),
        3 => handle_arm_3(),
        4 => handle_arm_4(),
        5 => handle_arm_5(),
        6 => handle_arm_6(),
        7 => handle_arm_7(),
        8 => handle_arm_8(),
        9 => handle_arm_9(),
        10 => handle_arm_10(),
        11 => handle_arm_11(),
        12 => handle_arm_12(),
        13 => handle_arm_13(),
        14 => handle_arm_14(),
        15 => handle_arm_15(),
        16 => handle_arm_16(),
        _ => handle_fallback(),
    }
}

macro_rules! hot_loop {
    ($inputs:ident, $iterations:ident, $value:ident => $dispatch:expr) => {{
        let mut checksum = 0_usize;
        let mut cursor = 0_usize;
        for _ in 0..$iterations {
            let $value = black_box(&$inputs[cursor]);
            checksum = checksum.wrapping_add(black_box($dispatch));
            cursor += 1;
            if cursor == $inputs.len() {
                cursor = 0;
            }
        }
        black_box(cursor);
        black_box(checksum)
    }};
}

#[inline(never)]
fn noop(inputs: &[ObjectRef], iterations: u64) -> usize {
    hot_loop!(inputs, iterations, _value => 0)
}

#[inline(never)]
fn diagnostic_single(inputs: &[ObjectRef], iterations: u64) -> usize {
    hot_loop!(inputs, iterations, value => {
        match TryFromTemp::<Leaf0>::try_from(AnyView::from(value)) {
            Ok(_) => 0,
            Err(rejected) => {
                drop(rejected);
                MISS
            }
        }
    })
}

#[inline(never)]
fn light_single(inputs: &[ObjectRef], iterations: u64) -> usize {
    hot_loop!(inputs, iterations, value => {
        match TryInto::<Leaf0>::try_into(AnyView::from(value)) {
            Ok(_) => 0,
            Err(()) => MISS,
        }
    })
}

#[inline(never)]
fn diagnostic_chain(inputs: &[ObjectRef], iterations: u64) -> usize {
    hot_loop!(inputs, iterations, value => {
        let view = AnyView::from(value);
        match TryFromTemp::<Leaf0>::try_from(view) {
            Ok(_) => 0,
            Err(rejected) => {
                drop(rejected);
                match TryFromTemp::<Leaf1>::try_from(view) {
                    Ok(_) => 1,
                    Err(rejected) => {
                        drop(rejected);
                        MISS
                    }
                }
            }
        }
    })
}

#[inline(never)]
fn light_chain(inputs: &[ObjectRef], iterations: u64) -> usize {
    hot_loop!(inputs, iterations, value => {
        let view = AnyView::from(value);
        match TryInto::<Leaf0>::try_into(view) {
            Ok(_) => 0,
            Err(()) => match TryInto::<Leaf1>::try_into(view) {
                Ok(_) => 1,
                Err(()) => MISS,
            },
        }
    })
}

macro_rules! define_match_any_pair {
    (
        $arm_count:literal,
        $table_capacity:literal,
        $ordered:ident,
        $lookup:ident,
        $($matcher:ident => $arm_id:expr),+ $(,)?
    ) => {
        #[inline(never)]
        fn $ordered(inputs: &[ObjectRef], iterations: u64) -> usize {
            hot_loop!(inputs, iterations, value => {
                // Guards intentionally keep this call site on the ordered path.
                let selected = match_any! {
                    *value {
                        $($matcher(_) if true => $arm_id,)+
                        _ => MISS,
                    }
                };
                dispatch_arm(selected)
            })
        }

        #[inline(never)]
        fn $lookup(inputs: &[ObjectRef], iterations: u64) -> usize {
            static TABLES: ColdTablePair<LeafLookupTable<$table_capacity>> = ColdTablePair {
                warmup: LeafLookupTable::new(),
                target: LeafLookupTable::new(),
            };
            if iterations == 0 {
                let pattern_id = TypeId::of::<[(); $arm_count]>();
                if TABLES.warmup.pattern_list_id().is_none() {
                    let object_begin =
                        tvm_ffi_sys::TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32;
                    let type_indices: [i32; $arm_count] =
                        std::array::from_fn(|offset| object_begin + offset as i32);
                    TABLES
                        .warmup
                        .initialize(pattern_id, type_indices)
                        .unwrap();
                }
                black_box(unsafe {
                    TABLES.warmup.lookup_after_init(
                        tvm_ffi_sys::TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32,
                    )
                });
                return 0;
            }
            hot_loop!(inputs, iterations, value => {
                // Mirror the warmed leaf-table selection independently of the
                // production threshold so every arity remains measurable.
                let view = AnyView::from(value);
                let pattern_id = TypeId::of::<($($matcher,)+)>();
                let table = &TABLES.target;
                let selected = match table.pattern_list_id() {
                    Some(initialized_id) if initialized_id == pattern_id => unsafe {
                        table.lookup_after_init(view.type_index())
                    },
                    Some(_) => panic!("leaf table pattern ID changed"),
                    None => {
                        table.initialize(
                            pattern_id,
                            [
                        $(
                            <<$matcher as ObjectRefCore>::ContainerType as ObjectCore>
                                ::type_index(),
                        )+
                            ],
                        ).unwrap();
                        unsafe { table.lookup_after_init(view.type_index()) }
                    }
                };
                let selected = match selected {
                    $(
                        Some(selected) if selected == $arm_id as ArmId => {
                            $arm_id
                        }
                    )+
                    Some(_) => unreachable!("leaf table returned an unknown arm"),
                    None => MISS,
                };
                dispatch_arm(selected)
            })
        }
    };
}

define_match_any_pair!(
    2,
    4,
    ordered_2,
    lookup_2,
    Leaf0 => 0,
    Leaf1 => 1,
);
define_match_any_pair!(
    3,
    8,
    ordered_3,
    lookup_3,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
);
define_match_any_pair!(
    4,
    8,
    ordered_4,
    lookup_4,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
);
define_match_any_pair!(
    5,
    16,
    ordered_5,
    lookup_5,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
);
define_match_any_pair!(
    6,
    16,
    ordered_6,
    lookup_6,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
);
define_match_any_pair!(
    7,
    16,
    ordered_7,
    lookup_7,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
);
define_match_any_pair!(
    8,
    16,
    ordered_8,
    lookup_8,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
);
define_match_any_pair!(
    9,
    32,
    ordered_9,
    lookup_9,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
);
define_match_any_pair!(
    10,
    32,
    ordered_10,
    lookup_10,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
);
define_match_any_pair!(
    11,
    32,
    ordered_11,
    lookup_11,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
);
define_match_any_pair!(
    12,
    32,
    ordered_12,
    lookup_12,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
);
define_match_any_pair!(
    13,
    32,
    ordered_13,
    lookup_13,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
);
define_match_any_pair!(
    14,
    32,
    ordered_14,
    lookup_14,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
    Leaf13 => 13,
);
define_match_any_pair!(
    15,
    32,
    ordered_15,
    lookup_15,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
    Leaf13 => 13,
    Leaf14 => 14,
);
define_match_any_pair!(
    16,
    32,
    ordered_16,
    lookup_16,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
    Leaf13 => 13,
    Leaf14 => 14,
    Leaf15 => 15,
);
define_match_any_pair!(
    17,
    64,
    ordered_17,
    lookup_17,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
    Leaf13 => 13,
    Leaf14 => 14,
    Leaf15 => 15,
    Leaf16 => 16,
);

macro_rules! define_production_lookup_pair {
    (
        $wildcard:ident,
        $bound:ident,
        $($matcher:ident => $arm_id:expr),+ $(,)?
    ) => {
        #[inline(never)]
        fn $wildcard(inputs: &[ObjectRef], iterations: u64) -> usize {
            hot_loop!(inputs, iterations, value => {
                let selected = match_any! {
                    *value {
                        $($matcher(_) => $arm_id,)+
                        _ => MISS,
                    }
                };
                dispatch_arm(selected)
            })
        }

        #[inline(never)]
        fn $bound(inputs: &[ObjectRef], iterations: u64) -> usize {
            hot_loop!(inputs, iterations, value => {
                let selected = match_any! {
                    *value {
                        $($matcher(_value) => $arm_id,)+
                        _ => MISS,
                    }
                };
                dispatch_arm(selected)
            })
        }
    };
}

macro_rules! define_ordered_bound {
    (
        $name:ident,
        $($matcher:ident => $arm_id:expr),+ $(,)?
    ) => {
        #[inline(never)]
        fn $name(inputs: &[ObjectRef], iterations: u64) -> usize {
            hot_loop!(inputs, iterations, value => {
                let selected = match_any! {
                    *value {
                        $($matcher(_value) if true => $arm_id,)+
                        _ => MISS,
                    }
                };
                dispatch_arm(selected)
            })
        }
    };
}

macro_rules! define_mixed_binding_pair {
    (
        $ordered:ident,
        $automatic:ident,
        $($matcher:ident($binding:pat) => $arm_id:expr),+ $(,)?
    ) => {
        #[inline(never)]
        fn $ordered(inputs: &[ObjectRef], iterations: u64) -> usize {
            hot_loop!(inputs, iterations, value => {
                let selected = match_any! {
                    *value {
                        $($matcher($binding) if true => $arm_id,)+
                        _ => MISS,
                    }
                };
                dispatch_arm(selected)
            })
        }

        #[inline(never)]
        fn $automatic(inputs: &[ObjectRef], iterations: u64) -> usize {
            hot_loop!(inputs, iterations, value => {
                let selected = match_any! {
                    *value {
                        $($matcher($binding) => $arm_id,)+
                        _ => MISS,
                    }
                };
                dispatch_arm(selected)
            })
        }
    };
}

define_mixed_binding_pair!(
    ordered_one_bound_12,
    automatic_one_bound_12,
    Leaf0(_value) => 0,
    Leaf1(_) => 1,
    Leaf2(_) => 2,
    Leaf3(_) => 3,
    Leaf4(_) => 4,
    Leaf5(_) => 5,
    Leaf6(_) => 6,
    Leaf7(_) => 7,
    Leaf8(_) => 8,
    Leaf9(_) => 9,
    Leaf10(_) => 10,
    Leaf11(_) => 11,
);
define_mixed_binding_pair!(
    ordered_half_bound_12,
    automatic_half_bound_12,
    Leaf0(_value) => 0,
    Leaf1(_) => 1,
    Leaf2(_value) => 2,
    Leaf3(_) => 3,
    Leaf4(_value) => 4,
    Leaf5(_) => 5,
    Leaf6(_value) => 6,
    Leaf7(_) => 7,
    Leaf8(_value) => 8,
    Leaf9(_) => 9,
    Leaf10(_value) => 10,
    Leaf11(_) => 11,
);

define_ordered_bound!(
    ordered_bound_10,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
);
define_ordered_bound!(
    ordered_bound_11,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
);
define_ordered_bound!(
    ordered_bound_12,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
);
define_ordered_bound!(
    ordered_bound_17,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
    Leaf13 => 13,
    Leaf14 => 14,
    Leaf15 => 15,
    Leaf16 => 16,
);

define_production_lookup_pair!(
    macro_lookup_10,
    macro_lookup_bound_10,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
);
define_production_lookup_pair!(
    macro_lookup_11,
    macro_lookup_bound_11,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
);
define_production_lookup_pair!(
    macro_lookup_12,
    macro_lookup_bound_12,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
);
define_production_lookup_pair!(
    macro_lookup_15,
    macro_lookup_bound_15,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
    Leaf13 => 13,
    Leaf14 => 14,
);
define_production_lookup_pair!(
    macro_lookup_16,
    macro_lookup_bound_16,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
    Leaf13 => 13,
    Leaf14 => 14,
    Leaf15 => 15,
);
define_production_lookup_pair!(
    macro_lookup_17,
    macro_lookup_bound_17,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
    Leaf5 => 5,
    Leaf6 => 6,
    Leaf7 => 7,
    Leaf8 => 8,
    Leaf9 => 9,
    Leaf10 => 10,
    Leaf11 => 11,
    Leaf12 => 12,
    Leaf13 => 13,
    Leaf14 => 14,
    Leaf15 => 15,
    Leaf16 => 16,
);

fn runners_for_arity(arm_count: usize) -> (Runner, Runner) {
    match arm_count {
        2 => (ordered_2, lookup_2),
        3 => (ordered_3, lookup_3),
        4 => (ordered_4, lookup_4),
        5 => (ordered_5, lookup_5),
        6 => (ordered_6, lookup_6),
        7 => (ordered_7, lookup_7),
        8 => (ordered_8, lookup_8),
        9 => (ordered_9, lookup_9),
        10 => (ordered_10, lookup_10),
        11 => (ordered_11, lookup_11),
        12 => (ordered_12, lookup_12),
        13 => (ordered_13, lookup_13),
        14 => (ordered_14, lookup_14),
        15 => (ordered_15, lookup_15),
        16 => (ordered_16, lookup_16),
        17 => (ordered_17, lookup_17),
        _ => panic!("unsupported arm count: {arm_count}"),
    }
}

fn benchmark_arity(
    arm_count: usize,
    ordered: Runner,
    lookup: Runner,
    values: &[ObjectRef],
    miss: &ObjectRef,
) {
    let mut runners: Vec<(&str, Runner)> = vec![("ordered", ordered), ("lookup", lookup)];
    let ordered_bound = match arm_count {
        10 => Some(ordered_bound_10 as Runner),
        11 => Some(ordered_bound_11 as Runner),
        12 => Some(ordered_bound_12 as Runner),
        17 => Some(ordered_bound_17 as Runner),
        _ => None,
    };
    if let Some(ordered_bound) = ordered_bound {
        runners.push(("ordered + bind", ordered_bound));
    }
    if arm_count == 12 {
        runners.push(("ordered one bind", ordered_one_bound_12));
        runners.push(("auto one bind", automatic_one_bound_12));
        runners.push(("ordered half bind", ordered_half_bound_12));
        runners.push(("auto half bind", automatic_half_bound_12));
    }
    let production_runners = match arm_count {
        10 => Some((macro_lookup_10 as Runner, macro_lookup_bound_10 as Runner)),
        11 => Some((macro_lookup_11 as Runner, macro_lookup_bound_11 as Runner)),
        12 => Some((macro_lookup_12 as Runner, macro_lookup_bound_12 as Runner)),
        15 => Some((macro_lookup_15 as Runner, macro_lookup_bound_15 as Runner)),
        16 => Some((macro_lookup_16 as Runner, macro_lookup_bound_16 as Runner)),
        17 => Some((macro_lookup_17 as Runner, macro_lookup_bound_17 as Runner)),
        _ => None,
    };
    if let Some((wildcard, bound)) = production_runners {
        runners.push(("macro auto wildcard", wildcard));
        runners.push(("macro auto bound", bound));
    }
    for (expected, value) in values[..arm_count].iter().enumerate() {
        let input = [value.clone()];
        for &(name, run) in &runners {
            assert_eq!(run(&input, 1), expected, "arity-{arm_count}/{name}");
        }
    }
    let miss_input = [miss.clone()];
    assert_eq!(ordered(&miss_input, 1), MISS);
    assert_eq!(lookup(&miss_input, 1), MISS);

    let first = [values[0].clone()];
    print_case(&format!("arity-{arm_count}"), "first", 0, &runners, &first);
    let last = [values[arm_count - 1].clone()];
    print_case(
        &format!("arity-{arm_count}"),
        "last",
        arm_count - 1,
        &runners,
        &last,
    );
    print_case(
        &format!("arity-{arm_count}"),
        "miss",
        MISS,
        &runners,
        &miss_input,
    );

    let inputs = balanced_inputs(values, miss, arm_count);
    let mut mixed_runners: Vec<(&str, Runner)> = vec![("noop", noop)];
    mixed_runners.extend(runners);
    print_mixed_case(
        &format!("arity-{arm_count}"),
        "balanced",
        &mixed_runners,
        &inputs,
    );
}

fn balanced_inputs(values: &[ObjectRef], miss: &ObjectRef, arm_count: usize) -> Vec<ObjectRef> {
    const OCCURRENCES_PER_CASE: usize = 1_024;

    let mut inputs = Vec::with_capacity((arm_count + 1) * OCCURRENCES_PER_CASE);
    for _ in 0..OCCURRENCES_PER_CASE {
        inputs.extend(values[..arm_count].iter().cloned());
        inputs.push(miss.clone());
    }

    // Deterministic Fisher-Yates shuffle. The same prebuilt sequence is shared
    // by every strategy and is never modified during timing.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ arm_count as u64;
    for index in (1..inputs.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        inputs.swap(index, state as usize % (index + 1));
    }
    inputs
}

fn time_once(run: Runner, inputs: &[ObjectRef], iterations: u64) -> Duration {
    let start = Instant::now();
    let checksum = run(inputs, iterations);
    let elapsed = start.elapsed();
    black_box(checksum);
    elapsed
}

fn calibrate(run: Runner, inputs: &[ObjectRef]) -> u64 {
    let sequence_len = inputs.len() as u64;
    let mut iterations = sequence_len.max(1_024);
    iterations = iterations.div_ceil(sequence_len) * sequence_len;
    loop {
        let elapsed = time_once(run, inputs, iterations);
        if elapsed >= Duration::from_millis(3) {
            let scaled =
                iterations as u128 * TARGET_SAMPLE_TIME.as_nanos() / elapsed.as_nanos().max(1);
            let scaled = scaled.max(iterations as u128) as u64;
            return scaled.div_ceil(sequence_len) * sequence_len;
        }
        iterations = iterations.saturating_mul(4);
    }
}

fn median_and_mad(samples: &mut [f64]) -> (f64, f64) {
    samples.sort_unstable_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    let mut deviations = samples
        .iter()
        .map(|sample| (sample - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_unstable_by(f64::total_cmp);
    (median, deviations[deviations.len() / 2])
}

fn make_objects() -> ([ObjectRef; 17], ObjectRef) {
    let values = [
        Leaf0::new().try_cast().unwrap(),
        Leaf1::new().try_cast().unwrap(),
        Leaf2::new().try_cast().unwrap(),
        Leaf3::new().try_cast().unwrap(),
        Leaf4::new().try_cast().unwrap(),
        Leaf5::new().try_cast().unwrap(),
        Leaf6::new().try_cast().unwrap(),
        Leaf7::new().try_cast().unwrap(),
        Leaf8::new().try_cast().unwrap(),
        Leaf9::new().try_cast().unwrap(),
        Leaf10::new().try_cast().unwrap(),
        Leaf11::new().try_cast().unwrap(),
        Leaf12::new().try_cast().unwrap(),
        Leaf13::new().try_cast().unwrap(),
        Leaf14::new().try_cast().unwrap(),
        Leaf15::new().try_cast().unwrap(),
        Leaf16::new().try_cast().unwrap(),
    ];
    let miss = Shape::from([1_i64, 2, 3]).try_cast().unwrap();
    (values, miss)
}

fn warm_leaf_table_code(arm_count: usize) {
    let object_begin = tvm_ffi_sys::TVMFFITypeIndex::kTVMFFIStaticObjectBegin as i32;
    macro_rules! warm {
        ($arms:literal, $capacity:literal) => {{
            static TABLE: LeafLookupTable<$capacity> = LeafLookupTable::new();
            let pattern_id = TypeId::of::<[(); $arms]>();
            let type_indices: [i32; $arms] =
                std::array::from_fn(|offset| object_begin + offset as i32);
            match TABLE.pattern_list_id() {
                Some(initialized_id) => assert_eq!(initialized_id, pattern_id),
                None => TABLE.initialize(pattern_id, type_indices).unwrap(),
            }
            black_box(unsafe { TABLE.lookup_after_init(object_begin) });
        }};
    }

    match arm_count {
        2 => warm!(2, 4),
        3 => warm!(3, 8),
        4 => warm!(4, 8),
        5 => warm!(5, 16),
        6 => warm!(6, 16),
        7 => warm!(7, 16),
        8 => warm!(8, 16),
        9 => warm!(9, 32),
        10 => warm!(10, 32),
        11 => warm!(11, 32),
        12 => warm!(12, 32),
        13 => warm!(13, 32),
        14 => warm!(14, 32),
        15 => warm!(15, 32),
        16 => warm!(16, 32),
        17 => warm!(17, 64),
        _ => panic!("unsupported warmup arm count: {arm_count}"),
    }
}

fn warm_cold_dependencies(values: &[ObjectRef; 17]) {
    let entries = [
        (Leaf0Obj::type_index(), 0 as ArmId),
        (Leaf1Obj::type_index(), 1 as ArmId),
        (Leaf2Obj::type_index(), 2 as ArmId),
        (Leaf3Obj::type_index(), 3 as ArmId),
        (Leaf4Obj::type_index(), 4 as ArmId),
        (Leaf5Obj::type_index(), 5 as ArmId),
        (Leaf6Obj::type_index(), 6 as ArmId),
        (Leaf7Obj::type_index(), 7 as ArmId),
        (Leaf8Obj::type_index(), 8 as ArmId),
        (Leaf9Obj::type_index(), 9 as ArmId),
        (Leaf10Obj::type_index(), 10 as ArmId),
        (Leaf11Obj::type_index(), 11 as ArmId),
        (Leaf12Obj::type_index(), 12 as ArmId),
        (Leaf13Obj::type_index(), 13 as ArmId),
        (Leaf14Obj::type_index(), 14 as ArmId),
        (Leaf15Obj::type_index(), 15 as ArmId),
        (Leaf16Obj::type_index(), 16 as ArmId),
    ];
    black_box(entries);
    macro_rules! warm_cast {
        ($matcher:ty, $index:expr) => {
            black_box(
                TryInto::<$matcher>::try_into(AnyView::from(&values[$index]))
                    .expect("matching leaf conversion"),
            );
        };
    }
    warm_cast!(Leaf0, 0);
    warm_cast!(Leaf1, 1);
    warm_cast!(Leaf2, 2);
    warm_cast!(Leaf3, 3);
    warm_cast!(Leaf4, 4);
    warm_cast!(Leaf5, 5);
    warm_cast!(Leaf6, 6);
    warm_cast!(Leaf7, 7);
    warm_cast!(Leaf8, 8);
    warm_cast!(Leaf9, 9);
    warm_cast!(Leaf10, 10);
    warm_cast!(Leaf11, 11);
    warm_cast!(Leaf12, 12);
    warm_cast!(Leaf13, 13);
    warm_cast!(Leaf14, 14);
    warm_cast!(Leaf15, 15);
    warm_cast!(Leaf16, 16);
    for arm_id in 0..17 {
        black_box(dispatch_arm(arm_id));
    }
    black_box(dispatch_arm(MISS));

    // Warm the timer without touching any target lookup_N call-site static.
    black_box(Instant::now().elapsed());
}

fn run_cold_child(spec: &str) {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let (values, miss) = make_objects();
    warm_cold_dependencies(&values);

    let (run, input, expected) = if spec == "noop" {
        (noop as Runner, [values[0].clone()], 0)
    } else {
        let (arm_count, case) = spec
            .split_once(':')
            .unwrap_or_else(|| panic!("invalid cold child spec: {spec}"));
        let arm_count = arm_count
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("invalid cold arm count: {arm_count}"));
        let (_, lookup) = runners_for_arity(arm_count);
        warm_leaf_table_code(arm_count);
        match case {
            "first" => (lookup, [values[0].clone()], 0),
            "last" => (lookup, [values[arm_count - 1].clone()], arm_count - 1),
            "miss" => (lookup, [miss], MISS),
            _ => panic!("invalid cold input case: {case}"),
        }
    };

    // Warm the exact runner's code and adjacent table storage without touching
    // the target table. `iterations == 0` initializes a separate warmup table.
    black_box(run(&input, 0));
    let start = Instant::now();
    let checksum = run(&input, 1);
    let elapsed = start.elapsed();
    assert_eq!(checksum, expected);
    println!("{}", elapsed.as_nanos());
}

fn run_metadata_cold_child(spec: &str) {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let arm_count = spec
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid metadata-cold arm count: {spec}"));
    let (_, lookup) = runners_for_arity(arm_count);

    // Construct only a non-matching input. This keeps every leaf-pattern
    // type-index LazyLock untouched until the lookup table is initialized.
    let miss: ObjectRef = Shape::from([1_i64, 2, 3]).try_cast().unwrap();
    let input = [miss];
    warm_leaf_table_code(arm_count);
    black_box(Instant::now().elapsed());
    // Warm the exact runner's code and adjacent table storage without touching
    // the target table. The pattern metadata remains cold.
    black_box(lookup(&input, 0));

    let start = Instant::now();
    let checksum = lookup(&input, 1);
    let elapsed = start.elapsed();
    assert_eq!(checksum, MISS);
    println!("{}", elapsed.as_nanos());
}

fn cold_sample(env_name: &str, spec: &str) -> f64 {
    let output = Command::new(env::current_exe().expect("benchmark executable path"))
        .env(env_name, spec)
        .output()
        .expect("run cold benchmark child");
    assert!(
        output.status.success(),
        "cold child {spec} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("cold child output must be UTF-8")
        .trim()
        .parse::<f64>()
        .expect("cold child must print elapsed nanoseconds")
}

fn measure_cold(env_name: &str, spec: &str) -> (f64, f64) {
    let mut samples = (0..COLD_SAMPLE_COUNT)
        .map(|_| cold_sample(env_name, spec))
        .collect::<Vec<_>>();
    median_and_mad(&mut samples)
}

fn run_cold_benchmark() {
    println!("Rust leaf-table first-call benchmark (-O3)");
    println!("Each sample runs in a fresh process, so the target call-site OnceLock is empty.");
    println!(
        "Object construction, type registration, leaf metadata, runner code, and timer are warmed."
    );
    println!("Timing includes forced table initialization, lookup, and selected no-op handler.");
    println!(
        "The production macro keeps its first eligible object call ordered and initializes later."
    );
    println!("Process startup is outside the child-process timer.");
    println!("Each result is the median of {COLD_SAMPLE_COUNT} samples; MAD is also reported.");
    println!();
    println!("arity      case      strategy               ns/call       MAD");

    let (median, mad) = measure_cold(COLD_CHILD_ENV, "noop");
    println!(
        "{:<10} {:<9} {:<20} {:>10.3} {:>9.3}",
        "-", "first", "timer + noop", median, mad
    );

    for arm_count in 2..=17 {
        let spec = format!("{arm_count}:first");
        let (median, mad) = measure_cold(COLD_CHILD_ENV, &spec);
        println!(
            "{:<10} {:<9} {:<20} {:>10.3} {:>9.3}",
            arm_count, "first", "cold leaf-table", median, mad
        );
        if arm_count == 2 || arm_count == 12 || arm_count == 17 {
            for case in ["last", "miss"] {
                let spec = format!("{arm_count}:{case}");
                let (median, mad) = measure_cold(COLD_CHILD_ENV, &spec);
                println!(
                    "{:<10} {:<9} {:<20} {:>10.3} {:>9.3}",
                    arm_count, case, "cold leaf-table", median, mad
                );
            }
        }
    }

    println!();
    println!("Worst-case first call with leaf-pattern TypeIndex metadata also cold:");
    println!("Object construction and process startup remain outside the timer.");
    println!("arity      case      strategy               ns/call       MAD");
    for arm_count in 2..=17 {
        let spec = arm_count.to_string();
        let (median, mad) = measure_cold(METADATA_COLD_CHILD_ENV, &spec);
        println!(
            "{:<10} {:<9} {:<20} {:>10.3} {:>9.3}",
            arm_count, "miss", "metadata + table", median, mad
        );
    }
}

fn measure_group(
    runners: &[(&'static str, Runner)],
    inputs: &[ObjectRef],
) -> Vec<(&'static str, f64, f64)> {
    // Warm type metadata, conversion paths, and the lookup call site's
    // OnceLock before calibration and timing.
    for &(_, run) in runners {
        black_box(run(inputs, inputs.len() as u64));
    }

    let iterations = runners
        .iter()
        .map(|&(_, run)| calibrate(run, inputs))
        .collect::<Vec<_>>();
    let mut samples = runners
        .iter()
        .map(|_| Vec::with_capacity(SAMPLE_COUNT))
        .collect::<Vec<_>>();

    // Rotate measurement order so no strategy always runs first.
    for sample in 0..SAMPLE_COUNT {
        for offset in 0..runners.len() {
            let index = (sample + offset) % runners.len();
            let elapsed = time_once(runners[index].1, inputs, iterations[index]);
            samples[index].push(elapsed.as_nanos() as f64 / iterations[index] as f64);
        }
    }

    runners
        .iter()
        .zip(samples)
        .map(|(&(name, _), mut samples)| {
            let (median, mad) = median_and_mad(&mut samples);
            (name, median, mad)
        })
        .collect()
}

fn print_case(
    group: &str,
    case: &str,
    expected: usize,
    runners: &[(&'static str, Runner)],
    inputs: &[ObjectRef],
) {
    for &(name, run) in runners {
        if name != "noop" {
            assert_eq!(run(inputs, 1), expected, "{group}/{case}/{name}");
        }
    }
    for (name, median, mad) in measure_group(runners, inputs) {
        println!("{group:<10} {case:<7} {name:<18} {median:>10.3} {mad:>9.3}");
    }
}

fn print_mixed_case(
    group: &str,
    case: &str,
    runners: &[(&'static str, Runner)],
    inputs: &[ObjectRef],
) {
    let expected = runners[1].1(inputs, inputs.len() as u64);
    for &(name, run) in &runners[2..] {
        assert_eq!(
            run(inputs, inputs.len() as u64),
            expected,
            "{group}/{case}/{name}"
        );
    }
    for (name, median, mad) in measure_group(runners, inputs) {
        println!("{group:<10} {case:<7} {name:<18} {median:>10.3} {mad:>9.3}");
    }
}

fn main() {
    if let Ok(spec) = env::var(METADATA_COLD_CHILD_ENV) {
        run_metadata_cold_child(&spec);
        return;
    }
    if let Ok(spec) = env::var(COLD_CHILD_ENV) {
        run_cold_child(&spec);
        return;
    }
    if cfg!(debug_assertions) {
        eprintln!("match_any_threshold requires `cargo bench` (release opt-level=3)");
        return;
    }
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);

    if env::args().any(|arg| arg == "--cold") {
        run_cold_benchmark();
        return;
    }

    // Object construction and runtime type registration happen before timing.
    let (values, miss) = make_objects();
    if env::args().any(|arg| arg == "--indices") {
        let indices = [
            Leaf0Obj::type_index(),
            Leaf1Obj::type_index(),
            Leaf2Obj::type_index(),
            Leaf3Obj::type_index(),
            Leaf4Obj::type_index(),
            Leaf5Obj::type_index(),
            Leaf6Obj::type_index(),
            Leaf7Obj::type_index(),
            Leaf8Obj::type_index(),
            Leaf9Obj::type_index(),
            Leaf10Obj::type_index(),
            Leaf11Obj::type_index(),
            Leaf12Obj::type_index(),
            Leaf13Obj::type_index(),
            Leaf14Obj::type_index(),
            Leaf15Obj::type_index(),
            Leaf16Obj::type_index(),
        ];
        println!("{indices:?}");
        return;
    }
    if let Some(arm_count) = env::args().find_map(|arg| {
        arg.strip_prefix("--arity=")
            .and_then(|value| value.parse::<usize>().ok())
    }) {
        let (ordered, lookup) = runners_for_arity(arm_count);
        println!("group      case    strategy               ns/op       MAD");
        benchmark_arity(arm_count, ordered, lookup, &values, &miss);
        return;
    }
    let first = [values[0].clone()];
    let second = [values[1].clone()];
    let miss_input = [miss.clone()];

    let single: &[(&str, Runner)] = &[
        ("noop", noop),
        ("TypeError path", diagnostic_single),
        ("Result<T, ()>", light_single),
    ];
    let chain: &[(&str, Runner)] = &[
        ("noop", noop),
        ("TypeError chain", diagnostic_chain),
        ("Result chain", light_chain),
        ("ordered match_any", ordered_2),
        ("leaf-table path", lookup_2),
    ];

    println!("Rust object-cast hot-loop benchmark (-O3)");
    println!("Objects and type indices are initialized before timing.");
    println!("The lookup OnceLock is warmed before timing.");
    println!(
        "Each arm calls a distinct noinline integer handler so dispatch is not optimized away."
    );
    println!("Each timed dispatch starts from a prebuilt ObjectRef handle.");
    println!("TypeError reproduces the old diagnostic path; Result uses AnyView::try_into.");
    println!("Each result is the median of {SAMPLE_COUNT} samples; MAD is also reported.");
    println!();
    println!("group      case    strategy               ns/op       MAD");

    print_case("single", "hit", 0, single, &first);
    print_case("single", "miss", MISS, single, &second);
    print_case("two-arm", "first", 0, chain, &first);
    print_case("two-arm", "second", 1, chain, &second);
    print_case("two-arm", "miss", MISS, chain, &miss_input);

    println!();
    println!("Leaf-only arity sweep with the same warmed hot-loop setup:");
    println!("balanced = uniform over every arm plus one miss, in a shuffled sequence.");
    benchmark_arity(2, ordered_2, lookup_2, &values, &miss);
    benchmark_arity(3, ordered_3, lookup_3, &values, &miss);
    benchmark_arity(4, ordered_4, lookup_4, &values, &miss);
    benchmark_arity(5, ordered_5, lookup_5, &values, &miss);
    benchmark_arity(6, ordered_6, lookup_6, &values, &miss);
    benchmark_arity(7, ordered_7, lookup_7, &values, &miss);
    benchmark_arity(8, ordered_8, lookup_8, &values, &miss);
    benchmark_arity(9, ordered_9, lookup_9, &values, &miss);
    benchmark_arity(10, ordered_10, lookup_10, &values, &miss);
    benchmark_arity(11, ordered_11, lookup_11, &values, &miss);
    benchmark_arity(12, ordered_12, lookup_12, &values, &miss);
    benchmark_arity(13, ordered_13, lookup_13, &values, &miss);
    benchmark_arity(14, ordered_14, lookup_14, &values, &miss);
    benchmark_arity(15, ordered_15, lookup_15, &values, &miss);
    benchmark_arity(16, ordered_16, lookup_16, &values, &miss);
    benchmark_arity(17, ordered_17, lookup_17, &values, &miss);
}
