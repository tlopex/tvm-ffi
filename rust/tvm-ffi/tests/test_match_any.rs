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

use tvm_ffi::match_any_internal::{ArmId, LeafLookupTable, LeafPatternMetadata, LeafPatternProbe};
use tvm_ffi::{match_any, Any, AnyView, Array, Function, Map, Module, Shape, Tensor, TypeIndex};

struct DirectShapeMatcher(Shape);

impl<'a> TryInto<DirectShapeMatcher> for AnyView<'a> {
    type Error = ();

    fn try_into(self) -> Result<DirectShapeMatcher, Self::Error> {
        self.try_as::<Shape>().map(DirectShapeMatcher).ok_or(())
    }
}

#[test]
fn matches_concrete_object_containers_in_source_order() {
    fn classify(expr: Any) -> (&'static str, usize) {
        match_any! {
            expr {
                Tensor(tensor)
                    if tensor.shape().len() == 2 => ("matrix", tensor.shape().len()),
                Tensor(tensor) => ("tensor", tensor.shape().len()),
                Shape(shape) => ("shape", shape.len()),
                Array::<i64>(array) => ("array", array.len()),
                _ => ("unsupported", 0),
            }
        }
    }

    let matrix = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let volume = Tensor::from_slice(&[0_f32; 24], &[2, 3, 4]).unwrap();
    let shape = Shape::from([2_i64, 3, 4, 5]);
    let array = [1_i64, 2, 3].into_iter().collect::<Array<i64>>();

    assert_eq!(classify(Any::from(matrix)), ("matrix", 2));
    assert_eq!(classify(Any::from(volume)), ("tensor", 3));
    assert_eq!(classify(Any::from(shape)), ("shape", 4));
    assert_eq!(classify(Any::from(array)), ("array", 3));
    assert_eq!(
        classify(Any::from(Map::<i64, i64>::default())),
        ("unsupported", 0)
    );
    assert_eq!(classify(Any::from(1_i64)), ("unsupported", 0));

    let tensor = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let view = AnyView::from(&tensor);
    let matched_view = match_any! {
        view {
            Tensor(tensor) => ("tensor", tensor.shape().len()),
            _ => ("unsupported", 0),
        }
    };
    assert_eq!(matched_view, ("tensor", 2));
}

#[test]
fn parameterized_containers_keep_ordered_conversion() {
    let integer_array = [1_i64, 2].into_iter().collect::<Array<i64>>();
    let converted = match_any! {
        Any::from(integer_array) {
            Array::<f64>(array) => array.iter().collect::<Vec<_>>(),
            _ => Vec::new(),
        }
    };
    assert_eq!(converted, vec![1.0, 2.0]);

    let integer_array = [1_i64, 2].into_iter().collect::<Array<i64>>();
    let wildcard_matched = match_any! {
        Any::from(integer_array) {
            Array::<f64>(_) => true,
            _ => false,
        }
    };
    assert!(wildcard_matched);

    let integer_map = [(1_i64, 10_i64)].into_iter().collect::<Map<i64, i64>>();
    let converted = match_any! {
        Any::from(integer_map) {
            Map::<f64, f64>(map) => map.get(&1.0).unwrap(),
            _ => None,
        }
    };
    assert_eq!(converted, Some(10.0));

    let array = [1.5_f64, 2.5].into_iter().collect::<Array<f64>>();
    let selected = match_any! {
        Any::from(array) {
            Array::<i64>(_) => "integer array",
            Tensor(_) => "tensor",
            Shape(_) => "shape",
            Array::<f64>(_) => "float array",
            _ => "unsupported",
        }
    };

    assert_eq!(selected, "float array");
}

#[test]
fn custom_try_into_matchers_keep_the_ordered_fallback() {
    let selected = match_any! {
        Any::from(Shape::from([2_i64, 3, 4])) {
            DirectShapeMatcher(shape) => shape.0.len(),
            Tensor(_) => 0,
            _ => 0,
        }
    };

    assert_eq!(selected, 3);

    let tensor = Tensor::from_slice(&[0_f32; 6], &[2, 3]).unwrap();
    let selected = match_any! {
        Any::from(tensor) {
            DirectShapeMatcher(_) => 0,
            Tensor(tensor) => tensor.shape().len(),
            _ => 0,
        }
    };
    assert_eq!(selected, 2);
}

#[test]
fn duplicate_patterns_keep_the_first_arm() {
    fn classify(value: Any) -> usize {
        match_any! {
            value {
                Module(_) => 0,
                Module(_) => 1,
                Module(_) => 2,
                Module(_) => 3,
                Module(_) => 4,
                Module(_) => 5,
                Module(_) => 6,
                Module(_) => 7,
                Module(_) => 8,
                Module(_) => 9,
                Module(_) => 10,
                Module(_) => 11,
                Module(_) => 12,
                Module(_) => 13,
                Module(_) => 14,
                Module(_) => 15,
                _ => 16,
            }
        }
    }

    let module: Module = Function::get_global("ffi.SystemLib")
        .unwrap()
        .call_tuple_with_len::<0, _>(())
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(classify(Any::from(module.clone())), 0);
    assert_eq!(classify(Any::from(Array::<i64>::default())), 16);
    assert_eq!(classify(Any::from(module)), 0);
    assert_eq!(classify(Any::from(1_i64)), 16);
}

#[test]
fn lookup_table_maps_runtime_indices_to_local_arm_ids() {
    const ARM_0: ArmId = 0;
    const ARM_2: ArmId = 2;
    let pattern_list_id = TypeId::of::<(i32, i64, f32)>();
    static TABLE: LeafLookupTable<8> = LeafLookupTable::new();

    assert_eq!(TABLE.initialize(pattern_list_id, [73, 73, 75]), Ok(()));
    assert_eq!(TABLE.pattern_list_id(), Some(pattern_list_id));
    assert_eq!(unsafe { TABLE.lookup_after_init(73) }, Some(ARM_0));
    assert_eq!(unsafe { TABLE.lookup_after_init(72) }, None);
    assert_eq!(unsafe { TABLE.lookup_after_init(74) }, None);
    assert_eq!(unsafe { TABLE.lookup_after_init(75) }, Some(ARM_2));
    assert_eq!(unsafe { TABLE.lookup_after_init(76) }, None);
}

#[test]
fn a_generic_pattern_list_cannot_reuse_another_lists_table() {
    let pattern_list_id = TypeId::of::<(i32, i64)>();
    static TABLE: LeafLookupTable<4> = LeafLookupTable::new();
    assert_eq!(TABLE.initialize(pattern_list_id, [73, 75]), Ok(()));
    assert_eq!(unsafe { TABLE.lookup_after_init(73) }, Some(0));

    assert_eq!(
        TABLE.initialize(TypeId::of::<(u8, u16)>(), [76, 77]),
        Err(())
    );
}

#[test]
fn lookup_table_handles_hash_collisions_when_indices_are_sparse() {
    let pattern_list_id = TypeId::of::<(i32, i64)>();
    static TABLE: LeafLookupTable<4> = LeafLookupTable::new();

    // These indices collide under the table hash, and their span is too wide
    // for the dense representation.
    assert_eq!(TABLE.initialize(pattern_list_id, [64, 103]), Ok(()));
    assert_eq!(unsafe { TABLE.lookup_after_init(64) }, Some(0));
    assert_eq!(unsafe { TABLE.lookup_after_init(103) }, Some(1));
    assert_eq!(unsafe { TABLE.lookup_after_init(104) }, None);
}

#[test]
fn failed_validation_does_not_partially_initialize_the_table() {
    let pattern_list_id = TypeId::of::<(i32, i64)>();
    static TABLE: LeafLookupTable<4> = LeafLookupTable::new();

    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        TABLE.initialize(pattern_list_id, [0, 73])
    }));
    assert!(failed.is_err());
    assert_eq!(TABLE.pattern_list_id(), None);

    assert_eq!(TABLE.initialize(pattern_list_id, [73, 75]), Ok(()));
    assert_eq!(unsafe { TABLE.lookup_after_init(73) }, Some(0));
    assert_eq!(unsafe { TABLE.lookup_after_init(75) }, Some(1));
}

#[test]
fn lookup_table_delays_initialization_once_and_publishes_concurrently() {
    let pattern_list_id = TypeId::of::<(u32, u64)>();
    static TABLE: LeafLookupTable<4> = LeafLookupTable::new();

    assert!(!TABLE.should_initialize());
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                if TABLE.should_initialize() {
                    assert_eq!(TABLE.initialize(pattern_list_id, [73, 75]), Ok(()));
                    assert_eq!(unsafe { TABLE.lookup_after_init(73) }, Some(0));
                    assert_eq!(unsafe { TABLE.lookup_after_init(75) }, Some(1));
                }
            });
        }
    });
    assert_eq!(TABLE.pattern_list_id(), Some(pattern_list_id));
}

#[test]
fn metadata_only_accepts_exact_leaf_patterns() {
    type Leaf = (Module, ());
    let leaf = LeafPatternProbe::<Leaf>::new();
    let mut type_indices = [0; 1];
    assert!((&leaf).leaf_pattern_list_id().is_some());
    (&leaf).fill_leaf_type_indices(&mut type_indices);
    assert!(type_indices[0] >= TypeIndex::kTVMFFIStaticObjectBegin as i32);

    type Parameterized = (Array<i64>, ());
    let parameterized = LeafPatternProbe::<Parameterized>::new();
    assert!((&parameterized).leaf_pattern_list_id().is_none());

    type NonFinal = (Tensor, ());
    let non_final = LeafPatternProbe::<NonFinal>::new();
    assert!((&non_final).leaf_pattern_list_id().is_none());

    struct NoAnyCompatibleMetadata;
    type Custom = (NoAnyCompatibleMetadata, ());
    let custom = LeafPatternProbe::<Custom>::new();
    assert!((&custom).leaf_pattern_list_id().is_none());
}
