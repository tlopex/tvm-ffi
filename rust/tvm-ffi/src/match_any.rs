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
use std::marker::PhantomData;

use tvm_ffi_sys::TVMFFITypeIndex as TypeIndex;

use crate::object::{ObjectCore, ObjectRefCore};
use crate::type_traits::AnyCompatible;

// A short comparison chain is cheaper than allocating and indexing a table.
const MIN_INDEXED_ARMS: usize = 4;
// Avoid a large mostly-empty table when registered type indices are sparse.
const MAX_INDEX_SPACE_PER_ARM: usize = 4;
const MISSING_ARM: usize = usize::MAX;

struct DenseDispatch {
    first_type_index: i32,
    arm_by_offset: Box<[usize]>,
}

impl DenseDispatch {
    fn build(type_indices: &[Option<i32>]) -> Option<Self> {
        if type_indices.len() < MIN_INDEXED_ARMS {
            return None;
        }

        let mut first_type_index = i32::MAX;
        let mut last_type_index = i32::MIN;
        for type_index in type_indices.iter().copied() {
            let type_index = type_index?;
            if type_index < TypeIndex::kTVMFFIStaticObjectBegin as i32 {
                return None;
            }
            first_type_index = first_type_index.min(type_index);
            last_type_index = last_type_index.max(type_index);
        }

        let mut unique_type_indices = Vec::with_capacity(type_indices.len());
        for type_index in type_indices.iter().copied().flatten() {
            if !unique_type_indices.contains(&type_index) {
                unique_type_indices.push(type_index);
            }
        }
        if unique_type_indices.len() < MIN_INDEXED_ARMS {
            return None;
        }

        let index_space =
            usize::try_from(i64::from(last_type_index) - i64::from(first_type_index) + 1).ok()?;
        if index_space
            > unique_type_indices
                .len()
                .saturating_mul(MAX_INDEX_SPACE_PER_ARM)
        {
            return None;
        }

        let mut arm_by_offset = vec![MISSING_ARM; index_space].into_boxed_slice();
        for (arm_id, type_index) in type_indices.iter().copied().enumerate() {
            let offset = usize::try_from(type_index? - first_type_index).ok()?;
            if arm_by_offset[offset] == MISSING_ARM {
                arm_by_offset[offset] = arm_id;
            }
        }

        Some(Self {
            first_type_index,
            arm_by_offset,
        })
    }

    #[inline(always)]
    fn lookup(&self, type_index: i32) -> Option<usize> {
        let offset = i64::from(type_index) - i64::from(self.first_type_index);
        let arm_id = *self.arm_by_offset.get(usize::try_from(offset).ok()?)?;
        (arm_id != MISSING_ARM).then_some(arm_id)
    }
}

/// A lazily initialized direct-dispatch plan used by `match_any!`.
#[doc(hidden)]
pub struct ExactDispatchPlan {
    pattern_types: TypeId,
    direct: Option<DenseDispatch>,
}

impl ExactDispatchPlan {
    #[doc(hidden)]
    pub fn build(pattern_types: TypeId, type_indices: &[Option<i32>]) -> Self {
        Self {
            pattern_types,
            direct: DenseDispatch::build(type_indices),
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn lookup(&self, pattern_types: TypeId, type_index: i32) -> Result<Option<usize>, ()> {
        if self.pattern_types != pattern_types {
            return Err(());
        }
        let direct = self.direct.as_ref().ok_or(())?;
        Ok(direct.lookup(type_index))
    }
}

/// Type-level list used to collect exact object-pattern metadata.
#[doc(hidden)]
pub trait ObjectPatternList: 'static {
    #[doc(hidden)]
    const ALL_EXACT: bool;

    #[doc(hidden)]
    fn fill_type_indices(out: &mut [Option<i32>]);
}

impl ObjectPatternList for () {
    const ALL_EXACT: bool = true;

    fn fill_type_indices(out: &mut [Option<i32>]) {
        debug_assert!(out.is_empty());
    }
}

impl<Head, Tail> ObjectPatternList for (Head, Tail)
where
    Head: AnyCompatible + ObjectRefCore + 'static,
    Tail: ObjectPatternList,
{
    const ALL_EXACT: bool =
        Head::TYPE_CONTAINER_IS_EXACT && Head::ContainerType::TYPE_FINAL && Tail::ALL_EXACT;

    fn fill_type_indices(out: &mut [Option<i32>]) {
        let (head, tail) = out
            .split_first_mut()
            .expect("match_any! pattern metadata length mismatch");
        if Head::TYPE_CONTAINER_IS_EXACT && Head::ContainerType::TYPE_FINAL {
            *head = Some(Head::ContainerType::type_index());
        }
        Tail::fill_type_indices(tail);
    }
}

/// Probe used by the macro to retain ordered matching for non-object patterns.
#[doc(hidden)]
pub struct PatternListProbe<T>(PhantomData<fn() -> T>);

impl<T> PatternListProbe<T> {
    #[doc(hidden)]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Fallback selected when the arm types do not form an `ObjectPatternList`.
#[doc(hidden)]
pub trait DynamicPatternListProbe {
    #[doc(hidden)]
    fn pattern_types(&self) -> Option<TypeId> {
        None
    }

    #[doc(hidden)]
    fn fill_type_indices(&self, _out: &mut [Option<i32>]) {}
}

impl<T> DynamicPatternListProbe for &PatternListProbe<T> {}

/// Specialized probe selected when every arm type is an object reference.
#[doc(hidden)]
pub trait ExactPatternListProbe {
    #[doc(hidden)]
    fn pattern_types(&self) -> Option<TypeId>;

    #[doc(hidden)]
    fn fill_type_indices(&self, out: &mut [Option<i32>]);
}

impl<T: ObjectPatternList> ExactPatternListProbe for PatternListProbe<T> {
    fn pattern_types(&self) -> Option<TypeId> {
        T::ALL_EXACT.then(TypeId::of::<T>)
    }

    fn fill_type_indices(&self, out: &mut [Option<i32>]) {
        T::fill_type_indices(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Array, Module, Tensor};

    #[test]
    fn dense_plan_preserves_source_order() {
        let pattern_types = TypeId::of::<(i32, i64, f32, f64, u32)>();
        let plan = ExactDispatchPlan::build(
            pattern_types,
            &[Some(69), Some(69), Some(70), Some(71), Some(72)],
        );

        assert_eq!(plan.lookup(pattern_types, 69), Ok(Some(0)));
        assert_eq!(plan.lookup(pattern_types, 70), Ok(Some(2)));
        assert_eq!(plan.lookup(pattern_types, 71), Ok(Some(3)));
        assert_eq!(plan.lookup(pattern_types, 73), Ok(None));
    }

    #[test]
    fn ineligible_plans_request_ordered_matching() {
        let pattern_types = TypeId::of::<(i32, i64, f32, f64)>();

        let dynamic =
            ExactDispatchPlan::build(pattern_types, &[Some(69), None, Some(70), Some(71)]);
        assert_eq!(dynamic.lookup(pattern_types, 69), Err(()));

        let sparse =
            ExactDispatchPlan::build(pattern_types, &[Some(64), Some(96), Some(128), Some(160)]);
        assert_eq!(sparse.lookup(pattern_types, 64), Err(()));

        let short = ExactDispatchPlan::build(pattern_types, &[Some(69), Some(70), Some(71)]);
        assert_eq!(short.lookup(pattern_types, 69), Err(()));
    }

    #[test]
    fn probe_only_enables_all_exact_object_lists() {
        type Exact = (Module, (Module, (Module, (Module, ()))));
        let exact = PatternListProbe::<Exact>::new();
        assert!((&exact).pattern_types().is_some());

        type Parameterized = (Array<i64>, (Module, (Module, (Module, ()))));
        let parameterized = PatternListProbe::<Parameterized>::new();
        assert_eq!((&parameterized).pattern_types(), None);

        type NonFinal = (Tensor, (Module, (Module, (Module, ()))));
        let non_final = PatternListProbe::<NonFinal>::new();
        assert_eq!((&non_final).pattern_types(), None);

        type NonObject = (i64, (Module, (Module, (Module, ()))));
        let non_object = PatternListProbe::<NonObject>::new();
        assert_eq!((&non_object).pattern_types(), None);
    }
}
