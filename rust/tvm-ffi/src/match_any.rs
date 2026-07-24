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

use crate::AnyCompatible;

const MISSING_ARM: usize = usize::MAX;

struct DenseDispatch {
    base: i32,
    arm_by_offset: Box<[usize]>,
}

impl DenseDispatch {
    fn build(type_indices: &[i32]) -> Option<Self> {
        let base = *type_indices.iter().min()?;
        let last = *type_indices.iter().max()?;
        if base < TypeIndex::kTVMFFIStaticObjectBegin as i32 {
            return None;
        }

        let index_space = usize::try_from(i64::from(last) - i64::from(base) + 1).ok()?;
        // Sparse indices are cheaper to handle with the existing ordered path.
        if index_space > type_indices.len() {
            return None;
        }
        let mut arm_by_offset = vec![MISSING_ARM; index_space].into_boxed_slice();
        for (arm_id, &type_index) in type_indices.iter().enumerate() {
            let offset = usize::try_from(i64::from(type_index) - i64::from(base)).ok()?;
            if arm_by_offset[offset] == MISSING_ARM {
                arm_by_offset[offset] = arm_id;
            }
        }
        Some(Self {
            base,
            arm_by_offset,
        })
    }

    #[inline(always)]
    fn lookup(&self, type_index: i32) -> Option<usize> {
        let offset = usize::try_from(i64::from(type_index) - i64::from(self.base)).ok()?;
        let arm_id = *self.arm_by_offset.get(offset)?;
        (arm_id != MISSING_ARM).then_some(arm_id)
    }
}

/// Lazily initialized direct-dispatch plan used by `match_any!`.
///
/// The pattern-list identity protects a function-local static shared by
/// different generic monomorphizations. A mismatch uses ordered dispatch.
#[doc(hidden)]
pub struct ExactDispatchPlan {
    pattern_list_id: TypeId,
    direct: Option<DenseDispatch>,
}

impl ExactDispatchPlan {
    #[doc(hidden)]
    pub fn build(pattern_list_id: TypeId, type_indices: &[i32]) -> Self {
        Self {
            pattern_list_id,
            direct: DenseDispatch::build(type_indices),
        }
    }

    #[doc(hidden)]
    #[inline(always)]
    pub fn lookup(&self, pattern_list_id: TypeId, type_index: i32) -> Result<Option<usize>, ()> {
        if self.pattern_list_id != pattern_list_id {
            return Err(());
        }
        Ok(self.direct.as_ref().ok_or(())?.lookup(type_index))
    }
}

/// Type-level list used to collect exact object-pattern metadata.
#[doc(hidden)]
pub trait ObjectPatternList: 'static {
    #[doc(hidden)]
    const ALL_EXACT: bool;

    #[doc(hidden)]
    fn fill_exact_type_indices(out: &mut [i32]);
}

impl ObjectPatternList for () {
    const ALL_EXACT: bool = true;

    fn fill_exact_type_indices(out: &mut [i32]) {
        debug_assert!(out.is_empty());
    }
}

impl<Head, Tail> ObjectPatternList for (Head, Tail)
where
    Head: AnyCompatible + 'static,
    Tail: ObjectPatternList,
{
    const ALL_EXACT: bool = Head::MATCH_ANY_EXACT && Tail::ALL_EXACT;

    fn fill_exact_type_indices(out: &mut [i32]) {
        let (head, tail) = out
            .split_first_mut()
            .expect("match_any! pattern metadata length mismatch");
        *head = Head::match_any_exact_type_index();
        Tail::fill_exact_type_indices(tail);
    }
}

/// Probe used by the macro to retain ordered matching for dynamic patterns.
#[doc(hidden)]
pub struct PatternListProbe<T>(PhantomData<fn() -> T>);

impl<T> PatternListProbe<T> {
    #[doc(hidden)]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Pattern-list metadata with an autoref fallback for non-object patterns.
#[doc(hidden)]
pub trait PatternListMetadata {
    #[doc(hidden)]
    fn pattern_list_id(&self) -> Option<TypeId> {
        None
    }

    #[doc(hidden)]
    fn fill_exact_type_indices(&self, _out: &mut [i32]) {}
}

impl<T> PatternListMetadata for &PatternListProbe<T> {}

impl<T: ObjectPatternList> PatternListMetadata for PatternListProbe<T> {
    fn pattern_list_id(&self) -> Option<TypeId> {
        T::ALL_EXACT.then(TypeId::of::<T>)
    }

    fn fill_exact_type_indices(&self, out: &mut [i32]) {
        T::fill_exact_type_indices(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Array, Module, Tensor};

    #[test]
    fn dense_plan_uses_base_and_preserves_source_order() {
        let pattern_list_id = TypeId::of::<(i32, i64, f32)>();
        let plan = ExactDispatchPlan::build(pattern_list_id, &[73, 73, 75]);

        assert_eq!(plan.lookup(pattern_list_id, 73), Ok(Some(0)));
        assert_eq!(plan.lookup(pattern_list_id, 74), Ok(None));
        assert_eq!(plan.lookup(pattern_list_id, 75), Ok(Some(2)));
        assert_eq!(plan.lookup(pattern_list_id, 72), Ok(None));
    }

    #[test]
    fn sparse_or_different_pattern_lists_use_ordered_dispatch() {
        let pattern_list_id = TypeId::of::<(i32, i64)>();
        let sparse = ExactDispatchPlan::build(pattern_list_id, &[73, 75]);
        let dense = ExactDispatchPlan::build(pattern_list_id, &[73, 74]);

        assert_eq!(sparse.lookup(pattern_list_id, 73), Err(()));
        assert_eq!(dense.lookup(TypeId::of::<(u8, u16)>(), 73), Err(()));
    }

    #[test]
    fn metadata_only_accepts_exact_object_patterns() {
        type Exact = (Module, ());
        let exact = PatternListProbe::<Exact>::new();
        let mut type_indices = [0; 1];
        assert!((&exact).pattern_list_id().is_some());
        (&exact).fill_exact_type_indices(&mut type_indices);
        assert!(type_indices[0] >= TypeIndex::kTVMFFIStaticObjectBegin as i32);

        type Parameterized = (Array<i64>, ());
        let parameterized = PatternListProbe::<Parameterized>::new();
        assert!((&parameterized).pattern_list_id().is_none());

        type NonFinal = (Tensor, ());
        let non_final = PatternListProbe::<NonFinal>::new();
        assert!((&non_final).pattern_list_id().is_none());
    }
}
