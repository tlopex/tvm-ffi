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

use crate::{AnyCompatible, AnyView};

/// Conversion adapter used by `match_any!` typed arms.
#[doc(hidden)]
pub struct PatternConversionProbe<T>(PhantomData<fn() -> T>);

impl<T> PatternConversionProbe<T> {
    #[doc(hidden)]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Prefer the lightweight `AnyCompatible` conversion while retaining a
/// `TryInto` fallback for custom matcher types.
#[doc(hidden)]
pub trait PatternConversion<'a, T> {
    #[doc(hidden)]
    fn try_convert(&self, view: AnyView<'a>) -> Result<T, ()>;
}

impl<'a, T: AnyCompatible> PatternConversion<'a, T> for PatternConversionProbe<T> {
    #[inline(always)]
    fn try_convert(&self, view: AnyView<'a>) -> Result<T, ()> {
        if T::MATCH_ANY_EXACT {
            if view.type_index() == T::match_any_exact_type_index() {
                Ok(unsafe { crate::any::copy_from_any_view_after_check::<T>(&view) })
            } else {
                Err(())
            }
        } else {
            crate::any::try_cast_from_any_view::<T>(&view)
        }
    }
}

impl<'a, T> PatternConversion<'a, T> for &PatternConversionProbe<T>
where
    AnyView<'a>: TryInto<T>,
{
    #[inline(always)]
    fn try_convert(&self, view: AnyView<'a>) -> Result<T, ()> {
        view.try_into().map_err(|_| ())
    }
}
