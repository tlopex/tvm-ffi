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

use tvm_ffi_sys::TVMFFIAny;

use crate::any::{Any, AnyView, TryFromTemp};
use crate::error::Result;
use crate::object::{get_object_field, ObjectCore, ObjectRefCore};
use crate::type_traits::AnyCompatible;

/// An object reference whose reflected `ty` field has type `Expected`.
///
/// `Base` preserves the expression's runtime object type. `Expected` is a
/// refinement checked through reflection whenever an unrefined value enters
/// this wrapper; it adds no Rust-side object layout.
#[repr(transparent)]
pub struct TypedExpr<Base, Expected> {
    base: Base,
    _expected: PhantomData<fn() -> Expected>,
}

impl<Base: Clone, Expected> Clone for TypedExpr<Base, Expected> {
    fn clone(&self) -> Self {
        Self::from_validated_base(self.base.clone())
    }
}

impl<Base, Expected> TypedExpr<Base, Expected> {
    #[inline]
    fn from_validated_base(base: Base) -> Self {
        Self {
            base,
            _expected: PhantomData,
        }
    }

    /// Borrow the underlying expression handle.
    #[inline]
    pub fn as_base(&self) -> &Base {
        &self.base
    }

    /// Remove the checked refinement without changing the object reference.
    #[inline]
    pub fn into_base(self) -> Base {
        self.base
    }
}

impl<Base, Expected> TypedExpr<Base, Expected>
where
    Base: ObjectRefCore,
    Expected: AnyCompatible,
{
    fn reflected_ty(base: &Base) -> Result<Expected> {
        get_object_field::<Expected, _>(
            base,
            <Base::ContainerType as ObjectCore>::type_index(),
            "ty",
        )
    }

    /// Return the expression's result type with the refinement preserved.
    pub fn ty(&self) -> Result<Expected> {
        Self::reflected_ty(&self.base)
    }

    /// Check `base`'s reflected result type and add the refinement.
    pub fn try_from_base(base: Base) -> Result<Self> {
        Self::reflected_ty(&base)?;
        Ok(Self::from_validated_base(base))
    }
}

impl<Base, Expected> Deref for TypedExpr<Base, Expected> {
    type Target = Base;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

unsafe impl<Base, Expected> ObjectRefCore for TypedExpr<Base, Expected>
where
    Base: ObjectRefCore,
    Expected: AnyCompatible,
{
    type ContainerType = Base::ContainerType;

    fn data(this: &Self) -> &crate::ObjectArc<Self::ContainerType> {
        Base::data(&this.base)
    }

    fn into_data(this: Self) -> crate::ObjectArc<Self::ContainerType> {
        Base::into_data(this.base)
    }

    fn from_data(data: crate::ObjectArc<Self::ContainerType>) -> Self {
        Self::try_from_base(Base::from_data(data))
            .unwrap_or_else(|error| panic!("invalid TypedExpr object reference: {error}"))
    }
}

unsafe impl<Base, Expected> AnyCompatible for TypedExpr<Base, Expected>
where
    Base: ObjectRefCore + AnyCompatible,
    Expected: AnyCompatible,
{
    const MATCH_ANY_EXACT: bool = false;

    unsafe fn copy_to_any_view(src: &Self, data: &mut TVMFFIAny) {
        Base::copy_to_any_view(&src.base, data);
    }

    unsafe fn move_to_any(src: Self, data: &mut TVMFFIAny) {
        Base::move_to_any(src.base, data);
    }

    unsafe fn check_any_strict(data: &TVMFFIAny) -> bool {
        if !Base::check_any_strict(data) {
            return false;
        }
        let base = Base::copy_from_any_view_after_check(data);
        Self::reflected_ty(&base).is_ok()
    }

    unsafe fn copy_from_any_view_after_check(data: &TVMFFIAny) -> Self {
        Self::from_validated_base(Base::copy_from_any_view_after_check(data))
    }

    unsafe fn move_from_any_after_check(data: &mut TVMFFIAny) -> Self {
        Self::from_validated_base(Base::move_from_any_after_check(data))
    }

    unsafe fn try_cast_from_any_view(data: &TVMFFIAny) -> std::result::Result<Self, ()> {
        let base = Base::try_cast_from_any_view(data)?;
        Self::try_from_base(base).map_err(|_| ())
    }

    fn get_mismatch_type_info(data: &TVMFFIAny) -> String {
        format!(
            "{} with incompatible `ty`",
            Base::get_mismatch_type_info(data)
        )
    }

    fn type_str() -> String {
        format!("TypedExpr<{}, {}>", Base::type_str(), Expected::type_str())
    }
}

impl<Base, Expected> TryFrom<Any> for TypedExpr<Base, Expected>
where
    Base: ObjectRefCore + AnyCompatible,
    Expected: AnyCompatible,
{
    type Error = crate::Error;

    fn try_from(value: Any) -> Result<Self> {
        TryFromTemp::<Self>::try_from(value).map(TryFromTemp::into_value)
    }
}

impl<'a, Base, Expected> TryFrom<AnyView<'a>> for TypedExpr<Base, Expected>
where
    Base: ObjectRefCore + AnyCompatible,
    Expected: AnyCompatible,
{
    type Error = crate::Error;

    fn try_from(value: AnyView<'a>) -> Result<Self> {
        TryFromTemp::<Self>::try_from(value).map(TryFromTemp::into_value)
    }
}
