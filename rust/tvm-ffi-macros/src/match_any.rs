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

use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{braced, parenthesized, Expr, Pat, Path, Result, Token};

use crate::utils::get_tvm_ffi_crate;

// For two or three arms, the ordered chain is generally the smaller plan.
const MIN_DIRECT_ARMS: usize = 4;

struct MatchAnyInput {
    scrutinee: Expr,
    arms: Vec<TypedArm>,
    fallback: Expr,
}

struct TypedArm {
    matcher: Path,
    binding: Pat,
    guard: Option<Expr>,
    body: Expr,
}

impl Parse for MatchAnyInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let scrutinee = input.call(Expr::parse_without_eager_brace)?;
        let content;
        braced!(content in input);

        let mut arms = Vec::new();
        let mut fallback = None;
        while !content.is_empty() {
            if fallback.is_some() {
                return Err(content.error("the `_` fallback must be the final arm"));
            }

            if content.peek(Token![_]) {
                content.parse::<Token![_]>()?;
                if content.peek(Token![if]) {
                    return Err(content.error("the `_` fallback cannot have a guard"));
                }
                content.parse::<Token![=>]>()?;
                fallback = Some(content.parse::<Expr>()?);
            } else {
                let matcher = content.parse::<Path>()?;
                let binding_content;
                parenthesized!(binding_content in content);
                let binding = binding_content.parse::<Pat>()?;
                if !binding_content.is_empty() {
                    return Err(binding_content.error("expected one binding pattern"));
                }
                let guard = if content.peek(Token![if]) {
                    content.parse::<Token![if]>()?;
                    Some(content.parse::<Expr>()?)
                } else {
                    None
                };
                content.parse::<Token![=>]>()?;
                let body = content.parse::<Expr>()?;
                arms.push(TypedArm {
                    matcher,
                    binding,
                    guard,
                    body,
                });
            }

            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            } else if !content.is_empty() {
                return Err(content.error("expected `,` between match_any! arms"));
            }
        }

        let fallback = fallback
            .ok_or_else(|| content.error("match_any! requires a final `_` fallback arm"))?;
        Ok(Self {
            scrutinee,
            arms,
            fallback,
        })
    }
}

pub fn expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as MatchAnyInput);
    expand_match_any(input).into()
}

fn expand_match_any(input: MatchAnyInput) -> TokenStream {
    let tvm_ffi = get_tvm_ffi_crate();
    let scrutinee = input.scrutinee;
    let fallback = input.fallback;
    let arms = input.arms;
    let can_attempt_direct_dispatch = arms.len() >= MIN_DIRECT_ARMS
        && arms
            .iter()
            .all(|arm| arm.guard.is_none() && is_simple_binding(&arm.binding));

    if can_attempt_direct_dispatch {
        expand_direct_match(&tvm_ffi, &scrutinee, &arms, &fallback)
    } else {
        expand_ordered_match(&tvm_ffi, &scrutinee, &arms, &fallback)
    }
}

fn expand_ordered_match(
    tvm_ffi: &TokenStream,
    scrutinee: &Expr,
    arms: &[TypedArm],
    fallback: &Expr,
) -> TokenStream {
    let span = Span::mixed_site();
    let source = Ident::new("__tvm_ffi_match_any_source", span);
    let converted = Ident::new("__tvm_ffi_match_any_converted", span);
    let view = Ident::new("__tvm_ffi_match_any_view", span);
    let rejected = Ident::new("__tvm_ffi_match_any_rejected", span);
    let dispatch_fallback = fallback.clone();
    let dispatch = arms
        .iter()
        .rev()
        .fold(quote!({ #dispatch_fallback }), |next, arm| {
            let matcher = &arm.matcher;
            let binding = &arm.binding;
            let body = &arm.body;
            let matched = if let Some(guard) = &arm.guard {
                quote!(::core::result::Result::Ok(#binding) if #guard)
            } else {
                quote!(::core::result::Result::Ok(#binding))
            };

            quote! {
                match ::core::convert::TryInto::<#matcher>::try_into(#view) {
                    #matched => { #body },
                    #rejected => {
                        ::core::mem::drop(#rejected);
                        #next
                    }
                }
            }
        });

    quote! {
        {
            let #source = &(#scrutinee);
            let #converted: ::core::result::Result<
                #tvm_ffi::AnyView<'_>,
                ::core::convert::Infallible,
            > = ::core::convert::TryInto::<#tvm_ffi::AnyView<'_>>::try_into(#source);
            let #view = match #converted {
                ::core::result::Result::Ok(view) => view,
                ::core::result::Result::Err(error) => match error {},
            };
            if #view.type_index()
                >= #tvm_ffi::TypeIndex::kTVMFFIStaticObjectBegin as i32
            {
                #dispatch
            } else {
                #fallback
            }
        }
    }
}

fn expand_direct_match(
    tvm_ffi: &TokenStream,
    scrutinee: &Expr,
    arms: &[TypedArm],
    fallback: &Expr,
) -> TokenStream {
    let span = Span::mixed_site();
    let source = Ident::new("__tvm_ffi_match_any_source", span);
    let converted = Ident::new("__tvm_ffi_match_any_converted", span);
    let view = Ident::new("__tvm_ffi_match_any_view", span);
    let rejected = Ident::new("__tvm_ffi_match_any_rejected", span);
    let probe = Ident::new("__tvm_ffi_match_any_probe", span);
    let pattern_list_id = Ident::new("__tvm_ffi_match_any_pattern_list_id", span);
    let type_indices = Ident::new("__tvm_ffi_match_any_type_indices", span);
    let static_plan = Ident::new("__TVM_FFI_MATCH_ANY_PLAN", span);
    let plan = Ident::new("__tvm_ffi_match_any_plan", span);
    let arm_id = Ident::new("__tvm_ffi_match_any_arm_id", span);
    let matched = Ident::new("__tvm_ffi_match_any_matched", span);
    let matched_value = Ident::new("__tvm_ffi_match_any_matched_value", span);
    let matched_enum = Ident::new("__TvmFfiMatchAnyArm", span);
    let fallback_variant = Ident::new("Fallback", span);
    let arm_count = arms.len();
    let matchers = arms.iter().map(|arm| &arm.matcher).collect::<Vec<_>>();
    let arm_types = (0..arm_count)
        .map(|arm_id| Ident::new(&format!("__TvmFfiMatchAnyType{arm_id}"), span))
        .collect::<Vec<_>>();
    let arm_variants = (0..arm_count)
        .map(|arm_id| Ident::new(&format!("Arm{arm_id}"), span))
        .collect::<Vec<_>>();
    let pattern_list = matchers
        .iter()
        .rev()
        .fold(quote!(()), |tail, matcher| quote!((#matcher, #tail)));

    let select_arm = quote! {
        {
            use #tvm_ffi::match_any::PatternListMetadata as _;

            let #probe =
                #tvm_ffi::match_any::PatternListProbe::<#pattern_list>::new();
            match (&#probe).pattern_list_id() {
                ::core::option::Option::Some(#pattern_list_id) => {
                    static #static_plan: ::std::sync::OnceLock<
                        #tvm_ffi::match_any::ExactDispatchPlan,
                    > = ::std::sync::OnceLock::new();
                    let #plan = #static_plan.get_or_init(|| {
                        let mut #type_indices = [0_i32; #arm_count];
                        (&#probe).fill_exact_type_indices(&mut #type_indices);
                        #tvm_ffi::match_any::ExactDispatchPlan::build(
                            #pattern_list_id,
                            &#type_indices,
                        )
                    });
                    #plan.lookup(#pattern_list_id, #view.type_index())
                }
                ::core::option::Option::None => {
                    ::core::result::Result::Err(())
                }
            }
        }
    };

    let ordered_selection = arms.iter().enumerate().rev().fold(
        quote!(#matched_enum::#fallback_variant),
        |next, (arm_id, arm)| {
            let matcher = &arm.matcher;
            let variant = &arm_variants[arm_id];

            quote! {
                match ::core::convert::TryInto::<#matcher>::try_into(#view) {
                    ::core::result::Result::Ok(#matched_value) => {
                        #matched_enum::#variant(#matched_value)
                    }
                    #rejected => {
                        ::core::mem::drop(#rejected);
                        #next
                    }
                }
            }
        },
    );

    let direct_selection = arms.iter().enumerate().map(|(arm_id, arm)| {
        let matcher = &arm.matcher;
        let variant = &arm_variants[arm_id];

        quote! {
            #arm_id => {
                match ::core::convert::TryInto::<#matcher>::try_into(#view) {
                    ::core::result::Result::Ok(#matched_value) => {
                        #matched_enum::#variant(#matched_value)
                    }
                    #rejected => {
                        ::core::mem::drop(#rejected);
                        ::core::panic!(
                            "match_any! exact TypeIndex selected an incompatible arm"
                        )
                    }
                }
            }
        }
    });

    let body_dispatch = arms.iter().enumerate().map(|(arm_id, arm)| {
        let binding = &arm.binding;
        let body = &arm.body;
        let variant = &arm_variants[arm_id];

        quote! {
            #matched_enum::#variant(#binding) => {
                #body
            }
        }
    });

    quote! {
        {
            enum #matched_enum<#(#arm_types),*> {
                #(#arm_variants(#arm_types),)*
                #fallback_variant,
            }

            let #source = &(#scrutinee);
            let #converted: ::core::result::Result<
                #tvm_ffi::AnyView<'_>,
                ::core::convert::Infallible,
            > = ::core::convert::TryInto::<#tvm_ffi::AnyView<'_>>::try_into(#source);
            let #view = match #converted {
                ::core::result::Result::Ok(view) => view,
                ::core::result::Result::Err(error) => match error {},
            };
            let #matched =
                if #view.type_index()
                    >= #tvm_ffi::TypeIndex::kTVMFFIStaticObjectBegin as i32
                {
                    match #select_arm {
                        ::core::result::Result::Ok(
                            ::core::option::Option::Some(#arm_id),
                        ) => {
                            match #arm_id {
                                #(#direct_selection,)*
                                _ => ::core::unreachable!(),
                            }
                        }
                        ::core::result::Result::Ok(
                            ::core::option::Option::None,
                        ) => {
                            #matched_enum::#fallback_variant
                        }
                        ::core::result::Result::Err(()) => {
                            #ordered_selection
                        }
                    }
                } else {
                    #matched_enum::#fallback_variant
                };
            match #matched {
                #(#body_dispatch,)*
                #matched_enum::#fallback_variant => {
                    #fallback
                }
            }
        }
    }
}

fn is_simple_binding(binding: &Pat) -> bool {
    match binding {
        Pat::Ident(binding) => binding.subpat.is_none(),
        Pat::Wild(_) => true,
        _ => false,
    }
}
