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
use std::hint::black_box;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use tvm_ffi::any::TryFromTemp;
use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::match_any_internal::{ArmId, LeafLookupTable};
use tvm_ffi::object::{Object, ObjectArc, ObjectCore, ObjectRef, ObjectRefCore};
use tvm_ffi::{match_any, AnyView, ObjectRefCast, Shape};

const MISS: usize = usize::MAX;
const SAMPLE_COUNT: usize = 31;
const TARGET_SAMPLE_TIME: Duration = Duration::from_millis(20);

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

type Runner = fn(&[ObjectRef], u64) -> usize;

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
        $ordered:ident,
        $lookup:ident,
        $($matcher:ident => $arm_id:expr),+ $(,)?
    ) => {
        #[inline(never)]
        fn $ordered(inputs: &[ObjectRef], iterations: u64) -> usize {
            hot_loop!(inputs, iterations, value => {
                // Guards intentionally keep this call site on the ordered path.
                match_any! {
                    *value {
                        $($matcher(_) if true => $arm_id,)+
                        _ => MISS,
                    }
                }
            })
        }

        #[inline(never)]
        fn $lookup(inputs: &[ObjectRef], iterations: u64) -> usize {
            hot_loop!(inputs, iterations, value => {
                // Mirror the warmed leaf-table selection independently of the
                // production threshold so every arity remains measurable.
                let view = AnyView::from(value);
                let pattern_id = TypeId::of::<($($matcher,)+)>();
                static TABLE: OnceLock<LeafLookupTable> = OnceLock::new();
                let table = TABLE.get_or_init(|| {
                    LeafLookupTable::build(
                        pattern_id,
                        &[
                            $(
                                (
                                    <<$matcher as ObjectRefCore>::ContainerType as ObjectCore>
                                        ::type_index(),
                                    $arm_id as ArmId,
                                ),
                            )+
                        ],
                    )
                });
                match table.lookup(pattern_id, view.type_index()).unwrap() {
                    $(
                        Some(selected) if selected == $arm_id as ArmId => {
                            match TryInto::<$matcher>::try_into(view) {
                                Ok(_) => $arm_id,
                                Err(()) => unreachable!(
                                    "leaf table selected an incompatible arm"
                                ),
                            }
                        }
                    )+
                    Some(_) => unreachable!("leaf table returned an unknown arm"),
                    None => MISS,
                    }
            })
        }
    };
}

define_match_any_pair!(
    ordered_2,
    lookup_2,
    Leaf0 => 0,
    Leaf1 => 1,
);
define_match_any_pair!(
    ordered_3,
    lookup_3,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
);
define_match_any_pair!(
    ordered_4,
    lookup_4,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
);
define_match_any_pair!(
    ordered_5,
    lookup_5,
    Leaf0 => 0,
    Leaf1 => 1,
    Leaf2 => 2,
    Leaf3 => 3,
    Leaf4 => 4,
);
define_match_any_pair!(
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

fn benchmark_arity(
    arm_count: usize,
    ordered: Runner,
    lookup: Runner,
    values: &[ObjectRef],
    miss: &ObjectRef,
) {
    let runners: &[(&str, Runner)] = &[("ordered", ordered), ("lookup", lookup)];
    for (expected, value) in values[..arm_count].iter().enumerate() {
        let input = [value.clone()];
        assert_eq!(ordered(&input, 1), expected);
        assert_eq!(lookup(&input, 1), expected);
    }
    let miss_input = [miss.clone()];
    assert_eq!(ordered(&miss_input, 1), MISS);
    assert_eq!(lookup(&miss_input, 1), MISS);

    let first = [values[0].clone()];
    print_case(&format!("arity-{arm_count}"), "first", 0, runners, &first);
    let last = [values[arm_count - 1].clone()];
    print_case(
        &format!("arity-{arm_count}"),
        "last",
        arm_count - 1,
        runners,
        &last,
    );
    print_case(
        &format!("arity-{arm_count}"),
        "miss",
        MISS,
        runners,
        &miss_input,
    );

    let inputs = balanced_inputs(values, miss, arm_count);
    let mixed_runners: &[(&str, Runner)] =
        &[("noop", noop), ("ordered", ordered), ("lookup", lookup)];
    print_mixed_case(
        &format!("arity-{arm_count}"),
        "balanced",
        mixed_runners,
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
    if cfg!(debug_assertions) {
        eprintln!("match_any_threshold requires `cargo bench` (release opt-level=3)");
        return;
    }
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);

    // Object construction and runtime type registration happen before timing.
    let values: [ObjectRef; 12] = [
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
    ];
    let miss: ObjectRef = Shape::from([1_i64, 2, 3]).try_cast().unwrap();
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
    println!("The lookup OnceLock is warmed before timing; arm bodies only return an integer.");
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
}
