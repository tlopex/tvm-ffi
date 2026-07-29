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

use std::env;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::object::{Object, ObjectArc, ObjectCore, ObjectRef};
use tvm_ffi::{match_any, AnyView, ObjectRefCast, Shape};

const MISS: usize = usize::MAX;
const SAMPLE_COUNT: usize = 31;
const COLD_SAMPLE_COUNT: usize = 31;
const TARGET_SAMPLE_TIME: Duration = Duration::from_millis(20);
// Keep this boundary set in sync with the proc-macro heuristic.
const LEAF_LOOKUP_THRESHOLD_ARMS: usize = 20;

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

define_leaf!(Leaf0Obj, Leaf0, "benchmark.match_any.Leaf0");
define_leaf!(Leaf1Obj, Leaf1, "benchmark.match_any.Leaf1");
define_leaf!(Leaf2Obj, Leaf2, "benchmark.match_any.Leaf2");
define_leaf!(Leaf3Obj, Leaf3, "benchmark.match_any.Leaf3");
define_leaf!(Leaf4Obj, Leaf4, "benchmark.match_any.Leaf4");
define_leaf!(Leaf5Obj, Leaf5, "benchmark.match_any.Leaf5");
define_leaf!(Leaf6Obj, Leaf6, "benchmark.match_any.Leaf6");
define_leaf!(Leaf7Obj, Leaf7, "benchmark.match_any.Leaf7");
define_leaf!(Leaf8Obj, Leaf8, "benchmark.match_any.Leaf8");
define_leaf!(Leaf9Obj, Leaf9, "benchmark.match_any.Leaf9");
define_leaf!(Leaf10Obj, Leaf10, "benchmark.match_any.Leaf10");
define_leaf!(Leaf11Obj, Leaf11, "benchmark.match_any.Leaf11");
define_leaf!(Leaf12Obj, Leaf12, "benchmark.match_any.Leaf12");
define_leaf!(Leaf13Obj, Leaf13, "benchmark.match_any.Leaf13");
define_leaf!(Leaf14Obj, Leaf14, "benchmark.match_any.Leaf14");
define_leaf!(Leaf15Obj, Leaf15, "benchmark.match_any.Leaf15");
define_leaf!(Leaf16Obj, Leaf16, "benchmark.match_any.Leaf16");
define_leaf!(Leaf17Obj, Leaf17, "benchmark.match_any.Leaf17");
define_leaf!(Leaf18Obj, Leaf18, "benchmark.match_any.Leaf18");
define_leaf!(Leaf19Obj, Leaf19, "benchmark.match_any.Leaf19");
define_leaf!(Leaf20Obj, Leaf20, "benchmark.match_any.Leaf20");

unsafe extern "C" {
    fn TVMFFITypeGetOrAllocIndex(
        type_key: *const tvm_ffi_sys::TVMFFIByteArray,
        static_type_index: i32,
        type_depth: i32,
        num_child_slots: i32,
        child_slots_can_overflow: i32,
        parent_type_index: i32,
    ) -> i32;
}

fn register_type(type_key: &str) -> i32 {
    let type_key = unsafe { tvm_ffi_sys::TVMFFIByteArray::from_str(type_key) };
    unsafe { TVMFFITypeGetOrAllocIndex(&type_key, -1, 1, 0, 1, Object::type_index()) }
}

fn register_leaf_types() -> [i32; 21] {
    let type_keys = [
        Leaf0Obj::TYPE_KEY,
        Leaf1Obj::TYPE_KEY,
        Leaf2Obj::TYPE_KEY,
        Leaf3Obj::TYPE_KEY,
        Leaf4Obj::TYPE_KEY,
        Leaf5Obj::TYPE_KEY,
        Leaf6Obj::TYPE_KEY,
        Leaf7Obj::TYPE_KEY,
        Leaf8Obj::TYPE_KEY,
        Leaf9Obj::TYPE_KEY,
        Leaf10Obj::TYPE_KEY,
        Leaf11Obj::TYPE_KEY,
        Leaf12Obj::TYPE_KEY,
        Leaf13Obj::TYPE_KEY,
        Leaf14Obj::TYPE_KEY,
        Leaf15Obj::TYPE_KEY,
        Leaf16Obj::TYPE_KEY,
        Leaf17Obj::TYPE_KEY,
        Leaf18Obj::TYPE_KEY,
        Leaf19Obj::TYPE_KEY,
        Leaf20Obj::TYPE_KEY,
    ];
    let mut type_indices = [0_i32; 21];
    for (leaf, type_key) in type_keys.into_iter().enumerate() {
        type_indices[leaf] = register_type(type_key);
        assert!(type_indices[leaf] > 0);
    }
    type_indices
}

fn print_type_index_layout(type_indices: &[i32; 21]) {
    println!("leaf TypeIndex values: {type_indices:?}");
}

type Runner = fn(&[ObjectRef], u64) -> usize;
type ColdRunner = for<'a> fn(AnyView<'a>) -> usize;

macro_rules! run_hot_loop {
    ($inputs:ident, $rounds:ident, $value:ident => $body:expr) => {{
        let mut checksum = 0_usize;
        for _ in 0..$rounds {
            for $value in $inputs {
                let $value = black_box($value);
                checksum = checksum.wrapping_add($body);
            }
        }
        black_box(checksum)
    }};
}

#[inline(never)]
fn noop(inputs: &[ObjectRef], rounds: u64) -> usize {
    run_hot_loop!(inputs, rounds, value => {
        let _ = value;
        0
    })
}

#[inline(never)]
fn single_any_view_try_into(inputs: &[ObjectRef], rounds: u64) -> usize {
    run_hot_loop!(inputs, rounds, value => {
        match TryInto::<Leaf0>::try_into(AnyView::from(value)) {
            Ok(_matched) => 0,
            Err(()) => MISS,
        }
    })
}

#[inline(never)]
fn two_item_any_view_try_into_chain(inputs: &[ObjectRef], rounds: u64) -> usize {
    run_hot_loop!(inputs, rounds, value => {
        match TryInto::<Leaf0>::try_into(AnyView::from(value)) {
            Ok(_matched) => 0,
            Err(()) => match TryInto::<Leaf1>::try_into(AnyView::from(value)) {
                Ok(_matched) => 1,
                Err(()) => MISS,
            },
        }
    })
}

macro_rules! define_match_pair {
    (
        $ordered:ident,
        $automatic:ident,
        $($matcher:ident => $arm_id:expr),+ $(,)?
    ) => {
        #[inline(never)]
        fn $ordered(inputs: &[ObjectRef], rounds: u64) -> usize {
            run_hot_loop!(inputs, rounds, value => {
                match_any! {
                    *value {
                        $($matcher(_matched) if true => $arm_id,)+
                        _ => MISS,
                    }
                }
            })
        }

        #[inline(never)]
        fn $automatic(inputs: &[ObjectRef], rounds: u64) -> usize {
            run_hot_loop!(inputs, rounds, value => {
                match_any! {
                    *value {
                        $($matcher(_matched) => $arm_id,)+
                        _ => MISS,
                    }
                }
            })
        }
    };
}

define_match_pair!(
    ordered_19,
    automatic_19,
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
    Leaf17 => 17,
    Leaf18 => 18,
);
define_match_pair!(
    ordered_20,
    automatic_20,
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
    Leaf17 => 17,
    Leaf18 => 18,
    Leaf19 => 19,
);
define_match_pair!(
    ordered_21,
    automatic_21,
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
    Leaf17 => 17,
    Leaf18 => 18,
    Leaf19 => 19,
    Leaf20 => 20,
);

struct Arity {
    count: usize,
    ordered: Runner,
    automatic: Runner,
}

const ARITIES: &[Arity] = &[
    Arity {
        count: 19,
        ordered: ordered_19,
        automatic: automatic_19,
    },
    Arity {
        count: 20,
        ordered: ordered_20,
        automatic: automatic_20,
    },
    Arity {
        count: 21,
        ordered: ordered_21,
        automatic: automatic_21,
    },
];

fn make_objects() -> ([ObjectRef; 21], ObjectRef) {
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
        Leaf17::new().try_cast().unwrap(),
        Leaf18::new().try_cast().unwrap(),
        Leaf19::new().try_cast().unwrap(),
        Leaf20::new().try_cast().unwrap(),
    ];
    let miss = Shape::from([1_i64, 2, 3]).try_cast().unwrap();
    (values, miss)
}

fn shuffled_inputs(values: &[ObjectRef], miss: Option<&ObjectRef>, seed: u64) -> Vec<ObjectRef> {
    let occurrences_per_arm = 8_192_usize.div_ceil(values.len());

    let mut inputs = Vec::with_capacity(values.len() * occurrences_per_arm);
    for _ in 0..occurrences_per_arm {
        inputs.extend(values.iter().cloned());
    }
    if let Some(miss) = miss {
        inputs.extend(std::iter::repeat_n(miss.clone(), occurrences_per_arm));
    }

    let mut state = seed ^ values.len() as u64;
    for index in (1..inputs.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        inputs.swap(index, state as usize % (index + 1));
    }
    inputs
}

fn time_once(run: Runner, inputs: &[ObjectRef], rounds: u64) -> Duration {
    let start = Instant::now();
    let checksum = run(inputs, rounds);
    let elapsed = start.elapsed();
    black_box(checksum);
    elapsed
}

fn calibrate(run: Runner, inputs: &[ObjectRef]) -> u64 {
    let mut rounds = 1_u64;
    loop {
        let elapsed = time_once(run, inputs, rounds);
        if elapsed >= Duration::from_millis(3) {
            let scaled = rounds as u128 * TARGET_SAMPLE_TIME.as_nanos() / elapsed.as_nanos().max(1);
            return scaled.max(rounds as u128) as u64;
        }
        rounds = rounds.saturating_mul(4);
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

fn measure_group<'a>(
    runners: &'a [(&'a str, Runner)],
    inputs: &[ObjectRef],
) -> Vec<(&'a str, f64, f64)> {
    let calibrated_rounds = runners
        .iter()
        .map(|(_, run)| {
            black_box(run(inputs, 1));
            calibrate(*run, inputs)
        })
        .collect::<Vec<_>>();
    let comparison_rounds = calibrated_rounds[1..]
        .iter()
        .copied()
        .max()
        .expect("ordered and automatic runners");
    let rounds = calibrated_rounds
        .into_iter()
        .enumerate()
        .map(|(runner, rounds)| {
            if runner == 0 {
                rounds
            } else {
                comparison_rounds
            }
        })
        .collect::<Vec<_>>();
    let mut samples = vec![Vec::with_capacity(SAMPLE_COUNT); runners.len()];

    for sample in 0..SAMPLE_COUNT {
        for offset in 0..runners.len() {
            let runner = (sample + offset) % runners.len();
            let elapsed = time_once(runners[runner].1, inputs, rounds[runner]);
            let operations = rounds[runner] as usize * inputs.len();
            samples[runner].push(elapsed.as_nanos() as f64 / operations as f64);
        }
    }

    runners
        .iter()
        .zip(samples)
        .map(|((name, _), mut samples)| {
            let (median, mad) = median_and_mad(&mut samples);
            (*name, median, mad)
        })
        .collect()
}

fn print_case(arity: usize, case: &str, runners: &[(&str, Runner)], inputs: &[ObjectRef]) {
    for (strategy, median, mad) in measure_group(runners, inputs) {
        println!("{arity}\t{case}\t{strategy}\t{median:.3}\t{mad:.3}");
    }
}

fn benchmark_try_into(values: &[ObjectRef; 21], miss: &ObjectRef) {
    println!("ObjectRef -> AnyView -> TryInto baseline");
    println!("case\tstrategy\tns/call\tMAD(ns)");
    let single = [
        ("noop", noop as Runner),
        ("single", single_any_view_try_into as Runner),
    ];
    let chain = [
        ("noop", noop as Runner),
        ("two-item-chain", two_item_any_view_try_into_chain as Runner),
    ];
    let first = [values[0].clone()];
    let second = [values[1].clone()];
    let missed = [miss.clone()];
    for (case, inputs, runners) in [
        ("single-hit", &first[..], &single[..]),
        ("single-miss", &second[..], &single[..]),
        ("chain-first", &first[..], &chain[..]),
        ("chain-second", &second[..], &chain[..]),
        ("chain-miss", &missed[..], &chain[..]),
    ] {
        for (strategy, median, mad) in measure_group(runners, inputs) {
            println!("{case}\t{strategy}\t{median:.3}\t{mad:.3}");
        }
    }
    println!();
}

fn benchmark_hot(values: &[ObjectRef; 21], miss: &ObjectRef, seed: u64) {
    println!("arms\tworkload\tstrategy\tns/call\tMAD(ns)");
    for arity in ARITIES {
        if !selected_arity(arity.count) {
            continue;
        }
        let automatic_strategy = if arity.count < LEAF_LOOKUP_THRESHOLD_ARMS {
            "auto-ordered"
        } else {
            "leaf-lookup"
        };
        let runners = [
            ("noop", noop as Runner),
            ("ordered", arity.ordered),
            (automatic_strategy, arity.automatic),
        ];
        for (expected, value) in values[..arity.count].iter().enumerate() {
            assert_eq!((arity.ordered)(&[value.clone()], 1), expected);
            assert_eq!((arity.automatic)(&[value.clone()], 1), expected);
        }
        assert_eq!((arity.ordered)(&[miss.clone()], 1), MISS);
        assert_eq!((arity.automatic)(&[miss.clone()], 1), MISS);

        if arity.count == LEAF_LOOKUP_THRESHOLD_ARMS {
            let first = [values[0].clone()];
            let middle = [values[arity.count / 2].clone()];
            let last = [values[arity.count - 1].clone()];
            let missed = [miss.clone()];
            print_case(arity.count, "T0-first", &runners, &first);
            print_case(arity.count, "T0-middle", &runners, &middle);
            print_case(arity.count, "T0-last", &runners, &last);
            print_case(arity.count, "T0-miss", &runners, &missed);
        }

        let hits = shuffled_inputs(&values[..arity.count], None, seed);
        print_case(arity.count, "T1-uniform-hits", &runners, &hits);
        let hits_and_misses = shuffled_inputs(
            &values[..arity.count],
            Some(miss),
            seed ^ 0xa5a5_a5a5_a5a5_a5a5,
        );
        print_case(
            arity.count,
            "T1-uniform-all-outcomes",
            &runners,
            &hits_and_misses,
        );
    }
}

fn selected_arity(arity: usize) -> bool {
    let Ok(filter) = env::var("TVM_FFI_MATCH_ANY_ARITIES") else {
        return true;
    };
    filter
        .split(',')
        .any(|candidate| candidate.parse::<usize>() == Ok(arity))
}

#[inline(never)]
fn cold_floor(view: AnyView<'_>) -> usize {
    black_box(view);
    MISS
}

#[inline(never)]
fn cold_ordered(view: AnyView<'_>) -> usize {
    match_any! {
        view {
            Leaf0(_matched) if true => 0,
            Leaf1(_matched) if true => 1,
            Leaf2(_matched) if true => 2,
            Leaf3(_matched) if true => 3,
            Leaf4(_matched) if true => 4,
            Leaf5(_matched) if true => 5,
            Leaf6(_matched) if true => 6,
            Leaf7(_matched) if true => 7,
            Leaf8(_matched) if true => 8,
            Leaf9(_matched) if true => 9,
            Leaf10(_matched) if true => 10,
            Leaf11(_matched) if true => 11,
            Leaf12(_matched) if true => 12,
            Leaf13(_matched) if true => 13,
            Leaf14(_matched) if true => 14,
            Leaf15(_matched) if true => 15,
            Leaf16(_matched) if true => 16,
            Leaf17(_matched) if true => 17,
            Leaf18(_matched) if true => 18,
            Leaf19(_matched) if true => 19,
            _ => MISS,
        }
    }
}

#[inline(never)]
fn cold_automatic(view: AnyView<'_>) -> usize {
    match_any! {
        view {
            Leaf0(_matched) => 0,
            Leaf1(_matched) => 1,
            Leaf2(_matched) => 2,
            Leaf3(_matched) => 3,
            Leaf4(_matched) => 4,
            Leaf5(_matched) => 5,
            Leaf6(_matched) => 6,
            Leaf7(_matched) => 7,
            Leaf8(_matched) => 8,
            Leaf9(_matched) => 9,
            Leaf10(_matched) => 10,
            Leaf11(_matched) => 11,
            Leaf12(_matched) => 12,
            Leaf13(_matched) => 13,
            Leaf14(_matched) => 14,
            Leaf15(_matched) => 15,
            Leaf16(_matched) => 16,
            Leaf17(_matched) => 17,
            Leaf18(_matched) => 18,
            Leaf19(_matched) => 19,
            _ => MISS,
        }
    }
}

fn warm_cold_dependencies(values: &[ObjectRef; 21]) {
    black_box([
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
        Leaf17Obj::type_index(),
        Leaf18Obj::type_index(),
        Leaf19Obj::type_index(),
        Leaf20Obj::type_index(),
    ]);

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
    warm_cast!(Leaf17, 17);
    warm_cast!(Leaf18, 18);
    warm_cast!(Leaf19, 19);

    let scalar = 1_i64;
    assert_eq!(black_box(cold_floor(AnyView::from(&scalar))), MISS);
    assert_eq!(black_box(cold_ordered(AnyView::from(&scalar))), MISS);
    assert_eq!(black_box(cold_automatic(AnyView::from(&scalar))), MISS);
    black_box(Instant::now().elapsed());
}

fn run_cold_child(strategy: &str, case: &str) {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let _type_indices = register_leaf_types();
    let (values, miss) = make_objects();
    warm_cold_dependencies(&values);
    let view = match case {
        "hit" => AnyView::from(&values[19]),
        "miss" => AnyView::from(&miss),
        _ => panic!("unknown cold case: {case}"),
    };
    let run: ColdRunner = match strategy {
        "floor" => cold_floor,
        "ordered" => cold_ordered,
        "leaf-lookup" => cold_automatic,
        _ => panic!("unknown cold strategy: {strategy}"),
    };

    let start = Instant::now();
    let selected = run(view);
    let elapsed = start.elapsed();
    let expected = if strategy == "floor" || case == "miss" {
        MISS
    } else {
        19
    };
    assert_eq!(black_box(selected), expected);
    println!("{}", elapsed.as_nanos());
}

fn measure_cold_once(strategy: &str, case: &str) -> f64 {
    let executable = env::current_exe().expect("current benchmark executable");
    let mut command = Command::new(&executable);
    command.args(["--cold-child", strategy, case]);
    let output = command.output().expect("run cold benchmark child");
    assert!(
        output.status.success(),
        "cold child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("cold child output is UTF-8")
        .trim()
        .parse::<f64>()
        .expect("cold child output is nanoseconds")
}

fn benchmark_cold() {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let type_indices = register_leaf_types();

    println!("Rust match_any! O3 cold benchmark");
    print_type_index_layout(&type_indices);
    println!(
        "first object-eligible call (20 arms; objects constructed before timing, \
         type metadata/code warm, leaf-lookup OnceLock cold)"
    );
    println!("results are median and MAD over {COLD_SAMPLE_COUNT} rotated child-process samples");
    println!("case\tstrategy\tns/call\tMAD(ns)");

    let strategies = ["floor", "ordered", "leaf-lookup"];
    for (case_index, (case_label, child_case)) in [("last-arm-hit", "hit"), ("miss", "miss")]
        .into_iter()
        .enumerate()
    {
        let mut samples = vec![Vec::with_capacity(COLD_SAMPLE_COUNT); strategies.len()];
        for sample in 0..COLD_SAMPLE_COUNT {
            for offset in 0..strategies.len() {
                let strategy = (case_index + sample + offset) % strategies.len();
                samples[strategy].push(measure_cold_once(strategies[strategy], child_case));
            }
        }
        for (strategy, mut samples) in strategies.into_iter().zip(samples) {
            let (median, mad) = median_and_mad(&mut samples);
            println!("{case_label}\t{strategy}\t{median:.1}\t{mad:.1}");
        }
    }
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    assert!(
        !cfg!(debug_assertions),
        "this microbenchmark must be built with `cargo bench` (optimized benchmark profile)"
    );
    if let Some(child) = args.iter().position(|arg| arg == "--cold-child") {
        run_cold_child(&args[child + 1], &args[child + 2]);
        return;
    }
    if args.iter().any(|arg| arg == "--cold") {
        benchmark_cold();
        return;
    }

    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let type_indices = register_leaf_types();
    let (values, miss) = make_objects();
    let seed = env::var("TVM_FFI_MATCH_ANY_SEED")
        .ok()
        .and_then(|seed| seed.parse().ok())
        .unwrap_or(0x9e37_79b9_7f4a_7c15);

    println!("Rust match_any! O3 benchmark");
    print_type_index_layout(&type_indices);
    println!("seed: {seed:#x}");
    println!("objects and shuffled input sequences are constructed before timing");
    println!("results are median and MAD over {SAMPLE_COUNT} rotated samples\n");
    benchmark_try_into(&values, &miss);
    benchmark_hot(&values, &miss, seed);
}
