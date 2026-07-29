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

#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using tvm::ffi::make_object;
using tvm::ffi::Object;
using tvm::ffi::ObjectRef;

#define TVM_FFI_DEFINE_BENCH_LEAF(ObjectName, RefName, TypeKey)                    \
  class ObjectName : public Object {                                               \
   public:                                                                         \
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL(TypeKey, ObjectName, Object);                \
  };                                                                               \
                                                                                   \
  class RefName : public ObjectRef {                                               \
   public:                                                                         \
    RefName() { data_ = make_object<ObjectName>(); }                               \
    TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(RefName, ObjectRef, ObjectName); \
  }

TVM_FFI_DEFINE_BENCH_LEAF(Leaf0Obj, Leaf0, "benchmark.Leaf0");
TVM_FFI_DEFINE_BENCH_LEAF(Leaf1Obj, Leaf1, "benchmark.Leaf1");
TVM_FFI_DEFINE_BENCH_LEAF(MissObj, Miss, "benchmark.Miss");

constexpr std::size_t kMiss = std::numeric_limits<std::size_t>::max();
constexpr int kSampleCount = 31;
constexpr auto kTargetSampleTime = std::chrono::milliseconds(20);

using Runner = std::size_t (*)(const ObjectRef&, std::uint64_t);

template <typename T>
inline void DoNotOptimize(const T& value) {
  asm volatile("" : : "g"(&value) : "memory");
}

TVM_FFI_NO_INLINE std::size_t Noop(const ObjectRef& input, std::uint64_t iterations) {
  std::size_t checksum = 0;
  for (std::uint64_t i = 0; i < iterations; ++i) {
    DoNotOptimize(input);
    const std::size_t selected = kMiss;
    DoNotOptimize(selected);
    checksum += selected;
  }
  DoNotOptimize(checksum);
  return checksum;
}

TVM_FFI_NO_INLINE std::size_t SingleAs(const ObjectRef& input, std::uint64_t iterations) {
  std::size_t checksum = 0;
  for (std::uint64_t i = 0; i < iterations; ++i) {
    DoNotOptimize(input);
    auto matched = input.as<Leaf0>();
    DoNotOptimize(matched);
    checksum += matched.has_value() ? 0 : kMiss;
  }
  DoNotOptimize(checksum);
  return checksum;
}

TVM_FFI_NO_INLINE std::size_t TwoItemAsChain(const ObjectRef& input, std::uint64_t iterations) {
  std::size_t checksum = 0;
  for (std::uint64_t i = 0; i < iterations; ++i) {
    DoNotOptimize(input);
    std::size_t selected;
    if (auto matched = input.as<Leaf0>()) {
      DoNotOptimize(matched);
      selected = 0;
    } else if (auto matched = input.as<Leaf1>()) {
      DoNotOptimize(matched);
      selected = 1;
    } else {
      selected = kMiss;
    }
    DoNotOptimize(selected);
    checksum += selected;
  }
  DoNotOptimize(checksum);
  return checksum;
}

struct Measurement {
  std::string name;
  double median_ns;
  double mad_ns;
};

std::chrono::nanoseconds TimeOnce(Runner run, const ObjectRef& input, std::uint64_t iterations) {
  const auto start = std::chrono::steady_clock::now();
  const std::size_t checksum = run(input, iterations);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  DoNotOptimize(checksum);
  return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
}

std::uint64_t Calibrate(Runner run, const ObjectRef& input) {
  std::uint64_t iterations = 1024;
  while (true) {
    const auto elapsed = TimeOnce(run, input, iterations);
    if (elapsed >= std::chrono::milliseconds(3)) {
      const auto elapsed_ns = std::max<std::int64_t>(elapsed.count(), 1);
      const auto target_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(kTargetSampleTime).count();
      const auto scaled =
          static_cast<std::uint64_t>(static_cast<long double>(iterations) * target_ns / elapsed_ns);
      return std::max(iterations, scaled);
    }
    iterations *= 4;
  }
}

std::pair<double, double> MedianAndMad(std::vector<double>* samples) {
  std::sort(samples->begin(), samples->end());
  const double median = (*samples)[samples->size() / 2];
  std::vector<double> deviations;
  deviations.reserve(samples->size());
  for (double sample : *samples) {
    deviations.push_back(std::abs(sample - median));
  }
  std::sort(deviations.begin(), deviations.end());
  return {median, deviations[deviations.size() / 2]};
}

std::vector<Measurement> MeasureGroup(const std::vector<std::pair<std::string, Runner>>& runners,
                                      const ObjectRef& input) {
  for (const auto& [_, run] : runners) {
    DoNotOptimize(run(input, 1));
  }

  std::vector<std::uint64_t> iterations;
  std::vector<std::vector<double>> samples(runners.size());
  iterations.reserve(runners.size());
  for (const auto& [_, run] : runners) {
    iterations.push_back(Calibrate(run, input));
  }
  for (auto& strategy_samples : samples) {
    strategy_samples.reserve(kSampleCount);
  }

  for (int sample = 0; sample < kSampleCount; ++sample) {
    for (std::size_t offset = 0; offset < runners.size(); ++offset) {
      const std::size_t index = (static_cast<std::size_t>(sample) + offset) % runners.size();
      const auto elapsed = TimeOnce(runners[index].second, input, iterations[index]);
      samples[index].push_back(static_cast<double>(elapsed.count()) /
                               static_cast<double>(iterations[index]));
    }
  }

  std::vector<Measurement> result;
  result.reserve(runners.size());
  for (std::size_t i = 0; i < runners.size(); ++i) {
    const auto [median, mad] = MedianAndMad(&samples[i]);
    result.push_back({runners[i].first, median, mad});
  }
  return result;
}

void PrintCase(const std::string& group, const std::string& name, const ObjectRef& input,
               const std::vector<std::pair<std::string, Runner>>& runners) {
  for (const Measurement& result : MeasureGroup(runners, input)) {
    std::cout << group << '\t' << name << '\t' << result.name << '\t' << result.median_ns << '\t'
              << result.mad_ns << '\n';
  }
}

}  // namespace

int main() {
  // Object construction and runtime type registration happen before timing.
  ObjectRef first = Leaf0();
  ObjectRef second = Leaf1();
  ObjectRef miss = Miss();

  const std::vector<std::pair<std::string, Runner>> single = {
      {"noop", Noop},
      {"as<T>()", SingleAs},
  };
  const std::vector<std::pair<std::string, Runner>> chain = {
      {"noop", Noop},
      {"two-item as chain", TwoItemAsChain},
  };

  // Warm every runtime type and conversion path before any samples.
  DoNotOptimize(SingleAs(first, 1));
  DoNotOptimize(SingleAs(second, 1));
  DoNotOptimize(TwoItemAsChain(first, 1));
  DoNotOptimize(TwoItemAsChain(second, 1));
  DoNotOptimize(TwoItemAsChain(miss, 1));

  std::cout << "C++ ObjectRef::as<ObjectRefType> optimized hot-loop benchmark\n";
  std::cout << "Objects and runtime type indices are initialized before timing.\n";
  std::cout << "Each arm only returns an integer; results are median and MAD over " << kSampleCount
            << " samples.\n\n";
  std::cout << "group\tcase\tstrategy\tns/op\tMAD\n";
  PrintCase("single", "hit", first, single);
  PrintCase("single", "miss", second, single);
  PrintCase("two-arm", "first", first, chain);
  PrintCase("two-arm", "second", second, chain);
  PrintCase("two-arm", "miss", miss, chain);
  return 0;
}
