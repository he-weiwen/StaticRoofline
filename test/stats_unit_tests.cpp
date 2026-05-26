//===- stats_unit_tests.cpp - Self-contained tests for lib/Stats -------===//
//
// Standalone test binary for the Stats query helper introduced in PR 3.
// Stats has no production consumer yet (PR 4 wires it in); these tests
// are the entire surface validating the API.
//
// Run via the `test-stats-unit` CMake target.
//
// Harness: small self-contained EXPECT macros + a registry-driven main.
// (Could be deduplicated against ptx_unit_tests.cpp's harness in a later
// cleanup PR; kept inline here for self-containment and zero churn on
// existing tests.)
//
//===---------------------------------------------------------------------===//

#include "Measurement.h"
#include "Stats.h"

#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace ptxai;

// =============================================================
// Assertion utility (mirrors ptx_unit_tests.cpp; see note above)
// =============================================================

static int g_failures = 0;
static const char *g_current_test = "<none>";

#define EXPECT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "  FAIL [%s:%d] in %s: %s\n",                 \
                         __FILE__, __LINE__, g_current_test, #cond);           \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define EXPECT_EQ(actual, expected)                                            \
    do {                                                                       \
        auto _a = (actual);                                                    \
        auto _e = (expected);                                                  \
        if (!(_a == _e)) {                                                     \
            std::fprintf(stderr, "  FAIL [%s:%d] in %s: %s != %s\n",           \
                         __FILE__, __LINE__, g_current_test, #actual, #expected); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

// =============================================================
// Small builders to keep tests readable
// =============================================================

// Address space numbers (PTX ABI; duplicated from PTX/Classifier.cpp to
// keep this test file self-contained — no dependency on internal headers).
enum : unsigned {
    AS_GENERIC = 0, AS_GLOBAL = 1, AS_SHARED = 3,
    AS_CONST   = 4, AS_LOCAL  = 5, AS_PARAM  = 101,
};

static Measurement flop(FpPrecision p, uint64_t count,
                        InvocationScope s = InvocationScope::PerThread) {
    Measurement m{Measurement::Kind::Flop};
    m.scope = s;
    m.precision = p;
    m.count = count;
    return m;
}

static Measurement memLoad(unsigned addrSpace, uint64_t count,
                            InvocationScope s = InvocationScope::PerThread) {
    Measurement m{Measurement::Kind::Memory};
    m.scope = s;
    m.addrSpace = addrSpace;
    m.isLoad = true;
    m.count = count;
    return m;
}

static Measurement memStore(unsigned addrSpace, uint64_t count,
                             InvocationScope s = InvocationScope::PerThread) {
    Measurement m{Measurement::Kind::Memory};
    m.scope = s;
    m.addrSpace = addrSpace;
    m.isStore = true;
    m.count = count;
    return m;
}

static Measurement memAtomic(unsigned addrSpace, uint64_t count) {
    Measurement m{Measurement::Kind::Memory};
    m.addrSpace = addrSpace;
    m.isLoad = true;
    m.isStore = true;
    m.count = count;
    return m;
}

// =============================================================
// Basic queries
// =============================================================

static void test_stats_empty() {
    std::vector<Measurement> v;
    Stats s(v);
    EXPECT_EQ((int)s.flops(), 0);
    EXPECT_EQ((int)s.bytes(), 0);
    Filter g; g.addrSpace = AS_GLOBAL;
    EXPECT(std::isnan(s.ai({}, g)));
}

static void test_stats_single_flop() {
    std::vector<Measurement> v = {flop(FpPrecision::F32, 2)};
    Stats s(v);
    EXPECT_EQ((int)s.flops(), 2);
    EXPECT_EQ((int)s.bytes(), 0);
}

static void test_stats_single_byte_load() {
    std::vector<Measurement> v = {memLoad(AS_GLOBAL, 4)};
    Stats s(v);
    EXPECT_EQ((int)s.flops(), 0);
    EXPECT_EQ((int)s.bytes(), 4);
}

static void test_stats_sums_across_kinds() {
    // flops() ignores Memory; bytes() ignores Flop.
    std::vector<Measurement> v = {
        flop(FpPrecision::F32, 2),
        flop(FpPrecision::F16, 4),
        memLoad(AS_GLOBAL, 8),
        memStore(AS_SHARED, 16),
    };
    Stats s(v);
    EXPECT_EQ((int)s.flops(), 6);
    EXPECT_EQ((int)s.bytes(), 24);
}

// =============================================================
// flops(Filter) — every filter dimension
// =============================================================

static void test_flops_filter_by_precision() {
    std::vector<Measurement> v = {
        flop(FpPrecision::F16, 4),
        flop(FpPrecision::F32, 2),
        flop(FpPrecision::F32, 2),
        flop(FpPrecision::F64, 1),
    };
    Stats s(v);
    Filter f16; f16.precision = FpPrecision::F16;
    Filter f32; f32.precision = FpPrecision::F32;
    Filter f64; f64.precision = FpPrecision::F64;
    EXPECT_EQ((int)s.flops(f16), 4);
    EXPECT_EQ((int)s.flops(f32), 4);
    EXPECT_EQ((int)s.flops(f64), 1);
}

static void test_flops_filter_by_scope_per_thread() {
    std::vector<Measurement> v = {
        flop(FpPrecision::F32, 2, InvocationScope::PerThread),
        flop(FpPrecision::F16, 4, InvocationScope::PerWarp),
    };
    Stats s(v);
    Filter pt; pt.scope = InvocationScope::PerThread;
    EXPECT_EQ((int)s.flops(pt), 2);
}

static void test_flops_filter_by_scope_per_warp() {
    std::vector<Measurement> v = {
        flop(FpPrecision::F32, 2, InvocationScope::PerThread),
        flop(FpPrecision::F16, 4096, InvocationScope::PerWarp),
    };
    Stats s(v);
    Filter pw; pw.scope = InvocationScope::PerWarp;
    EXPECT_EQ((int)s.flops(pw), 4096);
}

static void test_flops_filter_combined_precision_scope() {
    std::vector<Measurement> v = {
        flop(FpPrecision::F16, 4, InvocationScope::PerThread),
        flop(FpPrecision::F16, 8192, InvocationScope::PerWarp),
        flop(FpPrecision::F32, 2, InvocationScope::PerWarp),
    };
    Stats s(v);
    Filter f; f.precision = FpPrecision::F16; f.scope = InvocationScope::PerWarp;
    EXPECT_EQ((int)s.flops(f), 8192);
}

static void test_flops_filter_no_match() {
    std::vector<Measurement> v = {flop(FpPrecision::F32, 2)};
    Stats s(v);
    Filter f; f.precision = FpPrecision::F64;
    EXPECT_EQ((int)s.flops(f), 0);
}

static void test_flops_filter_addrspace_ignored() {
    // Documented behaviour: Memory-only fields on a flop query are
    // silently ignored. flops({addrSpace=AS_GLOBAL}) → "all flops" not 0.
    std::vector<Measurement> v = {
        flop(FpPrecision::F32, 2),
        flop(FpPrecision::F16, 4),
    };
    Stats s(v);
    Filter f; f.addrSpace = AS_GLOBAL;
    EXPECT_EQ((int)s.flops(f), 6);
}

// =============================================================
// bytes(Filter) — every filter dimension
// =============================================================

static void test_bytes_filter_by_addrspace_global() {
    std::vector<Measurement> v = {
        memLoad(AS_GLOBAL, 4),
        memLoad(AS_SHARED, 16),
        memStore(AS_GLOBAL, 4),
    };
    Stats s(v);
    Filter f; f.addrSpace = AS_GLOBAL;
    EXPECT_EQ((int)s.bytes(f), 8);
}

static void test_bytes_filter_by_addrspace_shared() {
    std::vector<Measurement> v = {
        memLoad(AS_GLOBAL, 4),
        memLoad(AS_SHARED, 16),
        memStore(AS_SHARED, 4),
    };
    Stats s(v);
    Filter f; f.addrSpace = AS_SHARED;
    EXPECT_EQ((int)s.bytes(f), 20);
}

static void test_bytes_filter_by_direction_load() {
    std::vector<Measurement> v = {
        memLoad(AS_GLOBAL, 4),
        memStore(AS_GLOBAL, 4),
        memLoad(AS_SHARED, 8),
    };
    Stats s(v);
    Filter f; f.isLoad = true;
    EXPECT_EQ((int)s.bytes(f), 12);
}

static void test_bytes_filter_by_direction_store() {
    std::vector<Measurement> v = {
        memLoad(AS_GLOBAL, 4),
        memStore(AS_GLOBAL, 4),
        memStore(AS_SHARED, 8),
    };
    Stats s(v);
    Filter f; f.isStore = true;
    EXPECT_EQ((int)s.bytes(f), 12);
}

static void test_bytes_filter_by_direction_atomic() {
    // An RMW atomic measurement has both isLoad and isStore set. The
    // filter isLoad=true AND isStore=true should match only it.
    std::vector<Measurement> v = {
        memLoad(AS_GLOBAL, 4),
        memStore(AS_GLOBAL, 4),
        memAtomic(AS_GLOBAL, 4),
    };
    Stats s(v);
    Filter f; f.isLoad = true; f.isStore = true;
    EXPECT_EQ((int)s.bytes(f), 4);
}

static void test_bytes_filter_by_scope_per_warp() {
    std::vector<Measurement> v = {
        memLoad(AS_SHARED, 4, InvocationScope::PerThread),
        memLoad(AS_SHARED, 512, InvocationScope::PerWarp),
    };
    Stats s(v);
    Filter f; f.scope = InvocationScope::PerWarp;
    EXPECT_EQ((int)s.bytes(f), 512);
}

static void test_bytes_filter_combined() {
    // Three-way AND: AS_GLOBAL × isLoad × PerThread.
    std::vector<Measurement> v = {
        memLoad (AS_GLOBAL, 4, InvocationScope::PerThread),  // match
        memLoad (AS_GLOBAL, 8, InvocationScope::PerWarp),    // wrong scope
        memStore(AS_GLOBAL, 4, InvocationScope::PerThread),  // wrong direction
        memLoad (AS_SHARED, 4, InvocationScope::PerThread),  // wrong AS
    };
    Stats s(v);
    Filter f;
    f.addrSpace = AS_GLOBAL;
    f.isLoad = true;
    f.scope = InvocationScope::PerThread;
    EXPECT_EQ((int)s.bytes(f), 4);
}

static void test_bytes_filter_precision_ignored() {
    // Symmetric to flops_filter_addrspace_ignored. Precision is not
    // consulted by bytes(); query returns all bytes regardless.
    std::vector<Measurement> v = {
        memLoad(AS_GLOBAL, 4),
        memLoad(AS_SHARED, 8),
    };
    Stats s(v);
    Filter f; f.precision = FpPrecision::F16;
    EXPECT_EQ((int)s.bytes(f), 12);
}

// =============================================================
// ai(flopFilter, byteFilter) — division semantics
// =============================================================

static void test_ai_zero_bytes_returns_nan() {
    std::vector<Measurement> v = {flop(FpPrecision::F32, 2)};
    Stats s(v);
    EXPECT(std::isnan(s.ai({}, {})));
}

static void test_ai_simple_ratio() {
    std::vector<Measurement> v = {
        flop(FpPrecision::F32, 8),
        memLoad(AS_GLOBAL, 4),
    };
    Stats s(v);
    double a = s.ai({}, {});
    EXPECT(!std::isnan(a));
    EXPECT(a == 2.0);
}

static void test_ai_zero_flops_finite_bytes() {
    std::vector<Measurement> v = {memLoad(AS_GLOBAL, 4)};
    Stats s(v);
    double a = s.ai({}, {});
    EXPECT(!std::isnan(a));
    EXPECT(a == 0.0);
}

static void test_ai_per_precision_per_level() {
    // The motivating case: f16 AI vs f32 AI, global vs shared.
    std::vector<Measurement> v = {
        flop(FpPrecision::F16, 4096),
        flop(FpPrecision::F32, 8),
        memLoad(AS_GLOBAL, 512),
        memLoad(AS_SHARED, 1024),
    };
    Stats s(v);
    Filter f16; f16.precision = FpPrecision::F16;
    Filter f32; f32.precision = FpPrecision::F32;
    Filter g;   g.addrSpace   = AS_GLOBAL;
    Filter sh;  sh.addrSpace  = AS_SHARED;

    EXPECT(s.ai(f16, g)  == 8.0);     // 4096 / 512
    EXPECT(s.ai(f32, g)  == 8.0/512); // 8 / 512
    EXPECT(s.ai(f16, sh) == 4.0);     // 4096 / 1024
}

static void test_ai_multi_term_realistic() {
    // Synthetic kernel-shaped workload. Loosely models a small matmul
    // body: one wmma tile (mma.sync m16n8k16, f16 → f32 accum, 4096
    // flops PerWarp) plus 3 f32 loads + 1 f32 store.
    std::vector<Measurement> v = {
        flop(FpPrecision::F16, 4096, InvocationScope::PerWarp),
        memLoad (AS_GLOBAL, 4),
        memLoad (AS_GLOBAL, 4),
        memLoad (AS_GLOBAL, 4),
        memStore(AS_GLOBAL, 4),
    };
    Stats s(v);
    EXPECT_EQ((int)s.flops(), 4096);
    EXPECT_EQ((int)s.bytes(), 16);
    EXPECT(s.ai({}, {}) == 256.0);
}

// =============================================================
// View / aggregation semantics
// =============================================================

static void test_bytes_aggregated_across_addrspaces() {
    // bytes({}) sums across all addrspaces (no filter = no restriction).
    std::vector<Measurement> v = {
        memLoad(AS_GLOBAL, 4),
        memLoad(AS_SHARED, 16),
        memLoad(AS_LOCAL,  8),
        memLoad(AS_CONST,  4),
        memLoad(AS_PARAM,  32),
    };
    Stats s(v);
    EXPECT_EQ((int)s.bytes(), 64);
}

static void test_flops_aggregated_across_precisions() {
    std::vector<Measurement> v = {
        flop(FpPrecision::F16,  4),
        flop(FpPrecision::BF16, 2),
        flop(FpPrecision::F32,  2),
        flop(FpPrecision::F64,  1),
        flop(FpPrecision::Other, 1),
    };
    Stats s(v);
    EXPECT_EQ((int)s.flops(), 10);
}

static void test_stats_is_a_view() {
    // Stats does not own the measurement vector. Mutating it after
    // constructing Stats is reflected on subsequent queries — important
    // contract for PR 4 (incremental per-BB collection).
    std::vector<Measurement> v = {flop(FpPrecision::F32, 2)};
    Stats s(v);
    EXPECT_EQ((int)s.flops(), 2);
    v.push_back(flop(FpPrecision::F32, 5));
    Stats s2(v);   // construct a fresh view after the vector grew
    EXPECT_EQ((int)s2.flops(), 7);
}

// =============================================================
// Hardening
// =============================================================

static void test_stats_measurement_size_within_budget() {
    // Duplicate of the assertion in ptx_unit_tests.cpp; kept here so the
    // stats binary is self-contained and a Measurement bloat caught
    // either side fails CI.
    EXPECT(sizeof(Measurement) <= 24);
}

// =============================================================
// Driver
// =============================================================

struct Test { const char *name; std::function<void()> fn; };

int main() {
    std::vector<Test> tests = {
        // basic
        {"stats_empty",                            test_stats_empty},
        {"stats_single_flop",                      test_stats_single_flop},
        {"stats_single_byte_load",                 test_stats_single_byte_load},
        {"stats_sums_across_kinds",                test_stats_sums_across_kinds},
        // flops filter
        {"flops_filter_by_precision",              test_flops_filter_by_precision},
        {"flops_filter_by_scope_per_thread",       test_flops_filter_by_scope_per_thread},
        {"flops_filter_by_scope_per_warp",         test_flops_filter_by_scope_per_warp},
        {"flops_filter_combined_precision_scope",  test_flops_filter_combined_precision_scope},
        {"flops_filter_no_match",                  test_flops_filter_no_match},
        {"flops_filter_addrspace_ignored",         test_flops_filter_addrspace_ignored},
        // bytes filter
        {"bytes_filter_by_addrspace_global",       test_bytes_filter_by_addrspace_global},
        {"bytes_filter_by_addrspace_shared",       test_bytes_filter_by_addrspace_shared},
        {"bytes_filter_by_direction_load",         test_bytes_filter_by_direction_load},
        {"bytes_filter_by_direction_store",        test_bytes_filter_by_direction_store},
        {"bytes_filter_by_direction_atomic",       test_bytes_filter_by_direction_atomic},
        {"bytes_filter_by_scope_per_warp",         test_bytes_filter_by_scope_per_warp},
        {"bytes_filter_combined",                  test_bytes_filter_combined},
        {"bytes_filter_precision_ignored",         test_bytes_filter_precision_ignored},
        // ai
        {"ai_zero_bytes_returns_nan",              test_ai_zero_bytes_returns_nan},
        {"ai_simple_ratio",                        test_ai_simple_ratio},
        {"ai_zero_flops_finite_bytes",             test_ai_zero_flops_finite_bytes},
        {"ai_per_precision_per_level",             test_ai_per_precision_per_level},
        {"ai_multi_term_realistic",                test_ai_multi_term_realistic},
        // semantics
        {"bytes_aggregated_across_addrspaces",     test_bytes_aggregated_across_addrspaces},
        {"flops_aggregated_across_precisions",     test_flops_aggregated_across_precisions},
        {"stats_is_a_view",                        test_stats_is_a_view},
        // hardening
        {"stats_measurement_size_within_budget",   test_stats_measurement_size_within_budget},
    };

    int passes = 0;
    for (const Test &t : tests) {
        g_current_test = t.name;
        int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        t.fn();
        if (g_failures == before) {
            std::printf("[  OK  ] %s\n", t.name);
            ++passes;
        } else {
            std::printf("[ FAIL ] %s\n", t.name);
        }
    }
    std::printf("\nSummary: %d/%zu passed, %d assertion(s) failed.\n",
                passes, tests.size(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
