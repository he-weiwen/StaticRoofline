# Measurement-Based Architecture Refactor

Design document and execution plan for a refactor of `nvptx_analyzer`'s
internal data flow. Splits the analyzer into three layers — classify,
collect, report — with a single value type (`Measurement`) as the contract
between them.

Status: **proposed, not implemented**. Captured for future reference.

---

## 1. Motivation

Two pressures point at the same architectural seam.

### 1.1 The `wmma.*` classification bug

The PTX-side classifier (`lib/PTX/Classifier.cpp:272-275`) routes every
mnemonic equal to `wmma` to `classifyMMA`, which finds the `m<M>n<N>k<K>`
shape modifier and synthesizes `2·M·N·K` FLOPs. But the WMMA family has
three roles, discriminated by the first modifier:

- `wmma.load.{a|b|c}.sync.…` — warp-cooperative memory load.
- `wmma.store.d.sync.…` — warp-cooperative memory store.
- `wmma.mma.sync.…` — the actual compute.

Loads and stores carry the same shape modifier (because the fragment role
needs to know its tile dimensions), so the current dispatch hands them
`8192` phantom FLOPs and zero memory bytes. A typical 5-instruction WMMA
tile (4 loads + 1 mma + 1 store) reports ~6× the true FLOP count and 0
bytes of traffic.

Empirically verified — the probe in
`tmp/probe_wmma` produces:

```
MMAOp  flops=8192  <-- wmma.load.a.sync.aligned.row.m16n16k16.global.f16 …
MMAOp  flops=8192  <-- wmma.load.b.sync.aligned.col.m16n16k16.global.f16 …
MMAOp  flops=8192  <-- wmma.load.c.sync.aligned.row.m16n16k16.global.f32 …
MMAOp  flops=8192  <-- wmma.store.d.sync.aligned.row.m16n16k16.global.f32 …
MMAOp  flops=8192  <-- wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32 …
```

The fix is mechanical (split dispatch by `modifiers[0]`), but it surfaces a
deeper issue: WMMA load/store byte counts are *per-warp*, and the current
`BlockStats`/`MemStats` god-object has no place to put them. The current
convention for warp-cooperative ops (`LdMatrix`, `AsyncCopy`) is to flatten
their bytes into the per-thread global/shared buckets — wrong by a factor
of 32, but visible. Adding wmma load/store via the same kludge propagates
the inconsistency.

### 1.2 Roadmap pressure

`STATUS.md` queues four phases of work that all push against the same
seams:

| Phase | What it needs |
|---|---|
| 3 — tensor cores on MIR side | per-scope flop buckets; collapses the `assert(PerThread)` tripwire |
| 4 — loop / SCEV attribution | hierarchical region stats (kernel → loop → BB) |
| 5 — JSON output / reporter abstraction | separate compute from emit |
| Per-precision / per-memory-level AI | AI as a function of (flop-filter, byte-filter), not a scalar |

Each touches `BlockStats` (god-object), the printer code (fused with
compute), or both. Doing them serially in status-quo style means paying
the structural cost four times.

### 1.3 The thesis

> **Measurements are values, not side effects.**

If the classifier *produces* a small value representing "what was measured"
and the aggregator simply *collects* those values, every queued change
lands as a one-file edit in the classifier. The aggregator and printers
become consumers of a stable value stream rather than co-conspirators in
mutating shared state.

This is the same pattern that LLVM's `OptimizationRemarkEmitter` (2016)
applied to optimization decisions, that rustc's `--message-format=json`
applied to diagnostics, that DTrace's `aggregation` applied to kernel
events, and that pandoc's AST applied to document conversion. Section 4
expands on the precedent.

---

## 2. Current architecture

### 2.1 Layout

```
                              INPUT
                    .ll → llc → MIR (MachineFunction)
                                  │
                  ┌───────────────┴───────────────┐
                  │     NVPTXArithIntensityPass    │   ← lib/NVPTXArithIntensityPass.cpp
                  │     (MachineFunctionPass)      │
                  └───────────────┬───────────────┘
                                  │ walks MBB → MI
              ┌───────────────────┼───────────────────┐
              │                   │                   │
          MI is FLOP        MI is INLINEASM       MI is anything
              │                   │                   │
   ptxai::classify(opcode)   parse + ptx::classify   recordMemory
   (lib/OpClassifier.cpp)    (lib/PTX/*.cpp)         (MMO + opcode-name fallback)
              │                   │                   │
       addFlops()        applyInlinePtxOpClass    bucketBytesByAddrSpace
              │                   │                   │
              └─────────┬─────────┴───────────────────┘
                        ▼
                ┌──────────────┐
                │  BlockStats  │   ← struct: 7 flop fields, 12 mem fields,
                │              │     opcode histogram, instr count
                └──────┬───────┘
                       ▼
        printBlockStats / printFlopsAndBytes / printMemoryStats
                       ▼
                     errs() text
```

### 2.2 Where it creaks

**`BlockStats` is a god-object.** Every new measurement requires editing
the struct, the `operator+=`, the printer, *and* at least one dispatcher
arm. Today's struct has 19 numeric fields plus a histogram. Per-scope
buckets push it past 30; per-precision × per-scope × per-memory-level
goes combinatorial.

**Two parallel `OpClass` types.** `ptxai::OpClass` (MIR-side struct, one
shape, `kind` enum) and `ptxai::ptx::OpClass` (PTX-side variant, 9 arms)
are wired into the aggregator via separate dispatch paths (`addFlops` vs
`applyInlinePtxOpClass`). MIR-side MMA support will force this divergence
to grow. `STATUS.md` defers unification.

**Compute, aggregation, and emission are fused.** `printFlopsAndBytes`
divides `flops / global_bytes` inline at print time. There's no
AI-as-a-value. Per-precision AI is impossible — the printer just shows
one number. JSON output is blocked not by the printer alone but by the
absence of a "stats are a value" boundary.

**Scope handling is half-built.** `InvocationScope` enum exists; every
classifier output struct has a `scope` field; the aggregator hard-asserts
PerThread on the MIR side and silently routes PerWarp into generic
buckets (`FlopsOther`, `SharedLoadBytes`) on the PTX side. No per-scope
accumulation, no warp/CTA multiplier.

### 2.3 What the current architecture is fit for

It correctly resisted over-design while the problem was being scoped. The
PTX subsystem in particular is well-structured: pure functions, exhaustive
`std::visit` arms, parametric-table-driven tests. The MIR-opcode classifier
is small and pure. The `parseMemoryOpcodeName` fallback for LDG opcodes is
exactly the right shape.

What's there is fit for purpose for what's shipped. It starts to creak
precisely at what's queued.

---

## 3. Target architecture

### 3.1 Shape

```
classify(MI or PTX Stmt) → vector<Measurement>      pure functions
                ↓
per-BB vector<Measurement>                          collected by pass
                ↓
Stats.flops({precision: F16, scope: PerWarp})       query helper
                ↓
TextReporter / JsonReporter                          swap-in writers
```

Three layers, three responsibilities, no framework, no visitor registry,
no reducer DSL.

### 3.2 The value

```cpp
// lib/Measurement.h
namespace ptxai {

struct Measurement {
    enum class Kind : uint8_t { Flop, Memory };
    Kind            kind;
    InvocationScope scope     = InvocationScope::PerThread;
    FpPrecision     precision = FpPrecision::Other;   // Flop only
    unsigned        addrSpace = 0;                    // Memory only
    bool            isLoad    = false;                // Memory only
    bool            isStore   = false;                // Memory only
    uint64_t        count     = 0;                    // FLOPs or bytes
};

} // namespace ptxai
```

Seven fields, ~24 bytes. Covers every field in today's `BlockStats` and
every dimension `STATUS.md` anticipates. If a future need wants a new
dimension (per-architecture? per-cluster?), add a field; existing
queries naturally ignore it.

`Sync` / `Barrier` / `Region` kinds are deliberately NOT added yet —
they contribute nothing to FLOPs or bytes and are already visible via
the opcode histogram (which stays as a per-BB side-channel; it's
diagnostic, not analytical). Add a `Kind` only when something needs to
aggregate it.

### 3.3 The query helper

```cpp
// lib/Stats.h
namespace ptxai {

struct Filter {
    std::optional<FpPrecision>     precision;
    std::optional<InvocationScope> scope;
    std::optional<unsigned>        addrSpace;
    std::optional<bool>            isLoad;
    std::optional<bool>            isStore;
};

class Stats {
public:
    explicit Stats(llvm::ArrayRef<Measurement> ms) : Ms(ms) {}
    uint64_t flops(const Filter &f = {}) const;
    uint64_t bytes(const Filter &f = {}) const;
    double   ai(const Filter &flopF, const Filter &byteF) const;  // NaN if bytes==0
private:
    llvm::ArrayRef<Measurement> Ms;
};

} // namespace ptxai
```

Per-precision AI: `stats.ai({F16}, {{}, {}, AS_GLOBAL})`.
Per-region AI: pass a `Stats` constructed from a region's measurement
subset.

### 3.4 The reporter

```cpp
// lib/Reporter.h
namespace ptxai {

struct BBRecord {
    const llvm::MachineBasicBlock *MBB;
    llvm::SmallVector<Measurement, 32> Ms;
    llvm::DenseMap<unsigned, uint64_t> OpcodeCounts;
    uint64_t Instrs = 0;
};

class Reporter {
public:
    virtual ~Reporter() = default;
    virtual void report(llvm::StringRef kernelName,
                        llvm::ArrayRef<BBRecord> blocks) = 0;
};

} // namespace ptxai
```

`TextReporter` reproduces today's `errs()` output byte-for-byte.
`JsonReporter` lands when a consumer exists; it implements the same
interface and has zero interaction with `TextReporter`.

### 3.5 How each queued change becomes local

| Change | Diff |
|---|---|
| wmma load/store fix | Classifier emits `Measurement{Memory, PerWarp, Global, isLoad, M*K*eltBytes}`. Reporter adds one query line. ~80 LOC. |
| stmatrix sense fix | Existing classifier flips `isStore`. ~10 LOC. |
| wmma.mma input/accum precision | Classifier writes the right `precision`. ~30 LOC. |
| Phase 3 — per-scope MMA | Classifier produces `Measurement{Flop, PerWarp, F16, 8192}`. No aggregator edits. |
| Phase 4 — hierarchical regions | `Stats.forLoop(MachineLoop*)` filters by MBB set. No measurement-shape change. |
| Phase 5 — JSON output | One new `JsonReporter` file. |
| Per-precision AI | One query: `flops({F16}) / bytes({{},{},Global})`. ~5 LOC. |
| tcgen05 / multimem | One classifier function each. |

The pattern "each addition touches 5 files" becomes "each addition is
one classifier function."

---

## 4. Grounding in precedent

The same shape has been validated in production tools for 15+ years.

**LLVM `OptimizationRemarkEmitter`** (Adam Nemet, 2016) — closest direct
analog. Pre-2016 passes printed decisions via `dbgs() <<`; consumers
regex-parsed stderr. Post-2016: typed `OptimizationRemark` values; multiple
consumers (`opt-viewer.py` HTML reports, `llvm-opt-report` CLI, Clang
`-Rpass=` diagnostics, profile-guided LTO tooling). None of those touched
pass code. Reference: `llvm/include/llvm/IR/DiagnosticInfo.h`,
`llvm/lib/Analysis/OptimizationRemarkEmitter.cpp`.

**Rustc `--message-format=json`** — same pattern, different domain.
Enabled rustfix, rust-analyzer diagnostics, cargo-clippy integration,
GitHub PR inline-comment bots — all without modifying the compiler.

**DTrace aggregations** (Cantrill et al., ~2005) and Linux `perf` events —
typed event records + aggregation primitive. The D one-liner
`syscall::read:entry { @[execname] = quantize(arg2); }` produces per-process
latency histograms. Before: custom instrumentation per process. After:
queries over a universal event schema.

**Pandoc AST** — `M readers → 1 AST → N writers`, so adding a writer
reaches every input format. Economics are M+N, not M×N.

**Bazel Build Event Protocol** and **`compile_commands.json`** — small,
stable, typed emission API at the producer side acts as a narrow waist
through which an unbounded set of consumers can be built (BuildBuddy,
clangd, clang-tidy, IDEs). Saltzer/Reed/Clark's "End-to-End Arguments"
(1984) is the canonical articulation of the narrow-waist principle.

**Out of the Tar Pit** (Moseley & Marks, 2006) — argues that separating
"essential state" (a relational core) from "accidental control" (presentation,
derived data, side effects) is the principled response to incidental
complexity. The Measurement / Stats / Reporter split is a small instance.

Honest disclaimer: those tools are larger than ours with more diverse
consumers. The dividend is correspondingly smaller. The argument for
doing it anyway is that the cost is also small, and the structural debt
of not doing it compounds with every Phase 3/4/5 change.

---

## 5. PR sequence

Each PR keeps both old and new code paths alive in parallel; each is
independently shippable; FileCheck output stays byte-identical through PR 8.

| # | What | Source LOC | New tests | Behaviour change |
|---|---|---|---|---|
| 1 | `Measurement` type + PTX dispatcher refactor | ~80 | ~30 | none |
| 2 | MIR dispatcher refactor | ~70 | ~10 | none |
| — | **(optional sidecar) wmma load/store fix** | ~80 | ~30-40 | yes (bug fix) |
| 3 | `Stats` query helper + unit tests | ~90 | ~30 | none |
| 4 | Pass collects Measurements per BB | ~50 | ~3 | none (debug assertion only) |
| 5 | Flops/AI printer to Stats | ~70 | ~5 | none |
| 6 | Memory printer to Stats | ~80 | ~6 | none |
| 7 | Delete `BlockStats` / `MemStats` | ~70 | +1 | none |
| 8 | Reporter interface + TextReporter | ~80 | ~5 | none |
| 9 | (deferred) JsonReporter | ~80 | TBD | new JSON output |

Net source delta: roughly flat (~0 to -100 LOC). New test code: ~90
(refactor) + ~30-40 (wmma sidecar). The win is *shape*, not line count.

### 5.1 PR 1 — `Measurement` type + use in PTX dispatcher

**New file**: `lib/Measurement.h` (~30 LOC) — the value type from §3.2.

**Modified**: `applyInlinePtxOpClass` in `NVPTXArithIntensityPass.cpp`
split into:

```cpp
// New: pure converter
static llvm::SmallVector<Measurement, 2>
toMeasurements(const ptxai::ptx::OpClass &op) {
    // std::visit dispatch, one arm per OpClass variant, producing 0..2 Measurements
}

// New: pure dispatcher
static void applyToBlockStats(const Measurement &m, BlockStats &stats) {
    // switch on m.kind, m.scope, m.precision, m.addrSpace
}

// Modified: now a thin shim
static void applyInlinePtxOpClass(BlockStats &Stats,
                                  const ptxai::ptx::OpClass &PtxOp) {
    for (const Measurement &m : toMeasurements(PtxOp))
        applyToBlockStats(m, Stats);
}
```

**Preserves**: `BlockStats` unchanged. All FileCheck output byte-identical.
All 76 PTX unit tests pass.

**Buys**: `Measurement` seam open on PTX path. The wmma sidecar PR
(below) becomes landable on top of this with no further aggregator work.

### 5.2 PR 2 — Use `Measurement` in MIR dispatcher

**Modified**: `addFlops` and `recordMemory` refactored to build
`SmallVector<Measurement>` first, then funnel through the same
`applyToBlockStats` from PR 1.

**Preserves**: `BlockStats` unchanged. All output byte-identical.

**Buys**: both dispatch paths now feed through `Measurement`. The
aggregator god-object is decoupled from the classifier outputs even
though it still exists.

### 5.3 Sidecar — wmma load/store fix

Optional but recommended to land between PR 2 and PR 3. The architecture
work doesn't become visible to users until then anyway; landing the
bug fix here demonstrates the new shape pays for itself immediately.

**Modified**: `lib/PTX/Classifier.cpp` — split `wmma.*` dispatch by
`modifiers[0]`:

```cpp
if (m == "wmma") {
    if (S.modifiers.empty()) return OpClass{Unknown{m}};
    StringRef role = S.modifiers[0];
    if (role == "load")  return classifyWmmaLoadStore(S, /*isLoad=*/true);
    if (role == "store") return classifyWmmaLoadStore(S, /*isLoad=*/false);
    if (role == "mma")   return classifyWmmaMma(S);  // existing classifyMMA renamed
    return OpClass{Unknown{m}};
}
if (m == "mma" || m == "wgmma") return classifyMma(S);
```

**New**: `classifyWmmaLoadStore` derives per-warp tile bytes from PTX-ISA
fragment dimensions:

```cpp
OpClass classifyWmmaLoadStore(const Stmt &S, bool isLoad) {
    if (S.modifiers.size() < 2) return OpClass{Unknown{S.mnemonic}};
    StringRef frag = S.modifiers[1];                  // "a" / "b" / "c" / "d"
    MMAShape sh = parseMmaShape(S.modifiers);
    if (sh.M == 0) return OpClass{Unknown{S.mnemonic}};

    unsigned eltBits = typeWidthBitsFromMods(S.modifiers);
    if (eltBits == 0) return OpClass{Unknown{S.mnemonic}};

    uint64_t elts;
    if      (frag == "a")                     elts = (uint64_t)sh.M * sh.K;
    else if (frag == "b")                     elts = (uint64_t)sh.K * sh.N;
    else if (frag == "c" || frag == "d")      elts = (uint64_t)sh.M * sh.N;
    else return OpClass{Unknown{S.mnemonic}};

    MemoryOp mem;
    mem.addrSpace = addrSpaceFromMods(S.modifiers);
    mem.bytes     = (elts * eltBits + 7) / 8;
    mem.isLoad    = isLoad;
    mem.isStore   = !isLoad;
    mem.scope     = InvocationScope::PerWarp;          // new field on MemoryOp
    return OpClass{mem};
}
```

**Extends** `typeWidthBitsFromMods` for missing types:

```cpp
if (m == "b1")                                   return 1;
if (m == "s4" || m == "u4" || m == "e2m1")       return 4;
if (m == "e3m2" || m == "e2m3")                  return 8;
// ... existing cases unchanged, plus add tf32 → 32
```

Authoritative grounding:

- Asm template grammar: `llvm/lib/Target/NVPTX/NVPTXIntrinsics.td:5198-5337`.
- Legal `(geometry, fragment, elt_type)` tuples:
  `llvm/include/llvm/IR/IntrinsicsNVVM.td:655-797`
  (specifically `NVVM_MMA_OPS.{all_ld_ops, all_st_ops}`).
- Fragment tile dimensions (A = M×K, B = K×N, C/D = M×N) — PTX ISA §9.7.14.

**Tests** — see §6 (~30-40 parametric cases).

**Behaviour change**: previously misclassified loads/stores now produce
correct `MemoryOp` measurements. The phantom FLOPs are gone. The bytes
are routed to the appropriate per-thread global/shared bucket via the
existing dispatcher (PerWarp scope is set on the Measurement; until
per-scope buckets land, the dispatcher silently flattens it — matching
the existing `LdMatrix` convention). When per-scope buckets land (post
PR 6), the flattening goes away automatically.

### 5.4 PR 3 — `Stats` query helper

**New**: `lib/Stats.{h,cpp}` per §3.3.

**Tests**: full unit-test coverage (no production caller yet; not dead
code because tests exercise the API).

**Preserves**: pass unchanged.

**Buys**: the query layer exists and is exercised. Reviewers can evaluate
the API independently of the migration.

### 5.5 PR 4 — Pass collects Measurements per BB

**Modified**: pass maintains `SmallVector<Measurement, 32>` per BB alongside
`BlockStats`. `applyToBlockStats` from PR 1 also appends to the vector.

**Added**: debug assertion:

```cpp
assert(Stats(measurements).flops({}) == bs.Flops);
assert(Stats(measurements).bytes({{},{},AS_GLOBAL}) ==
       bs.Mem.GlobalLoadBytes + bs.Mem.GlobalStoreBytes);
```

Fires immediately on any divergence between the two data paths. Catches
migration bugs in PRs 5-7 before they reach users.

**Preserves**: `BlockStats` still drives all output.

### 5.6 PR 5 — Flops/AI printer to Stats

**Modified**: `printFlopsAndBytes` reads from `Stats` instead of
`BlockStats`. Output text unchanged.

**Preserves**: FileCheck byte-identical. `BlockStats` still used for
memory printer + opcode histogram + Instrs count.

**Buys**: half the printer is now driven by Stats. If the wmma sidecar
landed, this PR is where its per-warp bytes become visible (a new query
line in TextReporter, gated on `bytes({scope: PerWarp})` being nonzero).

### 5.7 PR 6 — Memory printer to Stats

**Modified**: `printMemoryStats` reads from `Stats`.

**Preserves**: FileCheck byte-identical.

**Buys**: per-precision / per-scope memory queries become trivial to
add when consumers want them.

### 5.8 PR 7 — Delete `BlockStats` / `MemStats`

**Removed**: god-object structs, their `operator+=` overloads, the
`applyToBlockStats` shim, the PR-4 debug assertion (its purpose is
served).

**Replaced**: `Instrs` becomes a standalone counter on `BBRecord`.
Opcode histogram (`DenseMap<unsigned, uint64_t>`) also lives on
`BBRecord` as a side-channel — it's a literal per-opcode count, not
something to aggregate / filter, so doesn't belong in the
`Measurement` stream.

**Preserves**: FileCheck byte-identical.

### 5.9 PR 8 — Reporter interface + TextReporter

**New**: `lib/Reporter.h` + `lib/TextReporter.{h,cpp}` per §3.4.

**Modified**: pass instantiates a `TextReporter` and calls it; inline
`errs() << …` are gone from the pass.

**Preserves**: FileCheck byte-identical.

**Buys**: compute/emit boundary is a real seam. PR 9 (JSON) becomes a
new file with zero pass changes.

### 5.10 PR 9 — JsonReporter (deferred until needed)

Lands only when there's a consumer. Could be skipped entirely.

---

## 6. Test plan

### 6.1 Coverage target

"100% line coverage" is the weakest meaningful metric. The real target:

- **Every public function** exercised with at least one positive case.
- **Every branch** (every `if`, every variant arm, every switch case).
- **Every edge case** (empty input, zero denominator, max value, missing fields).
- **Every negative path** returning the documented fallback.
- **Behavioural parity** for migrations (new code path produces identical
  output to old, verified via FileCheck golden + debug assertions).

Some code — the pass walking real `MachineFunction` objects — is
impractical to unit-test without heavy LLVM fixture scaffolding. Those
gaps are called out explicitly per PR.

### 6.2 File organisation

```
test/
├── ptx_unit_tests.cpp           # existing — tokenizer/parser/PTX classifier
├── measurement_unit_tests.cpp   # PR 1 — Measurement + converters + applyToBlockStats
├── stats_unit_tests.cpp         # PR 3 — Stats query helper
└── reporter_unit_tests.cpp      # PR 8 — TextReporter (golden) + future JsonReporter
```

All link against the same `ptxai_ptx` object library; each produces a
separate `test-*` CMake target; the `check` aggregator runs them all.

### 6.3 Per-PR test design

**PR 1 — ~30 tests in `measurement_unit_tests.cpp`**

`toMeasurements` arm coverage (one per `ptx::OpClass` variant):

| Test | Asserts |
|---|---|
| `to_meas_flop_f32` | `FlopOp{F32, 2, PerThread}` → 1 measurement `{Flop, F32, PerThread, count=2}` |
| `to_meas_flop_f16x2` | `FlopOp{F16, 4}` → count=4 |
| `to_meas_mma_per_warp` | `MMAOp{16,8,16, F16, F32, 4096, PerWarp}` → 1 measurement, scope=PerWarp |
| `to_meas_memop_global_load` | `MemoryOp{1, 4, isLoad}` → 1 measurement |
| `to_meas_memop_atomic_rmw` | `MemoryOp{1, 4, isLoad+isStore}` → 1 measurement with both flags |
| `to_meas_async_copy_known_bytes` | `AsyncCopy{Shared, Global, 16}` → 2 measurements (load+store) |
| `to_meas_async_copy_unknown_bytes` | `AsyncCopy{..., bytes=nullopt}` → 0 measurements |
| `to_meas_ldmatrix` | `LdMatrix{Shared, 512, PerWarp}` → 1 measurement, scope=PerWarp |
| `to_meas_warpsync_empty` | `WarpSync{}` → 0 |
| `to_meas_barrier_empty` | `Barrier{}` → 0 |
| `to_meas_ignore_empty` | `Ignore{}` → 0 |
| `to_meas_unknown_empty` | `Unknown{"foo"}` → 0 |

`applyToBlockStats` dispatch coverage:

| Test | Asserts |
|---|---|
| `apply_flop_{f16, bf16, f32, f64, other}` | each precision → correct field |
| `apply_flop_per_warp_routes_to_other` | PerWarp Flop → `FlopsOther` |
| `apply_mem_{global, shared, local, const, param}_{load, store}` | parametric |
| `apply_mem_shared_cluster_aliases_shared` | AS_SHARED_CLUSTER routes to SharedLoadBytes |
| `apply_mem_atomic_increments_both` | isLoad+isStore bumps both sides |
| `apply_mem_unknown_addrspace` | unknown AS → UnknownBytes + UnknownAccesses |
| `apply_mem_size_zero` | count=0 → no-op, no assert |

Integration:

| Test | Asserts |
|---|---|
| `pr1_inline_asm_parity` | `inline_asm_hfma2.cu` FileCheck output byte-identical to pre-PR-1 baseline |

Coverage gap: `applyInlinePtxOpClass`'s `MI.isInlineAsm()` integration —
FileCheck only.

**PR 2 — ~10 tests**

MIR-side OpClass → Measurement:

| Test | Asserts |
|---|---|
| `mir_to_meas_scalar_flop_{f16, bf16, f32, f64}` | parametric |
| `mir_to_meas_flop_lanes` | lane-doubled FMA → count=2 |
| `mir_to_meas_none_kind_empty` | `OpClass{None}` → 0 measurements |
| `mir_to_meas_assert_per_thread` | `OpClass{ScalarFLOP, *, *, PerWarp}` → `EXPECT_DEATH` |
| `mir_recordmem_opcode_fallback_ldg` | fake opcode name `LD_GLOBAL_NC_i32` → 1 Memory measurement |
| `mir_recordmem_opcode_fallback_no_match` | `FMA_F32rrr` → 0 measurements |

Integration:

| Test | Asserts |
|---|---|
| `pr2_filecheck_parity` | all 8 green FileCheck tests byte-identical |

Coverage gap: `recordMemory`'s MMO walk needs a real MachineInstr — covered
by FileCheck.

**PR 3 — ~30 tests in `stats_unit_tests.cpp`**

Basic queries:

| Test | Asserts |
|---|---|
| `stats_empty` | empty vector → all queries return 0; `ai` returns NaN |
| `stats_single_flop` | one Flop measurement → `flops({}) == count` |
| `stats_single_byte_load` | one Memory measurement → `bytes({}) == count` |
| `stats_sums_across_kinds` | mixed Flop+Memory → flops query ignores Memory, vice versa |

Filter dimensions for flops:

| Test | Asserts |
|---|---|
| `flops_filter_by_precision` | precision-only filter |
| `flops_filter_by_scope_{per_thread, per_warp}` | scope-only filter |
| `flops_filter_combined_precision_scope` | both required (AND) |
| `flops_filter_no_match` | nothing matches → 0 |
| `flops_filter_addrspace_ignored` | Memory-only fields ignored for Flop query |

Filter dimensions for bytes:

| Test | Asserts |
|---|---|
| `bytes_filter_by_addrspace_{global, shared}` | parametric |
| `bytes_filter_by_direction_{load, store, both}` | parametric (both = atomics) |
| `bytes_filter_by_scope_per_warp` | scope filter |
| `bytes_filter_combined` | AS × isLoad × scope three-way AND |

AI semantics:

| Test | Asserts |
|---|---|
| `ai_zero_bytes_returns_nan` | division by zero handled |
| `ai_simple_ratio` | 8 FLOPs / 4 bytes → 2.0 |
| `ai_per_precision_per_level` | f16-flops / global-bytes ignores shared-bytes |
| `ai_zero_flops_finite_bytes` | 0/4 → 0.0 (not NaN) |

Hardening:

| Test | Asserts |
|---|---|
| `stats_overflow_safety` | uint64 sums near max — documented, no UB |
| `stats_measurement_size` | `sizeof(Measurement) <= 24` — guards against field bloat |
| `stats_const_correctness` | query methods are `const` |

**PR 4 — ~3 tests**

Mostly integration (the debug assertion is the real test):

| Test | Asserts |
|---|---|
| `pr4_debug_assertion_holds_smoke` | run smoke.ll; `Stats(ms).flops({}) == BS.Flops` |
| `pr4_debug_assertion_holds_inline_asm` | same for inline_asm_hfma2.cu |
| `pr4_filecheck_parity` | all 8 byte-identical |

**PR 5 — ~5 tests (printer)**

```cpp
std::string captured;
llvm::raw_string_ostream os(captured);
printFlopsAndBytes(os, Stats(/* known measurements */));
EXPECT_EQ(captured, " instrs=… flops=… …");
```

| Test | Asserts |
|---|---|
| `print_flops_empty` | empty Stats → `"flops=0 … ai=n/a"` |
| `print_flops_single_precision` | 4 FLOPs F16, 16 bytes → `"flops=4 flops_f16=4 … ai=0.250000"` |
| `print_flops_mixed_precision` | totals sum correctly |
| `print_flops_per_warp_routes_to_other` | PerWarp Flop → `flops_other` |
| `print_flops_ai_zero_bytes` | `ai=n/a` |

Plus `pr5_filecheck_parity`.

**PR 6 — ~6 tests (printer)**

| Test | Asserts |
|---|---|
| `print_mem_empty` | all zeros |
| `print_mem_global_only` | only global_load/global_store non-zero |
| `print_mem_shared_load_store_distinct` | load and store independently bumped |
| `print_mem_per_warp_diagnostic_line` | (post wmma sidecar) `per_warp_global_load=…` appears on a separate line |
| `print_mem_unknown_bytes_and_accesses` | unknown bucket surfaces both fields |

Plus `pr6_filecheck_parity`.

**PR 7 — 1 new test + standing regression bar**

| Test | Asserts |
|---|---|
| `pr7_instrs_counted_correctly` | `BBRecord.Instrs == sum of non-debug MIs`, verified via FileCheck |

All 76 + 8 existing tests must remain green.

**PR 8 — ~5 tests in `reporter_unit_tests.cpp`**

| Test | Asserts |
|---|---|
| `reporter_text_empty_kernel` | TextReporter over empty data → existing zero-kernel format |
| `reporter_text_single_block` | one BB → exact expected text |
| `reporter_text_multi_block_summary` | per-BB lines + summary, matching golden |
| `reporter_text_opcode_histogram_sorted` | histogram printed alphabetically |
| `reporter_interface_is_polymorphic` | `Reporter*` holds TextReporter; static_assert + runtime call |

Plus `pr8_filecheck_parity`.

### 6.4 Sidecar — wmma load/store fix (~30-40 parametric cases)

```cpp
struct WmmaLdStCase {
    const char *asm_;
    uint64_t   expectedBytes;
    unsigned   expectedAddrSpace;
    bool       expectedIsLoad;
    // scope is always PerWarp; not parameterised
};

static const WmmaLdStCase kWmmaLdStCases[] = {
    // m16n16k16 f16 — A is 16×16 f16 = 512B per warp
    {"wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];", 512, 1, true},
    // m16n16k16 f32 — C is 16×16 f32 = 1024B per warp
    {"wmma.load.c.sync.aligned.row.m16n16k16.global.f32 {%0,…,%7}, [%8];",                 1024, 1, true},
    {"wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%0], {%1,…,%8};",                1024, 1, false},
    // m32n8k16 s8 — A is 32×16 s8 = 512B per warp
    {"wmma.load.a.sync.aligned.row.m32n8k16.global.s8 {%0,%1,%2,%3}, [%4];",                512, 1, true},
    // m8n8k128 b1 — A is 8×128 bits = 128B per warp
    {"wmma.load.a.sync.aligned.row.m8n8k128.global.b1 {%0}, [%1];",                          128, 1, true},
    // m8n8k32 s4 — A is 8×32 × 4 bits = 128B per warp
    {"wmma.load.a.sync.aligned.row.m8n8k32.global.s4 {%0}, [%1];",                           128, 1, true},
    // m8n8k4 f64 — A is 8×4 × 8 bytes = 256B per warp
    {"wmma.load.a.sync.aligned.row.m8n8k4.global.f64 {%0}, [%1];",                           256, 1, true},
    // m16n16k8 tf32 — A is 16×8 × 4 bytes = 512B per warp
    {"wmma.load.a.sync.aligned.row.m16n16k8.global.tf32 {%0,%1,%2,%3}, [%4];",               512, 1, true},
    // … (~30-40 representative tuples drawn from IntrinsicsNVVM.td:771-794)
};
```

Categorical coverage:

| Category | Tests |
|---|---|
| Standard fp (f16, bf16, tf32, f32, f64) × A/B/C/D × global/shared | ~16 |
| Integer (s8, u8, s32) | ~6 |
| Sub-byte (s4, u4, b1, e2m1) — exercise bit→byte rounding | ~4 |
| FP8 (e4m3, e5m2) | ~2 |
| Address space defaults (modifier absent → generic) | ~2 |
| Negative cases (malformed shape, missing role, unknown role) | ~3 |

Bug-regression tests:

| Test | Asserts |
|---|---|
| `wmma_load_a_does_not_produce_flops` | `wmma.load.a.…` → Memory measurement, NOT MMAOp (the headline bug) |
| `wmma_store_d_per_warp_byte_count` | bytes match M×N × eltBytes |

### 6.5 Total

| Source | Tests |
|---|---|
| Existing PTX unit | 76 |
| Existing FileCheck (green) | 8 |
| PR 1-8 new unit | ~90 |
| Sidecar wmma | ~30-40 |
| **Post-refactor total** | **~200** |

---

## 7. What's explicitly NOT in scope

- **Per-region (loop / kernel) hierarchical stats** — Phase 4. Becomes
  trivial post-refactor (`Stats.forLoop(MachineLoop*)`) but adds no value
  until trip-count metadata threading lands.
- **JSON output** — PR 9, deferred until a consumer exists.
- **Unifying `ptxai::OpClass` (MIR struct) and `ptxai::ptx::OpClass`
  (variant)** — `STATUS.md` says defer until MIR-side MMA forces the
  issue. Post-refactor, both classifiers convert to the same `Measurement`
  stream, so unification stops being load-bearing.
- **`stmatrix` sense fix** — separate small bug fix (~10 LOC); should land
  as its own PR, not bundled here.
- **`wmma.mma` input/accum precision split** — separate bug fix
  (~30 LOC); the positional modifier walker is a notable style departure
  that deserves its own review surface.
- **MIR-side WMMA opcode classification** (`WMMA_LOAD_*` etc.) — most
  WMMA usage we've observed is inline-asm via CCCL; the C++ `nvcuda::wmma`
  API path goes through TableGen patterns that typically attach MMOs, so
  the existing MMO walk should cover it. Verify empirically when a corpus
  run shows `unknown_accesses` on a WMMA C++ kernel.
- **`tcgen05.*` (Blackwell)**, **`multimem.*` (NVLink)** — new classifier
  arms when needed; trivial under the new architecture.
- **Per-arch FLOP/byte machine tables for roofline** — Python tool, not
  in-tree (`STATUS.md` Phase 5 punts to a separate project).

---

## 8. Stop-points

If priorities change mid-sequence:

| After | State |
|---|---|
| PR 1 alone | Measurement seam open on PTX side; no other consumer; ~80 LOC of debt. Cheapest possible bet on the architecture; revert cost is the file plus one function rewrite. |
| PR 2 | Both dispatch paths feed through Measurement. Wmma sidecar becomes one-file-edit cheap. |
| PR 3 | Stats API exists and is exercised but unused in production. Useful even alone (someone could query measurements in a debugger). |
| PR 4 | Production data flows through Measurement vector + debug assertion validates parity. Strongest "land and stop" point if Phase 3-5 don't materialise. |
| PR 6 | All output driven by Stats; BlockStats is vestigial. Logical stop-point if PR 7's deletions feel risky. |
| PR 8 | Reporter abstraction complete. JSON or other formats are a new-file addition away. |

---

## 9. References

### In-tree
- `lib/NVPTXArithIntensityPass.cpp` — the pass; `applyInlinePtxOpClass`
  is the primary refactor surface.
- `lib/PTX/Classifier.cpp:272-285` — wmma dispatch (the bug site).
- `lib/PTX/Classifier.cpp:72-92` — `typeWidthBitsFromMods`,
  `vectorWidthFromMods` (extend for sub-byte types).
- `lib/OpClassifier.{h,cpp}` — MIR-side classifier.
- `test/ptx_unit_tests.cpp` — existing 76 unit tests (model for new
  test files).
- `STATUS.md` — Phase 3-5 specifications.

### External authoritative sources
- `llvm/lib/Target/NVPTX/NVPTXIntrinsics.td:5198-5337` — WMMA asm
  template grammar.
- `llvm/include/llvm/IR/IntrinsicsNVVM.td:177-470`, `:655-797` —
  `WMMA_REGS` class and `NVVM_MMA_OPS` enumeration.
- NVIDIA PTX ISA Reference §9.7.13-14 — `mma` / `wmma` semantics.

### Precedent (the architectural pattern)
- `llvm/include/llvm/IR/DiagnosticInfo.h`,
  `llvm/lib/Analysis/OptimizationRemarkEmitter.cpp` — the closest direct
  analog (LLVM Optimization Remarks).
- Cantrill et al., "Dynamic Instrumentation of Production Systems"
  (USENIX 2004) — DTrace aggregations.
- Moseley & Marks, "Out of the Tar Pit" (2006) — the underlying
  intellectual frame.
- Saltzer, Reed, Clark, "End-to-End Arguments in System Design" (1984) —
  the narrow-waist principle.

---

## 10. Open questions for future-me

1. **Should `BBRecord` carry the `MachineLoop` nest** or should consumers
   query `MachineLoopInfo` separately? Carrying it makes `Stats.forLoop()`
   trivial; querying keeps `BBRecord` smaller. Decide based on whether
   Phase 4 needs the per-BB list of enclosing loops in many places.
2. **Filter as struct of optionals vs. predicate function?** Struct is
   simpler and serialisable; predicates are more flexible. Start with
   struct; switch if a real consumer wants composition.
3. **Should `Stats` materialise sums lazily or eagerly?** Lazy (each query
   scans the vector) is simplest and likely fast enough at per-BB scale
   (hundreds of measurements). Eager (precomputed sums per bucket) is
   the optimisation if profiling shows it matters.
4. **`Measurement` size cap?** Currently 24 bytes. If a future field
   pushes past 32, reconsider whether `Kind`-specific layouts (via a
   tagged union or variant) buy enough to justify the complexity. Today
   the flat record wins.
