# Onboarding Walkthrough

A guided tour of `nvptx_analyzer` for engineers with entry-level compiler
and GPU background. By the end you should be able to: explain why this
project exists, run it, read its output, locate the relevant code, and
pick a contribution.

The reading order is top to bottom. Every shell command is meant to be
typed; outputs shown in code blocks are real captures, so you can
verify your environment matches.

> **Prerequisites.** A built project (see Part 2). LLVM 23 trunk at
> `/home/whe302/compilers/llvm-project/build`. CUDA 13 toolkit. You do
> NOT need a physical GPU — this is a static analyzer; nothing executes
> on hardware.

---

## Part 1 — Why this project exists

### The question we're answering

For a GPU kernel, two numbers determine whether it can possibly be
fast:

- **FLOPs**: how much arithmetic work is required.
- **Bytes**: how much data has to move through memory.

Their ratio — **arithmetic intensity (AI)** — tells you whether the
kernel is fundamentally compute-bound or memory-bound on a given
machine. The classic *roofline* picture says: if your AI lies below
the machine's "balance point" (peak FLOPs ÷ peak bandwidth, e.g.
~590 FLOP/byte for fp16 tensor cores on H100), no implementation
can be compute-bound — you're capped by memory bandwidth.

### Why not just use NCU?

NVIDIA's Nsight Compute (`ncu`) gives you measured AI by running the
kernel on real hardware and reading performance counters. Three reasons
that's not always what you want:

1. **It requires running the kernel.** Not great for CI on machines
   without GPUs, for design-space exploration ("what if I tile
   differently?"), or for comparing two implementations before either
   has been benchmarked.
2. **Measured numbers are an *execution* property, not a *kernel*
   property.** They depend on launch config, input shape, cache state.
   Different runs give different AI.
3. **Static analysis catches things measurement masks.** A static
   FLOP-per-byte count is a *theoretical* upper bound on what any
   measurement could see. Comparing the two surfaces gaps (cache
   reuse, coalescing, divergence) you'd otherwise have to guess at.

This project produces the static side: how many FLOPs and how many
bytes the kernel *requests*, derived from compiled machine IR without
ever running anything.

### What "static" means here, honestly

The numbers you'll see are **requested** quantities, not transferred
quantities. Specifically:

- FLOPs come from counting arithmetic instructions in the post-codegen
  MIR. FMA = 2 FLOPs. Packed-vector `f16x2` = 2× lane multiplier.
- Bytes come from `MachineMemOperand` sizes, bucketed by address space.

Things this **cannot** see:

- Cache reuse — every load counts at face value, no L1/L2 modeling.
- Coalescing — a 4-byte access counts as 4 bytes whether it's perfectly
  coalesced or fully scattered.
- Branch divergence — every instruction is counted as if all warp lanes
  execute it.
- ptxas spills — the closed-source assembler downstream of LLVM does
  its own register allocation; spills there are invisible to us.

That's not a flaw to fix; it's the boundary of static analysis. The
honest framing: these are the numbers any implementation must do *at
least*, not the numbers a real run will see.

### Self-check
Before continuing, you should be able to answer: *for a kernel `c[i] =
a[i] + b[i]` where each element is a 32-bit float, what AI does the
analyzer produce, and is the kernel compute-bound or memory-bound on a
GPU with peak ~10 FLOP/byte?*

Answer: 1 FLOP / 12 bytes ≈ 0.083 FLOP/byte. Memory-bound by ~2 orders
of magnitude. We'll verify this in Part 3.

---

## Part 2 — Build it and run a test

Set up:

```bash
cd nvptx_analyzer
cmake -S . -B build \
      -DLLVM_DIR=/home/whe302/compilers/llvm-project/build/lib/cmake/llvm
cmake --build build --target check
```

`check` is the aggregator that builds the plugin and runs all tests.
Expected (six tests, all green):

```
[ 30%] Built target test-fma
[ 45%] Built target test-fma_v2half
[ 60%] Built target test-smoke
[ 70%] Built target test-fsub
[ 80%] Built target test-add_f64
[ 90%] Built target test-local
[100%] Built target check
```

What just happened:

1. `cmake` configured the project against your installed LLVM.
2. `cmake --build` compiled `lib/NVPTXArithIntensity.so` — a shared
   *plugin* meant to be loaded by `llc`, not run on its own.
3. For each test (`smoke`, `fsub`, `add_f64`, `fma`, `local`,
   `fma_v2half`):
   - Lowered a `.ll` (or `.cu`) source down to MIR using `llc -stop-before=nvptx-asm-printer`.
   - Loaded the plugin into `llc` and ran the pass on the MIR.
   - Piped the analyzer's output through `FileCheck` to verify expected
     numbers.

If any of that was magic, the next two parts unpack it.

---

## Part 3 — The pipeline, with concrete artifacts

CUDA → ptx happens through several intermediate forms. The analyzer
runs at one specific point. To know where, you need the picture:

```
┌──────────────────────────────────────────────────────────────────┐
│  CUDA source (.cu)        ← what the user wrote                 │
│            │                                                     │
│            ↓ clang -emit-llvm                                    │
│  LLVM IR (.ll)            ← target-independent, SSA, types       │
│            │                                                     │
│            ↓ llc / NVPTXTargetMachine                            │
│  Machine IR (.mir)        ← post-codegen, target-specific        │
│            │                       opcodes, MachineMemOperands   │
│            │                       ★ THE ANALYZER RUNS HERE ★    │
│            ↓ NVPTXAsmPrinter                                     │
│  PTX text (.ptx)          ← NVIDIA's virtual ISA                 │
│            │                                                     │
│            ↓ ptxas (closed source, NVIDIA toolkit)               │
│  SASS / cubin             ← actual machine code per arch         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

Let's see each form for the smoke kernel.

### LLVM IR (target-independent)

```bash
cat test/smoke.ll
```

What the kernel says, written in LLVM IR. Note `addrspace(1)` (global
memory), `fadd`, `<2 x half>` would-be-here-if-we-cared, etc. This is
what `clang` produced before any NVPTX-specific lowering.

### MIR (target-specific, what we analyze)

```bash
sed -n '/^body:/,/^\.\.\./p' build/test/smoke.mir | head -16
```

```
body:             |
  bb.0.entry:
    %0:b64 = LD_i64 0, 0, 101, 3, 64, -1, &add_kernel_param_0, 0 :: (dereferenceable invariant load (s64), addrspace 101)
    %1:b64 = LD_i64 0, 0, 101, 3, 64, -1, &add_kernel_param_1, 0 :: (dereferenceable invariant load (s64), addrspace 101)
    %2:b32 = INT_PTX_SREG_TID_x
    %3:b64 = LD_i64 0, 0, 101, 3, 64, -1, &add_kernel_param_2, 0 :: (...)
    %4:b64 = MUL_WIDEu32_ri %2, 4
    %5:b64 = ADD64rr %0, %4
    ...
    %8:b32 = LD_i32 0, 0, 1, 3, 32, -1, %5, 0 :: (load (s32) from %ir.ap, addrspace 1)
    %9:b32 = LD_i32 0, 0, 1, 3, 32, -1, %6, 0 :: (load (s32) from %ir.bp, addrspace 1)
    %10:b32 = FADD_rnf32rr %8, %9, 0
    ST_i32 %10, 0, 0, 1, 32, %7, 0 :: (store (s32) into %ir.cp, addrspace 1)
    Return
```

This is what the analyzer iterates over. Notice:

- Each `LD_i32` has a trailing `:: (load (s32), addrspace 1)` — that's
  the **MachineMemOperand (MMO)**. The analyzer reads MMO size and
  addrspace to bucket bytes.
- `FADD_rnf32rr` is the actual NVPTX MIR opcode — the `_rn` is the
  rounding mode, `f32` is the type, `rr` means register+register.
  These string suffixes are what the classifier prefix-matches.
- `INT_PTX_SREG_TID_x` is reading `%tid.x`. Counted in the instruction
  histogram but not as a FLOP.

### PTX (the next step, not what we analyze)

```bash
/home/whe302/compilers/llvm-project/build/bin/llc \
  -march=nvptx64 -mcpu=sm_80 -O2 test/smoke.ll -o - 2>/dev/null \
  | sed -n '/\.entry add_kernel/,/^}/p'
```

```
.visible .entry add_kernel(
    .param .u64 .ptr .global .align 1 add_kernel_param_0,
    .param .u64 .ptr .global .align 1 add_kernel_param_1,
    .param .u64 .ptr .global .align 1 add_kernel_param_2
)
{
    .reg .b32   %r<5>;
    .reg .b64   %rd<8>;
    ld.param.b64    %rd1, [add_kernel_param_0];
    mov.u32         %r1, %tid.x;
    ...
    ld.global.b32   %r2, [%rd5];
    ld.global.b32   %r3, [%rd6];
    add.rn.f32      %r4, %r2, %r3;
    st.global.b32   [%rd7], %r4;
    ret;
}
```

This is the textual PTX that downstream tools (`ptxas`) consume. The
same `add.rn.f32` we saw as `FADD_rnf32rr` in MIR. The same `ld.global`
we saw with `addrspace(1)` MMOs.

> **Why analyze MIR rather than IR or PTX?** IR is too high — types and
> addrspaces aren't yet target-specific, FMA fusion hasn't happened. PTX
> text is too low — MMOs are gone, parsing is annoying. MIR is the
> sweet spot: target-specific opcodes already chosen, but MMOs and
> structural form still preserved.

### Self-check
You should be able to point at `FADD_rnf32rr` in the MIR and explain:
*this is one f32 fused-FLOP-producer instruction; the analyzer counts
it as 1 FLOP in the f32 bucket.*

---

## Part 4 — Read the analyzer output, field by field

```bash
/home/whe302/compilers/llvm-project/build/bin/llc \
  -load build/lib/NVPTXArithIntensity.so \
  -run-pass=ptx-ai \
  build/test/smoke.mir -o /dev/null
```

Output:

```
kernel add_kernel
  bb.0.entry instrs=13 flops=1 flops_f16=0 flops_bf16=0 flops_f32=1 flops_f64=0 global_bytes=12 local_bytes=0 ai=0.083333
    ADD64rr: 3
    FADD_rnf32rr: 1
    INT_PTX_SREG_TID_x: 1
    LD_i32: 2
    LD_i64: 3
    MUL_WIDEu32_ri: 1
    Return: 1
    ST_i32: 1
    memory: global_load=8 global_store=4 ... param_load=24 ...
summary: add_kernel blocks=1 instrs=13 flops=1 flops_f16=0 flops_bf16=0 flops_f32=1 flops_f64=0 global_bytes=12 local_bytes=0 ai=0.083333
```

Walk through it:

| Field | Meaning | Where it comes from |
|---|---|---|
| `kernel add_kernel` | name of the MachineFunction being analyzed | `MF.getName()` |
| `bb.0.entry` | one MachineBasicBlock — the analyzer reports per-BB, then summarizes | `MachineBasicBlock` traversal |
| `instrs=13` | non-debug `MachineInstr`s in this block | counter incremented per `MI` |
| `flops=1` | total FLOPs (sum of per-precision) | `OpClassifier::classify` returned `flopsPerInvocation=1` for the FADD |
| `flops_f32=1` | FLOPs in the f32 bucket | precision detected from opcode name `FADD_rnf32rr` |
| `flops_f16/_bf16/_f64=0` | other precision buckets | no matching opcodes |
| `global_bytes=12` | sum of MMO sizes for `addrspace(1)` loads + stores | 2× `LD_i32` (8B) + 1× `ST_i32` (4B) |
| `local_bytes=0` | `addrspace(5)` traffic — diagnostic only, not in AI denominator | no allocas in this kernel |
| `ai=0.083333` | `flops / global_bytes` = 1/12 | computed at print time |

Below the headline are two further blocks:

- The opcode histogram (`ADD64rr: 3`, etc.) — *every* opcode that
  appeared, sorted alphabetically. Useful when something doesn't add up
  and you want to see what's actually in the MIR.
- The `memory:` line — full per-address-space byte breakdown.

Key things to notice:

- **`ADD64rr: 3`, `MUL_WIDEu32_ri: 1` — these are integer ALU ops for
  address arithmetic.** Correctly *not* counted as FLOPs. The classifier
  is conservative: only `FADD/FSUB/FMUL/FMA` prefixes contribute.
- **`param_load=24` doesn't enter `global_bytes`.** Kernel parameter
  loads come from constant memory at launch, not DRAM during execution.
  This is a deliberate exclusion.
- **AI is computed against `global_bytes`, not "DRAM bytes".** Local
  memory is reported separately. Reasoning is in `STATUS.md` — local
  rarely matters in well-tuned kernels and folding it in noises the
  global-memory signal.

### Self-check
Run the same command on `build/test/add_f64.mir`. Predict the output
before reading: how many FLOPs, what precision bucket, what
`global_bytes`, what AI?

Answer: 1 FLOP in `flops_f64` (one `FADD_rnf64rr`), `global_bytes=24`
(three doubles at 8 bytes each), AI = 1/24 ≈ 0.0417.

---

## Part 5 — A more interesting example: the `f16x2` packed FMA

Half-precision is the workhorse of modern ML kernels. NVPTX has a
*packed* form (`f16x2`) where one instruction operates on two halves
side-by-side — a 2× lane multiplier. Under-counting these would
silently halve every fp16 kernel's reported FLOPs.

The CUDA test exercises this:

```bash
cat test/fma_v2half.cu
```

```cuda
typedef _Float16 v2half __attribute__((ext_vector_type(2)));

extern "C" __global__ void fma_v2half_kernel(
        const v2half* a, const v2half* b, const v2half* c, v2half* d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx] + c[idx];
}
```

The `_Float16 ext_vector_type(2)` syntax is clang's portable way to
write `<2 x half>` in C++. With `-ffp-contract=fast`, the multiply +
add fuses into a packed FMA.

Run the analyzer:

```bash
/home/whe302/compilers/llvm-project/build/bin/llc \
  -load build/lib/NVPTXArithIntensity.so \
  -run-pass=ptx-ai \
  build/test/fma_v2half.mir -o /dev/null
```

```
kernel fma_v2half_kernel
  bb.0.entry instrs=23 flops=4 flops_f16=4 flops_bf16=0 flops_f32=0 flops_f64=0 global_bytes=16 local_bytes=0 ai=0.250000
    ...
    FMA_F16x2rrr: 1
    ...
```

Notice:

- **One** `FMA_F16x2rrr` opcode → **4** FLOPs in `flops_f16`. That's the
  lane multiplier at work: 2 lanes × (1 mul + 1 add) per FMA.
- AI = 4/16 = 0.25, four times higher than the scalar f32 baseline.

### The hidden subtlety: what happens with `__hfma2` from `cuda_fp16.h`?

If the CUDA were written with NVIDIA's recommended intrinsic instead
of clang's vector type:

```cuda
#include <cuda_fp16.h>
d[idx] = __hfma2(a[idx], b[idx], c[idx]);
```

…the analyzer would report `flops=0`. Why?

`__hfma2` lowers to **inline PTX assembly** — an `INLINEASM` MachineInstr
whose body is the literal string `"fma.rn.f16x2 %0, %1, %2, %3;"`. The
classifier sees `INLINEASM` and has no idea what's inside. Today the
analyzer correctly *flags* this gap (the integration point at line ~266
of `lib/NVPTXArithIntensityPass.cpp` has a TODO marker), but the
contribution to FLOPs is zero.

This is the motivation for **Phase 1** of the roadmap (see `STATUS.md`):
add a small inline-PTX parser and route what it finds back into the
same FLOP/byte counters.

### Self-check
You should now be able to explain to someone else: *the analyzer
reports FLOPs accurately for code lowered through LLVM's normal
codegen, but undercounts when the source uses CUDA-header intrinsics
that lower to inline asm. Phase 1 of the project closes that gap.*

---

## Part 6 — Architecture tour

A guided look at the source. File sizes for orientation:

```
lib/NVPTXArithIntensityPass.cpp   301 lines   the pass + memory accounting + reporting
lib/OpClassifier.{h,cpp}          157 lines   MIR opcode → OpClass dispatch
lib/PTX/{Tokenizer,Parser,Classifier}.{h,cpp}  275 lines  scaffold (stubs)
test/...                          ~190 lines  six tests
STATUS.md, TUTORIAL.md, CMakeLists.txt        ~600 lines  meta
```

Roughly 1200 lines total. Small project; you can read all of it in one
sitting.

### `lib/NVPTXArithIntensityPass.cpp` — the pass

This file does five things in order:

1. **Defines aggregation types** (`MemStats`, `BlockStats`) — the
   counters we accumulate per-BB.
2. **Records memory** (`recordMemory`, `addLoadStoreBytes`) — walks
   `MI.memoperands()` and buckets by address space.
3. **Records FLOPs** (`addFlops`) — calls `OpClassifier` and bumps the
   right precision bucket.
4. **Prints results** (`printBlockStats`, `printMemoryStats`,
   `printFlopsAndBytes`) — the textual output you saw.
5. **The pass itself** (`NVPTXArithIntensityPass`) — registered with
   LLVM's PassManager, dispatches `runOnMachineFunction`.

The interesting per-MI loop is around line 253:

```cpp
for (auto &MI : MBB) {
    if (MI.isDebugInstr()) continue;
    ++Stats.Instrs;
    ++Stats.OpcodeCounts[MI.getOpcode()];

    // INLINEASM integration point: parser+classifier from lib/PTX/
    // will plug in here.
    if (MI.isInlineAsm()) {
        // TODO: parse asm body, classify each statement, route into
        //       addFlops/recordMemory.
        recordMemory(Stats, MI);
        continue;
    }

    ptxai::OpClass Op = ptxai::classify(TII->getName(MI.getOpcode()));
    if (Op.isFlopProducer()) addFlops(Stats, Op);
    recordMemory(Stats, MI);
}
```

This is the only place where you need to think about "what kind of
work does an instruction do." Everything else flows from that
classification.

### `lib/OpClassifier.{h,cpp}` — the MIR-side classifier

`OpClassifier.h` defines the public types:

- `FpPrecision` enum — `F16 / BF16 / F32 / F64 / Other`.
- `OpKind` enum — current and planned categories. Today only
  `ScalarFLOP` is implemented; `MMA / SpecialMath / AsyncCopy /
  LdMatrix / WarpOp` are reserved.
- `InvocationScope` enum — `PerThread / PerWarp / PerCTA`. Critical
  for correctness: an MMA instruction is *one* MIR opcode but
  produces work for the whole warp. Counting it as per-thread would be
  off by 32×.
- `OpClass` struct — the aggregated result.
- `classify(StringRef name)` — the only entry point.

`OpClassifier.cpp` is ~70 lines. The whole file is two prefix matches
and a precision substring lookup, plus a packed-vector lane multiplier
for `f16x2` / `bf16x2` / `f32x2`.

> **Read this file.** It's the simplest demonstration of the project's
> architectural principle: pure-function dispatch on opcode names,
> producing a single well-typed value the aggregator consumes. Any
> change to "what counts as a FLOP" lives here.

### `lib/PTX/` — the scaffold for inline-asm support

Three files, all stubs today:

```
PTX/Tokenizer.{h,cpp}   Token kinds + tokenize()         → returns {}
PTX/Parser.{h,cpp}      Stmt + Operand variant + parse() → returns {}
PTX/Classifier.{h,cpp}  OpClass variant + classify()     → returns Unknown{}
```

The headers define the types (token kinds, AST shape, classification
output). The .cpp files are placeholder implementations. When Phase 1
lands, these get filled in and the TODO marker in the pass routes to
them.

The PTX-side `OpClass` is a `std::variant` (sum type), not a struct —
because inline PTX has more diverse semantic categories than scalar
FLOPs, and exhaustive pattern-matching via `std::visit` keeps the
aggregator honest when new categories land.

### `test/` — how integration tests work

Each test is a `.ll` (or `.cu`) file with FileCheck patterns embedded
as comments at the top:

```bash
head -15 test/smoke.ll
```

```
; CHECK: kernel add_kernel
; CHECK: bb.0
; CHECK-SAME: flops=1
; CHECK-SAME: flops_f32=1
...
```

The `add_ptxai_test` CMake helper wires this into a 4-step pipeline:

```
test/foo.ll
   ↓ llc -stop-before=nvptx-asm-printer
build/test/foo.mir
   ↓ llc -load NVPTXArithIntensity.so -run-pass=ptx-ai
plugin output
   ↓ FileCheck against test/foo.ll
pass / fail
```

Adding a new test is 3 lines: drop a `.ll` into `test/`, add
`add_ptxai_test(name)` and the corresponding `test-name` to the
`check` aggregator's DEPENDS list.

CUDA tests use `add_ptxai_cuda_test` which prepends a `clang
--cuda-device-only` step; otherwise identical.

### Self-check
Pop quiz: where would you go to add support for `FDIV` (floating-point
divide) as a FLOP-producing instruction? What would change?

Answer: `lib/OpClassifier.cpp`'s `classify()`. Add a third arm:

```cpp
if (Name.starts_with("FDIV")) {
    C.kind = OpKind::ScalarFLOP;
    C.flopsPerInvocation = 1 * Lanes;     // by convention; div is 1 FLOP
    C.scope = InvocationScope::PerThread;
    C.precision = detectPrecision(Name);
    return C;
}
```

Then add a test: `test/fdiv.ll` (copy `fsub.ll`, change the op),
register it in `test/CMakeLists.txt`. About 5 minutes of work.

---

## Part 7 — Where to contribute

`STATUS.md` is the source of truth for the roadmap. Phases 1–5, in
priority order. For an entry-level first PR, here's a triage by
difficulty:

### Tiny (a few hours)

- **`FDIV` and `FSQRT` in the FP classifier.** See the self-check
  answer above. Adds two FLOP-producing op families with a precision
  bucket each, plus tests. Good first PR — touches the smallest amount
  of code while teaching the classifier pattern.
- **`FNEG` (negation).** Decide whether to count as 1 FLOP or 0; the
  literature is split. Worth a small writeup in the commit message.

### Small (a day)

- **Special-math classifier expansion.** `EX2`, `LG2`, `SIN`, `COS`,
  `RSQRT`, `RCP` — six opcodes, all PerThread, 1 FLOP by convention.
  Same shape as the FDIV exercise, but more of them, and one new
  `OpKind::SpecialMath` arm that's currently a stub.

### Medium (a few days)

- ~~**The LDG-without-MMO byte-undercount.**~~ **Done.** The
  `LD_GLOBAL_NC_*` and `LDU_GLOBAL_*` families intentionally ship
  without `MachineMemOperand`s and without the `mayLoad` flag — see
  the in-source rationale at `NVPTXIntrinsics.td:2786-2788` (D17471,
  2016) and the 2023 patch D112466 ("[NVPTX] Drop memory references
  of LDG/LDU") that actively removed any leftover MMO attachments.
  NVPTX models them as constant-materialization, letting MachineLICM
  / Sink / Scheduler freely reorder them. Our analyzer correctly
  recovers byte traffic via `parseMemoryOpcodeName` in `OpClassifier`,
  which pattern-matches the full LD/ST opcode taxonomy. Both
  `fma_v2half.cu` (with `__restrict__`) and `ldg_restrict.cu`
  exercise this end-to-end.
- **The `cccl-scraper` (Phase 1, item 1).** Walk
  `<cuda>/include/cccl/cuda/__ptx/instructions/generated/`, pull out
  asm template strings and their canonical-PTX comments, write
  `test/ptx-corpus.json`. ~100 lines of Python. Bounded and well-defined
  output.

### Larger (1–2 weeks, more architectural)

- **Tokenizer + Parser implementation** (Phase 1, items 2–3). The
  scaffold types are already there; you fill in `tokenize()` and
  `parse()` against the corpus from the scraper.
- **Per-family Classifier handlers** (Phase 1, item 4). Six families
  in priority order. Each family is mechanical — read CCCL docs, write
  the dispatch arm, add corpus-coverage assertions.
- **Tensor-core support on the MIR side** (Phase 3). MMA/WMMA/WGMMA
  opcode classifier with per-arch shape table, plus extending the
  aggregator to handle `PerWarp` scope. The `assert(scope ==
  PerThread)` in `addFlops` is the canary that goes off; replacing it
  with per-scope buckets is the change.

### What I wouldn't recommend as a first PR

- Anything in Phase 4 (loop / SCEV) — touches both IR-level and
  MIR-level passes and needs metadata round-tripping. Real architecture
  work.
- The Reporter abstraction in Phase 5 — meaningful only after JSON
  output is decided, and that's a UX call best left to a maintainer.

---

## Quick reference

| Task | Command |
|---|---|
| Build everything | `cmake --build build` |
| Run all tests | `cmake --build build --target check` |
| Run one test | `cmake --build build --target test-smoke` |
| Run analyzer manually | `llc -load build/lib/NVPTXArithIntensity.so -run-pass=ptx-ai <file.mir>` |
| Inspect MIR for a test | `cat build/test/<name>.mir` |
| Look at PTX for a kernel | `llc -march=nvptx64 -mcpu=sm_80 -O2 test/<name>.ll -o -` |

| File | Read it when you want to | 
|---|---|
| `STATUS.md` | …know what's done and what's planned |
| `lib/NVPTXArithIntensityPass.cpp` | …understand how the pass walks MIR and produces output |
| `lib/OpClassifier.cpp` | …understand how opcode → FLOPs+precision works |
| `lib/PTX/Classifier.h` | …see the planned shape of inline-PTX support |
| `test/smoke.ll` | …see the simplest possible test |
| `test/fma_v2half.cu` | …see how the CUDA test pipeline works end-to-end |

---

## A final note on framing

This project is a **conservative under-counter** by design. When the
analyzer reports `flops=N global_bytes=B`, it's saying "I can prove the
kernel does at least N FLOPs and reads at least B bytes; the real
numbers may be higher (cache reuse can lower bytes; a known-unknown
opcode contributes zero)." The diagnostic line for unknown opcodes
exists so you can always see what wasn't classified and decide if it
matters.

That's the right epistemic stance for a static analyzer paired with
a measurement tool. The analyzer is for theoretical bounds and
design-space exploration; `ncu` is for ground truth. Neither replaces
the other.

When you contribute, keep that framing visible: it's better to log an
"unknown" and produce conservative numbers than to guess and lie.
