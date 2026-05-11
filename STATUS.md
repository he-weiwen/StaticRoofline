# nvptx_analyzer — Status

A static arithmetic-intensity analyzer for NVPTX, implemented as an
out-of-tree LLVM `MachineFunctionPass`. Loaded into `llc` via `-load`,
runs against MIR dumped with `-stop-before=nvptx-asm-printer`.

## Done

### Build & infrastructure
- `MachineFunctionPass` registered as `ptx-ai`, loaded by `llc -load …`.
- Built against local LLVM 23 trunk
  (`/home/whe302/compilers/llvm-project/build`).
- Top-level + `lib/` + `test/` CMake; `add_ptxai_test` and
  `add_ptxai_cuda_test` helpers + `run-test.sh` shell wrapper to drive
  FileCheck without CMake's `bash -c` escaping issues.
- Aggregator target `check` runs all tests; `smoke-test` kept as alias
  for backward compatibility.

### Analysis features (MIR side)
- Per-MachineBasicBlock and per-kernel statistics: instruction count,
  per-opcode histogram, FLOPs, memory bytes per address space.
- FLOP detector (prefix match): `FADD`, `FSUB`, `FMUL`, `FMA`. Handles
  both LLVM 18 (`FMA32rrr`, bare width) and LLVM 23 (`FMA_F32rrr`,
  underscored) opcode forms.
- Precision split into `flops_f16` / `flops_bf16` / `flops_f32` /
  `flops_f64` / `flops_other` via case-insensitive substring + bare-
  width fallback for FMA.
- Packed-vector lane multiplier: `f16x2` / `bf16x2` / `f32x2` ⇒ 2× FLOPs
  per instruction. Defensive substring (`f16x2` etc.) rather than raw
  `x2`.
- Memory accounting via `MachineMemOperand` size + addrspace, bucketed
  into global / shared / local / const / param / unknown, split by
  load vs store. Atomics counted as both.
- **Opcode-name fallback** (`parseMemoryOpcodeName` in `OpClassifier`)
  for NVPTX opcode families that deliberately ship without
  `MachineMemOperand`s — the `LD_GLOBAL_NC_*` (LDG) and `LDU_GLOBAL_*`
  read-only-cache families. Recognizes the full prefix taxonomy
  (LD/ST × GLOBAL/SHARED/LOCAL/CONST/PARAM × scalar/v2/v4/v8 × every
  width). This is **not a workaround for an upstream bug**: the LDG
  family's lack of `mayLoad` / MMOs is intentional NVPTX design
  (D17471, D112466) — they're modeled as constant-materialization
  for optimization purposes. Our analyzer needs to know they move
  bytes, so we recover that target-side.
- AI denominator is `global_bytes` only. `local_bytes` reported as a
  separate diagnostic field (visible if non-zero, not folded into AI).
- `InvocationScope` (`PerThread` / `PerWarp` / `PerCTA`) baked into
  `OpClass`. Aggregator asserts `PerThread` today; assertion will fire
  when MMA emits `PerWarp`, signalling time to extend per-scope buckets.

### Tests (6, all green on LLVM 23)
| Test            | What it pins                                         | flops | precision | global_bytes | AI       |
|-----------------|------------------------------------------------------|-------|-----------|--------------|----------|
| `smoke.ll`      | f32 vector add baseline                              | 1     | f32       | 12           | 0.083333 |
| `fsub.ll`       | `fsub` counts as 1 FLOP                              | 1     | f32       | 12           | 0.083333 |
| `add_f64.ll`    | precision bucket isolation                           | 1     | f64       | 24           | 0.041667 |
| `fma.ll`        | scalar f32 FMA = 2 FLOPs                             | 2     | f32       | 16           | 0.125    |
| `local.ll`      | alloca w/ dynamic index ⇒ local-mem accounting       | 1     | f32       | 12 (+8 loc)  | 0.083333 |
| `fma_v2half.cu` | real CUDA: `FMA_F16x2rrr` = 4 FLOPs (lane multiplier)| 4     | f16       | 16           | 0.25     |

### Refactor / scaffold
- `lib/OpClassifier.{h,cpp}` extracted from monolithic pass file.
  Handles MIR opcode classification; pure function on `StringRef`.
- `lib/PTX/` subdirectory scaffolded for inline-PTX handling:
  - `Tokenizer.{h,cpp}` — token kinds + `tokenize()` (stub).
  - `Parser.{h,cpp}` — `Operand` `std::variant`, `Stmt`, `parse()`
    (stub).
  - `Classifier.{h,cpp}` — `OpClass` `std::variant` covering FlopOp,
    MMAOp, MemoryOp, AsyncCopy, LdMatrix, WarpSync, Barrier, Ignore,
    Unknown. `classify()` (stub returns Unknown).
- Pass integration point marked at `MI.isInlineAsm()` branch with
  TODO; behavior unchanged until parser/classifier land.

### Verified findings
- NVPTX backend itself never emits inline asm: zero `INLINEASM`
  references in any `lib/Target/NVPTX/*.td`. Every `llvm.nvvm.*`
  intrinsic lowers via TableGen patterns to a regular MachineInstr.
- The CUDA-shipped inline-asm universe is bounded:
  `cccl/cuda/__ptx/instructions/generated/` ships ~60 auto-generated
  headers covering essentially every modern PTX instruction (TMA,
  mbarrier, tcgen05, multimem, async copies, fences). Each wrapper
  carries its canonical PTX form in a doc comment — machine-readable.
- `cuda_fp{16,bf16,fp8,fp6,fp4}.hpp` use macro-driven inline asm.
  Bounded, predictable, scrapeable.
- `__hfma2` (and similar `cuda_fp16.h` operators) lower to inline asm
  ⇒ invisible to MIR classifier ⇒ requires the PTX parser.
- `__restrict__` triggers the LDG path. `LD_GLOBAL_NC_*` opcodes
  **deliberately** lack `MachineMemOperand`s and have `mayLoad = 0`
  in their MCInstrDesc — this is intentional upstream design (D17471
  in 2016 and D112466 in 2023), documented at
  `llvm/lib/Target/NVPTX/NVPTXIntrinsics.td:2786-2788`. NVPTX models
  these as constant-materialization rather than loads because the
  hardware texture-cache path guarantees the data is read-only for
  the kernel lifetime, which lets MachineLICM/Sink/Scheduler reorder
  them freely. Our analyzer correctly recovers byte traffic via an
  opcode-name fallback (`parseMemoryOpcodeName`) — not by fighting
  the upstream design.
- PTX modifiers use both `.` and `::` separators
  (e.g. `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`).
  Tokenizer must allow `::` inside identifiers.

## To do

### Phase 1 — inline-PTX parser (next)
1. **`tools/cccl-scraper`** — walk CCCL `generated/*.h`, extract canonical
   PTX from doc comments + asm template strings, output
   `test/ptx-corpus.json`. Run once per CUDA version, commit result.
2. **Tokenizer** — full grammar including `::`-bearing identifiers,
   register refs, operand refs, brackets, braces, predicate prefix.
3. **Parser** — recursive descent on tokens; produce `Stmt` array;
   per-statement error recovery via `parseError` flag.
4. **Classifier** — per-family handlers, in priority order:
   1. FP arithmetic from `cuda_fp{16,bf16}.hpp`
   2. `ld.*` / `st.*` (CCCL `ld.h` / `st.h`)
   3. `cp.async.*` and TMA family
   4. `mma` / `wmma` / `wgmma`
   5. `tcgen05.*` (Blackwell)
   6. `mbarrier`, `fence`, `bar.*`, `red.async`, `multimem.*`
   7. `shfl`, `vote`, `match`, `prmt`, `bfind`, `bmsk`, `shr`, `shl`,
      `getctarank`, `get_sreg.*`, `cvta`, `exit`, `trap`
5. **Corpus coverage test** — assert every entry in `ptx-corpus.json`
   classifies to non-`Unknown` OR is in the explicit `Ignore` allow-
   list. Failing target is the gate for "phase 1 done".
6. **Wire into pass** at the existing `MI.isInlineAsm()` TODO marker.
   Aggregator dispatches `OpClass` variant arms via `std::visit` into
   the existing `BlockStats` counters.

### Phase 2 — memory accounting gaps
- ~~**LDG opcode-name fallback**~~: **DONE.** `LD_GLOBAL_NC_*` /
  `LDU_GLOBAL_*` etc. deliberately lack MMOs (see Verified findings
  above for the rationale and upstream citations). Recovered via
  `parseMemoryOpcodeName` in `OpClassifier.{h,cpp}` covering all
  documented LD/ST opcode families. Empirically validated against
  the CUTLASS corpus: aggregate `global_bytes` increased by exactly
  1200 × 4 = 4800 bytes, matching the histogram count of
  `LD_GLOBAL_NC_i32` instances.
- Atomic byte accounting refinement (RMW semantics — currently OK as
  load+store, worth a comment).

### Phase 3 — tensor cores (MIR side)
- Extend `OpClassifier` for `MMA_*` / `WMMA_*` / `WGMMA_*` MIR opcodes
  with per-arch shape table (FLOPs = 2·M·N·K).
- Replace `assert(scope == PerThread)` in `addFlops` with per-scope
  buckets: `flops_per_thread_*` vs `flops_per_warp_*`.
- Mirrors the PTX-side MMA handler so both paths converge.

### Phase 4 — loop / SCEV attribution
- IR-level pre-pass: annotate `Loop`s with `BackedgeTakenCount` from
  `ScalarEvolution` as `!ptxai.trip_count` metadata.
- MIR-level consumption: read metadata via `MachineLoop::getLoopID()`,
  fall back to `Unknown`/`Lost` when codegen split or duplicated blocks.
- Replace flat `BlockStats` with hierarchical `RegionStats` tree:
  kernel → loop → BB.
- Symbolic count expressions parametric in shape variables.

### Phase 5 — output / tooling
- JSON output mode (probably gated by `-ptx-ai-json` cl::opt).
- Reporter abstraction so TTY pretty / JSON / roofline-export can
  coexist.
- Python roofline engine (separate project): consumes JSON +
  per-arch machine table, plots roofline, optionally cross-references
  NCU CSV for measured vs theoretical.

## Open / triage

- **Unify `ptxai::OpClass` (struct, MIR side) with `ptxai::ptx::OpClass`
  (variant, PTX side)?** Defer until MMA support lands on either side
  forces the issue. Keep the two distinct for now; aggregator routes
  into the same `BlockStats` counters via small adapters.
- **`Memory::base` storage**: currently `std::shared_ptr<Operand>` for
  recursive variant. Switch to an arena/handle scheme if profiling
  later shows it matters; not worth solving up front.
- **CUDA-version pinning for `ptx-corpus.json`**: commit the file vs
  regenerate at build time. Leaning toward commit-and-regenerate-on-
  toolkit-upgrade for reproducibility (avoids runtime Python
  dependency).
- ~~**`fma_v2half.cu` test**~~: now uses `__restrict__` again (since
  the LDG opcode-name fallback landed). Same `flops=4 global_bytes=16
  ai=0.25` as the non-`__restrict__` version. Plus a dedicated
  `ldg_restrict.cu` integration test exercising the LDG path.
- **FA4-paper-level roofline**: explicitly distinguishes static-
  requested vs measured-transferred vs achieved. This tool produces
  the first; pair with NCU for the others. Naming convention in
  output should keep the distinction visible.

## Known structural limitations

These don't have fixes; they're properties of the chosen analysis level.

- **ptxas spills are invisible.** NVPTX in LLVM uses virtual registers
  through to PTX text; real register allocation and spills happen in
  closed-source `ptxas` downstream of our analyzer. SASS-level analysis
  (via `cuobjdump --dump-sass`) is the only way to see them.
- **`global_bytes` is requested, not transferred.** No cache-reuse
  model. Compulsory bytes ≤ our number ≤ ncu-measured DRAM bytes;
  exact only with an idealized cache.
- **Coalescing and divergence are unknown statically.** Counts assume
  fully-converged warps and perfect coalescing. Deviations require
  symbolic dataflow (coalescing) or measurement (divergence).
- **AI is not a single number.** Per memory level, per shape, per
  launch config, per architecture. Symbolic per-region output is the
  honest framing; a scalar AI is a shortcut.

## Repository layout

```
nvptx_analyzer/
├── STATUS.md                         # this file
├── CMakeLists.txt
├── lib/
│   ├── CMakeLists.txt
│   ├── NVPTXArithIntensityPass.cpp   # the pass; INLINEASM TODO marker inside
│   ├── OpClassifier.{h,cpp}          # MIR opcode classification
│   └── PTX/
│       ├── Tokenizer.{h,cpp}         # stub
│       ├── Parser.{h,cpp}            # stub
│       └── Classifier.{h,cpp}        # stub
└── test/
    ├── CMakeLists.txt                # add_ptxai_test, add_ptxai_cuda_test
    ├── run-test.sh                   # FileCheck driver
    ├── smoke.ll
    ├── fsub.ll
    ├── add_f64.ll
    ├── local.ll
    ├── fma.ll
    └── fma_v2half.cu
```
