# CUTLASS corpus

A curated set of [CUTLASS](https://github.com/NVIDIA/cutlass) examples
exercised against the analyzer to drive coverage closure on the gap
categories surfaced by the canonical-opcode calibration (see
`STATUS.md` and tutorial §7).

## Layout

```
corpus/real/cutlass/
├── README.md           # this file
├── targets.txt         # which CUTLASS examples to compile, with arch
├── build.sh            # compile each → MIR → run analyzer → aggregate report
└── build/              # generated (gitignore): .ll, .mir, per-target reports
```

## Quickstart

```bash
# One-time: shallow-clone CUTLASS as a sibling of nvptx_analyzer
git clone --depth 1 https://github.com/NVIDIA/cutlass.git ~/compilers/cutlass

# Build the analyzer plugin (if you haven't already)
cd ../../..              # nvptx_analyzer/
cmake -S . -B build -DLLVM_DIR=$HOME/compilers/llvm-project/build/lib/cmake/llvm
cmake --build build

# Run the corpus
cd corpus/real/cutlass
./build.sh
```

Configure paths via env vars (defaults shown):

```bash
CUTLASS_PATH=$HOME/compilers/cutlass         \
CUDA_PATH=/usr/local/cuda-13.0               \
LLVM_PATH=$HOME/compilers/llvm-project/build \
ANALYZER_LIB=$HOME/compilers/nvptx_analyzer/build/lib/NVPTXArithIntensity.so \
./build.sh
```

## Current status

10 examples picked spanning Volta → Blackwell tensor cores plus fused MHA.
**8 compile cleanly and produce analyzer output**; 2 are blocked on
toolchain-side issues outside this project's control.

| # | Example | Arch | Status |
|---|---|---|---|
| 00 | basic_gemm | sm_80 | ✓ pass |
| 07 | volta_tensorop_gemm | sm_70 | ✓ pass |
| 08 | turing_tensorop_gemm | sm_75 | ✓ pass |
| 14 | ampere_tf32_tensorop_gemm | sm_80 | ✓ pass |
| 15 | ampere_sparse_tensorop_gemm | sm_80 | ✓ pass |
| 18 | ampere_fp64_tensorop_affine2_gemm | sm_80 | ✗ MIR parser crash |
| 41 | fused_multi_head_attention | sm_80 | ✓ pass (10 kernels) |
| 48 | hopper_warp_specialized_gemm | sm_90a | ✓ pass (4 kernels) |
| 54 | hopper_fp8_warp_specialized_gemm | sm_90a | ✗ IR verifier failure |
| 70 | blackwell_gemm | sm_100a | ⚠ empty (no kernels) |

### Known stuck cases

These are not bugs in our analyzer; they're upstream issues:

- **18 / Ampere FP64** — LLVM's MIR parser asserts on jump-table round-trip:
  ```
  llc: MachineFunction.cpp:1395: createJumpTableIndex:
       Assertion `!DestBBs.empty()' failed.
  ```
  This kernel emits an empty jump table that survives to MIR text but
  fails to re-parse. It's a vanilla LLVM round-trip bug. Workaround:
  none in this project; reportable upstream.

- **54 / Hopper fp8** — clang produces LLVM IR that fails the verifier
  ("input module cannot be verified"). Likely a CUTLASS template
  pattern interacting badly with our `__CUDACC_VER_*` defines. Not
  worth chasing — example 48 already covers Hopper WGMMA + TMA.

- **70 / Blackwell** — clang's CUDA frontend doesn't currently support
  the Blackwell-specific tensor-core intrinsics (`tcgen05.*`), so
  `sm_100a`-gated CUTLASS templates SFINAE-fail and produce zero
  kernels. Will become unblocked when clang adds Blackwell frontend
  support; until then the `tcgen05` family must be exercised by
  hand-written test kernels (Tier 4).

## What this corpus tells us empirically

Aggregate opcode histogram across the eight passing examples (top of
`build/` reports):

| Rank | Opcode | Count | Status in our table |
|---|---|---|---|
| 1 | `INLINEASM` | 3001 | **Gap** (Phase 1) |
| 2 | `ADD64rr` | 1937 | Ignored (correct) |
| 3 | `FMA_F32rrr` | 1390 | Handled |
| 4 | `MOV_B32_r` | 1349 | Ignored (correct) |
| 5 | `FMULf32rr` | 1280 | Handled |
| 6 | `XOR_b32rr` | 1264 | Ignored (correct) |
| 7 | `SHL64_ri` | 1226 | Ignored (correct) |
| 8 | `CBranch` | 1217 | Ignored (correct) |
| 9 | **`LD_GLOBAL_NC_i32`** | **1200** | **Gap (LDG without MMO)** |
| 10 | `CVT_u64_u32` | 838 | Gap (`CVT_*` undecided) |

Two things this confirms with hard numbers:

1. **`INLINEASM` dominates** — 3001 occurrences, more than any other
   opcode. Phase 1 (the inline-PTX parser) is the highest-priority
   gap by raw frequency.
2. **`LD_GLOBAL_NC_*` is not a corner case** — 1200 occurrences in just
   eight CUTLASS examples (driven by `__restrict__` on input
   pointers). The Phase-2 byte-undercount fix has measurable impact.

## Compile flags worth knowing

The `CFLAGS` block in `build.sh` carries three non-obvious requirements:

1. **No `-nocudalib`.** We need the CUDA runtime headers visible so
   `<<<grid,block>>>` host-side launch sites parse cleanly even with
   `--cuda-device-only`. Without them, `cudaConfigureCall` is
   undeclared.
2. **`__CUDACC_VER_MAJOR__=12, MINOR=7`.** NVCC defines these; clang
   doesn't. CUTLASS gates many features on them. Claim 12.7
   specifically — recent enough to enable Hopper/Ampere code paths
   but **below the 12.8 threshold** that activates
   `__nv_atomic_load_n` (an NVCC-only intrinsic clang can't compile).
3. **`-DPFN_cuTensorMapEncodeTiled=PFN_cuTensorMapEncodeTiled_v12000`
   (and `Im2col`)**. CUDA 13's headers versioned these typedef names;
   CUTLASS still uses the unversioned forms. The redirect macro
   bridges the gap.

## Adding a new target

Append a line to `targets.txt`:

```
<example_dir>:<source_file>:<gpu_arch>:<comment>
```

Re-run `./build.sh`. New compile errors usually fall into one of:

- "missing helper.h" → likely a new include path; add to `INCLUDES`.
- "use of undeclared identifier `__CUDACC_VER_*`" → already handled.
- "use of undeclared identifier `cudaConfigureCall`" → re-check that
  `-nocudalib` isn't being added.
- "input module cannot be verified" → clang/CUDA-header interaction
  issue; usually unfixable without picking a different example.

## What this corpus does NOT cover

- **Blackwell tcgen05.** Frontend-blocked; needs hand-written tests.
- **Triton-emitted kernels.** Different path: capture `kernel.asm['llir']`
  from Python. To be added under `corpus/real/triton/` later.
- **PyTorch ATen kernels.** Same shape, different sources. Future
  `corpus/real/aten/`.
- **TMA inline-asm bodies.** Structurally an `INLINEASM` count today;
  the inline-PTX parser is what closes this.
