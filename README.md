# ptxroof

Static roofline analysis for PTX kernels. Point it at a `.ptx` file
(from nvcc, or any producer emitting standard PTX) and it reports, per
loop, the steady-state flops, bytes, and arithmetic intensity — as
*symbolic expressions* over the kernel's parameters, so the answer
holds for every problem size — plus a compute-vs-memory verdict
against real machine tables.

```text
$ ptxroof analyze kernel.ptx
kernel void hgemm_2d_blocktiling<64, 64, 8, 8, 8>(int, int, int, float, ...)
  hot loop: 5_2d_blocktiling.cuh:39
  verdict @ sm_80 (A100-SXM4-40GB, from target-directive): compute-bound —
      loop 5_2d_blocktiling.cuh:39 AI(global) 32 flop/B vs f32 knee 12.5
  shared memory [static]: 2048 B per CTA
  loop 5_2d_blocktiling.cuh:39 ($L__BB0_2)
    trips = ceildiv(param_2, 8)
    per iteration:
      flops = 1024  (f32 1024)
      global bytes: load 32 B, store 0 B
      AI(global) = 32 flop/B
    ...
```

## Install

```sh
cargo install --path .       # from this directory; needs stable Rust
```

## Usage

```sh
ptxroof analyze kernel.ptx                 # text report
ptxroof analyze kernel.ptx --json          # the same result tree as JSON
ptxroof analyze kernel.ptx --arch sm_80 --arch sm_86
                                           # verdicts on chosen parts
ptxroof analyze kernel.ptx --bind 2:K=4096 # numeric columns: bind kernel
                                           # param 2 (positional) to 4096
ptxroof analyze kernel.ptx --launch 16,16,1  # per-CTA totals
ptxroof analyze kernel.ptx --dump-ast      # parsed module, canonical PTX
```

Generate PTX with `nvcc -ptx -lineinfo kernel.cu` — `-lineinfo` is what
lets loops be named by source line instead of by label.

Everything the tool prints is labeled `[static]`: these are the
*requested* counts — the algorithm's demand as written in the PTX — not
measured traffic. ptxas may add spills the PTX never shows, and the
memory system moves more bytes than requested (uncoalesced overfetch)
or fewer (cache reuse). A static number is a bound and a design check,
not a measurement; for measured DRAM traffic and achieved rates, use
Nsight Compute.

The per-kernel `shared memory [static]` line is the same kind of
figure: the bytes the kernel declares in `.shared`, which is exactly
what ptxas reports as `bytes smem` and Nsight Compute as
`launch__shared_mem_per_block_static`. The CUDA driver's own reserved
shared memory and any launch-sized dynamic allocation (flagged
`+ dynamic`) are not included — the dynamic amount is knowable only at
launch.

Verdicts inherit this contract: "memory-bound" means the *requested*
AI sits below the machine's knee — a no-reuse worst case, since the
knee divides peak compute by DRAM bandwidth. Both halves of that
sentence were validated on hardware (RTX 4090, the k5 fixture at
4096³, Nsight Compute): the requested flop and byte counts matched the
hardware counters to the digit, yet the kernel ran at 37 TFLOP/s —
above its no-reuse ceiling of AI × bandwidth ≈ 32 TFLOP/s — because
the 72 MB L2 absorbed every re-read and DRAM moved only the compulsory
matrix bytes, at 3% of peak. Once the working set outgrows L2, the
ceiling is real again. The demand side is this tool's half of the
story; what the memory hierarchy does with the demand is Nsight
Compute's.

Flops are reported in three tables by the unit that runs them: `flops`
(CUDA cores), `tensor flops` (`wmma.mma`, `mma.sync`) and `sfu flops`
(`ex2`, `rsqrt`, `div.rn`, ... — one flop per result). AI(global)
counts all three; the verdict compares against the peak of whichever
bucket dominates, so a tensor-core GEMM is judged against the part's
tensor peak ("vs f16 tensor knee 200.6"), and every count is per
thread — a warp-collective instruction contributes its warp total
divided by the 32 lanes. `cp.async` is recorded on both sides, a global
read and a shared write. Every peak in `data/machine/*.toml` cites the
NVIDIA document it came from.

Counts marked `<=` are upper bounds (code behind data-dependent or
bounds-check branches). Whatever the tool cannot derive it reports as a
*named unknown* with a reason — an unknown is a result, not an error.
That includes statements the parser cannot read, integer, sparse and
block-scaled MMA kinds, and the Hopper bulk/TMA copies, which are
counted by name rather than guessed.

## Audience boundary

The target is regular/tiled kernels — GEMM, conv, stencils, attention —
from mainstream producers (nvcc, clang, Triton). Sparse/irregular
kernels whose loop bounds are data-dependent (CSR SpMV being the
canonical case) get honest, labeled unknowns; that is designed
behavior, not a gap awaiting a feature.

## Anti-scope

The maintenance budget. Each item trades "more useful in rare cases"
against "permanently more surface," and under this project's goals the
trade always resolves the same way. Revisiting any item means editing
this list in the same change that implements it:

- cache-reuse modeling, divergence, bank conflicts, occupancy/latency
  analysis — NCU owns those; we point at it;
- branch-probability modeling — conditional blocks carry `≤` bounds,
  never probabilities;
- SCEV-equivalent generality — trip counts come from a fixed shape
  catalog, grown demand-driven via tested coverage minimums;
- symbolic series for triangular nests — a note, never a solver;
- guard-implication analysis for loop versioning — variants are
  reported side by side, never auto-resolved;
- resolution of data-dependent bounds;
- SASS instruction semantics beyond line-join + `LDL`/`STL`/resource
  counting;
- `cvta`-provenance refinement of generic addressing — deferred until a
  fixture actually emits generic loads.

## Development

`./ci.sh` runs everything: rustfmt, clippy (warnings deny), unit and
corpus tests, and the CLI/acceptance suite (`tests/run.py`, stdlib-only
Python ≥ 3.11). The full design and execution plan live in `PLAN.md`.
