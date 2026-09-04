# ptxroof

Static roofline analysis for PTX kernels. Point it at a `.ptx` file
(from nvcc, or any producer emitting standard PTX) and it reports, per
loop, the steady-state flops, bytes, and arithmetic intensity — as
*symbolic expressions* over the kernel's parameters, so the answer
holds for every problem size — next to what a real part can sustain,
its peak over its DRAM bandwidth, with both numbers cited.

```text
$ ptxroof analyze kernel.ptx
kernel void hgemm_2d_blocktiling<64, 64, 8, 8, 8>(int, int, int, float, ...)
  heaviest loop (static weight): 5_2d_blocktiling.cuh:39
  machine @ sm_80 (A100-SXM4-40GB, from target-directive): f32 peak 19.5 TFLOPS
      / 1555 GB/s DRAM = 12.5 flop/B; loop 5_2d_blocktiling.cuh:39 AI(global) = 32 flop/B
  shared memory [static]: 2048 B per CTA
  loop 5_2d_blocktiling.cuh:39 ($L__BB0_2)
    trips = ceildiv(param_2, 8)
    per iteration:
      instructions = 1051
        workload: cuda-core f32 512, global load 2 B 16, shared load 2 B 128, shared store 2 B 16
        bookkeeping: compare / select 9, control 9, conversion 128, integer arithmetic 212, synchronization 2
        register moves (mostly removed by ptxas): 19
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
                                           # machine lines for chosen parts
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

The machine line inherits this contract, which is why it is a reference
number and not a verdict. It divides a part's peak by its DRAM
bandwidth, the flop/B above which the part is limited by its peak
rather than by memory; the loop's AI divides flops by *requested* bytes. Comparing
the two assumes every requested byte is a DRAM byte exactly once, and
that fails in both directions: uncoalesced access moves more, cache
reuse moves less. The second failure was measured on hardware (RTX
4090, the k5 fixture at 4096³, Nsight Compute): the requested flop and
byte counts matched the hardware counters to the digit, yet the kernel
ran at 37 TFLOP/s — above the AI × bandwidth ≈ 32 TFLOP/s a no-reuse
reading would allow — because the 72 MB L2 absorbed every re-read and
DRAM moved only the compulsory matrix bytes, at 3% of peak. The demand
side is this tool's half of the story; what the memory hierarchy does
with the demand is Nsight Compute's.

Every loop also lists the instructions it issues per iteration, by
kind, with memory kinds carrying their access width. That is a static
proxy for issue pressure (a warp scheduler issues one instruction per
cycle) and the cheapest coalescing signal there is (128 two-byte
shared loads per tile is a finding). They are PTX counts, not SASS:
register moves are mostly removed by ptxas, which is why they sit on
their own line, and one `.rn` divide becomes many machine instructions.

Flops are reported in three tables by the unit that runs them: `flops`
(CUDA cores), `tensor flops` (`wmma.mma`, `mma.sync`) and `sfu flops`
(`ex2`, `rsqrt`, `div.rn`, ... — one flop per result). AI(global)
counts all three; the machine line uses the peak of whichever bucket
dominates, so a tensor-core GEMM is shown against the part's tensor
peak ("f16 tensor peak 312 TFLOPS / 1555 GB/s DRAM = 200.6 flop/B"), and every count is per
thread — a warp-collective instruction contributes its warp total
divided by the 32 lanes. `cp.async` is recorded on both sides, a global
read and a shared write. Every peak in `data/machine/*.toml` cites the
NVIDIA document it came from.

Counts marked `<=` are upper bounds (code behind data-dependent or
bounds-check branches). A ratio of a bound is itself a bound in one
direction only: `AI(global) >= 64` means exact flops over bytes that
are an upper bound, and when both sides are bounds no AI is printed at
all. Whatever the tool cannot derive it reports as a
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
