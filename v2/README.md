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

Counts marked `<=` are upper bounds (code behind data-dependent or
bounds-check branches). Whatever the tool cannot derive it reports as a
*named unknown* with a reason — an unknown is a result, not an error.

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
