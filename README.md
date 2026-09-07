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
  blocks (program order; loops are named by their header block):
           block      lines                       instrs  successors            loop
    /----  <block 0>  5_2d_blocktiling.cuh:10-39      10  $L__BB0_5, <block 1>
    |      <block 1>  5_2d_blocktiling.cuh:23-24      74  $L__BB0_2
    |/-->  $L__BB0_2  5_2d_blocktiling.cuh:17-48     155  $L__BB0_3             5_2d_blocktiling.cuh:39 (header)
    ||<->  $L__BB0_3  5_2d_blocktiling.cuh:42-60     111  $L__BB0_3, <block 4>  5_2d_blocktiling.cuh:53 (header, latch)
    |\---  <block 4>  5_2d_blocktiling.cuh:39-62       8  $L__BB0_2, <block 5>  5_2d_blocktiling.cuh:39 (latch)
    | /--  <block 5>  5_2d_blocktiling.cuh:39          1  $L__BB0_6
    \--->  $L__BB0_5  (no line info)                  64  $L__BB0_6
      \->  $L__BB0_6  5_2d_blocktiling.cuh:13-72     412  (end)
  loop with the most instructions (static): 5_2d_blocktiling.cuh:39
  machine @ sm_80 (A100-SXM4-40GB, from target-directive): f32 peak 19.5 TFLOPS
      / 1555 GB/s DRAM = 12.5 flop/B; loop 5_2d_blocktiling.cuh:39 AI(global) = 32 flop/B
  shared memory [static]: 2048 B per CTA
  loop 5_2d_blocktiling.cuh:39 ($L__BB0_2)
    trips = ⌈param_2/8⌉
    per iteration:
      instructions = 1051
        cuda-core f32       512
          fma.rn.f32        512
        integer arithmetic  212
          add.s32            70
          shl.b32            49
          ...
        conversion          128
          cvt.f32.f16       128
        shared load 2 B     128
          ld.shared.u16     128
        ...
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

Generate PTX with `nvcc -ptx -lineinfo kernel.cu`; without
`-lineinfo`, loops are named by label instead of source line.

Every count is static and per thread: what the PTX requests, not what
the hardware moves (a warp-collective instruction contributes its warp
total over the 32 lanes). `<=` marks an upper bound from a conditional
path; whatever cannot be derived is reported as a named unknown. The
machine line is a reference, not a verdict: requested bytes are not
DRAM bytes, since cache reuse and uncoalesced access move the real
figure in either direction. For measured traffic use Nsight Compute.

## Development

`./ci.sh` runs everything: rustfmt, clippy (warnings deny), unit and
corpus tests, and the CLI/acceptance suite (`tests/run.py`, stdlib-only
Python ≥ 3.11). The design, the audience boundary, the anti-scope
list and the execution plan live in `PLAN.md`.
