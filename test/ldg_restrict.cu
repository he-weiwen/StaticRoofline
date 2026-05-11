// Regression test for the LDG (read-only / non-coherent) byte-undercount
// bug. Using __restrict__ on input pointers triggers clang/NVPTX to emit
// LD_GLOBAL_NC_* opcodes, which historically carried no MachineMemOperand
// and were silently dropped from byte accounting.
//
// Same kernel shape as smoke.ll (c[i] = a[i] + b[i] in fp32):
//   - 2 reads × 4 bytes via LDG    (now counted via opcode-name fallback)
//   - 1 write × 4 bytes via plain ST_i32 (still counted via MMO)
//   - 1 fadd_f32                   = 1 FLOP
// Expected: flops=1, global_bytes=12, ai=1/12 = 0.083333.
// Pre-fix: global_bytes=4 (LDG loads invisible), ai=0.25.
//
// CHECK: kernel ldg_restrict_kernel
// CHECK: bb.0
// CHECK-SAME: flops=1
// CHECK-SAME: flops_f32=1
// CHECK-SAME: global_bytes=12
// CHECK-SAME: ai=0.083333

extern "C" __global__ void ldg_restrict_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        float* __restrict__ c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
