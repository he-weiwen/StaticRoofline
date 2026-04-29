// CHECK: kernel fma_v2half_kernel
// CHECK: bb.0
// CHECK-SAME: flops=4
// CHECK-SAME: flops_f16=4
// CHECK-SAME: flops_f32=0
// CHECK-SAME: global_bytes=16
// CHECK-SAME: local_bytes=0
// CHECK-SAME: ai=0.250000

// CUDA kernel that exercises the NVPTX packed-half FMA opcode (FMA_F16x2rrr).
//
// We use clang's ext_vector_type(2) to get a native <2 x half> type, then
// rely on the fp-contract=fast flag so that "a*b + c" fuses into a packed
// FMA. This bypasses the cuda_fp16.h __hfma2 intrinsic which lowers to
// inline asm — which our static analyzer cannot see inside.
//
// We deliberately DO NOT use __restrict__ on the loads. With __restrict__,
// clang emits the LDG (read-only / non-coherent) form, which lowers to
// LD_GLOBAL_NC_i32 in MIR. That opcode family carries no MachineMemOperand
// in the NVPTX backend, so our MMO-based byte accounting misses it. The
// non-restrict form lowers to plain LD_GLOBAL_i32 with a proper MMO and
// gets bucketed correctly. Filed: LDG byte accounting needs an
// opcode-driven fallback when MMOs are absent.
//
// Per thread: one FMA_F16x2rrr (4 FLOPs in f16) and 4 global accesses of
// 4 bytes each (3 loads + 1 store of v2half = i32 width).
// Expected: flops=4 flops_f16=4 global_bytes=16 ai=0.25.

typedef _Float16 v2half __attribute__((ext_vector_type(2)));

extern "C" __global__ void fma_v2half_kernel(
        const v2half* a,
        const v2half* b,
        const v2half* c,
        v2half* d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx] + c[idx];
}
