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
// __restrict__ on the inputs triggers the LDG (read-only / non-coherent)
// path in NVPTX. Those opcodes (LD_GLOBAL_NC_*) carry no MachineMemOperand,
// so the MMO-walking path drops their bytes — we now recover them via the
// opcode-name fallback in OpClassifier::parseMemoryOpcodeName. Re-adding
// __restrict__ here is the regression check: byte counts must match the
// non-__restrict__ version.
//
// Per thread: one FMA_F16x2rrr (4 FLOPs in f16) and 4 global accesses of
// 4 bytes each (3 loads + 1 store of v2half = i32 width).
// Expected: flops=4 flops_f16=4 global_bytes=16 ai=0.25.

typedef _Float16 v2half __attribute__((ext_vector_type(2)));

extern "C" __global__ void fma_v2half_kernel(
        const v2half* __restrict__ a,
        const v2half* __restrict__ b,
        const v2half* __restrict__ c,
        v2half* __restrict__ d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx] + c[idx];
}
