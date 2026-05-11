// Regression test: kernel using __hfma2 from cuda_fp16.h, which lowers
// to INLINEASM "fma.rn.f16x2 %0,%1,%2,%3;". Before the inline-PTX parser
// landed, this kernel reported flops=0. Now it should report flops=4
// (one packed FMA × 2 lanes × FMA-counts-as-2).
//
// CHECK: kernel hfma2_kernel
// CHECK: bb.0
// CHECK-SAME: flops=4
// CHECK-SAME: flops_f16=4
// CHECK-SAME: flops_f32=0

#include <cuda_fp16.h>

extern "C" __global__ void hfma2_kernel(const __half2* a,
                                         const __half2* b,
                                         const __half2* c,
                                         __half2* d) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = __hfma2(a[idx], b[idx], c[idx]);
}
