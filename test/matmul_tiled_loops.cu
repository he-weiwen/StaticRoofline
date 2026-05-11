// CHECK: hgemm_1d_blocktiling
// CHECK: exec_count=L
//
// CHECK: hgemm_2d_blocktiling
// CHECK: exec_count=L
//
// CHECK: hgemm_vectorized
// CHECK: exec_count=L
//
// CHECK: hgemm_autotuned
// CHECK: exec_count=L
//
// Explicit template instantiations for the scalar/register-tiled matmul
// variants. These exercise nested machine loops and larger unrolled bodies
// without depending on WMMA, cp.async, or inline mma.sync accounting.

#include "4_1d_blocktiling.cuh"
#include "5_2d_blocktiling.cuh"
#include "6_vectorized.cuh"
#include "9_autotuned.cuh"

template __global__ void hgemm_1d_blocktiling<64, 64, 8, 8>(
    int, int, int, float, const half *, const half *, float, half *);

template __global__ void hgemm_2d_blocktiling<128, 128, 8, 8, 8>(
    int, int, int, float, const half *, const half *, float, half *);

template __global__ void hgemm_vectorized<128, 128, 8, 8, 8>(
    int, int, int, float, half *, half *, float, half *);

template __global__ void hgemm_autotuned<128, 128, 16, 8, 8>(
    int, int, int, float, half *, half *, float, half *);
