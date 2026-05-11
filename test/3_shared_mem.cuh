#pragma once

#include <cuda_fp16.h>

// Shared memory tiling for fp16. SMEM stores half to save space (2x more tiles).
// Accumulation in float registers.

#define HSM_BM 32
#define HSM_BN 32
#define HSM_BK 32

__global__ void hgemm_shared_mem(int M, int N, int K, float alpha,
                                  const half *A, const half *B, float beta, half *C) {
    __shared__ half As[HSM_BM][HSM_BK];
    __shared__ half Bs[HSM_BK][HSM_BN];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * HSM_BM + ty;
    const int col = bx * HSM_BN + tx;

    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += HSM_BK) {
        if (row < M && (bk + tx) < K)
            As[ty][tx] = A[row * K + (bk + tx)];
        else
            As[ty][tx] = __float2half(0.0f);

        if ((bk + ty) < K && col < N)
            Bs[ty][tx] = B[(bk + ty) * N + col];
        else
            Bs[ty][tx] = __float2half(0.0f);

        __syncthreads();

        for (int k = 0; k < HSM_BK; k++) {
            sum += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(alpha * sum + beta * __half2float(C[row * N + col]));
    }
}
