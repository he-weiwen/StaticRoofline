#pragma once

#include <cuda_fp16.h>

#define H_BLOCKSIZE 32

__global__ void hgemm_coalesced(int M, int N, int K, float alpha,
                                const half *A, const half *B, float beta, half *C) {
    const int col = blockIdx.x * H_BLOCKSIZE + threadIdx.x;
    const int row = blockIdx.y * H_BLOCKSIZE + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(alpha * sum + beta * __half2float(C[row * N + col]));
    }
}
