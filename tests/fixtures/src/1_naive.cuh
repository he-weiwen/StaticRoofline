#pragma once

#include <cuda_fp16.h>

// Kernel 1: Naive HGEMM — fp16 inputs, fp32 accumulation, fp16 output
// Each thread computes one element of C.

__global__ void hgemm_naive(int M, int N, int K, float alpha,
                            const half *A, const half *B, float beta, half *C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(alpha * sum + beta * __half2float(C[row * N + col]));
    }
}
