#pragma once

#include <cuda_fp16.h>

// Kernel 5: 2D Blocktiling for fp16
// Each thread computes TM x TN outputs via outer product.
// SMEM stores half, accumulation in fp32.

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void hgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                      const half *A, const half *B, float beta,
                                      half *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    const uint numThreads = BM * BN / (TM * TN);

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreads / BK;

    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreads / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i)
                regM[i] = __half2float(As[(threadRow * TM + i) * BK + dotIdx]);
            for (uint i = 0; i < TN; ++i)
                regN[i] = __half2float(Bs[dotIdx * BN + threadCol * TN + i]);
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int idx = (threadRow * TM + resIdxM) * N + (threadCol * TN + resIdxN);
            C[idx] = __float2half(alpha * threadResults[resIdxM * TN + resIdxN] +
                                  beta * __half2float(C[idx]));
        }
    }
}
