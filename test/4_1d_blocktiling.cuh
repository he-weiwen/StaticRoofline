#pragma once

#include <cuda_fp16.h>

// Kernel 4: 1D Blocktiling for fp16
// Each thread computes TM outputs, accumulates in fp32.

template <const int BM, const int BN, const int BK, const int TM>
__global__ void hgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                      const half *A, const half *B, float beta,
                                      half *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int tid = threadIdx.x;
    const int threadCol = tid % BN;
    const int threadRowGrp = tid / BN;

    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerColA = tid % BK;
    const uint innerRowA = tid / BK;
    const uint innerColB = tid % BN;
    const uint innerRowB = tid / BN;

    float threadResults[TM] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            float tmpB = __half2float(Bs[dotIdx * BN + threadCol]);
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] +=
                    __half2float(As[(threadRowGrp * TM + resIdx) * BK + dotIdx]) * tmpB;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        int idx = (threadRowGrp * TM + resIdx) * N + threadCol;
        C[idx] = __float2half(alpha * threadResults[resIdx] +
                              beta * __half2float(C[idx]));
    }
}
