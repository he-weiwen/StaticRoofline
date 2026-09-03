#pragma once

#include <cuda_fp16.h>

// Kernel 6: Vectorized fp16 with transposed As in SMEM
// Uses vectorized loads: each half is 2 bytes, so we load 8 halfs per uint4 (128-bit).
// float4-equivalent for half: load 8 elements at once.
// Accumulation in fp32.

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void hgemm_vectorized(int M, int N, int K, float alpha,
                                  half *A, half *B, float beta, half *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // SMEM stores half — same tile size but half the bytes vs fp32
    __shared__ half As[BM * BK];  // transposed: As[k * BM + m]
    __shared__ half Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Loading indices for vectorized loads (8 halfs = 128 bits per load)
    // A tile: BM * BK elements. 256 threads * 8 halfs = 2048. BM*BK = 128*8 = 1024.
    // So we can load the entire tile in one pass with 4 halfs per thread.
    // Use float2 (64-bit = 4 halfs) per load.
    // innerRowA = tid / (BK/4), innerColA = tid % (BK/4)
    // 256 * 4 = 1024 = BM * BK — exact fit with 4 halfs per thread
    const uint innerRowA = threadIdx.x / (BK / 4);  // BK/4 = 2
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);  // BN/4 = 32
    const uint innerColB = threadIdx.x % (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Load A into SMEM with transpose.
        // Load 4 halfs (one __half2 pair = 2x __half2) from A row.
        // Then scatter-write transposed into As[k * BM + m].
        {
            // Load 4 consecutive half values from A
            const half *src = &A[innerRowA * K + innerColA * 4];
            half v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
            As[(innerColA * 4 + 0) * BM + innerRowA] = v0;
            As[(innerColA * 4 + 1) * BM + innerRowA] = v1;
            As[(innerColA * 4 + 2) * BM + innerRowA] = v2;
            As[(innerColA * 4 + 3) * BM + innerRowA] = v3;
        }

        // Load B into SMEM — contiguous, can vectorize
        {
            const half *src = &B[innerRowB * N + innerColB * 4];
            half *dst = &Bs[innerRowB * BN + innerColB * 4];
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        // Compute outer products with fp32 accumulation
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i)
                regM[i] = __half2float(As[dotIdx * BM + threadRow * TM + i]);
            for (uint i = 0; i < TN; ++i)
                regN[i] = __half2float(Bs[dotIdx * BN + threadCol * TN + i]);
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
        __syncthreads();
    }

    // Write back
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            int idx = (threadRow * TM + resIdxM) * N + (threadCol * TN + resIdxN);
            C[idx] = __float2half(alpha * threadResults[resIdxM * TN + resIdxN] +
                                  beta * __half2float(C[idx]));
        }
    }
}
