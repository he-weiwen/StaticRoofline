#pragma once

#include <cuda_fp16.h>

// Kernel 9: Autotuned HGEMM — fp16 inputs, fp32 accumulation, fp16 output
// Strided SMEM loading + warp iteration, same structure as fp32 kernel 9.

#define CEIL_DIV_H9(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256)
    hgemm_autotuned(int M, int N, int K, float alpha, half *A, half *B,
                    float beta, half *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr int WM = TM * 16;
    constexpr int WN = TN * 16;
    constexpr int WMITER = CEIL_DIV_H9(BM, WM);
    constexpr int WNITER = CEIL_DIV_H9(BN, WN);

    const int threadCol = threadIdx.x % (WN / TN);
    const int threadRow = threadIdx.x / (WN / TN);

    __shared__ half As[BM * BK];  // transposed: As[k * BM + m]
    __shared__ half Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Strided loading indices — 4 halfs per thread per load
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (256 * 4) / BK;
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = 256 / (BN / 4);

    float threadResults[WMITER * WNITER * TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // Strided load of A (transposed into SMEM)
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            const half *src = &A[(innerRowA + offset) * K + innerColA * 4];
            half v0 = src[0], v1 = src[1], v2 = src[2], v3 = src[3];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = v0;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = v1;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = v2;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = v3;
        }
        // Strided load of B
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            const half *src = &B[(innerRowB + offset) * N + innerColB * 4];
            half *dst = &Bs[(innerRowB + offset) * BN + innerColB * 4];
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
        }
        __syncthreads();

        // Warp iteration + outer products
        for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
            for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
                for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    for (uint i = 0; i < TM; ++i)
                        regM[i] = __half2float(
                            As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i]);
                    for (uint i = 0; i < TN; ++i)
                        regN[i] = __half2float(
                            Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i]);
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
                            threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                                          wnIdx * TN + resIdxN] +=
                                regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Write back
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
        for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
            half *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    int idx = (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN;
                    const int ri = (wmIdx * TM + resIdxM) * (WNITER * TN) +
                                   wnIdx * TN + resIdxN;
                    C_interim[idx] = __float2half(
                        alpha * threadResults[ri] +
                        beta * __half2float(C_interim[idx]));
                }
            }
        }
    }
}
