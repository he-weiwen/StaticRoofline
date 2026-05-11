#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "1_naive.cuh"
#include "2_coalesced.cuh"
#include "3_shared_mem.cuh"
#include "4_1d_blocktiling.cuh"
#include "5_2d_blocktiling.cuh"
#include "6_vectorized.cuh"
#include "9_autotuned.cuh"
#include "11_wmma.cuh"
#include "12_double_buffer.cuh"
#include "13_smem_padded.cuh"
#include "14_ldmatrix_mma.cuh"

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void run_kernel(int kernel_num, int M, int N, int K,
                float alpha, half *A, half *B, float beta, half *C) {
    switch (kernel_num) {
        case 1: {
            dim3 blockDim(32, 32);
            dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
            hgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 2: {
            dim3 blockDim(H_BLOCKSIZE, H_BLOCKSIZE);
            dim3 gridDim(CEIL_DIV(N, H_BLOCKSIZE), CEIL_DIV(M, H_BLOCKSIZE));
            hgemm_coalesced<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 3: {
            dim3 blockDim(HSM_BN, HSM_BM);
            dim3 gridDim(CEIL_DIV(N, HSM_BN), CEIL_DIV(M, HSM_BM));
            hgemm_shared_mem<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 4: {
            const int BM4 = 64, BN4 = 64, BK4 = 8, TM4 = 8;
            dim3 blockDim4((BM4 * BN4) / TM4);
            dim3 gridDim4(CEIL_DIV(N, BN4), CEIL_DIV(M, BM4));
            hgemm_1d_blocktiling<BM4, BN4, BK4, TM4><<<gridDim4, blockDim4>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 5: {
            const int BM5 = 128, BN5 = 128, BK5 = 8, TM5 = 8, TN5 = 8;
            dim3 blockDim5((BM5 * BN5) / (TM5 * TN5));
            dim3 gridDim5(CEIL_DIV(N, BN5), CEIL_DIV(M, BM5));
            hgemm_2d_blocktiling<BM5, BN5, BK5, TM5, TN5><<<gridDim5, blockDim5>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 6: {
            const int BM6 = 128, BN6 = 128, BK6 = 8, TM6 = 8, TN6 = 8;
            dim3 blockDim6((BM6 * BN6) / (TM6 * TN6));
            dim3 gridDim6(CEIL_DIV(N, BN6), CEIL_DIV(M, BM6));
            hgemm_vectorized<BM6, BN6, BK6, TM6, TN6><<<gridDim6, blockDim6>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 9: {
            const int BM9 = 128, BN9 = 128, BK9 = 16, TM9 = 8, TN9 = 8;
            dim3 blockDim9(256);
            dim3 gridDim9(CEIL_DIV(N, BN9), CEIL_DIV(M, BM9));
            hgemm_autotuned<BM9, BN9, BK9, TM9, TN9><<<gridDim9, blockDim9>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 11: {
            // Kernel 11: WMMA tensor core
            // BM=128, BN=128, BK=32, WM=64, WN=64, 4 warps (128 threads)
            // BK=32 gives 2 WMMA K-steps per tile (better compute/load ratio)
            const int BM11=128, BN11=128, BK11=32, WM11=64, WN11=64, NT11=128;
            dim3 blockDim11(NT11);
            dim3 gridDim11(CEIL_DIV(N, BN11), CEIL_DIV(M, BM11));
            hgemm_wmma<BM11, BN11, BK11, WM11, WN11, NT11>
                <<<gridDim11, blockDim11>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 12: {
            // Kernel 12: WMMA + double-buffered cp.async
            const int BM12=128, BN12=128, BK12=32, WM12=64, WN12=64, NT12=128;
            dim3 blockDim12(NT12);
            dim3 gridDim12(CEIL_DIV(N, BN12), CEIL_DIV(M, BM12));
            hgemm_double_buffer<BM12, BN12, BK12, WM12, WN12, NT12>
                <<<gridDim12, blockDim12>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 13: {
            // Kernel 13: WMMA + double-buffer + SMEM padding (anti bank-conflict)
            const int BM13=128, BN13=128, BK13=32, WM13=64, WN13=64, NT13=128;
            dim3 blockDim13(NT13);
            dim3 gridDim13(CEIL_DIV(N, BN13), CEIL_DIV(M, BM13));
            hgemm_smem_padded<BM13, BN13, BK13, WM13, WN13, NT13>
                <<<gridDim13, blockDim13>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        case 14: {
            // Kernel 14: ldmatrix + mma.sync PTX + double-buffer + SMEM padding
            const int BM14=128, BN14=128, BK14=32, WM14=64, WN14=64, NT14=128;
            dim3 blockDim14(NT14);
            dim3 gridDim14(CEIL_DIV(N, BN14), CEIL_DIV(M, BM14));
            hgemm_ldmatrix_mma<BM14, BN14, BK14, WM14, WN14, NT14>
                <<<gridDim14, blockDim14>>>(M, N, K, alpha, A, B, beta, C);
            break;
        }

        default:
            printf("Kernel %d not implemented yet!\n", kernel_num);
            break;
    }
}
