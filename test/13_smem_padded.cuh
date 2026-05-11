#pragma once

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>

using namespace nvcuda;

/*
 * Kernel 13: WMMA + Double-Buffer cp.async + SMEM Padding
 *
 * Builds on kernel 12 by adding padding to SMEM rows to eliminate bank
 * conflicts during wmma::load_matrix_sync.
 *
 * Bank conflict analysis (kernel 11/12, no padding):
 *   As: BK=32 halfs/row = 64 bytes = 16 banks. Rows 0,2,4,... all hit banks 0-15.
 *       wmma::load_matrix_sync reads 16 rows → 16/2 = 8-way conflict minimum.
 *       ncu measured 28.8-way average across 520M conflicts.
 *
 *   Bs: BN=128 halfs/row = 256 bytes = 64 banks, wraps to all 32 banks × 2.
 *       Every row starts at bank 0. Similar conflict pattern.
 *
 * Fix: Add PAD halfs to each row.
 *   As: stride = BK + PAD_K instead of BK. PAD_K = 8 halfs = 16 bytes = 4 banks.
 *       Now consecutive rows shift by 4 banks, so 8 consecutive rows cover all 32 banks.
 *   Bs: stride = BN + PAD_N. PAD_N = 8 halfs = 16 bytes = 4 banks.
 *       Same idea for B fragment loads.
 *
 * wmma::load_matrix_sync takes a leading dimension argument, so we just pass
 * the padded stride — no other code changes needed.
 *
 * SMEM increase:
 *   As: BM * (BK + PAD_K) * 2 = 128 * 40 * 2 = 10,240 bytes per stage
 *   Bs: BK * (BN + PAD_N) * 2 = 32 * 136 * 2 = 8,704 bytes per stage
 *   Per stage: 18,944 bytes. Double buffer: 37,888 bytes (~37 KB).
 *   Still fits: 101 KB optin SMEM, allows 2 blocks/SM.
 */

constexpr int W13_WMMA_M = 16;
constexpr int W13_WMMA_N = 16;
constexpr int W13_WMMA_K = 16;

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int NUM_THREADS,
          const int PAD_K = 8, const int PAD_N = 8>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_smem_padded(int M, int N, int K, float alpha,
                      const half *A, const half *B,
                      float beta, half *C) {

    // Padded strides for SMEM
    constexpr int LDA_S = BK + PAD_K;  // leading dim of As in SMEM (padded)
    constexpr int LDB_S = BN + PAD_N;  // leading dim of Bs in SMEM (padded)

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint warpIdx = threadIdx.x / 32;
    constexpr uint WARPS_PER_ROW = BN / WN;
    const uint warpRow = warpIdx / WARPS_PER_ROW;
    const uint warpCol = warpIdx % WARPS_PER_ROW;

    constexpr uint WMMA_TILES_M = WM / W13_WMMA_M;
    constexpr uint WMMA_TILES_N = WN / W13_WMMA_N;

    wmma::fragment<wmma::accumulator, W13_WMMA_M, W13_WMMA_N, W13_WMMA_K, float>
        acc[WMMA_TILES_M][WMMA_TILES_N];
    for (uint m = 0; m < WMMA_TILES_M; ++m)
        for (uint n = 0; n < WMMA_TILES_N; ++n)
            wmma::fill_fragment(acc[m][n], 0.0f);

    // Double-buffered SMEM with padding
    __shared__ half As[2][BM * LDA_S];  // As[buf][m * LDA_S + k]
    __shared__ half Bs[2][BK * LDB_S];  // Bs[buf][k * LDB_S + n]

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Loading indices: 8 halfs per cp.async (16 bytes)
    // For A: load along K dimension (BK=32 halfs per row).
    //   Each thread loads 8 halfs. BK/8 = 4 groups per row.
    //   tid maps to: row = tid / (BK/8) = tid / 4, col = (tid % 4) * 8
    //   rowStride = NUM_THREADS / (BK/8) = 128 / 4 = 32
    //   Passes needed: BM / 32 = 4
    const uint tid = threadIdx.x;
    const uint a_innerRow = tid / (BK / 8);
    const uint a_innerCol = (tid % (BK / 8)) * 8;
    constexpr uint a_rowStride = NUM_THREADS / (BK / 8);

    // For B: load along N dimension (BN=128 halfs per row).
    //   BN/8 = 16 groups per row. tid / 16 = row, tid % 16 = col group.
    //   rowStride = 128 / 16 = 8.  Passes: BK / 8 = 4.
    const uint b_innerRow = tid / (BN / 8);
    const uint b_innerCol = (tid % (BN / 8)) * 8;
    constexpr uint b_rowStride = NUM_THREADS / (BN / 8);

    const uint num_k_tiles = K / BK;

    // Load tile into padded SMEM buffer
    auto load_tile = [&](uint buf, uint tile_idx) {
        const half *A_tile = A + tile_idx * BK;
        const half *B_tile = B + tile_idx * BK * N;

        // Load A: 8 halfs per cp.async, write to padded layout
        for (uint offset = 0; offset < BM; offset += a_rowStride) {
            uint m = a_innerRow + offset;
            uint k = a_innerCol;
            // SMEM dest uses padded stride
            __pipeline_memcpy_async(
                &As[buf][m * LDA_S + k],
                &A_tile[m * K + k],
                16);
        }

        // Load B: 8 halfs per cp.async, write to padded layout
        for (uint offset = 0; offset < BK; offset += b_rowStride) {
            uint kk = b_innerRow + offset;
            uint n = b_innerCol;
            __pipeline_memcpy_async(
                &Bs[buf][kk * LDB_S + n],
                &B_tile[kk * N + n],
                16);
        }

        __pipeline_commit();
    };

    // Compute using padded SMEM — pass padded stride as leading dimension
    auto compute_tile = [&](uint buf) {
        for (uint kTile = 0; kTile < BK; kTile += W13_WMMA_K) {
            for (uint wm = 0; wm < WMMA_TILES_M; ++wm) {
                wmma::fragment<wmma::matrix_a, W13_WMMA_M, W13_WMMA_N, W13_WMMA_K,
                               half, wmma::row_major> a_frag;
                // As[m * LDA_S + k], leading dim = LDA_S
                wmma::load_matrix_sync(a_frag,
                    &As[buf][(warpRow * WM + wm * W13_WMMA_M) * LDA_S + kTile],
                    LDA_S);

                for (uint wn = 0; wn < WMMA_TILES_N; ++wn) {
                    wmma::fragment<wmma::matrix_b, W13_WMMA_M, W13_WMMA_N, W13_WMMA_K,
                                   half, wmma::row_major> b_frag;
                    // Bs[k * LDB_S + n], leading dim = LDB_S
                    wmma::load_matrix_sync(b_frag,
                        &Bs[buf][kTile * LDB_S + warpCol * WN + wn * W13_WMMA_N],
                        LDB_S);

                    wmma::mma_sync(acc[wm][wn], a_frag, b_frag, acc[wm][wn]);
                }
            }
        }
    };

    // Prologue: pre-load 2 tiles
    load_tile(0, 0);
    if (num_k_tiles > 1)
        load_tile(1, 1);

    // Main loop
    for (uint tile = 0; tile < num_k_tiles; ++tile) {
        uint buf = tile % 2;

        if (tile < num_k_tiles - 1)
            __pipeline_wait_prior(1);
        else
            __pipeline_wait_prior(0);
        __syncthreads();

        compute_tile(buf);

        __syncthreads();

        if (tile + 2 < num_k_tiles)
            load_tile(buf, tile + 2);
    }

    // Write back
    for (uint wm = 0; wm < WMMA_TILES_M; ++wm) {
        for (uint wn = 0; wn < WMMA_TILES_N; ++wn) {
            half *c_ptr = &C[(warpRow * WM + wm * W13_WMMA_M) * N +
                              warpCol * WN + wn * W13_WMMA_N];

            if (beta == 0.0f && alpha == 1.0f) {
                wmma::fragment<wmma::accumulator, W13_WMMA_M, W13_WMMA_N, W13_WMMA_K, half> c_half;
                for (uint i = 0; i < acc[wm][wn].num_elements; ++i)
                    c_half.x[i] = __float2half(acc[wm][wn].x[i]);
                wmma::store_matrix_sync(c_ptr, c_half, N, wmma::mem_row_major);
            } else {
                wmma::fragment<wmma::accumulator, W13_WMMA_M, W13_WMMA_N, W13_WMMA_K, half> c_frag;
                wmma::load_matrix_sync(c_frag, c_ptr, N, wmma::mem_row_major);
                wmma::fragment<wmma::accumulator, W13_WMMA_M, W13_WMMA_N, W13_WMMA_K, half> out;
                for (uint i = 0; i < acc[wm][wn].num_elements; ++i)
                    out.x[i] = __float2half(alpha * acc[wm][wn].x[i] +
                                            beta * __half2float(c_frag.x[i]));
                wmma::store_matrix_sync(c_ptr, out, N, wmma::mem_row_major);
            }
        }
    }
}
