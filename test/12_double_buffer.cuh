#pragma once

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>

using namespace nvcuda;

/*
 * Kernel 12: WMMA + Double-Buffered cp.async Pipeline
 *
 * Pipeline timeline (steady state):
 *
 *   Iter i:   [wait buf_i] [compute buf_i] [load tile i+2 → buf_i]
 *   Iter i+1: [wait buf_j] [compute buf_j] [load tile i+3 → buf_j]
 *
 * Overlap: while computing buf_i, the load for buf_j (issued end of iter i-1)
 * is completing via cp.async hardware. wait_prior(1) ensures at most 1
 * commit group is still in-flight.
 *
 * Prologue: load tile 0 → buf 0, load tile 1 → buf 1 (2 groups in-flight).
 */

constexpr int W12_WMMA_M = 16;
constexpr int W12_WMMA_N = 16;
constexpr int W12_WMMA_K = 16;

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_double_buffer(int M, int N, int K, float alpha,
                        const half *A, const half *B,
                        float beta, half *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint warpIdx = threadIdx.x / 32;
    constexpr uint WARPS_PER_ROW = BN / WN;
    const uint warpRow = warpIdx / WARPS_PER_ROW;
    const uint warpCol = warpIdx % WARPS_PER_ROW;

    constexpr uint WMMA_TILES_M = WM / W12_WMMA_M;
    constexpr uint WMMA_TILES_N = WN / W12_WMMA_N;

    wmma::fragment<wmma::accumulator, W12_WMMA_M, W12_WMMA_N, W12_WMMA_K, float>
        acc[WMMA_TILES_M][WMMA_TILES_N];
    for (uint m = 0; m < WMMA_TILES_M; ++m)
        for (uint n = 0; n < WMMA_TILES_N; ++n)
            wmma::fill_fragment(acc[m][n], 0.0f);

    __shared__ half As[2][BM * BK];
    __shared__ half Bs[2][BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint tid = threadIdx.x;
    const uint a_innerRow = tid / (BK / 8);
    const uint a_innerCol = (tid % (BK / 8)) * 8;
    constexpr uint a_rowStride = NUM_THREADS / (BK / 8);

    const uint b_innerRow = tid / (BN / 8);
    const uint b_innerCol = (tid % (BN / 8)) * 8;
    constexpr uint b_rowStride = NUM_THREADS / (BN / 8);

    const uint num_k_tiles = K / BK;

    auto load_tile = [&](uint buf, uint tile_idx) {
        const half *A_tile = A + tile_idx * BK;
        const half *B_tile = B + tile_idx * BK * N;

        for (uint offset = 0; offset < BM; offset += a_rowStride) {
            __pipeline_memcpy_async(
                &As[buf][(a_innerRow + offset) * BK + a_innerCol],
                &A_tile[(a_innerRow + offset) * K + a_innerCol],
                16);
        }
        for (uint offset = 0; offset < BK; offset += b_rowStride) {
            __pipeline_memcpy_async(
                &Bs[buf][(b_innerRow + offset) * BN + b_innerCol],
                &B_tile[(b_innerRow + offset) * N + b_innerCol],
                16);
        }
        __pipeline_commit();
    };

    auto compute_tile = [&](uint buf) {
        for (uint kTile = 0; kTile < BK; kTile += W12_WMMA_K) {
            for (uint wm = 0; wm < WMMA_TILES_M; ++wm) {
                wmma::fragment<wmma::matrix_a, W12_WMMA_M, W12_WMMA_N, W12_WMMA_K,
                               half, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag,
                    &As[buf][(warpRow * WM + wm * W12_WMMA_M) * BK + kTile], BK);

                for (uint wn = 0; wn < WMMA_TILES_N; ++wn) {
                    wmma::fragment<wmma::matrix_b, W12_WMMA_M, W12_WMMA_N, W12_WMMA_K,
                                   half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag,
                        &Bs[buf][kTile * BN + warpCol * WN + wn * W12_WMMA_N], BN);
                    wmma::mma_sync(acc[wm][wn], a_frag, b_frag, acc[wm][wn]);
                }
            }
        }
    };

    // ─── Prologue: pre-load 2 tiles ──────────────────────────────────
    load_tile(0, 0);
    if (num_k_tiles > 1)
        load_tile(1, 1);

    // ─── Main loop ───────────────────────────────────────────────────
    for (uint tile = 0; tile < num_k_tiles; ++tile) {
        uint buf = tile % 2;

        // Wait for this buffer. Keep at most 1 group in-flight (the other buf).
        if (tile < num_k_tiles - 1)
            __pipeline_wait_prior(1);
        else
            __pipeline_wait_prior(0);
        __syncthreads();

        // Compute on current buffer
        compute_tile(buf);

        // All threads done with buf — safe to overwrite it
        __syncthreads();

        // Issue load for tile+2 into this buffer (now free)
        if (tile + 2 < num_k_tiles)
            load_tile(buf, tile + 2);
    }

    // ─── Write back ──────────────────────────────────────────────────
    for (uint wm = 0; wm < WMMA_TILES_M; ++wm) {
        for (uint wn = 0; wn < WMMA_TILES_N; ++wn) {
            half *c_ptr = &C[(warpRow * WM + wm * W12_WMMA_M) * N +
                              warpCol * WN + wn * W12_WMMA_N];

            if (beta == 0.0f && alpha == 1.0f) {
                wmma::fragment<wmma::accumulator, W12_WMMA_M, W12_WMMA_N, W12_WMMA_K, half> c_half;
                for (uint i = 0; i < acc[wm][wn].num_elements; ++i)
                    c_half.x[i] = __float2half(acc[wm][wn].x[i]);
                wmma::store_matrix_sync(c_ptr, c_half, N, wmma::mem_row_major);
            } else {
                wmma::fragment<wmma::accumulator, W12_WMMA_M, W12_WMMA_N, W12_WMMA_K, half> c_frag;
                wmma::load_matrix_sync(c_frag, c_ptr, N, wmma::mem_row_major);
                wmma::fragment<wmma::accumulator, W12_WMMA_M, W12_WMMA_N, W12_WMMA_K, half> out;
                for (uint i = 0; i < acc[wm][wn].num_elements; ++i)
                    out.x[i] = __float2half(alpha * acc[wm][wn].x[i] +
                                            beta * __half2float(c_frag.x[i]));
                wmma::store_matrix_sync(c_ptr, out, N, wmma::mem_row_major);
            }
        }
    }
}
