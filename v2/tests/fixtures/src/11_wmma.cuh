#pragma once

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

/*
 * Kernel 11: WMMA Tensor Core HGEMM
 *
 * Block-level SMEM tiling + WMMA tensor core inner computation.
 *
 * Hierarchy:
 *   Block tile (BM × BN) in SMEM
 *     └─ Warp tile (WM × WN) — each warp covers WM×WN of the block tile
 *         └─ WMMA tile (16 × 16 × 16) — hardware unit
 *
 * SMEM layout (NO transpose — both row-major for vectorized contiguous writes):
 *   As[BM * BK] — row-major: As[m * BK + k], fragment loads with row_major
 *   Bs[BK * BN] — row-major: Bs[k * BN + n], fragment loads with row_major
 *
 * Parameters: BM=128, BN=128, BK=16, WM=64, WN=64, 128 threads (4 warps)
 */

#define CEIL_DIV_W(M, N) (((M) + (N) - 1) / (N))

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_wmma(int M, int N, int K, float alpha, const half *A, const half *B,
               float beta, half *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // Warp identification: 2×2 warp layout over the 128×128 block
    const uint warpIdx = threadIdx.x / 32;
    constexpr uint WARPS_PER_ROW = BN / WN;  // 2
    const uint warpRow = warpIdx / WARPS_PER_ROW;
    const uint warpCol = warpIdx % WARPS_PER_ROW;

    constexpr uint WMMA_TILES_M = WM / WMMA_M;  // 4
    constexpr uint WMMA_TILES_N = WN / WMMA_N;  // 4

    // Accumulator fragments: 4×4 = 16 per warp, fp32 accumulation
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WMMA_TILES_M][WMMA_TILES_N];
    for (uint m = 0; m < WMMA_TILES_M; ++m)
        for (uint n = 0; n < WMMA_TILES_N; ++n)
            wmma::fill_fragment(acc[m][n], 0.0f);

    // SMEM — both row-major, no transpose needed
    __shared__ half As[BM * BK];  // As[m * BK + k]
    __shared__ half Bs[BK * BN];  // Bs[k * BN + n]

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // Vectorized loading indices (int4 = 8 halfs = 128 bits per load)
    //
    // A: BM*BK = 128*16 = 2048 halfs. Load along K (8 consecutive K-values).
    //   BK=16, so 8 halfs covers half a row. Each thread loads one int4.
    //   tid maps to: row = tid / (BK/8) = tid / 2, col = (tid % 2) * 8
    //   stride = 128 / 2 = 64 rows per pass, need 128/64 = 2 passes.
    //   Store contiguous: As[m * BK + k] — the 8 halfs go into consecutive SMEM.
    const uint tid = threadIdx.x;
    const uint a_innerRow = tid / (BK / 8);
    const uint a_innerCol = (tid % (BK / 8)) * 8;
    constexpr uint a_rowStride = NUM_THREADS / (BK / 8);

    // B: BK*BN = 16*128 = 2048 halfs. Load along N (8 consecutive N-values).
    //   BN=128, so 8 halfs is 1/16 of a row. Each thread loads one int4.
    //   tid maps to: row = tid / (BN/8) = tid / 16, col = (tid % 16) * 8
    //   stride = 128 / 16 = 8 rows per pass, need 16/8 = 2 passes.
    const uint b_innerRow = tid / (BN / 8);
    const uint b_innerCol = (tid % (BN / 8)) * 8;
    constexpr uint b_rowStride = NUM_THREADS / (BN / 8);

    // ─── Main K-loop ─────────────────────────────────────────────────
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // Load A into SMEM — row-major, contiguous vectorized writes
        for (uint offset = 0; offset < BM; offset += a_rowStride) {
            uint m = a_innerRow + offset;
            uint k = a_innerCol;
            // GMEM: A[m * K + k], 8 consecutive halfs along K
            // SMEM: As[m * BK + k], 8 consecutive positions
            *reinterpret_cast<int4 *>(&As[m * BK + k]) =
                *reinterpret_cast<const int4 *>(&A[m * K + k]);
        }

        // Load B into SMEM — row-major, contiguous vectorized writes
        for (uint offset = 0; offset < BK; offset += b_rowStride) {
            uint k = b_innerRow + offset;
            uint n = b_innerCol;
            *reinterpret_cast<int4 *>(&Bs[k * BN + n]) =
                *reinterpret_cast<const int4 *>(&B[k * N + n]);
        }

        __syncthreads();

        // ─── WMMA compute ────────────────────────────────────────────
        // BK=16 = WMMA_K, so one K-step per tile.
        for (uint kTile = 0; kTile < BK; kTile += WMMA_K) {
            for (uint wm = 0; wm < WMMA_TILES_M; ++wm) {
                // Load A fragment once, reuse across all wn
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major> a_frag;
                // As[m * BK + k]: row = warpRow*WM + wm*16, starting at k = kTile
                // leading dimension = BK
                wmma::load_matrix_sync(a_frag,
                    &As[(warpRow * WM + wm * WMMA_M) * BK + kTile], BK);

                for (uint wn = 0; wn < WMMA_TILES_N; ++wn) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                                   half, wmma::row_major> b_frag;
                    // Bs[k * BN + n]: row = kTile, col = warpCol*WN + wn*16
                    // leading dimension = BN
                    wmma::load_matrix_sync(b_frag,
                        &Bs[kTile * BN + warpCol * WN + wn * WMMA_N], BN);

                    wmma::mma_sync(acc[wm][wn], a_frag, b_frag, acc[wm][wn]);
                }
            }
        }

        __syncthreads();

        A += BK;
        B += BK * N;
    }

    // ─── Write back ──────────────────────────────────────────────────
    for (uint wm = 0; wm < WMMA_TILES_M; ++wm) {
        for (uint wn = 0; wn < WMMA_TILES_N; ++wn) {
            half *c_ptr = &C[(warpRow * WM + wm * WMMA_M) * N +
                              warpCol * WN + wn * WMMA_N];

            if (beta == 0.0f && alpha == 1.0f) {
                // Fast path: just convert fp32 acc → fp16 and store
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_half;
                for (uint i = 0; i < acc[wm][wn].num_elements; ++i)
                    c_half.x[i] = __float2half(acc[wm][wn].x[i]);
                wmma::store_matrix_sync(c_ptr, c_half, N, wmma::mem_row_major);
            } else {
                // General path: alpha * acc + beta * C
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
                wmma::load_matrix_sync(c_frag, c_ptr, N, wmma::mem_row_major);
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out;
                for (uint i = 0; i < acc[wm][wn].num_elements; ++i)
                    out.x[i] = __float2half(alpha * acc[wm][wn].x[i] +
                                            beta * __half2float(c_frag.x[i]));
                wmma::store_matrix_sync(c_ptr, out, N, wmma::mem_row_major);
            }
        }
    }
}
