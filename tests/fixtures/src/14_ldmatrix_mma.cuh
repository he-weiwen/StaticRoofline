#pragma once

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>

/*
 * Kernel 14: ldmatrix + mma.sync PTX inline assembly
 *
 * Replaces wmma::load_matrix_sync with ldmatrix (single warp-cooperative
 * instruction, guaranteed conflict-free) and wmma::mma_sync with the native
 * mma.sync.m16n8k16 PTX instruction.
 *
 * Key changes from kernel 13:
 * - A fragments loaded with ldmatrix.sync.aligned.m8n8.x4.shared.b16
 * - B fragments loaded with ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
 *   (.trans converts row-major SMEM to col-major operand for mma.sync)
 * - Each 16×16 output tile = 2× mma.sync.m16n8k16 (producing 2× 16×8 halves)
 * - Accumulator: float[4] per 16×8 half, so 8 floats per 16×16 tile per thread
 *
 * Builds on kernel 13's cp.async double-buffer + SMEM padding.
 */

// ─── PTX helpers ─────────────────────────────────────────────────────

// Convert generic pointer to shared memory address for PTX instructions
__device__ __forceinline__ uint32_t smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Load 16×16 A fragment from row-major SMEM using ldmatrix.x4
//
// ldmatrix.x4 loads four 8×8 sub-matrices.  The warp is split into
// four groups of 8 consecutive threads, and each group supplies the
// 8 row-start addresses for one sub-matrix:
//
//   Threads  0- 7  → sub-matrix 0 (rows  0-7,  cols 0-7)
//   Threads  8-15  → sub-matrix 1 (rows  8-15, cols 0-7)
//   Threads 16-23  → sub-matrix 2 (rows  0-7,  cols 8-15)
//   Threads 24-31  → sub-matrix 3 (rows  8-15, cols 8-15)
//
// Each thread provides the address of the start of a 16-byte row
// (8 half values).  The address formula is therefore:
//   row = lane % 16          (wraps 0-15 for the 16 rows)
//   col = (lane / 16) * 8    (0 for lanes 0-15, 8 for lanes 16-31)
__device__ __forceinline__ void ldmatrix_a(uint32_t (&frag)[4],
                                           const half *smem_ptr, int ldm) {
    uint32_t lane = threadIdx.x % 32;
    uint32_t row = lane & 15;            // lane % 16
    uint32_t col = (lane >> 4) << 3;     // (lane / 16) * 8
    uint32_t addr = smem_addr(&smem_ptr[row * ldm + col]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(addr)
    );
}

// Load 16×8 B fragment from row-major SMEM using ldmatrix.x2.trans
// .trans converts row-major SMEM to col-major register layout for mma.sync
__device__ __forceinline__ void ldmatrix_b_trans(uint32_t (&frag)[2],
                                                 const half *smem_ptr, int ldm) {
    uint32_t lane = threadIdx.x % 32;
    // For x2: threads 0-7 load matrix 0 (rows 0-7), threads 8-15 load matrix 1 (rows 8-15)
    // Threads 16-31 mirror 0-15
    uint32_t row = (lane & 7) + (((lane >> 3) & 1) << 3);
    uint32_t addr = smem_addr(&smem_ptr[row * ldm]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag[0]), "=r"(frag[1])
        : "r"(addr)
    );
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// A: 16×16 (4 uint32), B: 16×8 (2 uint32), C/D: 16×8 (4 float)
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                              const uint32_t (&a)[4],
                                              const uint32_t (&b)[2],
                                              const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// ─── Kernel ──────────────────────────────────────────────────────────

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int NUM_THREADS,
          const int PAD_K = 8, const int PAD_N = 8>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_ldmatrix_mma(int M, int N, int K, float alpha,
                       const half *A, const half *B,
                       float beta, half *C) {

    constexpr int LDA_S = BK + PAD_K;
    constexpr int LDB_S = BN + PAD_N;

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint warpIdx = threadIdx.x / 32;
    const uint lane = threadIdx.x % 32;
    constexpr uint WARPS_PER_ROW = BN / WN;
    const uint warpRow = warpIdx / WARPS_PER_ROW;
    const uint warpCol = warpIdx % WARPS_PER_ROW;

    // WMMA tiles per warp tile
    constexpr uint TILES_M = WM / 16;  // 4
    constexpr uint TILES_N = WN / 16;  // 4
    // Each 16×16 output = 2× mma.sync(16×8), so 2 acc halves per tile
    // Accumulators: TILES_M × TILES_N × 2 (left/right) × 4 floats
    float acc[TILES_M][TILES_N][2][4];
    #pragma unroll
    for (uint m = 0; m < TILES_M; ++m)
        for (uint n = 0; n < TILES_N; ++n)
            for (uint h = 0; h < 2; ++h)
                for (uint i = 0; i < 4; ++i)
                    acc[m][n][h][i] = 0.0f;

    __shared__ half As[2][BM * LDA_S];
    __shared__ half Bs[2][BK * LDB_S];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // cp.async loading indices (same as kernel 13)
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
                &As[buf][(a_innerRow + offset) * LDA_S + a_innerCol],
                &A_tile[(a_innerRow + offset) * K + a_innerCol],
                16);
        }
        for (uint offset = 0; offset < BK; offset += b_rowStride) {
            __pipeline_memcpy_async(
                &Bs[buf][(b_innerRow + offset) * LDB_S + b_innerCol],
                &B_tile[(b_innerRow + offset) * N + b_innerCol],
                16);
        }
        __pipeline_commit();
    };

    auto compute_tile = [&](uint buf) {
        for (uint kTile = 0; kTile < BK; kTile += 16) {
            // Load all B fragments for this K-step (reuse across wm)
            // Each B fragment is 16×8 (half of a 16×16 B tile)
            uint32_t b_frag[TILES_N][2][2]; // [tile_n][left/right][2 regs]
            #pragma unroll
            for (uint wn = 0; wn < TILES_N; ++wn) {
                // Left 8 columns
                ldmatrix_b_trans(b_frag[wn][0],
                    &Bs[buf][kTile * LDB_S + warpCol * WN + wn * 16],
                    LDB_S);
                // Right 8 columns
                ldmatrix_b_trans(b_frag[wn][1],
                    &Bs[buf][kTile * LDB_S + warpCol * WN + wn * 16 + 8],
                    LDB_S);
            }

            #pragma unroll
            for (uint wm = 0; wm < TILES_M; ++wm) {
                // Load A fragment (16×16), reuse across all wn
                uint32_t a_frag[4];
                ldmatrix_a(a_frag,
                    &As[buf][(warpRow * WM + wm * 16) * LDA_S + kTile],
                    LDA_S);

                #pragma unroll
                for (uint wn = 0; wn < TILES_N; ++wn) {
                    // Left half: A × B_left → acc[wm][wn][0]
                    mma_m16n8k16(acc[wm][wn][0], a_frag, b_frag[wn][0], acc[wm][wn][0]);
                    // Right half: A × B_right → acc[wm][wn][1]
                    mma_m16n8k16(acc[wm][wn][1], a_frag, b_frag[wn][1], acc[wm][wn][1]);
                }
            }
        }
    };

    // Prologue
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

    // ─── Write back ──────────────────────────────────────────────────
    // mma.sync.m16n8k16 output mapping per thread:
    //   groupID = lane >> 2 (0..7)
    //   tidInGroup = lane & 3 (0..3)
    //   row0 = groupID * 2,      row1 = row0 + 1
    //   col0 = tidInGroup * 2,   col1 = col0 + 1
    //   D[0] = out[row0][col0], D[1] = out[row0][col1]
    //   D[2] = out[row1][col0], D[3] = out[row1][col1]

    // mma.sync.m16n8k16 output layout per thread:
    //   groupID = lane / 4 (0..7)
    //   tidInGroup = lane % 4 (0..3)
    //   D[0] → out[groupID,     tidInGroup*2    ]
    //   D[1] → out[groupID,     tidInGroup*2 + 1]
    //   D[2] → out[groupID + 8, tidInGroup*2    ]
    //   D[3] → out[groupID + 8, tidInGroup*2 + 1]
    uint32_t groupID = lane >> 2;
    uint32_t tidInGroup = lane & 3;
    uint32_t frag_row_top = groupID;          // rows 0-7
    uint32_t frag_row_bot = groupID + 8;      // rows 8-15
    uint32_t frag_col0 = tidInGroup * 2;

    for (uint wm = 0; wm < TILES_M; ++wm) {
        for (uint wn = 0; wn < TILES_N; ++wn) {
            uint32_t tile_row = warpRow * WM + wm * 16;
            uint32_t tile_col = warpCol * WN + wn * 16;

            for (uint h = 0; h < 2; ++h) {
                uint32_t col_offset = h * 8;
                uint32_t rt = tile_row + frag_row_top;
                uint32_t rb = tile_row + frag_row_bot;
                uint32_t c0 = tile_col + col_offset + frag_col0;

                if (beta == 0.0f && alpha == 1.0f) {
                    C[rt * N + c0]     = __float2half(acc[wm][wn][h][0]);
                    C[rt * N + c0 + 1] = __float2half(acc[wm][wn][h][1]);
                    C[rb * N + c0]     = __float2half(acc[wm][wn][h][2]);
                    C[rb * N + c0 + 1] = __float2half(acc[wm][wn][h][3]);
                } else {
                    C[rt * N + c0]     = __float2half(alpha * acc[wm][wn][h][0] + beta * __half2float(C[rt * N + c0]));
                    C[rt * N + c0 + 1] = __float2half(alpha * acc[wm][wn][h][1] + beta * __half2float(C[rt * N + c0 + 1]));
                    C[rb * N + c0]     = __float2half(alpha * acc[wm][wn][h][2] + beta * __half2float(C[rb * N + c0]));
                    C[rb * N + c0 + 1] = __float2half(alpha * acc[wm][wn][h][3] + beta * __half2float(C[rb * N + c0 + 1]));
                }
            }
        }
    }
}
