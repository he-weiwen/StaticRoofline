// Naive one-kernel softmax over row-major input.
//
// Launch convention:
//   gridDim.x  = number of rows
//   blockDim.x = power-of-two threads per row
//   dynamic shared memory = blockDim.x * sizeof(float)
//
// The implementation is deliberately simple:
//   1. reduce row max into shared memory
//   2. compute exp(x - max), write temporary output, reduce row sum
//   3. normalize output in place
//
// It is "fused" in the sense that all softmax stages happen in one kernel,
// but it is not optimized: it reads input twice and reads/writes output during
// normalization.

extern "C" __device__ float __nv_expf(float);

extern "C" __global__ void naive_fused_softmax_kernel(const float *input,
                                                       float *output,
                                                       int cols) {
    extern __shared__ float scratch[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int base = row * cols;

    float local_max = -3.4028234663852886e38f;
    for (int col = tid; col < cols; col += stride) {
        float x = input[base + col];
        local_max = x > local_max ? x : local_max;
    }

    scratch[tid] = local_max;
    __syncthreads();

    for (int offset = stride >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = scratch[tid + offset];
            float self = scratch[tid];
            scratch[tid] = other > self ? other : self;
        }
        __syncthreads();
    }

    float row_max = scratch[0];
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += stride) {
        float e = __nv_expf(input[base + col] - row_max);
        output[base + col] = e;
        local_sum += e;
    }

    scratch[tid] = local_sum;
    __syncthreads();

    for (int offset = stride >> 1; offset > 0; offset >>= 1) {
        if (tid < offset)
            scratch[tid] += scratch[tid + offset];
        __syncthreads();
    }

    float inv_sum = 1.0f / scratch[0];
    for (int col = tid; col < cols; col += stride)
        output[base + col] = output[base + col] * inv_sum;
}
