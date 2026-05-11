// CHECK: hgemm_naive
// CHECK: loop_depth=1 exec_count=L
// CHECK: FMA
//
// CHECK: hgemm_coalesced
// CHECK: loop_depth=1 exec_count=L
// CHECK: FMA
//
// CHECK: hgemm_shared_mem
// CHECK: loop_depth=1 exec_count=L
// CHECK: shared_load=
// CHECK: shared_store=
//
// This file intentionally includes the simpler matmul kernels first. They
// exercise ordinary scalar FMA, global memory, shared memory, and machine-loop
// attribution without depending on tensor-core or inline-PTX classification.

#include "1_naive.cuh"
#include "2_coalesced.cuh"
#include "3_shared_mem.cuh"
