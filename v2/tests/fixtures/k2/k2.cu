// Fixture wrapper (PLAN.md §3 fixture policy): includes the v1 ladder
// kernel unchanged. hgemm_coalesced is a plain __global__ function, so
// the include alone emits its PTX body.
#include "2_coalesced.cuh"
