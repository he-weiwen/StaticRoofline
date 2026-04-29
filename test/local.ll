; Forces an alloca that survives SROA (dynamic index via tid) and uses
; volatile to prevent the loads/stores from being optimized away.
; Lands in addrspace(5) (local).
;
; Expected per-thread traffic:
;   - 2 global loads (a, b)        =  8 bytes  (global_load)
;   - 1 global store (out)         =  4 bytes  (global_store)
;   - 1 local store (volatile)     =  4 bytes  (local_store)
;   - 1 local load  (volatile)     =  4 bytes  (local_load)
; Plus 1 fadd_f32 = 1 FLOP.
;
; AI denominator is global only (local is reported separately for
; diagnostic visibility): ai = 1/12 = 0.083333.

; CHECK: kernel local_kernel
; CHECK: bb.0
; CHECK-SAME: flops=1
; CHECK-SAME: flops_f32=1
; CHECK-SAME: global_bytes=12
; CHECK-SAME: local_bytes=8
; CHECK-SAME: ai=0.083333

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @local_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out) {
entry:
  %arr = alloca [4 x float]
  %va = load float, ptr addrspace(1) %a
  %vb = load float, ptr addrspace(1) %b
  %sum = fadd float %va, %vb
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idx = and i32 %tid, 3
  %slot = getelementptr [4 x float], ptr %arr, i32 0, i32 %idx
  store volatile float %sum, ptr %slot
  %r = load volatile float, ptr %slot
  store float %r, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @local_kernel, !"kernel", i32 1}
