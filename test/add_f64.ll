; Same shape as smoke but with double precision.
; 3 doubles in/out = 24 global bytes; 1 fadd_f64 = 1 flop in the f64 bucket.

; CHECK: kernel add_kernel_f64
; CHECK: bb.0
; CHECK-SAME: flops=1
; CHECK-SAME: flops_f32=0
; CHECK-SAME: flops_f64=1
; CHECK-SAME: global_bytes=24
; CHECK-SAME: local_bytes=0
; CHECK-SAME: ai=0.041667

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @add_kernel_f64(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idx = sext i32 %tid to i64
  %ap = getelementptr double, ptr addrspace(1) %a, i64 %idx
  %bp = getelementptr double, ptr addrspace(1) %b, i64 %idx
  %cp = getelementptr double, ptr addrspace(1) %c, i64 %idx
  %av = load double, ptr addrspace(1) %ap
  %bv = load double, ptr addrspace(1) %bp
  %sum = fadd double %av, %bv
  store double %sum, ptr addrspace(1) %cp
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @add_kernel_f64, !"kernel", i32 1}
