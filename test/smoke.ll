; Minimal NVPTX kernel: c[i] = a[i] + b[i]
; Exercises: global-address-space loads/stores, fadd, tid intrinsic.

; CHECK: kernel add_kernel
; CHECK: bb.0
; CHECK-SAME: flops=1
; CHECK-SAME: flops_f32=1
; CHECK-SAME: flops_f64=0
; CHECK-SAME: global_bytes=12
; CHECK-SAME: local_bytes=0
; CHECK-SAME: ai=0.083333
; CHECK: summary: add_kernel
; CHECK-SAME: flops=1
; CHECK-SAME: global_bytes=12
; CHECK-SAME: ai=0.083333

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @add_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idx = sext i32 %tid to i64
  %ap = getelementptr float, ptr addrspace(1) %a, i64 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i64 %idx
  %cp = getelementptr float, ptr addrspace(1) %c, i64 %idx
  %av = load float, ptr addrspace(1) %ap
  %bv = load float, ptr addrspace(1) %bp
  %sum = fadd float %av, %bv
  store float %sum, ptr addrspace(1) %cp
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @add_kernel, !"kernel", i32 1}
