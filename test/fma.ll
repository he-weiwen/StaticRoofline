; Regression test for the FMA opcode-naming variation across LLVM versions.
; LLVM 18 emits "FMA32rrr"; newer LLVM emits "FMA_F32rrr". Both must be
; counted as 2 FLOPs in the f32 bucket.
;
; Kernel: d = a*b + c (fmul + fadd marked `contract` so ptxas/LLVM fuses
; them into a single FMA).

; CHECK: kernel fma_kernel
; CHECK: bb.0
; CHECK-SAME: flops=2
; CHECK-SAME: flops_f32=2
; CHECK-SAME: flops_f64=0
; CHECK-SAME: global_bytes=16
; CHECK-SAME: local_bytes=0
; CHECK-SAME: ai=0.125

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @fma_kernel(ptr addrspace(1) %a, ptr addrspace(1) %b,
                        ptr addrspace(1) %c, ptr addrspace(1) %d) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %idx = sext i32 %tid to i64
  %ap = getelementptr float, ptr addrspace(1) %a, i64 %idx
  %bp = getelementptr float, ptr addrspace(1) %b, i64 %idx
  %cp = getelementptr float, ptr addrspace(1) %c, i64 %idx
  %dp = getelementptr float, ptr addrspace(1) %d, i64 %idx
  %av = load float, ptr addrspace(1) %ap
  %bv = load float, ptr addrspace(1) %bp
  %cv = load float, ptr addrspace(1) %cp
  %prod = fmul contract float %av, %bv
  %sum = fadd contract float %prod, %cv
  store float %sum, ptr addrspace(1) %dp
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

!nvvm.annotations = !{!0}
!0 = !{ptr @fma_kernel, !"kernel", i32 1}
