; ModuleID = '/home/whe302/compilers/nvptx_analyzer/test/fma_v2half.cu'
source_filename = "/home/whe302/compilers/nvptx_analyzer/test/fma_v2half.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local ptx_kernel void @fma_v2half_kernel(ptr noundef readonly captures(none) %a, ptr noundef readonly captures(none) %b, ptr noundef readonly captures(none) %c, ptr noundef writeonly captures(none) %d) local_unnamed_addr #0 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul = mul i32 %1, %2
  %add = add i32 %mul, %0
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds [4 x i8], ptr %a, i64 %idxprom
  %3 = load <2 x half>, ptr %arrayidx, align 4, !tbaa !7
  %arrayidx4 = getelementptr inbounds [4 x i8], ptr %b, i64 %idxprom
  %4 = load <2 x half>, ptr %arrayidx4, align 4, !tbaa !7
  %mul5 = fmul contract <2 x half> %3, %4
  %arrayidx7 = getelementptr inbounds [4 x i8], ptr %c, i64 %idxprom
  %5 = load <2 x half>, ptr %arrayidx7, align 4, !tbaa !7
  %add8 = fadd contract <2 x half> %mul5, %5
  %arrayidx10 = getelementptr inbounds [4 x i8], ptr %d, i64 %idxprom
  store <2 x half> %add8, ptr %arrayidx10, align 4, !tbaa !7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx88,+sm_80" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}
!llvm.errno.tbaa = !{!3}

!0 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 23.0.0git (https://github.com/llvm/llvm-project.git 8f1e24a207294f6454aa79ac9389c1093654cee5)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!5, !5, i64 0}
