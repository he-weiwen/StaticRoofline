; ModuleID = '/home/whe302/compilers/nvptx_analyzer/test/matmul_loops.cu'
source_filename = "/home/whe302/compilers/nvptx_analyzer/test/matmul_loops.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__half = type { i16 }

@_ZZ16hgemm_shared_memiiifPK6__halfS1_fPS_E2As = internal unnamed_addr addrspace(3) global [32 x [32 x %struct.__half]] undef, align 2
@_ZZ16hgemm_shared_memiiifPK6__halfS1_fPS_E2Bs = internal unnamed_addr addrspace(3) global [32 x [32 x %struct.__half]] undef, align 2

; Function Attrs: convergent mustprogress noinline norecurse nounwind memory(argmem: readwrite)
define dso_local ptx_kernel void @_Z11hgemm_naiveiiifPK6__halfS1_fPS_(i32 noundef %M, i32 noundef %N, i32 noundef %K, float noundef %alpha, ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, float noundef %beta, ptr noundef captures(none) %C) local_unnamed_addr #0 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul = mul nuw nsw i32 %0, %1
  %2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add = add nuw nsw i32 %mul, %2
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul5 = mul i32 %3, %4
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add7 = add i32 %mul5, %5
  %cmp = icmp slt i32 %add, %M
  %cmp8 = icmp slt i32 %add7, %N
  %or.cond = and i1 %cmp, %cmp8
  br i1 %or.cond, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %entry
  %cmp950 = icmp sgt i32 %K, 0
  br i1 %cmp950, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %mul10 = mul nuw nsw i32 %K, %add
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %for.cond.preheader
  %sum.0.lcssa = phi float [ 0.000000e+00, %for.cond.preheader ], [ %add20, %for.body ]
  %mul21 = fmul contract float %alpha, %sum.0.lcssa
  %mul23 = mul nsw i32 %N, %add
  %add24 = add nsw i32 %mul23, %add7
  %idxprom25 = sext i32 %add24 to i64
  %arrayidx26 = getelementptr inbounds [2 x i8], ptr %C, i64 %idxprom25
  %6 = load i16, ptr %arrayidx26, align 2, !tbaa !7
  %7 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %6) #4, !srcloc !9
  %mul28 = fmul contract float %beta, %7
  %add29 = fadd contract float %mul21, %mul28
  %8 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add29) #4, !srcloc !10
  store i16 %8, ptr %arrayidx26, align 2, !tbaa !7
  br label %if.end

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %sum.052 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add20, %for.body ]
  %k.051 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %add11 = add nuw nsw i32 %k.051, %mul10
  %idxprom = zext nneg i32 %add11 to i64
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %A, i64 %idxprom
  %9 = load i16, ptr %arrayidx, align 2, !tbaa !7
  %10 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %9) #4, !srcloc !9
  %mul14 = mul nsw i32 %k.051, %N
  %add15 = add nsw i32 %mul14, %add7
  %idxprom16 = sext i32 %add15 to i64
  %arrayidx17 = getelementptr inbounds [2 x i8], ptr %B, i64 %idxprom16
  %11 = load i16, ptr %arrayidx17, align 2, !tbaa !7
  %12 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %11) #4, !srcloc !9
  %mul19 = fmul contract float %10, %12
  %add20 = fadd contract float %sum.052, %mul19
  %inc = add nuw nsw i32 %k.051, 1
  %exitcond.not = icmp eq i32 %inc, %K
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !11

if.end:                                           ; preds = %for.cond.cleanup, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind memory(argmem: readwrite)
define dso_local ptx_kernel void @_Z15hgemm_coalescediiifPK6__halfS1_fPS_(i32 noundef %M, i32 noundef %N, i32 noundef %K, float noundef %alpha, ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, float noundef %beta, ptr noundef captures(none) %C) local_unnamed_addr #0 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %mul = shl i32 %0, 5
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add = add i32 %mul, %1
  %2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %mul3 = shl nuw nsw i32 %2, 5
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add5 = add nuw nsw i32 %mul3, %3
  %cmp = icmp slt i32 %add5, %M
  %cmp6 = icmp slt i32 %add, %N
  %or.cond = and i1 %cmp, %cmp6
  br i1 %or.cond, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %entry
  %cmp748 = icmp sgt i32 %K, 0
  br i1 %cmp748, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %mul8 = mul nuw nsw i32 %K, %add5
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %for.cond.preheader
  %sum.0.lcssa = phi float [ 0.000000e+00, %for.cond.preheader ], [ %add18, %for.body ]
  %mul19 = fmul contract float %alpha, %sum.0.lcssa
  %mul21 = mul nsw i32 %N, %add5
  %add22 = add nsw i32 %mul21, %add
  %idxprom23 = sext i32 %add22 to i64
  %arrayidx24 = getelementptr inbounds [2 x i8], ptr %C, i64 %idxprom23
  %4 = load i16, ptr %arrayidx24, align 2, !tbaa !7
  %5 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %4) #4, !srcloc !9
  %mul26 = fmul contract float %beta, %5
  %add27 = fadd contract float %mul19, %mul26
  %6 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add27) #4, !srcloc !10
  store i16 %6, ptr %arrayidx24, align 2, !tbaa !7
  br label %if.end

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %sum.050 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add18, %for.body ]
  %k.049 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %add9 = add nuw nsw i32 %k.049, %mul8
  %idxprom = zext nneg i32 %add9 to i64
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %A, i64 %idxprom
  %7 = load i16, ptr %arrayidx, align 2, !tbaa !7
  %8 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %7) #4, !srcloc !9
  %mul12 = mul nsw i32 %k.049, %N
  %add13 = add nsw i32 %mul12, %add
  %idxprom14 = sext i32 %add13 to i64
  %arrayidx15 = getelementptr inbounds [2 x i8], ptr %B, i64 %idxprom14
  %9 = load i16, ptr %arrayidx15, align 2, !tbaa !7
  %10 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %9) #4, !srcloc !9
  %mul17 = fmul contract float %8, %10
  %add18 = fadd contract float %sum.050, %mul17
  %inc = add nuw nsw i32 %k.049, 1
  %exitcond.not = icmp eq i32 %inc, %K
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !13

if.end:                                           ; preds = %for.cond.cleanup, %entry
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z16hgemm_shared_memiiifPK6__halfS1_fPS_(i32 noundef %M, i32 noundef %N, i32 noundef %K, float noundef %alpha, ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, float noundef %beta, ptr noundef captures(none) %C) local_unnamed_addr #1 {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %3 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %mul = shl nuw nsw i32 %1, 5
  %add = add nuw nsw i32 %mul, %3
  %mul4 = shl nsw i32 %0, 5
  %add5 = add nuw nsw i32 %mul4, %2
  %cmp128 = icmp sgt i32 %K, 0
  br i1 %cmp128, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %cmp6 = icmp slt i32 %add, %M
  %idxprom17 = zext nneg i32 %3 to i64
  %arrayidx18 = getelementptr inbounds nuw [64 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_shared_memiiifPK6__halfS1_fPS_E2As to ptr), i64 %idxprom17
  %idxprom19 = zext nneg i32 %2 to i64
  %arrayidx20 = getelementptr inbounds nuw [2 x i8], ptr %arrayidx18, i64 %idxprom19
  %mul9 = mul nuw nsw i32 %K, %add
  %cmp24 = icmp slt i32 %add5, %N
  %arrayidx39 = getelementptr inbounds nuw [64 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_shared_memiiifPK6__halfS1_fPS_E2Bs to ptr), i64 %idxprom17
  %arrayidx41 = getelementptr inbounds nuw [2 x i8], ptr %arrayidx39, i64 %idxprom19
  %invariant.gep = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_shared_memiiifPK6__halfS1_fPS_E2Bs to ptr), i64 %idxprom19
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup45, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add59.1, %for.cond.cleanup45 ]
  %cmp63 = icmp slt i32 %add, %M
  %cmp65 = icmp slt i32 %add5, %N
  %or.cond123 = select i1 %cmp63, i1 %cmp65, i1 false
  br i1 %or.cond123, label %if.then66, label %if.end82

for.body:                                         ; preds = %for.body.lr.ph, %for.cond.cleanup45
  %sum.0131 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add59.1, %for.cond.cleanup45 ]
  %bk.0129 = phi i32 [ 0, %for.body.lr.ph ], [ %add61, %for.cond.cleanup45 ]
  br i1 %cmp6, label %land.lhs.true, label %if.else

land.lhs.true:                                    ; preds = %for.body
  %add7 = add nuw nsw i32 %bk.0129, %2
  %cmp8 = icmp slt i32 %add7, %K
  br i1 %cmp8, label %if.then, label %if.else

if.then:                                          ; preds = %land.lhs.true
  %add11 = add nuw nsw i32 %add7, %mul9
  %idxprom = zext nneg i32 %add11 to i64
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %A, i64 %idxprom
  %4 = load i16, ptr %arrayidx, align 2, !tbaa !7
  br label %if.end

if.else:                                          ; preds = %land.lhs.true, %for.body
  %5 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float 0.000000e+00) #4, !srcloc !10
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %storemerge = phi i16 [ %5, %if.else ], [ %4, %if.then ]
  store i16 %storemerge, ptr %arrayidx20, align 2, !tbaa !7
  %add21 = add nuw nsw i32 %bk.0129, %3
  %cmp22 = icmp slt i32 %add21, %K
  %or.cond = select i1 %cmp22, i1 %cmp24, i1 false
  br i1 %or.cond, label %if.then25, label %if.else35

if.then25:                                        ; preds = %if.end
  %mul27 = mul nsw i32 %add21, %N
  %add28 = add nsw i32 %mul27, %add5
  %idxprom29 = sext i32 %add28 to i64
  %arrayidx30 = getelementptr inbounds [2 x i8], ptr %B, i64 %idxprom29
  %6 = load i16, ptr %arrayidx30, align 2, !tbaa !7
  br label %if.end42

if.else35:                                        ; preds = %if.end
  %7 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float 0.000000e+00) #4, !srcloc !10
  br label %if.end42

if.end42:                                         ; preds = %if.else35, %if.then25
  %storemerge132 = phi i16 [ %7, %if.else35 ], [ %6, %if.then25 ]
  store i16 %storemerge132, ptr %arrayidx41, align 2, !tbaa !7
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %for.body46

for.cond.cleanup45:                               ; preds = %for.body46
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %add61 = add nuw nsw i32 %bk.0129, 32
  %cmp = icmp slt i32 %add61, %K
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !14

for.body46:                                       ; preds = %for.body46, %if.end42
  %sum.1127 = phi float [ %sum.0131, %if.end42 ], [ %add59.1, %for.body46 ]
  %k.0126 = phi i32 [ 0, %if.end42 ], [ %inc.1, %for.body46 ]
  %idxprom49 = zext nneg i32 %k.0126 to i64
  %arrayidx50 = getelementptr inbounds nuw [2 x i8], ptr %arrayidx18, i64 %idxprom49
  %8 = load i16, ptr %arrayidx50, align 2, !tbaa !7
  %9 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %8) #4, !srcloc !9
  %gep = getelementptr inbounds nuw [64 x i8], ptr %invariant.gep, i64 %idxprom49
  %10 = load i16, ptr %gep, align 2, !tbaa !7
  %11 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %10) #4, !srcloc !9
  %mul58 = fmul contract float %9, %11
  %add59 = fadd contract float %sum.1127, %mul58
  %inc = or disjoint i32 %k.0126, 1
  %idxprom49.1 = zext nneg i32 %inc to i64
  %arrayidx50.1 = getelementptr inbounds nuw [2 x i8], ptr %arrayidx18, i64 %idxprom49.1
  %12 = load i16, ptr %arrayidx50.1, align 2, !tbaa !7
  %13 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %12) #4, !srcloc !9
  %gep.1 = getelementptr inbounds nuw [64 x i8], ptr %invariant.gep, i64 %idxprom49.1
  %14 = load i16, ptr %gep.1, align 2, !tbaa !7
  %15 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %14) #4, !srcloc !9
  %mul58.1 = fmul contract float %13, %15
  %add59.1 = fadd contract float %add59, %mul58.1
  %inc.1 = add nuw nsw i32 %k.0126, 2
  %exitcond.not.1 = icmp eq i32 %inc.1, 32
  br i1 %exitcond.not.1, label %for.cond.cleanup45, label %for.body46, !llvm.loop !15

if.then66:                                        ; preds = %for.cond.cleanup
  %mul68 = fmul contract float %alpha, %sum.0.lcssa
  %mul70 = mul nsw i32 %N, %add
  %add71 = add nuw nsw i32 %mul70, %add5
  %idxprom72 = zext nneg i32 %add71 to i64
  %arrayidx73 = getelementptr inbounds nuw [2 x i8], ptr %C, i64 %idxprom72
  %16 = load i16, ptr %arrayidx73, align 2, !tbaa !7
  %17 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %16) #4, !srcloc !9
  %mul75 = fmul contract float %beta, %17
  %add76 = fadd contract float %mul68, %mul75
  %18 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add76) #4, !srcloc !10
  store i16 %18, ptr %arrayidx73, align 2, !tbaa !7
  br label %if.end82

if.end82:                                         ; preds = %if.then66, %for.cond.cleanup
  ret void
}

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

attributes #0 = { convergent mustprogress noinline norecurse nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx88,+sm_80" "uniform-work-group-size" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx88,+sm_80" "uniform-work-group-size" }
attributes #2 = { convergent nocallback nounwind }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nounwind memory(none) }

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
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !5, i64 0}
!9 = !{i64 2157010155}
!10 = !{i64 2156945415}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
!14 = distinct !{!14, !12}
!15 = distinct !{!15, !12}
