; ModuleID = '/home/whe302/compilers/nvptx_analyzer/test/matmul_tiled_loops.cu'
source_filename = "/home/whe302/compilers/nvptx_analyzer/test/matmul_tiled_loops.cu"
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.__half = type { i16 }

$_Z20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_ = comdat any

$_Z20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_ = comdat any

$_Z16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_ = comdat any

$_Z15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_ = comdat any

$_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As = comdat any

$_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs = comdat any

$_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As = comdat any

$_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs = comdat any

$_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As = comdat any

$_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs = comdat any

$_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As = comdat any

$_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs = comdat any

@_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As = linkonce_odr dso_local local_unnamed_addr addrspace(3) global [512 x %struct.__half] undef, comdat, align 2
@_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs = linkonce_odr dso_local local_unnamed_addr addrspace(3) global [512 x %struct.__half] undef, comdat, align 2
@_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As = linkonce_odr dso_local local_unnamed_addr addrspace(3) global [1024 x %struct.__half] undef, comdat, align 2
@_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs = linkonce_odr dso_local local_unnamed_addr addrspace(3) global [1024 x %struct.__half] undef, comdat, align 2
@_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As = linkonce_odr dso_local local_unnamed_addr addrspace(3) global [1024 x %struct.__half] undef, comdat, align 2
@_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs = linkonce_odr dso_local local_unnamed_addr addrspace(3) global [1024 x %struct.__half] undef, comdat, align 2
@_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As = linkonce_odr dso_local local_unnamed_addr addrspace(3) global [2048 x %struct.__half] undef, comdat, align 2
@_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs = linkonce_odr dso_local local_unnamed_addr addrspace(3) global [2048 x %struct.__half] undef, comdat, align 2

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_(i32 noundef %M, i32 noundef %N, i32 noundef %K, float noundef %alpha, ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, float noundef %beta, ptr noundef captures(none) %C) local_unnamed_addr #0 comdat {
entry:
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %rem124125 = and i32 %2, 63
  %div126127132 = lshr i32 %2, 6
  %mul = shl nuw nsw i32 %0, 6
  %mul4 = shl i32 %1, 6
  %cmp136.not = icmp eq i32 %K, 0
  br i1 %cmp136.not, label %for.cond66.preheader, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %div13130131133 = lshr i32 %2, 3
  %rem12128129 = and i32 %2, 7
  %idx.ext5 = zext i32 %mul4 to i64
  %add.ptr6 = getelementptr inbounds nuw [2 x i8], ptr %B, i64 %idx.ext5
  %mul3 = mul i32 %K, %mul
  %idx.ext = zext i32 %mul3 to i64
  %add.ptr = getelementptr inbounds nuw [2 x i8], ptr %A, i64 %idx.ext
  %mul16 = mul i32 %K, %div13130131133
  %add17 = add i32 %mul16, %rem12128129
  %idxprom = zext i32 %add17 to i64
  %idxprom20 = zext nneg i32 %2 to i64
  %arrayidx21 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %idxprom20
  %mul22 = mul i32 %N, %div126127132
  %add23 = add i32 %mul22, %rem124125
  %idxprom24 = zext i32 %add23 to i64
  %arrayidx29 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %idxprom20
  %mul31 = shl nsw i32 %N, 3
  %idx.ext32 = sext i32 %mul31 to i64
  %3 = and i32 %2, 960
  br label %for.body

for.cond66.preheader:                             ; preds = %for.cond.cleanup36, %entry
  %threadResults.sroa.0.0 = phi float [ 0.000000e+00, %entry ], [ %add58, %for.cond.cleanup36 ]
  %threadResults.sroa.6.0 = phi float [ 0.000000e+00, %entry ], [ %add58.1, %for.cond.cleanup36 ]
  %threadResults.sroa.9.0 = phi float [ 0.000000e+00, %entry ], [ %add58.2, %for.cond.cleanup36 ]
  %threadResults.sroa.12.0 = phi float [ 0.000000e+00, %entry ], [ %add58.3, %for.cond.cleanup36 ]
  %threadResults.sroa.15.0 = phi float [ 0.000000e+00, %entry ], [ %add58.4, %for.cond.cleanup36 ]
  %threadResults.sroa.18.0 = phi float [ 0.000000e+00, %entry ], [ %add58.5, %for.cond.cleanup36 ]
  %threadResults.sroa.21.0 = phi float [ 0.000000e+00, %entry ], [ %add58.6, %for.cond.cleanup36 ]
  %threadResults.sroa.24.0 = phi float [ 0.000000e+00, %entry ], [ %add58.7, %for.cond.cleanup36 ]
  %mul8 = mul i32 %N, %mul
  %add = add i32 %mul8, %mul4
  %idx.ext10 = zext i32 %add to i64
  %add.ptr11 = getelementptr inbounds nuw [2 x i8], ptr %C, i64 %idx.ext10
  %mul70 = shl nuw nsw i32 %div126127132, 3
  %mul72 = mul i32 %mul70, %N
  %add73 = add i32 %mul72, %rem124125
  %mul76 = fmul contract float %alpha, %threadResults.sroa.0.0
  %idxprom78 = sext i32 %add73 to i64
  %arrayidx79 = getelementptr inbounds [2 x i8], ptr %add.ptr11, i64 %idxprom78
  %4 = load i16, ptr %arrayidx79, align 2, !tbaa !7
  %5 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %4) #6, !srcloc !9
  %mul81 = fmul contract float %beta, %5
  %add82 = fadd contract float %mul76, %mul81
  %6 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add82) #6, !srcloc !10
  store i16 %6, ptr %arrayidx79, align 2, !tbaa !7
  %add71.1 = or disjoint i32 %mul70, 1
  %mul72.1 = mul i32 %add71.1, %N
  %add73.1 = add i32 %mul72.1, %rem124125
  %mul76.1 = fmul contract float %alpha, %threadResults.sroa.6.0
  %idxprom78.1 = sext i32 %add73.1 to i64
  %arrayidx79.1 = getelementptr inbounds [2 x i8], ptr %add.ptr11, i64 %idxprom78.1
  %7 = load i16, ptr %arrayidx79.1, align 2, !tbaa !7
  %8 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %7) #6, !srcloc !9
  %mul81.1 = fmul contract float %beta, %8
  %add82.1 = fadd contract float %mul76.1, %mul81.1
  %9 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add82.1) #6, !srcloc !10
  store i16 %9, ptr %arrayidx79.1, align 2, !tbaa !7
  %add71.2 = or disjoint i32 %mul70, 2
  %mul72.2 = mul i32 %add71.2, %N
  %add73.2 = add i32 %mul72.2, %rem124125
  %mul76.2 = fmul contract float %alpha, %threadResults.sroa.9.0
  %idxprom78.2 = sext i32 %add73.2 to i64
  %arrayidx79.2 = getelementptr inbounds [2 x i8], ptr %add.ptr11, i64 %idxprom78.2
  %10 = load i16, ptr %arrayidx79.2, align 2, !tbaa !7
  %11 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %10) #6, !srcloc !9
  %mul81.2 = fmul contract float %beta, %11
  %add82.2 = fadd contract float %mul76.2, %mul81.2
  %12 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add82.2) #6, !srcloc !10
  store i16 %12, ptr %arrayidx79.2, align 2, !tbaa !7
  %add71.3 = or disjoint i32 %mul70, 3
  %mul72.3 = mul i32 %add71.3, %N
  %add73.3 = add i32 %mul72.3, %rem124125
  %mul76.3 = fmul contract float %alpha, %threadResults.sroa.12.0
  %idxprom78.3 = sext i32 %add73.3 to i64
  %arrayidx79.3 = getelementptr inbounds [2 x i8], ptr %add.ptr11, i64 %idxprom78.3
  %13 = load i16, ptr %arrayidx79.3, align 2, !tbaa !7
  %14 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %13) #6, !srcloc !9
  %mul81.3 = fmul contract float %beta, %14
  %add82.3 = fadd contract float %mul76.3, %mul81.3
  %15 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add82.3) #6, !srcloc !10
  store i16 %15, ptr %arrayidx79.3, align 2, !tbaa !7
  %add71.4 = or disjoint i32 %mul70, 4
  %mul72.4 = mul i32 %add71.4, %N
  %add73.4 = add i32 %mul72.4, %rem124125
  %mul76.4 = fmul contract float %alpha, %threadResults.sroa.15.0
  %idxprom78.4 = sext i32 %add73.4 to i64
  %arrayidx79.4 = getelementptr inbounds [2 x i8], ptr %add.ptr11, i64 %idxprom78.4
  %16 = load i16, ptr %arrayidx79.4, align 2, !tbaa !7
  %17 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %16) #6, !srcloc !9
  %mul81.4 = fmul contract float %beta, %17
  %add82.4 = fadd contract float %mul76.4, %mul81.4
  %18 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add82.4) #6, !srcloc !10
  store i16 %18, ptr %arrayidx79.4, align 2, !tbaa !7
  %add71.5 = or disjoint i32 %mul70, 5
  %mul72.5 = mul i32 %add71.5, %N
  %add73.5 = add i32 %mul72.5, %rem124125
  %mul76.5 = fmul contract float %alpha, %threadResults.sroa.18.0
  %idxprom78.5 = sext i32 %add73.5 to i64
  %arrayidx79.5 = getelementptr inbounds [2 x i8], ptr %add.ptr11, i64 %idxprom78.5
  %19 = load i16, ptr %arrayidx79.5, align 2, !tbaa !7
  %20 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %19) #6, !srcloc !9
  %mul81.5 = fmul contract float %beta, %20
  %add82.5 = fadd contract float %mul76.5, %mul81.5
  %21 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add82.5) #6, !srcloc !10
  store i16 %21, ptr %arrayidx79.5, align 2, !tbaa !7
  %add71.6 = or disjoint i32 %mul70, 6
  %mul72.6 = mul i32 %add71.6, %N
  %add73.6 = add i32 %mul72.6, %rem124125
  %mul76.6 = fmul contract float %alpha, %threadResults.sroa.21.0
  %idxprom78.6 = sext i32 %add73.6 to i64
  %arrayidx79.6 = getelementptr inbounds [2 x i8], ptr %add.ptr11, i64 %idxprom78.6
  %22 = load i16, ptr %arrayidx79.6, align 2, !tbaa !7
  %23 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %22) #6, !srcloc !9
  %mul81.6 = fmul contract float %beta, %23
  %add82.6 = fadd contract float %mul76.6, %mul81.6
  %24 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add82.6) #6, !srcloc !10
  store i16 %24, ptr %arrayidx79.6, align 2, !tbaa !7
  %add71.7 = or disjoint i32 %mul70, 7
  %mul72.7 = mul i32 %add71.7, %N
  %add73.7 = add i32 %mul72.7, %rem124125
  %mul76.7 = fmul contract float %alpha, %threadResults.sroa.24.0
  %idxprom78.7 = sext i32 %add73.7 to i64
  %arrayidx79.7 = getelementptr inbounds [2 x i8], ptr %add.ptr11, i64 %idxprom78.7
  %25 = load i16, ptr %arrayidx79.7, align 2, !tbaa !7
  %26 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %25) #6, !srcloc !9
  %mul81.7 = fmul contract float %beta, %26
  %add82.7 = fadd contract float %mul76.7, %mul81.7
  %27 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add82.7) #6, !srcloc !10
  store i16 %27, ptr %arrayidx79.7, align 2, !tbaa !7
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.cond.cleanup36
  %threadResults.sroa.0.1 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add58, %for.cond.cleanup36 ]
  %threadResults.sroa.6.1 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add58.1, %for.cond.cleanup36 ]
  %threadResults.sroa.9.1 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add58.2, %for.cond.cleanup36 ]
  %threadResults.sroa.12.1 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add58.3, %for.cond.cleanup36 ]
  %threadResults.sroa.15.1 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add58.4, %for.cond.cleanup36 ]
  %threadResults.sroa.18.1 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add58.5, %for.cond.cleanup36 ]
  %threadResults.sroa.21.1 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add58.6, %for.cond.cleanup36 ]
  %threadResults.sroa.24.1 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add58.7, %for.cond.cleanup36 ]
  %A.addr.0139 = phi ptr [ %add.ptr, %for.body.lr.ph ], [ %add.ptr30, %for.cond.cleanup36 ]
  %B.addr.0138 = phi ptr [ %add.ptr6, %for.body.lr.ph ], [ %add.ptr33, %for.cond.cleanup36 ]
  %bkIdx.0137 = phi i32 [ 0, %for.body.lr.ph ], [ %add63, %for.cond.cleanup36 ]
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %A.addr.0139, i64 %idxprom
  %28 = load i16, ptr %arrayidx, align 2, !tbaa !7
  store i16 %28, ptr %arrayidx21, align 2, !tbaa !7
  %arrayidx25 = getelementptr inbounds nuw [2 x i8], ptr %B.addr.0138, i64 %idxprom24
  %29 = load i16, ptr %arrayidx25, align 2, !tbaa !7
  store i16 %29, ptr %arrayidx29, align 2, !tbaa !7
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %for.body37

for.cond.cleanup36:                               ; preds = %for.body37
  %add.ptr30 = getelementptr inbounds nuw i8, ptr %A.addr.0139, i64 16
  %add.ptr33 = getelementptr inbounds [2 x i8], ptr %B.addr.0138, i64 %idx.ext32
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %add63 = add i32 %bkIdx.0137, 8
  %cmp = icmp ult i32 %add63, %K
  br i1 %cmp, label %for.body, label %for.cond66.preheader, !llvm.loop !11

for.body37:                                       ; preds = %for.body, %for.body37
  %threadResults.sroa.0.2 = phi float [ %threadResults.sroa.0.1, %for.body ], [ %add58, %for.body37 ]
  %threadResults.sroa.6.2 = phi float [ %threadResults.sroa.6.1, %for.body ], [ %add58.1, %for.body37 ]
  %threadResults.sroa.9.2 = phi float [ %threadResults.sroa.9.1, %for.body ], [ %add58.2, %for.body37 ]
  %threadResults.sroa.12.2 = phi float [ %threadResults.sroa.12.1, %for.body ], [ %add58.3, %for.body37 ]
  %threadResults.sroa.15.2 = phi float [ %threadResults.sroa.15.1, %for.body ], [ %add58.4, %for.body37 ]
  %threadResults.sroa.18.2 = phi float [ %threadResults.sroa.18.1, %for.body ], [ %add58.5, %for.body37 ]
  %threadResults.sroa.21.2 = phi float [ %threadResults.sroa.21.1, %for.body ], [ %add58.6, %for.body37 ]
  %threadResults.sroa.24.2 = phi float [ %threadResults.sroa.24.1, %for.body ], [ %add58.7, %for.body37 ]
  %dotIdx.0135 = phi i32 [ 0, %for.body ], [ %inc60, %for.body37 ]
  %mul38 = shl nuw nsw i32 %dotIdx.0135, 6
  %add39 = or disjoint i32 %mul38, %rem124125
  %idxprom40 = zext nneg i32 %add39 to i64
  %arrayidx41 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %idxprom40
  %30 = load i16, ptr %arrayidx41, align 2, !tbaa !7
  %31 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %30) #6, !srcloc !9
  %invariant.op = add nuw nsw i32 %3, %dotIdx.0135
  %idxprom52 = zext nneg i32 %invariant.op to i64
  %arrayidx53 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %idxprom52
  %32 = load i16, ptr %arrayidx53, align 2, !tbaa !7
  %33 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %32) #6, !srcloc !9
  %mul55 = fmul contract float %31, %33
  %add58 = fadd contract float %threadResults.sroa.0.2, %mul55
  %34 = zext nneg i32 %invariant.op to i64
  %35 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %34
  %arrayidx53.1 = getelementptr inbounds nuw i8, ptr %35, i64 16
  %36 = load i16, ptr %arrayidx53.1, align 2, !tbaa !7
  %37 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %36) #6, !srcloc !9
  %mul55.1 = fmul contract float %31, %37
  %add58.1 = fadd contract float %threadResults.sroa.6.2, %mul55.1
  %38 = zext nneg i32 %invariant.op to i64
  %39 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %38
  %arrayidx53.2 = getelementptr inbounds nuw i8, ptr %39, i64 32
  %40 = load i16, ptr %arrayidx53.2, align 2, !tbaa !7
  %41 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %40) #6, !srcloc !9
  %mul55.2 = fmul contract float %31, %41
  %add58.2 = fadd contract float %threadResults.sroa.9.2, %mul55.2
  %42 = zext nneg i32 %invariant.op to i64
  %43 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %42
  %arrayidx53.3 = getelementptr inbounds nuw i8, ptr %43, i64 48
  %44 = load i16, ptr %arrayidx53.3, align 2, !tbaa !7
  %45 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %44) #6, !srcloc !9
  %mul55.3 = fmul contract float %31, %45
  %add58.3 = fadd contract float %threadResults.sroa.12.2, %mul55.3
  %46 = zext nneg i32 %invariant.op to i64
  %47 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %46
  %arrayidx53.4 = getelementptr inbounds nuw i8, ptr %47, i64 64
  %48 = load i16, ptr %arrayidx53.4, align 2, !tbaa !7
  %49 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %48) #6, !srcloc !9
  %mul55.4 = fmul contract float %31, %49
  %add58.4 = fadd contract float %threadResults.sroa.15.2, %mul55.4
  %50 = zext nneg i32 %invariant.op to i64
  %51 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %50
  %arrayidx53.5 = getelementptr inbounds nuw i8, ptr %51, i64 80
  %52 = load i16, ptr %arrayidx53.5, align 2, !tbaa !7
  %53 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %52) #6, !srcloc !9
  %mul55.5 = fmul contract float %31, %53
  %add58.5 = fadd contract float %threadResults.sroa.18.2, %mul55.5
  %54 = zext nneg i32 %invariant.op to i64
  %55 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %54
  %arrayidx53.6 = getelementptr inbounds nuw i8, ptr %55, i64 96
  %56 = load i16, ptr %arrayidx53.6, align 2, !tbaa !7
  %57 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %56) #6, !srcloc !9
  %mul55.6 = fmul contract float %31, %57
  %add58.6 = fadd contract float %threadResults.sroa.21.2, %mul55.6
  %58 = zext nneg i32 %invariant.op to i64
  %59 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_1d_blocktilingILi64ELi64ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %58
  %arrayidx53.7 = getelementptr inbounds nuw i8, ptr %59, i64 112
  %60 = load i16, ptr %arrayidx53.7, align 2, !tbaa !7
  %61 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %60) #6, !srcloc !9
  %mul55.7 = fmul contract float %31, %61
  %add58.7 = fadd contract float %threadResults.sroa.24.2, %mul55.7
  %inc60 = add nuw nsw i32 %dotIdx.0135, 1
  %exitcond.not = icmp eq i32 %inc60, 8
  br i1 %exitcond.not, label %for.cond.cleanup36, label %for.body37, !llvm.loop !13
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_(i32 noundef %M, i32 noundef %N, i32 noundef %K, float noundef %alpha, ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, float noundef %beta, ptr noundef captures(none) %C) local_unnamed_addr #0 comdat {
entry:
  %threadResults = alloca [64 x float], align 4
  %regM = alloca [8 x float], align 4
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %rem = and i32 %2, 15
  %div207 = lshr i32 %2, 4
  %mul = shl nuw nsw i32 %0, 7
  %mul5 = shl i32 %1, 7
  %mul9 = mul i32 %N, %mul
  %add = add i32 %mul9, %mul5
  %idx.ext11 = zext i32 %add to i64
  %add.ptr12 = getelementptr inbounds nuw [2 x i8], ptr %C, i64 %idx.ext11
  call void @llvm.lifetime.start.p0(ptr nonnull %threadResults) #7
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(256) %threadResults, i8 0, i64 256, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %regM) #7
  %cmp217.not = icmp eq i32 %K, 0
  br i1 %cmp217.not, label %entry.for.cond124.preheader_crit_edge, label %for.cond21.preheader.lr.ph

entry.for.cond124.preheader_crit_edge:            ; preds = %entry
  %.pre = shl nuw nsw i32 %rem, 3
  br label %for.cond124.preheader

for.cond21.preheader.lr.ph:                       ; preds = %entry
  %rem20 = and i32 %2, 127
  %div18209 = lshr i32 %2, 7
  %rem16 = and i32 %2, 7
  %div14208 = lshr i32 %2, 3
  %idx.ext6 = zext i32 %mul5 to i64
  %add.ptr7 = getelementptr inbounds nuw [2 x i8], ptr %B, i64 %idx.ext6
  %mul4 = mul i32 %K, %mul
  %idx.ext = zext i32 %mul4 to i64
  %add.ptr = getelementptr inbounds nuw [2 x i8], ptr %A, i64 %idx.ext
  %mul53 = shl nsw i32 %N, 3
  %idx.ext54 = sext i32 %mul53 to i64
  %3 = shl nuw nsw i32 %div207, 6
  %mul82 = shl nuw nsw i32 %rem, 3
  %mul26 = mul i32 %div14208, %K
  %add27 = add i32 %mul26, %rem16
  %idxprom = zext i32 %add27 to i64
  %idxprom31 = zext nneg i32 %2 to i64
  %arrayidx32 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %idxprom31
  %add25.1 = add nuw nsw i32 %div14208, 32
  %mul26.1 = mul i32 %add25.1, %K
  %add27.1 = add i32 %mul26.1, %rem16
  %idxprom.1 = zext i32 %add27.1 to i64
  %mul29.1 = shl nuw nsw i32 %add25.1, 3
  %add30.1 = or disjoint i32 %mul29.1, %rem16
  %idxprom31.1 = zext nneg i32 %add30.1 to i64
  %arrayidx32.1 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %idxprom31.1
  %add25.2 = add nuw nsw i32 %div14208, 64
  %mul26.2 = mul i32 %add25.2, %K
  %add27.2 = add i32 %mul26.2, %rem16
  %idxprom.2 = zext i32 %add27.2 to i64
  %mul29.2 = shl nuw nsw i32 %add25.2, 3
  %add30.2 = or disjoint i32 %mul29.2, %rem16
  %idxprom31.2 = zext nneg i32 %add30.2 to i64
  %arrayidx32.2 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %idxprom31.2
  %add25.3 = add nuw nsw i32 %div14208, 96
  %mul26.3 = mul i32 %add25.3, %K
  %add27.3 = add i32 %mul26.3, %rem16
  %idxprom.3 = zext i32 %add27.3 to i64
  %mul29.3 = shl nuw nsw i32 %add25.3, 3
  %add30.3 = or disjoint i32 %mul29.3, %rem16
  %idxprom31.3 = zext nneg i32 %add30.3 to i64
  %arrayidx32.3 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %idxprom31.3
  %mul40 = mul i32 %div18209, %N
  %add41 = add i32 %mul40, %rem20
  %idxprom42 = zext i32 %add41 to i64
  %idxprom47 = zext nneg i32 %2 to i64
  %arrayidx48 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %idxprom47
  %add39.1 = add nuw nsw i32 %div18209, 2
  %mul40.1 = mul i32 %add39.1, %N
  %add41.1 = add i32 %mul40.1, %rem20
  %idxprom42.1 = zext i32 %add41.1 to i64
  %mul45.1 = shl nuw nsw i32 %add39.1, 7
  %add46.1 = or disjoint i32 %mul45.1, %rem20
  %idxprom47.1 = zext nneg i32 %add46.1 to i64
  %arrayidx48.1 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %idxprom47.1
  %add39.2 = add nuw nsw i32 %div18209, 4
  %mul40.2 = mul i32 %add39.2, %N
  %add41.2 = add i32 %mul40.2, %rem20
  %idxprom42.2 = zext i32 %add41.2 to i64
  %mul45.2 = shl nuw nsw i32 %add39.2, 7
  %add46.2 = or disjoint i32 %mul45.2, %rem20
  %idxprom47.2 = zext nneg i32 %add46.2 to i64
  %arrayidx48.2 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %idxprom47.2
  %add39.3 = add nuw nsw i32 %div18209, 6
  %mul40.3 = mul i32 %add39.3, %N
  %add41.3 = add i32 %mul40.3, %rem20
  %idxprom42.3 = zext i32 %add41.3 to i64
  %mul45.3 = shl nuw nsw i32 %add39.3, 7
  %add46.3 = or disjoint i32 %mul45.3, %rem20
  %idxprom47.3 = zext nneg i32 %add46.3 to i64
  %arrayidx48.3 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %idxprom47.3
  %arrayidx72.1 = getelementptr inbounds nuw i8, ptr %regM, i64 4
  %arrayidx72.2 = getelementptr inbounds nuw i8, ptr %regM, i64 8
  %arrayidx72.3 = getelementptr inbounds nuw i8, ptr %regM, i64 12
  %arrayidx72.4 = getelementptr inbounds nuw i8, ptr %regM, i64 16
  %arrayidx72.5 = getelementptr inbounds nuw i8, ptr %regM, i64 20
  %arrayidx72.6 = getelementptr inbounds nuw i8, ptr %regM, i64 24
  %arrayidx72.7 = getelementptr inbounds nuw i8, ptr %regM, i64 28
  br label %for.cond21.preheader

for.cond21.preheader:                             ; preds = %for.cond21.preheader.lr.ph, %for.cond.cleanup58
  %A.addr.0220 = phi ptr [ %add.ptr, %for.cond21.preheader.lr.ph ], [ %add.ptr52, %for.cond.cleanup58 ]
  %B.addr.0219 = phi ptr [ %add.ptr7, %for.cond21.preheader.lr.ph ], [ %add.ptr55, %for.cond.cleanup58 ]
  %bkIdx.0218 = phi i32 [ 0, %for.cond21.preheader.lr.ph ], [ %add121, %for.cond.cleanup58 ]
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %A.addr.0220, i64 %idxprom
  %4 = load i16, ptr %arrayidx, align 2, !tbaa !7
  store i16 %4, ptr %arrayidx32, align 2, !tbaa !7
  %arrayidx.1 = getelementptr inbounds nuw [2 x i8], ptr %A.addr.0220, i64 %idxprom.1
  %5 = load i16, ptr %arrayidx.1, align 2, !tbaa !7
  store i16 %5, ptr %arrayidx32.1, align 2, !tbaa !7
  %arrayidx.2 = getelementptr inbounds nuw [2 x i8], ptr %A.addr.0220, i64 %idxprom.2
  %6 = load i16, ptr %arrayidx.2, align 2, !tbaa !7
  store i16 %6, ptr %arrayidx32.2, align 2, !tbaa !7
  %arrayidx.3 = getelementptr inbounds nuw [2 x i8], ptr %A.addr.0220, i64 %idxprom.3
  %7 = load i16, ptr %arrayidx.3, align 2, !tbaa !7
  store i16 %7, ptr %arrayidx32.3, align 2, !tbaa !7
  %arrayidx43 = getelementptr inbounds nuw [2 x i8], ptr %B.addr.0219, i64 %idxprom42
  %8 = load i16, ptr %arrayidx43, align 2, !tbaa !7
  store i16 %8, ptr %arrayidx48, align 2, !tbaa !7
  %arrayidx43.1 = getelementptr inbounds nuw [2 x i8], ptr %B.addr.0219, i64 %idxprom42.1
  %9 = load i16, ptr %arrayidx43.1, align 2, !tbaa !7
  store i16 %9, ptr %arrayidx48.1, align 2, !tbaa !7
  %arrayidx43.2 = getelementptr inbounds nuw [2 x i8], ptr %B.addr.0219, i64 %idxprom42.2
  %10 = load i16, ptr %arrayidx43.2, align 2, !tbaa !7
  store i16 %10, ptr %arrayidx48.2, align 2, !tbaa !7
  %arrayidx43.3 = getelementptr inbounds nuw [2 x i8], ptr %B.addr.0219, i64 %idxprom42.3
  %11 = load i16, ptr %arrayidx43.3, align 2, !tbaa !7
  store i16 %11, ptr %arrayidx48.3, align 2, !tbaa !7
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %for.cond60.preheader

for.cond124.preheader:                            ; preds = %for.cond.cleanup58, %entry.for.cond124.preheader_crit_edge
  %mul136.pre-phi = phi i32 [ %.pre, %entry.for.cond124.preheader_crit_edge ], [ %mul82, %for.cond.cleanup58 ]
  %mul133 = shl nuw nsw i32 %div207, 3
  br label %for.cond129.preheader

for.cond60.preheader:                             ; preds = %for.cond21.preheader, %for.cond.cleanup95
  %dotIdx.0216 = phi i32 [ 0, %for.cond21.preheader ], [ %inc118, %for.cond.cleanup95 ]
  %invariant.op = add nuw nsw i32 %3, %dotIdx.0216
  %idxprom68 = zext nneg i32 %invariant.op to i64
  %arrayidx69 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %idxprom68
  %12 = load i16, ptr %arrayidx69, align 2, !tbaa !7
  %13 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %12) #6, !srcloc !9
  store float %13, ptr %regM, align 4, !tbaa !14
  %14 = zext nneg i32 %invariant.op to i64
  %15 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %14
  %arrayidx69.1 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %16 = load i16, ptr %arrayidx69.1, align 2, !tbaa !7
  %17 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %16) #6, !srcloc !9
  store float %17, ptr %arrayidx72.1, align 4, !tbaa !14
  %18 = zext nneg i32 %invariant.op to i64
  %19 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %18
  %arrayidx69.2 = getelementptr inbounds nuw i8, ptr %19, i64 32
  %20 = load i16, ptr %arrayidx69.2, align 2, !tbaa !7
  %21 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %20) #6, !srcloc !9
  store float %21, ptr %arrayidx72.2, align 4, !tbaa !14
  %22 = zext nneg i32 %invariant.op to i64
  %23 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %22
  %arrayidx69.3 = getelementptr inbounds nuw i8, ptr %23, i64 48
  %24 = load i16, ptr %arrayidx69.3, align 2, !tbaa !7
  %25 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %24) #6, !srcloc !9
  store float %25, ptr %arrayidx72.3, align 4, !tbaa !14
  %26 = zext nneg i32 %invariant.op to i64
  %27 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %26
  %arrayidx69.4 = getelementptr inbounds nuw i8, ptr %27, i64 64
  %28 = load i16, ptr %arrayidx69.4, align 2, !tbaa !7
  %29 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %28) #6, !srcloc !9
  store float %29, ptr %arrayidx72.4, align 4, !tbaa !14
  %30 = zext nneg i32 %invariant.op to i64
  %31 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %30
  %arrayidx69.5 = getelementptr inbounds nuw i8, ptr %31, i64 80
  %32 = load i16, ptr %arrayidx69.5, align 2, !tbaa !7
  %33 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %32) #6, !srcloc !9
  store float %33, ptr %arrayidx72.5, align 4, !tbaa !14
  %34 = zext nneg i32 %invariant.op to i64
  %35 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %34
  %arrayidx69.6 = getelementptr inbounds nuw i8, ptr %35, i64 96
  %36 = load i16, ptr %arrayidx69.6, align 2, !tbaa !7
  %37 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %36) #6, !srcloc !9
  store float %37, ptr %arrayidx72.6, align 4, !tbaa !14
  %38 = zext nneg i32 %invariant.op to i64
  %39 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2As to ptr), i64 %38
  %arrayidx69.7 = getelementptr inbounds nuw i8, ptr %39, i64 112
  %40 = load i16, ptr %arrayidx69.7, align 2, !tbaa !7
  %41 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %40) #6, !srcloc !9
  store float %41, ptr %arrayidx72.7, align 4, !tbaa !14
  %mul81 = shl nuw nsw i32 %dotIdx.0216, 7
  %add83 = or disjoint i32 %mul81, %mul82
  %idxprom85 = zext nneg i32 %add83 to i64
  %arrayidx86 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %idxprom85
  %42 = load i16, ptr %arrayidx86, align 2, !tbaa !7
  %43 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %42) #6, !srcloc !9
  %44 = zext nneg i32 %add83 to i64
  %45 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %44
  %arrayidx86.1 = getelementptr inbounds nuw i8, ptr %45, i64 2
  %46 = load i16, ptr %arrayidx86.1, align 2, !tbaa !7
  %47 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %46) #6, !srcloc !9
  %48 = zext nneg i32 %add83 to i64
  %49 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %48
  %arrayidx86.2 = getelementptr inbounds nuw i8, ptr %49, i64 4
  %50 = load i16, ptr %arrayidx86.2, align 2, !tbaa !7
  %51 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %50) #6, !srcloc !9
  %52 = zext nneg i32 %add83 to i64
  %53 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %52
  %arrayidx86.3 = getelementptr inbounds nuw i8, ptr %53, i64 6
  %54 = load i16, ptr %arrayidx86.3, align 2, !tbaa !7
  %55 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %54) #6, !srcloc !9
  %56 = zext nneg i32 %add83 to i64
  %57 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %56
  %arrayidx86.4 = getelementptr inbounds nuw i8, ptr %57, i64 8
  %58 = load i16, ptr %arrayidx86.4, align 2, !tbaa !7
  %59 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %58) #6, !srcloc !9
  %60 = zext nneg i32 %add83 to i64
  %61 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %60
  %arrayidx86.5 = getelementptr inbounds nuw i8, ptr %61, i64 10
  %62 = load i16, ptr %arrayidx86.5, align 2, !tbaa !7
  %63 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %62) #6, !srcloc !9
  %64 = zext nneg i32 %add83 to i64
  %65 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %64
  %arrayidx86.6 = getelementptr inbounds nuw i8, ptr %65, i64 12
  %66 = load i16, ptr %arrayidx86.6, align 2, !tbaa !7
  %67 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %66) #6, !srcloc !9
  %68 = zext nneg i32 %add83 to i64
  %69 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ20hgemm_2d_blocktilingILi128ELi128ELi8ELi8ELi8EEviiifPK6__halfS2_fPS0_E2Bs to ptr), i64 %68
  %arrayidx86.7 = getelementptr inbounds nuw i8, ptr %69, i64 14
  %70 = load i16, ptr %arrayidx86.7, align 2, !tbaa !7
  %71 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %70) #6, !srcloc !9
  br label %for.cond97.preheader

for.cond.cleanup58:                               ; preds = %for.cond.cleanup95
  %add.ptr52 = getelementptr inbounds nuw i8, ptr %A.addr.0220, i64 16
  %add.ptr55 = getelementptr inbounds [2 x i8], ptr %B.addr.0219, i64 %idx.ext54
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %add121 = add i32 %bkIdx.0218, 8
  %cmp = icmp ult i32 %add121, %K
  br i1 %cmp, label %for.cond21.preheader, label %for.cond124.preheader, !llvm.loop !16

for.cond97.preheader:                             ; preds = %for.cond60.preheader, %for.cond97.preheader
  %resIdxM.0215 = phi i32 [ %inc115, %for.cond97.preheader ], [ 0, %for.cond60.preheader ]
  %idxprom101 = zext nneg i32 %resIdxM.0215 to i64
  %arrayidx102 = getelementptr inbounds nuw [4 x i8], ptr %regM, i64 %idxprom101
  %72 = load float, ptr %arrayidx102, align 4, !tbaa !14
  %mul106 = shl nuw nsw i32 %resIdxM.0215, 3
  %mul105 = fmul contract float %72, %43
  %idxprom108 = zext nneg i32 %mul106 to i64
  %arrayidx109 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %idxprom108
  %73 = load float, ptr %arrayidx109, align 4, !tbaa !14
  %add110 = fadd contract float %73, %mul105
  store float %add110, ptr %arrayidx109, align 4, !tbaa !14
  %mul105.1 = fmul contract float %72, %47
  %74 = zext nneg i32 %mul106 to i64
  %75 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %74
  %arrayidx109.1 = getelementptr inbounds nuw i8, ptr %75, i64 4
  %76 = load float, ptr %arrayidx109.1, align 4, !tbaa !14
  %add110.1 = fadd contract float %76, %mul105.1
  store float %add110.1, ptr %arrayidx109.1, align 4, !tbaa !14
  %mul105.2 = fmul contract float %72, %51
  %77 = zext nneg i32 %mul106 to i64
  %78 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %77
  %arrayidx109.2 = getelementptr inbounds nuw i8, ptr %78, i64 8
  %79 = load float, ptr %arrayidx109.2, align 4, !tbaa !14
  %add110.2 = fadd contract float %79, %mul105.2
  store float %add110.2, ptr %arrayidx109.2, align 4, !tbaa !14
  %mul105.3 = fmul contract float %72, %55
  %80 = zext nneg i32 %mul106 to i64
  %81 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %80
  %arrayidx109.3 = getelementptr inbounds nuw i8, ptr %81, i64 12
  %82 = load float, ptr %arrayidx109.3, align 4, !tbaa !14
  %add110.3 = fadd contract float %82, %mul105.3
  store float %add110.3, ptr %arrayidx109.3, align 4, !tbaa !14
  %mul105.4 = fmul contract float %72, %59
  %83 = zext nneg i32 %mul106 to i64
  %84 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %83
  %arrayidx109.4 = getelementptr inbounds nuw i8, ptr %84, i64 16
  %85 = load float, ptr %arrayidx109.4, align 4, !tbaa !14
  %add110.4 = fadd contract float %85, %mul105.4
  store float %add110.4, ptr %arrayidx109.4, align 4, !tbaa !14
  %mul105.5 = fmul contract float %72, %63
  %86 = zext nneg i32 %mul106 to i64
  %87 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %86
  %arrayidx109.5 = getelementptr inbounds nuw i8, ptr %87, i64 20
  %88 = load float, ptr %arrayidx109.5, align 4, !tbaa !14
  %add110.5 = fadd contract float %88, %mul105.5
  store float %add110.5, ptr %arrayidx109.5, align 4, !tbaa !14
  %mul105.6 = fmul contract float %72, %67
  %89 = zext nneg i32 %mul106 to i64
  %90 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %89
  %arrayidx109.6 = getelementptr inbounds nuw i8, ptr %90, i64 24
  %91 = load float, ptr %arrayidx109.6, align 4, !tbaa !14
  %add110.6 = fadd contract float %91, %mul105.6
  store float %add110.6, ptr %arrayidx109.6, align 4, !tbaa !14
  %mul105.7 = fmul contract float %72, %71
  %92 = zext nneg i32 %mul106 to i64
  %93 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %92
  %arrayidx109.7 = getelementptr inbounds nuw i8, ptr %93, i64 28
  %94 = load float, ptr %arrayidx109.7, align 4, !tbaa !14
  %add110.7 = fadd contract float %94, %mul105.7
  store float %add110.7, ptr %arrayidx109.7, align 4, !tbaa !14
  %inc115 = add nuw nsw i32 %resIdxM.0215, 1
  %exitcond.not = icmp eq i32 %inc115, 8
  br i1 %exitcond.not, label %for.cond.cleanup95, label %for.cond97.preheader, !llvm.loop !17

for.cond.cleanup95:                               ; preds = %for.cond97.preheader
  %inc118 = add nuw nsw i32 %dotIdx.0216, 1
  %exitcond223.not = icmp eq i32 %inc118, 8
  br i1 %exitcond223.not, label %for.cond.cleanup58, label %for.cond60.preheader, !llvm.loop !18

for.cond129.preheader:                            ; preds = %for.cond124.preheader, %for.cond129.preheader
  %resIdxM123.0222 = phi i32 [ 0, %for.cond124.preheader ], [ %inc157, %for.cond129.preheader ]
  %add134 = add nuw nsw i32 %resIdxM123.0222, %mul133
  %mul135 = mul i32 %add134, %N
  %add137 = add i32 %mul135, %mul136.pre-phi
  %mul139 = shl nuw nsw i32 %resIdxM123.0222, 3
  %idxprom141 = zext nneg i32 %mul139 to i64
  %arrayidx142 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %idxprom141
  %95 = load float, ptr %arrayidx142, align 4, !tbaa !14
  %mul143 = fmul contract float %alpha, %95
  %idxprom145 = sext i32 %add137 to i64
  %arrayidx146 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom145
  %96 = load i16, ptr %arrayidx146, align 2, !tbaa !7
  %97 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %96) #6, !srcloc !9
  %mul148 = fmul contract float %beta, %97
  %add149 = fadd contract float %mul143, %mul148
  %98 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add149) #6, !srcloc !10
  store i16 %98, ptr %arrayidx146, align 2, !tbaa !7
  %add138.1 = add i32 %add137, 1
  %99 = zext nneg i32 %mul139 to i64
  %100 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %99
  %arrayidx142.1 = getelementptr inbounds nuw i8, ptr %100, i64 4
  %101 = load float, ptr %arrayidx142.1, align 4, !tbaa !14
  %mul143.1 = fmul contract float %alpha, %101
  %idxprom145.1 = sext i32 %add138.1 to i64
  %arrayidx146.1 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom145.1
  %102 = load i16, ptr %arrayidx146.1, align 2, !tbaa !7
  %103 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %102) #6, !srcloc !9
  %mul148.1 = fmul contract float %beta, %103
  %add149.1 = fadd contract float %mul143.1, %mul148.1
  %104 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add149.1) #6, !srcloc !10
  store i16 %104, ptr %arrayidx146.1, align 2, !tbaa !7
  %add138.2 = add i32 %add137, 2
  %105 = zext nneg i32 %mul139 to i64
  %106 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %105
  %arrayidx142.2 = getelementptr inbounds nuw i8, ptr %106, i64 8
  %107 = load float, ptr %arrayidx142.2, align 4, !tbaa !14
  %mul143.2 = fmul contract float %alpha, %107
  %idxprom145.2 = sext i32 %add138.2 to i64
  %arrayidx146.2 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom145.2
  %108 = load i16, ptr %arrayidx146.2, align 2, !tbaa !7
  %109 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %108) #6, !srcloc !9
  %mul148.2 = fmul contract float %beta, %109
  %add149.2 = fadd contract float %mul143.2, %mul148.2
  %110 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add149.2) #6, !srcloc !10
  store i16 %110, ptr %arrayidx146.2, align 2, !tbaa !7
  %add138.3 = add i32 %add137, 3
  %111 = zext nneg i32 %mul139 to i64
  %112 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %111
  %arrayidx142.3 = getelementptr inbounds nuw i8, ptr %112, i64 12
  %113 = load float, ptr %arrayidx142.3, align 4, !tbaa !14
  %mul143.3 = fmul contract float %alpha, %113
  %idxprom145.3 = sext i32 %add138.3 to i64
  %arrayidx146.3 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom145.3
  %114 = load i16, ptr %arrayidx146.3, align 2, !tbaa !7
  %115 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %114) #6, !srcloc !9
  %mul148.3 = fmul contract float %beta, %115
  %add149.3 = fadd contract float %mul143.3, %mul148.3
  %116 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add149.3) #6, !srcloc !10
  store i16 %116, ptr %arrayidx146.3, align 2, !tbaa !7
  %add138.4 = add i32 %add137, 4
  %117 = zext nneg i32 %mul139 to i64
  %118 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %117
  %arrayidx142.4 = getelementptr inbounds nuw i8, ptr %118, i64 16
  %119 = load float, ptr %arrayidx142.4, align 4, !tbaa !14
  %mul143.4 = fmul contract float %alpha, %119
  %idxprom145.4 = sext i32 %add138.4 to i64
  %arrayidx146.4 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom145.4
  %120 = load i16, ptr %arrayidx146.4, align 2, !tbaa !7
  %121 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %120) #6, !srcloc !9
  %mul148.4 = fmul contract float %beta, %121
  %add149.4 = fadd contract float %mul143.4, %mul148.4
  %122 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add149.4) #6, !srcloc !10
  store i16 %122, ptr %arrayidx146.4, align 2, !tbaa !7
  %add138.5 = add i32 %add137, 5
  %123 = zext nneg i32 %mul139 to i64
  %124 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %123
  %arrayidx142.5 = getelementptr inbounds nuw i8, ptr %124, i64 20
  %125 = load float, ptr %arrayidx142.5, align 4, !tbaa !14
  %mul143.5 = fmul contract float %alpha, %125
  %idxprom145.5 = sext i32 %add138.5 to i64
  %arrayidx146.5 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom145.5
  %126 = load i16, ptr %arrayidx146.5, align 2, !tbaa !7
  %127 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %126) #6, !srcloc !9
  %mul148.5 = fmul contract float %beta, %127
  %add149.5 = fadd contract float %mul143.5, %mul148.5
  %128 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add149.5) #6, !srcloc !10
  store i16 %128, ptr %arrayidx146.5, align 2, !tbaa !7
  %add138.6 = add i32 %add137, 6
  %129 = zext nneg i32 %mul139 to i64
  %130 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %129
  %arrayidx142.6 = getelementptr inbounds nuw i8, ptr %130, i64 24
  %131 = load float, ptr %arrayidx142.6, align 4, !tbaa !14
  %mul143.6 = fmul contract float %alpha, %131
  %idxprom145.6 = sext i32 %add138.6 to i64
  %arrayidx146.6 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom145.6
  %132 = load i16, ptr %arrayidx146.6, align 2, !tbaa !7
  %133 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %132) #6, !srcloc !9
  %mul148.6 = fmul contract float %beta, %133
  %add149.6 = fadd contract float %mul143.6, %mul148.6
  %134 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add149.6) #6, !srcloc !10
  store i16 %134, ptr %arrayidx146.6, align 2, !tbaa !7
  %add138.7 = add i32 %add137, 7
  %135 = zext nneg i32 %mul139 to i64
  %136 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %135
  %arrayidx142.7 = getelementptr inbounds nuw i8, ptr %136, i64 28
  %137 = load float, ptr %arrayidx142.7, align 4, !tbaa !14
  %mul143.7 = fmul contract float %alpha, %137
  %idxprom145.7 = sext i32 %add138.7 to i64
  %arrayidx146.7 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom145.7
  %138 = load i16, ptr %arrayidx146.7, align 2, !tbaa !7
  %139 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %138) #6, !srcloc !9
  %mul148.7 = fmul contract float %beta, %139
  %add149.7 = fadd contract float %mul143.7, %mul148.7
  %140 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add149.7) #6, !srcloc !10
  store i16 %140, ptr %arrayidx146.7, align 2, !tbaa !7
  %inc157 = add nuw nsw i32 %resIdxM123.0222, 1
  %exitcond225.not = icmp eq i32 %inc157, 8
  br i1 %exitcond225.not, label %for.cond.cleanup126, label %for.cond129.preheader, !llvm.loop !19

for.cond.cleanup126:                              ; preds = %for.cond129.preheader
  call void @llvm.lifetime.end.p0(ptr nonnull %regM) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %threadResults) #7
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_(i32 noundef %M, i32 noundef %N, i32 noundef %K, float noundef %alpha, ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, float noundef %beta, ptr noundef captures(none) %C) local_unnamed_addr #0 comdat {
entry:
  %threadResults = alloca [64 x float], align 4
  %regM = alloca [8 x float], align 4
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %rem = and i32 %2, 15
  %div237 = lshr i32 %2, 4
  %mul = shl nuw nsw i32 %0, 7
  %mul5 = shl i32 %1, 7
  %mul9 = mul i32 %N, %mul
  %add = add i32 %mul9, %mul5
  %idx.ext11 = zext i32 %add to i64
  %add.ptr12 = getelementptr inbounds nuw [2 x i8], ptr %C, i64 %idx.ext11
  call void @llvm.lifetime.start.p0(ptr nonnull %threadResults) #7
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(256) %threadResults, i8 0, i64 256, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %regM) #7
  %cmp245.not = icmp eq i32 %K, 0
  br i1 %cmp245.not, label %entry.for.cond141.preheader_crit_edge, label %for.body.lr.ph

entry.for.cond141.preheader_crit_edge:            ; preds = %entry
  %.pre = shl nuw nsw i32 %div237, 3
  %.pre254 = shl nuw nsw i32 %rem, 3
  br label %for.cond141.preheader

for.body.lr.ph:                                   ; preds = %entry
  %div18239 = lshr i32 %2, 5
  %div14238 = lshr i32 %2, 1
  %idx.ext6 = zext i32 %mul5 to i64
  %add.ptr7 = getelementptr inbounds nuw [2 x i8], ptr %B, i64 %idx.ext6
  %mul4 = mul i32 %K, %mul
  %idx.ext = zext i32 %mul4 to i64
  %add.ptr = getelementptr inbounds nuw [2 x i8], ptr %A, i64 %idx.ext
  %mul21 = mul i32 %K, %div14238
  %rem16 = shl nuw nsw i32 %2, 2
  %mul22 = and i32 %rem16, 4
  %add23 = add i32 %mul21, %mul22
  %idxprom = zext i32 %add23 to i64
  %mul30 = shl nuw nsw i32 %mul22, 7
  %add31 = or disjoint i32 %mul30, %div14238
  %idxprom32 = zext nneg i32 %add31 to i64
  %arrayidx33 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom32
  %mul36 = add nuw nsw i32 %div14238, 128
  %add37 = add nuw nsw i32 %mul36, %mul30
  %idxprom38 = zext nneg i32 %add37 to i64
  %arrayidx39 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom38
  %mul42 = add nuw nsw i32 %div14238, 256
  %add43 = add nuw nsw i32 %mul42, %mul30
  %idxprom44 = zext nneg i32 %add43 to i64
  %arrayidx45 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom44
  %mul48 = add nuw nsw i32 %div14238, 384
  %add49 = add nuw nsw i32 %mul48, %mul30
  %idxprom50 = zext nneg i32 %add49 to i64
  %arrayidx51 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom50
  %mul53 = mul i32 %N, %div18239
  %mul54 = and i32 %rem16, 124
  %add55 = add i32 %mul53, %mul54
  %idxprom56 = zext i32 %add55 to i64
  %mul58 = shl nuw nsw i32 %div18239, 7
  %add60 = or disjoint i32 %mul58, %mul54
  %idxprom61 = zext nneg i32 %add60 to i64
  %arrayidx62 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %idxprom61
  %arrayidx66 = getelementptr inbounds nuw i8, ptr %arrayidx62, i64 2
  %arrayidx68 = getelementptr inbounds nuw i8, ptr %arrayidx62, i64 4
  %arrayidx70 = getelementptr inbounds nuw i8, ptr %arrayidx62, i64 6
  %mul72 = shl nsw i32 %N, 3
  %idx.ext73 = sext i32 %mul72 to i64
  %mul84 = shl nuw nsw i32 %div237, 3
  %mul99 = shl nuw nsw i32 %rem, 3
  %arrayidx91.1 = getelementptr inbounds nuw i8, ptr %regM, i64 4
  %arrayidx91.2 = getelementptr inbounds nuw i8, ptr %regM, i64 8
  %arrayidx91.3 = getelementptr inbounds nuw i8, ptr %regM, i64 12
  %arrayidx91.4 = getelementptr inbounds nuw i8, ptr %regM, i64 16
  %arrayidx91.5 = getelementptr inbounds nuw i8, ptr %regM, i64 20
  %arrayidx91.6 = getelementptr inbounds nuw i8, ptr %regM, i64 24
  %arrayidx91.7 = getelementptr inbounds nuw i8, ptr %regM, i64 28
  br label %for.body

for.cond141.preheader:                            ; preds = %for.cond.cleanup77, %entry.for.cond141.preheader_crit_edge
  %mul153.pre-phi = phi i32 [ %.pre254, %entry.for.cond141.preheader_crit_edge ], [ %mul99, %for.cond.cleanup77 ]
  %mul150.pre-phi = phi i32 [ %.pre, %entry.for.cond141.preheader_crit_edge ], [ %mul84, %for.cond.cleanup77 ]
  br label %for.cond146.preheader

for.body:                                         ; preds = %for.body.lr.ph, %for.cond.cleanup77
  %A.addr.0248 = phi ptr [ %add.ptr, %for.body.lr.ph ], [ %add.ptr71, %for.cond.cleanup77 ]
  %B.addr.0247 = phi ptr [ %add.ptr7, %for.body.lr.ph ], [ %add.ptr74, %for.cond.cleanup77 ]
  %bkIdx.0246 = phi i32 [ 0, %for.body.lr.ph ], [ %add138, %for.cond.cleanup77 ]
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %A.addr.0248, i64 %idxprom
  %v0.sroa.0.0.copyload = load i16, ptr %arrayidx, align 2, !tbaa !7
  %arrayidx25 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 2
  %v1.sroa.0.0.copyload = load i16, ptr %arrayidx25, align 2, !tbaa !7
  %arrayidx26 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 4
  %v2.sroa.0.0.copyload = load i16, ptr %arrayidx26, align 2, !tbaa !7
  %arrayidx27 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 6
  %v3.sroa.0.0.copyload = load i16, ptr %arrayidx27, align 2, !tbaa !7
  store i16 %v0.sroa.0.0.copyload, ptr %arrayidx33, align 2, !tbaa !7
  store i16 %v1.sroa.0.0.copyload, ptr %arrayidx39, align 2, !tbaa !7
  store i16 %v2.sroa.0.0.copyload, ptr %arrayidx45, align 2, !tbaa !7
  store i16 %v3.sroa.0.0.copyload, ptr %arrayidx51, align 2, !tbaa !7
  %arrayidx57 = getelementptr inbounds nuw [2 x i8], ptr %B.addr.0247, i64 %idxprom56
  %3 = load i16, ptr %arrayidx57, align 2, !tbaa !7
  store i16 %3, ptr %arrayidx62, align 2, !tbaa !7
  %arrayidx65 = getelementptr inbounds nuw i8, ptr %arrayidx57, i64 2
  %4 = load i16, ptr %arrayidx65, align 2, !tbaa !7
  store i16 %4, ptr %arrayidx66, align 2, !tbaa !7
  %arrayidx67 = getelementptr inbounds nuw i8, ptr %arrayidx57, i64 4
  %5 = load i16, ptr %arrayidx67, align 2, !tbaa !7
  store i16 %5, ptr %arrayidx68, align 2, !tbaa !7
  %arrayidx69 = getelementptr inbounds nuw i8, ptr %arrayidx57, i64 6
  %6 = load i16, ptr %arrayidx69, align 2, !tbaa !7
  store i16 %6, ptr %arrayidx70, align 2, !tbaa !7
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %for.cond79.preheader

for.cond79.preheader:                             ; preds = %for.body, %for.cond.cleanup112
  %dotIdx.0244 = phi i32 [ 0, %for.body ], [ %inc135, %for.cond.cleanup112 ]
  %mul83 = shl nuw nsw i32 %dotIdx.0244, 7
  %add85 = add nuw nsw i32 %mul83, %mul84
  %idxprom87 = zext nneg i32 %add85 to i64
  %arrayidx88 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom87
  %7 = load i16, ptr %arrayidx88, align 2, !tbaa !7
  %8 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %7) #6, !srcloc !9
  store float %8, ptr %regM, align 4, !tbaa !14
  %9 = zext nneg i32 %add85 to i64
  %10 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %9
  %arrayidx88.1 = getelementptr inbounds nuw i8, ptr %10, i64 2
  %11 = load i16, ptr %arrayidx88.1, align 2, !tbaa !7
  %12 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %11) #6, !srcloc !9
  store float %12, ptr %arrayidx91.1, align 4, !tbaa !14
  %13 = zext nneg i32 %add85 to i64
  %14 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %13
  %arrayidx88.2 = getelementptr inbounds nuw i8, ptr %14, i64 4
  %15 = load i16, ptr %arrayidx88.2, align 2, !tbaa !7
  %16 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %15) #6, !srcloc !9
  store float %16, ptr %arrayidx91.2, align 4, !tbaa !14
  %17 = zext nneg i32 %add85 to i64
  %18 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %17
  %arrayidx88.3 = getelementptr inbounds nuw i8, ptr %18, i64 6
  %19 = load i16, ptr %arrayidx88.3, align 2, !tbaa !7
  %20 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %19) #6, !srcloc !9
  store float %20, ptr %arrayidx91.3, align 4, !tbaa !14
  %21 = zext nneg i32 %add85 to i64
  %22 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %21
  %arrayidx88.4 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %23 = load i16, ptr %arrayidx88.4, align 2, !tbaa !7
  %24 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %23) #6, !srcloc !9
  store float %24, ptr %arrayidx91.4, align 4, !tbaa !14
  %25 = zext nneg i32 %add85 to i64
  %26 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %25
  %arrayidx88.5 = getelementptr inbounds nuw i8, ptr %26, i64 10
  %27 = load i16, ptr %arrayidx88.5, align 2, !tbaa !7
  %28 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %27) #6, !srcloc !9
  store float %28, ptr %arrayidx91.5, align 4, !tbaa !14
  %29 = zext nneg i32 %add85 to i64
  %30 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %29
  %arrayidx88.6 = getelementptr inbounds nuw i8, ptr %30, i64 12
  %31 = load i16, ptr %arrayidx88.6, align 2, !tbaa !7
  %32 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %31) #6, !srcloc !9
  store float %32, ptr %arrayidx91.6, align 4, !tbaa !14
  %33 = zext nneg i32 %add85 to i64
  %34 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %33
  %arrayidx88.7 = getelementptr inbounds nuw i8, ptr %34, i64 14
  %35 = load i16, ptr %arrayidx88.7, align 2, !tbaa !7
  %36 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %35) #6, !srcloc !9
  store float %36, ptr %arrayidx91.7, align 4, !tbaa !14
  %add100 = or disjoint i32 %mul83, %mul99
  %idxprom102 = zext nneg i32 %add100 to i64
  %arrayidx103 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %idxprom102
  %37 = load i16, ptr %arrayidx103, align 2, !tbaa !7
  %38 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %37) #6, !srcloc !9
  %39 = zext nneg i32 %add100 to i64
  %40 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %39
  %arrayidx103.1 = getelementptr inbounds nuw i8, ptr %40, i64 2
  %41 = load i16, ptr %arrayidx103.1, align 2, !tbaa !7
  %42 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %41) #6, !srcloc !9
  %43 = zext nneg i32 %add100 to i64
  %44 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %43
  %arrayidx103.2 = getelementptr inbounds nuw i8, ptr %44, i64 4
  %45 = load i16, ptr %arrayidx103.2, align 2, !tbaa !7
  %46 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %45) #6, !srcloc !9
  %47 = zext nneg i32 %add100 to i64
  %48 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %47
  %arrayidx103.3 = getelementptr inbounds nuw i8, ptr %48, i64 6
  %49 = load i16, ptr %arrayidx103.3, align 2, !tbaa !7
  %50 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %49) #6, !srcloc !9
  %51 = zext nneg i32 %add100 to i64
  %52 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %51
  %arrayidx103.4 = getelementptr inbounds nuw i8, ptr %52, i64 8
  %53 = load i16, ptr %arrayidx103.4, align 2, !tbaa !7
  %54 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %53) #6, !srcloc !9
  %55 = zext nneg i32 %add100 to i64
  %56 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %55
  %arrayidx103.5 = getelementptr inbounds nuw i8, ptr %56, i64 10
  %57 = load i16, ptr %arrayidx103.5, align 2, !tbaa !7
  %58 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %57) #6, !srcloc !9
  %59 = zext nneg i32 %add100 to i64
  %60 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %59
  %arrayidx103.6 = getelementptr inbounds nuw i8, ptr %60, i64 12
  %61 = load i16, ptr %arrayidx103.6, align 2, !tbaa !7
  %62 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %61) #6, !srcloc !9
  %63 = zext nneg i32 %add100 to i64
  %64 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ16hgemm_vectorizedILi128ELi128ELi8ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %63
  %arrayidx103.7 = getelementptr inbounds nuw i8, ptr %64, i64 14
  %65 = load i16, ptr %arrayidx103.7, align 2, !tbaa !7
  %66 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %65) #6, !srcloc !9
  br label %for.cond114.preheader

for.cond.cleanup77:                               ; preds = %for.cond.cleanup112
  %add.ptr71 = getelementptr inbounds nuw i8, ptr %A.addr.0248, i64 16
  %add.ptr74 = getelementptr inbounds [2 x i8], ptr %B.addr.0247, i64 %idx.ext73
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %add138 = add i32 %bkIdx.0246, 8
  %cmp = icmp ult i32 %add138, %K
  br i1 %cmp, label %for.body, label %for.cond141.preheader, !llvm.loop !20

for.cond114.preheader:                            ; preds = %for.cond79.preheader, %for.cond114.preheader
  %resIdxM.0243 = phi i32 [ %inc132, %for.cond114.preheader ], [ 0, %for.cond79.preheader ]
  %idxprom118 = zext nneg i32 %resIdxM.0243 to i64
  %arrayidx119 = getelementptr inbounds nuw [4 x i8], ptr %regM, i64 %idxprom118
  %67 = load float, ptr %arrayidx119, align 4, !tbaa !14
  %mul123 = shl nuw nsw i32 %resIdxM.0243, 3
  %mul122 = fmul contract float %67, %38
  %idxprom125 = zext nneg i32 %mul123 to i64
  %arrayidx126 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %idxprom125
  %68 = load float, ptr %arrayidx126, align 4, !tbaa !14
  %add127 = fadd contract float %68, %mul122
  store float %add127, ptr %arrayidx126, align 4, !tbaa !14
  %mul122.1 = fmul contract float %67, %42
  %69 = zext nneg i32 %mul123 to i64
  %70 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %69
  %arrayidx126.1 = getelementptr inbounds nuw i8, ptr %70, i64 4
  %71 = load float, ptr %arrayidx126.1, align 4, !tbaa !14
  %add127.1 = fadd contract float %71, %mul122.1
  store float %add127.1, ptr %arrayidx126.1, align 4, !tbaa !14
  %mul122.2 = fmul contract float %67, %46
  %72 = zext nneg i32 %mul123 to i64
  %73 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %72
  %arrayidx126.2 = getelementptr inbounds nuw i8, ptr %73, i64 8
  %74 = load float, ptr %arrayidx126.2, align 4, !tbaa !14
  %add127.2 = fadd contract float %74, %mul122.2
  store float %add127.2, ptr %arrayidx126.2, align 4, !tbaa !14
  %mul122.3 = fmul contract float %67, %50
  %75 = zext nneg i32 %mul123 to i64
  %76 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %75
  %arrayidx126.3 = getelementptr inbounds nuw i8, ptr %76, i64 12
  %77 = load float, ptr %arrayidx126.3, align 4, !tbaa !14
  %add127.3 = fadd contract float %77, %mul122.3
  store float %add127.3, ptr %arrayidx126.3, align 4, !tbaa !14
  %mul122.4 = fmul contract float %67, %54
  %78 = zext nneg i32 %mul123 to i64
  %79 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %78
  %arrayidx126.4 = getelementptr inbounds nuw i8, ptr %79, i64 16
  %80 = load float, ptr %arrayidx126.4, align 4, !tbaa !14
  %add127.4 = fadd contract float %80, %mul122.4
  store float %add127.4, ptr %arrayidx126.4, align 4, !tbaa !14
  %mul122.5 = fmul contract float %67, %58
  %81 = zext nneg i32 %mul123 to i64
  %82 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %81
  %arrayidx126.5 = getelementptr inbounds nuw i8, ptr %82, i64 20
  %83 = load float, ptr %arrayidx126.5, align 4, !tbaa !14
  %add127.5 = fadd contract float %83, %mul122.5
  store float %add127.5, ptr %arrayidx126.5, align 4, !tbaa !14
  %mul122.6 = fmul contract float %67, %62
  %84 = zext nneg i32 %mul123 to i64
  %85 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %84
  %arrayidx126.6 = getelementptr inbounds nuw i8, ptr %85, i64 24
  %86 = load float, ptr %arrayidx126.6, align 4, !tbaa !14
  %add127.6 = fadd contract float %86, %mul122.6
  store float %add127.6, ptr %arrayidx126.6, align 4, !tbaa !14
  %mul122.7 = fmul contract float %67, %66
  %87 = zext nneg i32 %mul123 to i64
  %88 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %87
  %arrayidx126.7 = getelementptr inbounds nuw i8, ptr %88, i64 28
  %89 = load float, ptr %arrayidx126.7, align 4, !tbaa !14
  %add127.7 = fadd contract float %89, %mul122.7
  store float %add127.7, ptr %arrayidx126.7, align 4, !tbaa !14
  %inc132 = add nuw nsw i32 %resIdxM.0243, 1
  %exitcond.not = icmp eq i32 %inc132, 8
  br i1 %exitcond.not, label %for.cond.cleanup112, label %for.cond114.preheader, !llvm.loop !21

for.cond.cleanup112:                              ; preds = %for.cond114.preheader
  %inc135 = add nuw nsw i32 %dotIdx.0244, 1
  %exitcond251.not = icmp eq i32 %inc135, 8
  br i1 %exitcond251.not, label %for.cond.cleanup77, label %for.cond79.preheader, !llvm.loop !22

for.cond146.preheader:                            ; preds = %for.cond141.preheader, %for.cond146.preheader
  %resIdxM140.0250 = phi i32 [ 0, %for.cond141.preheader ], [ %inc174, %for.cond146.preheader ]
  %add151 = add nuw nsw i32 %resIdxM140.0250, %mul150.pre-phi
  %mul152 = mul i32 %add151, %N
  %add154 = add i32 %mul152, %mul153.pre-phi
  %mul156 = shl nuw nsw i32 %resIdxM140.0250, 3
  %idxprom158 = zext nneg i32 %mul156 to i64
  %arrayidx159 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %idxprom158
  %90 = load float, ptr %arrayidx159, align 4, !tbaa !14
  %mul160 = fmul contract float %alpha, %90
  %idxprom162 = sext i32 %add154 to i64
  %arrayidx163 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom162
  %91 = load i16, ptr %arrayidx163, align 2, !tbaa !7
  %92 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %91) #6, !srcloc !9
  %mul165 = fmul contract float %beta, %92
  %add166 = fadd contract float %mul160, %mul165
  %93 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add166) #6, !srcloc !10
  store i16 %93, ptr %arrayidx163, align 2, !tbaa !7
  %add155.1 = add i32 %add154, 1
  %94 = zext nneg i32 %mul156 to i64
  %95 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %94
  %arrayidx159.1 = getelementptr inbounds nuw i8, ptr %95, i64 4
  %96 = load float, ptr %arrayidx159.1, align 4, !tbaa !14
  %mul160.1 = fmul contract float %alpha, %96
  %idxprom162.1 = sext i32 %add155.1 to i64
  %arrayidx163.1 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom162.1
  %97 = load i16, ptr %arrayidx163.1, align 2, !tbaa !7
  %98 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %97) #6, !srcloc !9
  %mul165.1 = fmul contract float %beta, %98
  %add166.1 = fadd contract float %mul160.1, %mul165.1
  %99 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add166.1) #6, !srcloc !10
  store i16 %99, ptr %arrayidx163.1, align 2, !tbaa !7
  %add155.2 = add i32 %add154, 2
  %100 = zext nneg i32 %mul156 to i64
  %101 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %100
  %arrayidx159.2 = getelementptr inbounds nuw i8, ptr %101, i64 8
  %102 = load float, ptr %arrayidx159.2, align 4, !tbaa !14
  %mul160.2 = fmul contract float %alpha, %102
  %idxprom162.2 = sext i32 %add155.2 to i64
  %arrayidx163.2 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom162.2
  %103 = load i16, ptr %arrayidx163.2, align 2, !tbaa !7
  %104 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %103) #6, !srcloc !9
  %mul165.2 = fmul contract float %beta, %104
  %add166.2 = fadd contract float %mul160.2, %mul165.2
  %105 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add166.2) #6, !srcloc !10
  store i16 %105, ptr %arrayidx163.2, align 2, !tbaa !7
  %add155.3 = add i32 %add154, 3
  %106 = zext nneg i32 %mul156 to i64
  %107 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %106
  %arrayidx159.3 = getelementptr inbounds nuw i8, ptr %107, i64 12
  %108 = load float, ptr %arrayidx159.3, align 4, !tbaa !14
  %mul160.3 = fmul contract float %alpha, %108
  %idxprom162.3 = sext i32 %add155.3 to i64
  %arrayidx163.3 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom162.3
  %109 = load i16, ptr %arrayidx163.3, align 2, !tbaa !7
  %110 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %109) #6, !srcloc !9
  %mul165.3 = fmul contract float %beta, %110
  %add166.3 = fadd contract float %mul160.3, %mul165.3
  %111 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add166.3) #6, !srcloc !10
  store i16 %111, ptr %arrayidx163.3, align 2, !tbaa !7
  %add155.4 = add i32 %add154, 4
  %112 = zext nneg i32 %mul156 to i64
  %113 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %112
  %arrayidx159.4 = getelementptr inbounds nuw i8, ptr %113, i64 16
  %114 = load float, ptr %arrayidx159.4, align 4, !tbaa !14
  %mul160.4 = fmul contract float %alpha, %114
  %idxprom162.4 = sext i32 %add155.4 to i64
  %arrayidx163.4 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom162.4
  %115 = load i16, ptr %arrayidx163.4, align 2, !tbaa !7
  %116 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %115) #6, !srcloc !9
  %mul165.4 = fmul contract float %beta, %116
  %add166.4 = fadd contract float %mul160.4, %mul165.4
  %117 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add166.4) #6, !srcloc !10
  store i16 %117, ptr %arrayidx163.4, align 2, !tbaa !7
  %add155.5 = add i32 %add154, 5
  %118 = zext nneg i32 %mul156 to i64
  %119 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %118
  %arrayidx159.5 = getelementptr inbounds nuw i8, ptr %119, i64 20
  %120 = load float, ptr %arrayidx159.5, align 4, !tbaa !14
  %mul160.5 = fmul contract float %alpha, %120
  %idxprom162.5 = sext i32 %add155.5 to i64
  %arrayidx163.5 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom162.5
  %121 = load i16, ptr %arrayidx163.5, align 2, !tbaa !7
  %122 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %121) #6, !srcloc !9
  %mul165.5 = fmul contract float %beta, %122
  %add166.5 = fadd contract float %mul160.5, %mul165.5
  %123 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add166.5) #6, !srcloc !10
  store i16 %123, ptr %arrayidx163.5, align 2, !tbaa !7
  %add155.6 = add i32 %add154, 6
  %124 = zext nneg i32 %mul156 to i64
  %125 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %124
  %arrayidx159.6 = getelementptr inbounds nuw i8, ptr %125, i64 24
  %126 = load float, ptr %arrayidx159.6, align 4, !tbaa !14
  %mul160.6 = fmul contract float %alpha, %126
  %idxprom162.6 = sext i32 %add155.6 to i64
  %arrayidx163.6 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom162.6
  %127 = load i16, ptr %arrayidx163.6, align 2, !tbaa !7
  %128 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %127) #6, !srcloc !9
  %mul165.6 = fmul contract float %beta, %128
  %add166.6 = fadd contract float %mul160.6, %mul165.6
  %129 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add166.6) #6, !srcloc !10
  store i16 %129, ptr %arrayidx163.6, align 2, !tbaa !7
  %add155.7 = add i32 %add154, 7
  %130 = zext nneg i32 %mul156 to i64
  %131 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %130
  %arrayidx159.7 = getelementptr inbounds nuw i8, ptr %131, i64 28
  %132 = load float, ptr %arrayidx159.7, align 4, !tbaa !14
  %mul160.7 = fmul contract float %alpha, %132
  %idxprom162.7 = sext i32 %add155.7 to i64
  %arrayidx163.7 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom162.7
  %133 = load i16, ptr %arrayidx163.7, align 2, !tbaa !7
  %134 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %133) #6, !srcloc !9
  %mul165.7 = fmul contract float %beta, %134
  %add166.7 = fadd contract float %mul160.7, %mul165.7
  %135 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add166.7) #6, !srcloc !10
  store i16 %135, ptr %arrayidx163.7, align 2, !tbaa !7
  %inc174 = add nuw nsw i32 %resIdxM140.0250, 1
  %exitcond253.not = icmp eq i32 %inc174, 8
  br i1 %exitcond253.not, label %for.cond.cleanup143, label %for.cond146.preheader, !llvm.loop !23

for.cond.cleanup143:                              ; preds = %for.cond146.preheader
  call void @llvm.lifetime.end.p0(ptr nonnull %regM) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %threadResults) #7
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define dso_local ptx_kernel void @_Z15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_(i32 noundef %M, i32 noundef %N, i32 noundef %K, float noundef %alpha, ptr noundef readonly captures(none) %A, ptr noundef readonly captures(none) %B, float noundef %beta, ptr noundef captures(none) %C) local_unnamed_addr #4 comdat {
entry:
  %threadResults = alloca [64 x float], align 4
  %regM = alloca [8 x float], align 4
  %0 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %2 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %rem = and i32 %2, 15
  %div332 = lshr i32 %2, 4
  %mul = shl nuw nsw i32 %0, 7
  %mul5 = shl i32 %1, 7
  %mul9 = mul i32 %N, %mul
  %add = add i32 %mul9, %mul5
  %idx.ext11 = zext i32 %add to i64
  %add.ptr12 = getelementptr inbounds nuw [2 x i8], ptr %C, i64 %idx.ext11
  call void @llvm.lifetime.start.p0(ptr nonnull %threadResults) #7
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(256) %threadResults, i8 0, i64 256, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %regM) #7
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(32) %regM, i8 0, i64 32, i1 false)
  %cmp342.not = icmp eq i32 %K, 0
  br i1 %cmp342.not, label %entry.for.cond187.preheader_crit_edge, label %for.cond21.preheader.lr.ph

entry.for.cond187.preheader_crit_edge:            ; preds = %entry
  %.pre = shl nuw nsw i32 %div332, 3
  %.pre351 = shl nuw nsw i32 %rem, 3
  br label %for.cond187.preheader

for.cond21.preheader.lr.ph:                       ; preds = %entry
  %div18334 = lshr i32 %2, 5
  %div14333 = lshr i32 %2, 2
  %idx.ext6 = zext i32 %mul5 to i64
  %add.ptr7 = getelementptr inbounds nuw [2 x i8], ptr %B, i64 %idx.ext6
  %mul4 = mul i32 %K, %mul
  %idx.ext = zext i32 %mul4 to i64
  %add.ptr = getelementptr inbounds nuw [2 x i8], ptr %A, i64 %idx.ext
  %rem16 = shl nuw nsw i32 %2, 2
  %mul28 = and i32 %rem16, 12
  %mul36 = shl nuw nsw i32 %mul28, 7
  %add37 = or disjoint i32 %mul36, %div14333
  %mul43 = add nuw nsw i32 %div14333, 128
  %add44 = or disjoint i32 %mul43, %mul36
  %mul57 = add nuw nsw i32 %div14333, 384
  %add58 = add nuw nsw i32 %mul57, %mul36
  %mul72 = and i32 %rem16, 124
  %mul112 = shl nuw nsw i32 %div332, 3
  %mul131 = shl nuw nsw i32 %rem, 3
  %mul180 = shl nsw i32 %N, 4
  %idx.ext181 = sext i32 %mul180 to i64
  %mul27 = mul i32 %div14333, %K
  %add29 = add i32 %mul27, %mul28
  %idxprom = zext i32 %add29 to i64
  %idxprom39 = zext nneg i32 %add37 to i64
  %arrayidx40 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom39
  %idxprom46 = zext nneg i32 %add44 to i64
  %arrayidx47 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom46
  %3 = zext nneg i32 %add37 to i64
  %4 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %3
  %arrayidx54 = getelementptr inbounds nuw i8, ptr %4, i64 512
  %idxprom60 = zext nneg i32 %add58 to i64
  %arrayidx61 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom60
  %add26.1 = add nuw nsw i32 %div14333, 64
  %mul27.1 = mul i32 %add26.1, %K
  %add29.1 = add i32 %mul27.1, %mul28
  %idxprom.1 = zext i32 %add29.1 to i64
  %5 = zext nneg i32 %add37 to i64
  %6 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %5
  %arrayidx40.1 = getelementptr inbounds nuw i8, ptr %6, i64 128
  %7 = zext nneg i32 %add44 to i64
  %8 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %7
  %arrayidx47.1 = getelementptr inbounds nuw i8, ptr %8, i64 128
  %9 = zext nneg i32 %add37 to i64
  %10 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %9
  %arrayidx54.1 = getelementptr inbounds nuw i8, ptr %10, i64 640
  %11 = zext nneg i32 %add58 to i64
  %12 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %11
  %arrayidx61.1 = getelementptr inbounds nuw i8, ptr %12, i64 128
  %mul71 = mul i32 %div18334, %N
  %add73 = add i32 %mul71, %mul72
  %idxprom74 = zext i32 %add73 to i64
  %mul77 = shl nuw nsw i32 %div18334, 7
  %add79 = or disjoint i32 %mul77, %mul72
  %idxprom80 = zext nneg i32 %add79 to i64
  %arrayidx81 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %idxprom80
  %arrayidx85 = getelementptr inbounds nuw i8, ptr %arrayidx81, i64 2
  %arrayidx87 = getelementptr inbounds nuw i8, ptr %arrayidx81, i64 4
  %arrayidx89 = getelementptr inbounds nuw i8, ptr %arrayidx81, i64 6
  %add70.1 = add nuw nsw i32 %div18334, 8
  %mul71.1 = mul i32 %add70.1, %N
  %add73.1 = add i32 %mul71.1, %mul72
  %idxprom74.1 = zext i32 %add73.1 to i64
  %mul77.1 = shl nuw nsw i32 %add70.1, 7
  %add79.1 = or disjoint i32 %mul77.1, %mul72
  %idxprom80.1 = zext nneg i32 %add79.1 to i64
  %arrayidx81.1 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %idxprom80.1
  %arrayidx85.1 = getelementptr inbounds nuw i8, ptr %arrayidx81.1, i64 2
  %arrayidx87.1 = getelementptr inbounds nuw i8, ptr %arrayidx81.1, i64 4
  %arrayidx89.1 = getelementptr inbounds nuw i8, ptr %arrayidx81.1, i64 6
  %arrayidx119.1 = getelementptr inbounds nuw i8, ptr %regM, i64 4
  %arrayidx119.2 = getelementptr inbounds nuw i8, ptr %regM, i64 8
  %arrayidx119.3 = getelementptr inbounds nuw i8, ptr %regM, i64 12
  %arrayidx119.4 = getelementptr inbounds nuw i8, ptr %regM, i64 16
  %arrayidx119.5 = getelementptr inbounds nuw i8, ptr %regM, i64 20
  %arrayidx119.6 = getelementptr inbounds nuw i8, ptr %regM, i64 24
  %arrayidx119.7 = getelementptr inbounds nuw i8, ptr %regM, i64 28
  br label %for.cond21.preheader

for.cond21.preheader:                             ; preds = %for.cond21.preheader.lr.ph, %for.cond97.loopexit
  %A.addr.0345 = phi ptr [ %add.ptr, %for.cond21.preheader.lr.ph ], [ %add.ptr179, %for.cond97.loopexit ]
  %B.addr.0344 = phi ptr [ %add.ptr7, %for.cond21.preheader.lr.ph ], [ %add.ptr182, %for.cond97.loopexit ]
  %bkIdx.0343 = phi i32 [ 0, %for.cond21.preheader.lr.ph ], [ %add184, %for.cond97.loopexit ]
  %arrayidx = getelementptr inbounds nuw [2 x i8], ptr %A.addr.0345, i64 %idxprom
  %v0.sroa.0.0.copyload = load i16, ptr %arrayidx, align 2, !tbaa !7
  %arrayidx31 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 2
  %v1.sroa.0.0.copyload = load i16, ptr %arrayidx31, align 2, !tbaa !7
  %arrayidx32 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 4
  %v2.sroa.0.0.copyload = load i16, ptr %arrayidx32, align 2, !tbaa !7
  %arrayidx33 = getelementptr inbounds nuw i8, ptr %arrayidx, i64 6
  %v3.sroa.0.0.copyload = load i16, ptr %arrayidx33, align 2, !tbaa !7
  store i16 %v0.sroa.0.0.copyload, ptr %arrayidx40, align 2, !tbaa !7
  store i16 %v1.sroa.0.0.copyload, ptr %arrayidx47, align 2, !tbaa !7
  store i16 %v2.sroa.0.0.copyload, ptr %arrayidx54, align 2, !tbaa !7
  store i16 %v3.sroa.0.0.copyload, ptr %arrayidx61, align 2, !tbaa !7
  %arrayidx.1 = getelementptr inbounds nuw [2 x i8], ptr %A.addr.0345, i64 %idxprom.1
  %v0.sroa.0.0.copyload.1 = load i16, ptr %arrayidx.1, align 2, !tbaa !7
  %arrayidx31.1 = getelementptr inbounds nuw i8, ptr %arrayidx.1, i64 2
  %v1.sroa.0.0.copyload.1 = load i16, ptr %arrayidx31.1, align 2, !tbaa !7
  %arrayidx32.1 = getelementptr inbounds nuw i8, ptr %arrayidx.1, i64 4
  %v2.sroa.0.0.copyload.1 = load i16, ptr %arrayidx32.1, align 2, !tbaa !7
  %arrayidx33.1 = getelementptr inbounds nuw i8, ptr %arrayidx.1, i64 6
  %v3.sroa.0.0.copyload.1 = load i16, ptr %arrayidx33.1, align 2, !tbaa !7
  store i16 %v0.sroa.0.0.copyload.1, ptr %arrayidx40.1, align 2, !tbaa !7
  store i16 %v1.sroa.0.0.copyload.1, ptr %arrayidx47.1, align 2, !tbaa !7
  store i16 %v2.sroa.0.0.copyload.1, ptr %arrayidx54.1, align 2, !tbaa !7
  store i16 %v3.sroa.0.0.copyload.1, ptr %arrayidx61.1, align 2, !tbaa !7
  %arrayidx75 = getelementptr inbounds nuw [2 x i8], ptr %B.addr.0344, i64 %idxprom74
  %13 = load i16, ptr %arrayidx75, align 2, !tbaa !7
  store i16 %13, ptr %arrayidx81, align 2, !tbaa !7
  %arrayidx84 = getelementptr inbounds nuw i8, ptr %arrayidx75, i64 2
  %14 = load i16, ptr %arrayidx84, align 2, !tbaa !7
  store i16 %14, ptr %arrayidx85, align 2, !tbaa !7
  %arrayidx86 = getelementptr inbounds nuw i8, ptr %arrayidx75, i64 4
  %15 = load i16, ptr %arrayidx86, align 2, !tbaa !7
  store i16 %15, ptr %arrayidx87, align 2, !tbaa !7
  %arrayidx88 = getelementptr inbounds nuw i8, ptr %arrayidx75, i64 6
  %16 = load i16, ptr %arrayidx88, align 2, !tbaa !7
  store i16 %16, ptr %arrayidx89, align 2, !tbaa !7
  %arrayidx75.1 = getelementptr inbounds nuw [2 x i8], ptr %B.addr.0344, i64 %idxprom74.1
  %17 = load i16, ptr %arrayidx75.1, align 2, !tbaa !7
  store i16 %17, ptr %arrayidx81.1, align 2, !tbaa !7
  %arrayidx84.1 = getelementptr inbounds nuw i8, ptr %arrayidx75.1, i64 2
  %18 = load i16, ptr %arrayidx84.1, align 2, !tbaa !7
  store i16 %18, ptr %arrayidx85.1, align 2, !tbaa !7
  %arrayidx86.1 = getelementptr inbounds nuw i8, ptr %arrayidx75.1, i64 4
  %19 = load i16, ptr %arrayidx86.1, align 2, !tbaa !7
  store i16 %19, ptr %arrayidx87.1, align 2, !tbaa !7
  %arrayidx88.1 = getelementptr inbounds nuw i8, ptr %arrayidx75.1, i64 6
  %20 = load i16, ptr %arrayidx88.1, align 2, !tbaa !7
  store i16 %20, ptr %arrayidx89.1, align 2, !tbaa !7
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  br label %for.cond105.preheader

for.cond187.preheader:                            ; preds = %for.cond97.loopexit, %entry.for.cond187.preheader_crit_edge
  %mul216.pre-phi = phi i32 [ %.pre351, %entry.for.cond187.preheader_crit_edge ], [ %mul131, %for.cond97.loopexit ]
  %mul213.pre-phi = phi i32 [ %.pre, %entry.for.cond187.preheader_crit_edge ], [ %mul112, %for.cond97.loopexit ]
  br label %for.cond209.preheader

for.cond97.loopexit:                              ; preds = %for.cond.cleanup144
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  %add.ptr179 = getelementptr inbounds nuw i8, ptr %A.addr.0345, i64 32
  %add.ptr182 = getelementptr inbounds [2 x i8], ptr %B.addr.0344, i64 %idx.ext181
  %add184 = add i32 %bkIdx.0343, 16
  %cmp = icmp ult i32 %add184, %K
  br i1 %cmp, label %for.cond21.preheader, label %for.cond187.preheader, !llvm.loop !24

for.cond105.preheader:                            ; preds = %for.cond21.preheader, %for.cond.cleanup144
  %dotIdx.0341 = phi i32 [ 0, %for.cond21.preheader ], [ %inc171, %for.cond.cleanup144 ]
  %mul109 = shl nuw nsw i32 %dotIdx.0341, 7
  %add113 = add nuw nsw i32 %mul109, %mul112
  %idxprom115 = zext nneg i32 %add113 to i64
  %arrayidx116 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %idxprom115
  %21 = load i16, ptr %arrayidx116, align 2, !tbaa !7
  %22 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %21) #6, !srcloc !9
  store float %22, ptr %regM, align 4, !tbaa !14
  %23 = zext nneg i32 %add113 to i64
  %24 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %23
  %arrayidx116.1 = getelementptr inbounds nuw i8, ptr %24, i64 2
  %25 = load i16, ptr %arrayidx116.1, align 2, !tbaa !7
  %26 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %25) #6, !srcloc !9
  store float %26, ptr %arrayidx119.1, align 4, !tbaa !14
  %27 = zext nneg i32 %add113 to i64
  %28 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %27
  %arrayidx116.2 = getelementptr inbounds nuw i8, ptr %28, i64 4
  %29 = load i16, ptr %arrayidx116.2, align 2, !tbaa !7
  %30 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %29) #6, !srcloc !9
  store float %30, ptr %arrayidx119.2, align 4, !tbaa !14
  %31 = zext nneg i32 %add113 to i64
  %32 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %31
  %arrayidx116.3 = getelementptr inbounds nuw i8, ptr %32, i64 6
  %33 = load i16, ptr %arrayidx116.3, align 2, !tbaa !7
  %34 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %33) #6, !srcloc !9
  store float %34, ptr %arrayidx119.3, align 4, !tbaa !14
  %35 = zext nneg i32 %add113 to i64
  %36 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %35
  %arrayidx116.4 = getelementptr inbounds nuw i8, ptr %36, i64 8
  %37 = load i16, ptr %arrayidx116.4, align 2, !tbaa !7
  %38 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %37) #6, !srcloc !9
  store float %38, ptr %arrayidx119.4, align 4, !tbaa !14
  %39 = zext nneg i32 %add113 to i64
  %40 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %39
  %arrayidx116.5 = getelementptr inbounds nuw i8, ptr %40, i64 10
  %41 = load i16, ptr %arrayidx116.5, align 2, !tbaa !7
  %42 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %41) #6, !srcloc !9
  store float %42, ptr %arrayidx119.5, align 4, !tbaa !14
  %43 = zext nneg i32 %add113 to i64
  %44 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %43
  %arrayidx116.6 = getelementptr inbounds nuw i8, ptr %44, i64 12
  %45 = load i16, ptr %arrayidx116.6, align 2, !tbaa !7
  %46 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %45) #6, !srcloc !9
  store float %46, ptr %arrayidx119.6, align 4, !tbaa !14
  %47 = zext nneg i32 %add113 to i64
  %48 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2As to ptr), i64 %47
  %arrayidx116.7 = getelementptr inbounds nuw i8, ptr %48, i64 14
  %49 = load i16, ptr %arrayidx116.7, align 2, !tbaa !7
  %50 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %49) #6, !srcloc !9
  store float %50, ptr %arrayidx119.7, align 4, !tbaa !14
  %add132 = or disjoint i32 %mul109, %mul131
  %idxprom134 = zext nneg i32 %add132 to i64
  %arrayidx135 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %idxprom134
  %51 = load i16, ptr %arrayidx135, align 2, !tbaa !7
  %52 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %51) #6, !srcloc !9
  %53 = zext nneg i32 %add132 to i64
  %54 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %53
  %arrayidx135.1 = getelementptr inbounds nuw i8, ptr %54, i64 2
  %55 = load i16, ptr %arrayidx135.1, align 2, !tbaa !7
  %56 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %55) #6, !srcloc !9
  %57 = zext nneg i32 %add132 to i64
  %58 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %57
  %arrayidx135.2 = getelementptr inbounds nuw i8, ptr %58, i64 4
  %59 = load i16, ptr %arrayidx135.2, align 2, !tbaa !7
  %60 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %59) #6, !srcloc !9
  %61 = zext nneg i32 %add132 to i64
  %62 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %61
  %arrayidx135.3 = getelementptr inbounds nuw i8, ptr %62, i64 6
  %63 = load i16, ptr %arrayidx135.3, align 2, !tbaa !7
  %64 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %63) #6, !srcloc !9
  %65 = zext nneg i32 %add132 to i64
  %66 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %65
  %arrayidx135.4 = getelementptr inbounds nuw i8, ptr %66, i64 8
  %67 = load i16, ptr %arrayidx135.4, align 2, !tbaa !7
  %68 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %67) #6, !srcloc !9
  %69 = zext nneg i32 %add132 to i64
  %70 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %69
  %arrayidx135.5 = getelementptr inbounds nuw i8, ptr %70, i64 10
  %71 = load i16, ptr %arrayidx135.5, align 2, !tbaa !7
  %72 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %71) #6, !srcloc !9
  %73 = zext nneg i32 %add132 to i64
  %74 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %73
  %arrayidx135.6 = getelementptr inbounds nuw i8, ptr %74, i64 12
  %75 = load i16, ptr %arrayidx135.6, align 2, !tbaa !7
  %76 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %75) #6, !srcloc !9
  %77 = zext nneg i32 %add132 to i64
  %78 = getelementptr inbounds nuw [2 x i8], ptr addrspacecast (ptr addrspace(3) @_ZZ15hgemm_autotunedILi128ELi128ELi16ELi8ELi8EEviiifP6__halfS1_fS1_E2Bs to ptr), i64 %77
  %arrayidx135.7 = getelementptr inbounds nuw i8, ptr %78, i64 14
  %79 = load i16, ptr %arrayidx135.7, align 2, !tbaa !7
  %80 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %79) #6, !srcloc !9
  br label %for.cond146.preheader

for.cond146.preheader:                            ; preds = %for.cond105.preheader, %for.cond146.preheader
  %resIdxM.0340 = phi i32 [ %inc168, %for.cond146.preheader ], [ 0, %for.cond105.preheader ]
  %idxprom150 = zext nneg i32 %resIdxM.0340 to i64
  %arrayidx151 = getelementptr inbounds nuw [4 x i8], ptr %regM, i64 %idxprom150
  %81 = load float, ptr %arrayidx151, align 4, !tbaa !14
  %mul157 = shl nuw nsw i32 %resIdxM.0340, 3
  %mul154 = fmul contract float %81, %52
  %idxprom161 = zext nneg i32 %mul157 to i64
  %arrayidx162 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %idxprom161
  %82 = load float, ptr %arrayidx162, align 4, !tbaa !14
  %add163 = fadd contract float %82, %mul154
  store float %add163, ptr %arrayidx162, align 4, !tbaa !14
  %mul154.1 = fmul contract float %81, %56
  %83 = zext nneg i32 %mul157 to i64
  %84 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %83
  %arrayidx162.1 = getelementptr inbounds nuw i8, ptr %84, i64 4
  %85 = load float, ptr %arrayidx162.1, align 4, !tbaa !14
  %add163.1 = fadd contract float %85, %mul154.1
  store float %add163.1, ptr %arrayidx162.1, align 4, !tbaa !14
  %mul154.2 = fmul contract float %81, %60
  %86 = zext nneg i32 %mul157 to i64
  %87 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %86
  %arrayidx162.2 = getelementptr inbounds nuw i8, ptr %87, i64 8
  %88 = load float, ptr %arrayidx162.2, align 4, !tbaa !14
  %add163.2 = fadd contract float %88, %mul154.2
  store float %add163.2, ptr %arrayidx162.2, align 4, !tbaa !14
  %mul154.3 = fmul contract float %81, %64
  %89 = zext nneg i32 %mul157 to i64
  %90 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %89
  %arrayidx162.3 = getelementptr inbounds nuw i8, ptr %90, i64 12
  %91 = load float, ptr %arrayidx162.3, align 4, !tbaa !14
  %add163.3 = fadd contract float %91, %mul154.3
  store float %add163.3, ptr %arrayidx162.3, align 4, !tbaa !14
  %mul154.4 = fmul contract float %81, %68
  %92 = zext nneg i32 %mul157 to i64
  %93 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %92
  %arrayidx162.4 = getelementptr inbounds nuw i8, ptr %93, i64 16
  %94 = load float, ptr %arrayidx162.4, align 4, !tbaa !14
  %add163.4 = fadd contract float %94, %mul154.4
  store float %add163.4, ptr %arrayidx162.4, align 4, !tbaa !14
  %mul154.5 = fmul contract float %81, %72
  %95 = zext nneg i32 %mul157 to i64
  %96 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %95
  %arrayidx162.5 = getelementptr inbounds nuw i8, ptr %96, i64 20
  %97 = load float, ptr %arrayidx162.5, align 4, !tbaa !14
  %add163.5 = fadd contract float %97, %mul154.5
  store float %add163.5, ptr %arrayidx162.5, align 4, !tbaa !14
  %mul154.6 = fmul contract float %81, %76
  %98 = zext nneg i32 %mul157 to i64
  %99 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %98
  %arrayidx162.6 = getelementptr inbounds nuw i8, ptr %99, i64 24
  %100 = load float, ptr %arrayidx162.6, align 4, !tbaa !14
  %add163.6 = fadd contract float %100, %mul154.6
  store float %add163.6, ptr %arrayidx162.6, align 4, !tbaa !14
  %mul154.7 = fmul contract float %81, %80
  %101 = zext nneg i32 %mul157 to i64
  %102 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %101
  %arrayidx162.7 = getelementptr inbounds nuw i8, ptr %102, i64 28
  %103 = load float, ptr %arrayidx162.7, align 4, !tbaa !14
  %add163.7 = fadd contract float %103, %mul154.7
  store float %add163.7, ptr %arrayidx162.7, align 4, !tbaa !14
  %inc168 = add nuw nsw i32 %resIdxM.0340, 1
  %exitcond.not = icmp eq i32 %inc168, 8
  br i1 %exitcond.not, label %for.cond.cleanup144, label %for.cond146.preheader, !llvm.loop !25

for.cond.cleanup144:                              ; preds = %for.cond146.preheader
  %inc171 = add nuw nsw i32 %dotIdx.0341, 1
  %exitcond348.not = icmp eq i32 %inc171, 16
  br i1 %exitcond348.not, label %for.cond97.loopexit, label %for.cond105.preheader, !llvm.loop !26

for.cond192.loopexit:                             ; preds = %for.cond209.preheader
  call void @llvm.lifetime.end.p0(ptr nonnull %regM) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %threadResults) #7
  ret void

for.cond209.preheader:                            ; preds = %for.cond187.preheader, %for.cond209.preheader
  %resIdxM203.0347 = phi i32 [ 0, %for.cond187.preheader ], [ %inc241, %for.cond209.preheader ]
  %add214 = add nuw nsw i32 %resIdxM203.0347, %mul213.pre-phi
  %mul215 = mul i32 %add214, %N
  %add217 = add i32 %mul215, %mul216.pre-phi
  %mul221 = shl nuw nsw i32 %resIdxM203.0347, 3
  %idxprom225 = zext nneg i32 %mul221 to i64
  %arrayidx226 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %idxprom225
  %104 = load float, ptr %arrayidx226, align 4, !tbaa !14
  %mul227 = fmul contract float %alpha, %104
  %idxprom229 = sext i32 %add217 to i64
  %arrayidx230 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom229
  %105 = load i16, ptr %arrayidx230, align 2, !tbaa !7
  %106 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %105) #6, !srcloc !9
  %mul232 = fmul contract float %beta, %106
  %add233 = fadd contract float %mul227, %mul232
  %107 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add233) #6, !srcloc !10
  store i16 %107, ptr %arrayidx230, align 2, !tbaa !7
  %add218.1 = add i32 %add217, 1
  %108 = zext nneg i32 %mul221 to i64
  %109 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %108
  %arrayidx226.1 = getelementptr inbounds nuw i8, ptr %109, i64 4
  %110 = load float, ptr %arrayidx226.1, align 4, !tbaa !14
  %mul227.1 = fmul contract float %alpha, %110
  %idxprom229.1 = sext i32 %add218.1 to i64
  %arrayidx230.1 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom229.1
  %111 = load i16, ptr %arrayidx230.1, align 2, !tbaa !7
  %112 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %111) #6, !srcloc !9
  %mul232.1 = fmul contract float %beta, %112
  %add233.1 = fadd contract float %mul227.1, %mul232.1
  %113 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add233.1) #6, !srcloc !10
  store i16 %113, ptr %arrayidx230.1, align 2, !tbaa !7
  %add218.2 = add i32 %add217, 2
  %114 = zext nneg i32 %mul221 to i64
  %115 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %114
  %arrayidx226.2 = getelementptr inbounds nuw i8, ptr %115, i64 8
  %116 = load float, ptr %arrayidx226.2, align 4, !tbaa !14
  %mul227.2 = fmul contract float %alpha, %116
  %idxprom229.2 = sext i32 %add218.2 to i64
  %arrayidx230.2 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom229.2
  %117 = load i16, ptr %arrayidx230.2, align 2, !tbaa !7
  %118 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %117) #6, !srcloc !9
  %mul232.2 = fmul contract float %beta, %118
  %add233.2 = fadd contract float %mul227.2, %mul232.2
  %119 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add233.2) #6, !srcloc !10
  store i16 %119, ptr %arrayidx230.2, align 2, !tbaa !7
  %add218.3 = add i32 %add217, 3
  %120 = zext nneg i32 %mul221 to i64
  %121 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %120
  %arrayidx226.3 = getelementptr inbounds nuw i8, ptr %121, i64 12
  %122 = load float, ptr %arrayidx226.3, align 4, !tbaa !14
  %mul227.3 = fmul contract float %alpha, %122
  %idxprom229.3 = sext i32 %add218.3 to i64
  %arrayidx230.3 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom229.3
  %123 = load i16, ptr %arrayidx230.3, align 2, !tbaa !7
  %124 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %123) #6, !srcloc !9
  %mul232.3 = fmul contract float %beta, %124
  %add233.3 = fadd contract float %mul227.3, %mul232.3
  %125 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add233.3) #6, !srcloc !10
  store i16 %125, ptr %arrayidx230.3, align 2, !tbaa !7
  %add218.4 = add i32 %add217, 4
  %126 = zext nneg i32 %mul221 to i64
  %127 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %126
  %arrayidx226.4 = getelementptr inbounds nuw i8, ptr %127, i64 16
  %128 = load float, ptr %arrayidx226.4, align 4, !tbaa !14
  %mul227.4 = fmul contract float %alpha, %128
  %idxprom229.4 = sext i32 %add218.4 to i64
  %arrayidx230.4 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom229.4
  %129 = load i16, ptr %arrayidx230.4, align 2, !tbaa !7
  %130 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %129) #6, !srcloc !9
  %mul232.4 = fmul contract float %beta, %130
  %add233.4 = fadd contract float %mul227.4, %mul232.4
  %131 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add233.4) #6, !srcloc !10
  store i16 %131, ptr %arrayidx230.4, align 2, !tbaa !7
  %add218.5 = add i32 %add217, 5
  %132 = zext nneg i32 %mul221 to i64
  %133 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %132
  %arrayidx226.5 = getelementptr inbounds nuw i8, ptr %133, i64 20
  %134 = load float, ptr %arrayidx226.5, align 4, !tbaa !14
  %mul227.5 = fmul contract float %alpha, %134
  %idxprom229.5 = sext i32 %add218.5 to i64
  %arrayidx230.5 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom229.5
  %135 = load i16, ptr %arrayidx230.5, align 2, !tbaa !7
  %136 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %135) #6, !srcloc !9
  %mul232.5 = fmul contract float %beta, %136
  %add233.5 = fadd contract float %mul227.5, %mul232.5
  %137 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add233.5) #6, !srcloc !10
  store i16 %137, ptr %arrayidx230.5, align 2, !tbaa !7
  %add218.6 = add i32 %add217, 6
  %138 = zext nneg i32 %mul221 to i64
  %139 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %138
  %arrayidx226.6 = getelementptr inbounds nuw i8, ptr %139, i64 24
  %140 = load float, ptr %arrayidx226.6, align 4, !tbaa !14
  %mul227.6 = fmul contract float %alpha, %140
  %idxprom229.6 = sext i32 %add218.6 to i64
  %arrayidx230.6 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom229.6
  %141 = load i16, ptr %arrayidx230.6, align 2, !tbaa !7
  %142 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %141) #6, !srcloc !9
  %mul232.6 = fmul contract float %beta, %142
  %add233.6 = fadd contract float %mul227.6, %mul232.6
  %143 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add233.6) #6, !srcloc !10
  store i16 %143, ptr %arrayidx230.6, align 2, !tbaa !7
  %add218.7 = add i32 %add217, 7
  %144 = zext nneg i32 %mul221 to i64
  %145 = getelementptr inbounds nuw [4 x i8], ptr %threadResults, i64 %144
  %arrayidx226.7 = getelementptr inbounds nuw i8, ptr %145, i64 28
  %146 = load float, ptr %arrayidx226.7, align 4, !tbaa !14
  %mul227.7 = fmul contract float %alpha, %146
  %idxprom229.7 = sext i32 %add218.7 to i64
  %arrayidx230.7 = getelementptr inbounds [2 x i8], ptr %add.ptr12, i64 %idxprom229.7
  %147 = load i16, ptr %arrayidx230.7, align 2, !tbaa !7
  %148 = tail call contract noundef float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %147) #6, !srcloc !9
  %mul232.7 = fmul contract float %beta, %148
  %add233.7 = fadd contract float %mul227.7, %mul232.7
  %149 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add233.7) #6, !srcloc !10
  store i16 %149, ptr %arrayidx230.7, align 2, !tbaa !7
  %inc241 = add nuw nsw i32 %resIdxM203.0347, 1
  %exitcond350.not = icmp eq i32 %inc241, 8
  br i1 %exitcond350.not, label %for.cond192.loopexit, label %for.cond209.preheader, !llvm.loop !27
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5

attributes #0 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx88,+sm_80" "uniform-work-group-size" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { convergent nocallback nounwind }
attributes #4 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "nvvm.maxntid"="256" "stack-protector-buffer-size"="8" "target-cpu"="sm_80" "target-features"="+ptx88,+sm_80" "uniform-work-group-size" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { convergent nounwind memory(none) }
attributes #7 = { nounwind }

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
!9 = !{i64 2157011728}
!10 = !{i64 2156946988}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
!14 = !{!15, !15, i64 0}
!15 = !{!"float", !5, i64 0}
!16 = distinct !{!16, !12}
!17 = distinct !{!17, !12}
!18 = distinct !{!18, !12}
!19 = distinct !{!19, !12}
!20 = distinct !{!20, !12}
!21 = distinct !{!21, !12}
!22 = distinct !{!22, !12}
!23 = distinct !{!23, !12}
!24 = distinct !{!24, !12}
!25 = distinct !{!25, !12}
!26 = distinct !{!26, !12}
!27 = distinct !{!27, !12}
