//===- OpClassifier.cpp - NVPTX opcode classification ------------------===//

#include "OpClassifier.h"

using namespace llvm;

namespace ptxai {

FpPrecision detectPrecision(StringRef Name) {
    // Case-insensitive: covers lowercase suffixes (FADD_rnf32rr) and uppercase
    // (FMA_F32rrr in newer LLVM). bf16 first since it contains "f16".
    if (Name.contains_insensitive("bf16")) return FpPrecision::BF16;
    if (Name.contains_insensitive("f64"))  return FpPrecision::F64;
    if (Name.contains_insensitive("f32"))  return FpPrecision::F32;
    if (Name.contains_insensitive("f16"))  return FpPrecision::F16;

    // LLVM 18 FMA family uses bare-width naming with no 'f' prefix:
    // FMA32rrr / FMA64rrr / FMA16rrr (and BF16 variant if present).
    if (Name.starts_with("FMA")) {
        StringRef Rest = Name.substr(3);
        if (Rest.starts_with_insensitive("bf16")) return FpPrecision::BF16;
        if (Rest.starts_with("64"))               return FpPrecision::F64;
        if (Rest.starts_with("32"))               return FpPrecision::F32;
        if (Rest.starts_with("16"))               return FpPrecision::F16;
    }
    return FpPrecision::Other;
}

OpClass classify(StringRef Name) {
    OpClass C;

    // Packed-vector lane multiplier. NVPTX has f16x2 / bf16x2 / f32x2 forms
    // that process two FP lanes in one instruction (same scope, double the
    // FLOPs). Check the explicit type-with-x2 substring rather than bare
    // "x2" — defensive against any future opcode that incidentally has "x2"
    // in its name without being a packed-vector op. There is no x4 form.
    const uint64_t Lanes =
        (Name.contains_insensitive("f16x2")  ||
         Name.contains_insensitive("bf16x2") ||
         Name.contains_insensitive("f32x2")) ? 2 : 1;

    // Drop the trailing "_" requirement: LLVM 18 emits FMA32rrr (no
    // underscore) while newer LLVM emits FMA_F32rrr. Same story for the
    // FADDf32rr / FADD_rnf32rr split. Prefix-only is safe — no non-FP opcode
    // in NVPTXInstrInfo.td begins with FADD/FSUB/FMUL/FMA.
    //
    // Note: this scalar path treats FMA as PerThread. When tensor-core MMA
    // support lands it will be a separate prefix family (MMA_, WMMA_, WGMMA_)
    // dispatched to OpKind::MMA with PerWarp scope; do NOT collapse them
    // into the FMA path.
    if (Name.starts_with("FMA")) {
        C.kind = OpKind::ScalarFLOP;
        C.flopsPerInvocation = 2 * Lanes;
        C.scope = InvocationScope::PerThread;
        C.precision = detectPrecision(Name);
        return C;
    }
    if (Name.starts_with("FADD") ||
        Name.starts_with("FMUL") ||
        Name.starts_with("FSUB")) {
        C.kind = OpKind::ScalarFLOP;
        C.flopsPerInvocation = 1 * Lanes;
        C.scope = InvocationScope::PerThread;
        C.precision = detectPrecision(Name);
        return C;
    }

    // TODO: MMA / WMMA / WGMMA dispatch with shape-encoded FLOPs and
    // PerWarp scope. Per-arch shape table.
    // TODO: SpecialMath (FDIV, SQRT, RSQRT, EX2, LG2, SIN, COS, RCP).
    // TODO: AsyncCopy / TMA / ldmatrix / WarpOp categories.

    return C; // OpKind::None
}

} // namespace ptxai
