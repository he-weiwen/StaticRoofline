//===- OpClassifier.cpp - NVPTX opcode classification ------------------===//

#include "OpClassifier.h"

#include "llvm/Support/NVPTXAddrSpace.h"

using namespace llvm;

namespace ptxai {

namespace {

// Returns byte width of a PTX type token, or 0 if not a recognized type.
unsigned typeWidthBytes(StringRef word) {
    if (word == "i8" || word == "u8" || word == "s8" || word == "b8")  return 1;
    if (word == "i16" || word == "u16" || word == "s16" || word == "b16" ||
        word == "f16" || word == "bf16")                                 return 2;
    if (word == "i32" || word == "u32" || word == "s32" || word == "b32" ||
        word == "f32")                                                   return 4;
    if (word == "i64" || word == "u64" || word == "s64" || word == "b64" ||
        word == "f64")                                                   return 8;
    if (word == "i128" || word == "b128")                                return 16;
    return 0;
}

// Returns vector multiplier from a vector-prefix token, or 0 if not a known
// vector indicator. Recognizes both bare ("v2") and joined ("v2i16") forms.
//   - "v2" / "v4" / "v8"           → returns the multiplier
//   - "v2i16", "v4i32", "v8i32"    → returns the multiplier and writes
//                                    type bytes through *typeBytesOut
unsigned vectorMultiplier(StringRef word, unsigned *typeBytesOut) {
    if (word.size() < 2 || word[0] != 'v') return 0;
    char d = word[1];
    if (d != '2' && d != '4' && d != '8') return 0;
    unsigned mult = (unsigned)(d - '0');
    StringRef rest = word.drop_front(2);
    if (rest.empty()) return mult;          // bare "v2" / "v4" / "v8"
    unsigned w = typeWidthBytes(rest);
    if (w == 0) return 0;                   // "vXgarbage" — reject
    if (typeBytesOut) *typeBytesOut = w;
    return mult;                            // "v2i16" etc.
}

// Walk underscore-separated segments of the opcode name. Sets `bytes` to
// the largest type-width seen and `vec` to the vector multiplier seen.
struct WidthInfo { unsigned bytes = 0; unsigned vec = 1; };

WidthInfo scanWidth(StringRef name) {
    WidthInfo info;
    size_t pos = 0;
    while (pos < name.size()) {
        size_t under = name.find('_', pos);
        StringRef word = (under == StringRef::npos)
            ? name.substr(pos)
            : name.substr(pos, under - pos);

        unsigned joinedTypeBytes = 0;
        if (unsigned mult = vectorMultiplier(word, &joinedTypeBytes)) {
            info.vec = mult;
            if (joinedTypeBytes) info.bytes = joinedTypeBytes;
        } else if (unsigned w = typeWidthBytes(word)) {
            info.bytes = w;
        }

        if (under == StringRef::npos) break;
        pos = under + 1;
    }
    return info;
}

// Classify the address-space implied by the opcode-name prefix. Returns
// std::nullopt for unknown prefixes.
struct PrefixInfo {
    unsigned addrSpace;
    bool isLoad;
    bool isStore;
};

std::optional<PrefixInfo> classifyPrefix(StringRef Name) {
    using namespace NVPTXAS;
    // Order matters: more-specific first (LD_GLOBAL_NC before LD_GLOBAL).
    if (Name.starts_with("LD_GLOBAL_NC_") ||
        Name.starts_with("LDU_GLOBAL_")   ||
        Name.starts_with("LDG_")          ||
        Name.starts_with("LD_GLOBAL_")    ||
        Name.starts_with("LDU_"))
        return PrefixInfo{ADDRESS_SPACE_GLOBAL, true, false};

    if (Name.starts_with("LD_SHARED_") || Name.starts_with("LDS_"))
        return PrefixInfo{ADDRESS_SPACE_SHARED, true, false};

    if (Name.starts_with("LD_LOCAL_"))
        return PrefixInfo{ADDRESS_SPACE_LOCAL, true, false};

    if (Name.starts_with("LD_CONST_") || Name.starts_with("LDC_"))
        return PrefixInfo{ADDRESS_SPACE_CONST, true, false};

    if (Name.starts_with("LD_PARAM_"))
        return PrefixInfo{ADDRESS_SPACE_ENTRY_PARAM, true, false};

    if (Name.starts_with("ST_GLOBAL_"))
        return PrefixInfo{ADDRESS_SPACE_GLOBAL, false, true};

    if (Name.starts_with("ST_SHARED_") || Name.starts_with("STS_"))
        return PrefixInfo{ADDRESS_SPACE_SHARED, false, true};

    if (Name.starts_with("ST_LOCAL_"))
        return PrefixInfo{ADDRESS_SPACE_LOCAL, false, true};

    return std::nullopt;
}

} // anonymous namespace

std::optional<OpcodeNameMemInfo>
parseMemoryOpcodeName(StringRef Name) {
    auto pre = classifyPrefix(Name);
    if (!pre) return std::nullopt;

    WidthInfo w = scanWidth(Name);
    if (w.bytes == 0) return std::nullopt;

    return OpcodeNameMemInfo{
        pre->addrSpace,
        (uint64_t)w.bytes * w.vec,
        pre->isLoad,
        pre->isStore,
    };
}

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
