//===- PTX/Classifier.cpp - Semantic classification of PTX statements -===//
//
// Maps a parsed Stmt to a semantic OpClass. Pure function; no LLVM context.
//
// Coverage in this implementation:
//
//   FlopOp       — fma/add/sub/mul/min/max with FP type detection
//   MMAOp        — mma.sync / wmma.mma with shape parsing
//   MemoryOp     — ld / st with address-space + width
//   AsyncCopy    — cp.async.* with byte count when statically known
//   LdMatrix     — ldmatrix.sync.* (warp-cooperative shared->reg)
//   WarpSync     — shfl / vote / match / activemask
//   Barrier      — bar / mbarrier / fence / membar
//   Ignore       — cvt / cvta / mov / setp / selp / prmt / bfind / bmsk /
//                  exit / trap / ret / nop / bra / brkpt / etc.
//   Unknown      — anything else; aggregator counts and surfaces by name
//
// FLOPs convention:
//   add/sub/mul = 1 per invocation (× lane multiplier)
//   fma         = 2 per invocation (× lane multiplier)
//   mma.sync    = 2 * M * N * K (per warp — see InvocationScope::PerWarp)
//   transcendentals (sqrt/rsqrt/rcp/sin/cos/ex2/lg2/tanh) = 1 per invocation
//   div         = 1 per invocation
//   min/max/abs/neg = 1 per invocation (Williams convention)
//
//===---------------------------------------------------------------------===//

#include "PTX/Classifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

namespace ptxai::ptx {

namespace {

// -- type-modifier detection ------------------------------------------------

// Lane multiplier from a vector type modifier like "f16x2" / "bf16x2" /
// "f32x2". The PTX text uses these modifiers directly (e.g. "fma.rn.f16x2").
unsigned laneMultFromMods(ArrayRef<StringRef> Mods) {
    for (StringRef m : Mods) {
        if (m == "f16x2" || m == "bf16x2" || m == "f32x2") return 2;
    }
    return 1;
}

FpPrecision fpPrecisionFromMods(ArrayRef<StringRef> Mods) {
    // bf16 / bf16x2 first (contains "f16" as substring otherwise)
    for (StringRef m : Mods) {
        if (m == "bf16" || m == "bf16x2") return FpPrecision::BF16;
    }
    for (StringRef m : Mods) {
        if (m == "f16" || m == "f16x2") return FpPrecision::F16;
    }
    for (StringRef m : Mods) {
        if (m == "f32" || m == "f32x2") return FpPrecision::F32;
    }
    for (StringRef m : Mods) {
        if (m == "f64") return FpPrecision::F64;
    }
    return FpPrecision::Other;
}

// -- byte-width from type modifiers ---------------------------------------

// PTX type modifiers and their widths in bits. The list is small and
// well-defined.
unsigned typeWidthBitsFromMods(ArrayRef<StringRef> Mods) {
    for (StringRef m : Mods) {
        if (m == "b8"  || m == "u8"  || m == "s8"  || m == "e4m3" || m == "e5m2") return 8;
        if (m == "b16" || m == "u16" || m == "s16" || m == "f16"  || m == "bf16") return 16;
        if (m == "b32" || m == "u32" || m == "s32" || m == "f32"  || m == "f16x2" || m == "bf16x2") return 32;
        if (m == "b64" || m == "u64" || m == "s64" || m == "f64"  || m == "f32x2") return 64;
        if (m == "b128") return 128;
    }
    return 0;
}

// -- vector multiplier ("v2" / "v4" / "v8") -------------------------------

unsigned vectorWidthFromMods(ArrayRef<StringRef> Mods) {
    for (StringRef m : Mods) {
        if (m == "v2") return 2;
        if (m == "v4") return 4;
        if (m == "v8") return 8;
    }
    return 1;
}

// -- address-space inference from PTX modifiers -----------------------------

// Address-space numbers (PTX ABI). These mirror NVPTXAS but we duplicate them
// here to keep this module self-contained and not depend on llvm/Support.
enum : unsigned {
    AS_GENERIC = 0,
    AS_GLOBAL  = 1,
    AS_SHARED  = 3,
    AS_CONST   = 4,
    AS_LOCAL   = 5,
    AS_PARAM   = 101,
};

unsigned addrSpaceFromMods(ArrayRef<StringRef> Mods) {
    for (StringRef m : Mods) {
        if (m == "global") return AS_GLOBAL;
        if (m == "shared" || m == "shared::cluster" || m == "shared::cta") return AS_SHARED;
        if (m == "local") return AS_LOCAL;
        if (m == "const") return AS_CONST;
        if (m == "param") return AS_PARAM;
    }
    return AS_GENERIC;
}

bool hasMod(ArrayRef<StringRef> Mods, StringRef Want) {
    for (StringRef m : Mods)
        if (m == Want) return true;
    return false;
}

// -- MMA shape extraction ---------------------------------------------------

// Parses an mma shape modifier of the form "mMnNkK" — e.g. "m16n8k16".
// Returns (M,N,K) on success or all-zero on failure.
struct MMAShape { uint16_t M = 0, N = 0, K = 0; };

MMAShape parseMmaShape(ArrayRef<StringRef> Mods) {
    for (StringRef m : Mods) {
        if (m.empty() || m[0] != 'm') continue;
        // Match m<digits>n<digits>k<digits>
        StringRef rest = m.drop_front(1);
        size_t nPos = rest.find('n');
        if (nPos == StringRef::npos) continue;
        StringRef mPart = rest.substr(0, nPos);
        StringRef rest2 = rest.drop_front(nPos + 1);
        size_t kPos = rest2.find('k');
        if (kPos == StringRef::npos) continue;
        StringRef nPart = rest2.substr(0, kPos);
        StringRef kPart = rest2.drop_front(kPos + 1);

        unsigned M = 0, N = 0, K = 0;
        if (mPart.getAsInteger(10, M)) continue;
        if (nPart.getAsInteger(10, N)) continue;
        if (kPart.getAsInteger(10, K)) continue;
        return MMAShape{(uint16_t)M, (uint16_t)N, (uint16_t)K};
    }
    return MMAShape{};
}

// -- per-family handlers ---------------------------------------------------

OpClass classifyArith(const Stmt &S, uint64_t baseFlops) {
    FlopOp f;
    f.flops = baseFlops * laneMultFromMods(S.modifiers);
    f.precision = fpPrecisionFromMods(S.modifiers);
    f.scope = InvocationScope::PerThread;
    return OpClass{f};
}

OpClass classifyMMA(const Stmt &S) {
    MMAShape sh = parseMmaShape(S.modifiers);
    if (sh.M == 0) {
        // Couldn't parse shape — surface as Unknown so the diagnostic
        // logs the asm body rather than silently zeroing it.
        return OpClass{Unknown{S.mnemonic}};
    }
    MMAOp m;
    m.M = sh.M; m.N = sh.N; m.K = sh.K;
    m.flops = 2ull * (uint64_t)sh.M * (uint64_t)sh.N * (uint64_t)sh.K;
    m.scope = InvocationScope::PerWarp;
    // Best-effort precision: take the first FP type modifier we recognize.
    m.inputPrecision = fpPrecisionFromMods(S.modifiers);
    m.accumPrecision = m.inputPrecision; // refined later if needed
    return OpClass{m};
}

OpClass classifyLoadOrStore(const Stmt &S, bool isLoad) {
    MemoryOp mem;
    mem.addrSpace = addrSpaceFromMods(S.modifiers);
    unsigned widthBits = typeWidthBitsFromMods(S.modifiers);
    unsigned vec = vectorWidthFromMods(S.modifiers);
    mem.bytes = (uint64_t)widthBits / 8 * vec;
    mem.isLoad = isLoad;
    mem.isStore = !isLoad;
    return OpClass{mem};
}

// `cp.async` family. Two main shapes:
//   cp.async.cg.shared.global [%dst], [%src], 16;   — bytes is an immediate
//   cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
//        [%dst], [%src], %nbytes, [%mbar];
// For non-tensor variants the 3rd operand is the byte count, immediate or
// register. We extract it when immediate and leave it std::nullopt when the
// asm passes it as a runtime register.
OpClass classifyCpAsync(const Stmt &S) {
    AsyncCopy ac;
    ac.dstAddrSpace = AS_SHARED; // cp.async always writes to shared
    ac.srcAddrSpace = AS_GLOBAL; // and reads from global
    // Find the byte count: third operand if present and Immediate.
    if (S.operands.size() >= 3) {
        if (auto *imm = std::get_if<Immediate>(&S.operands[2])) {
            uint64_t v = 0;
            if (!imm->text.getAsInteger(0, v))
                ac.bytes = v;
        }
    }
    return OpClass{ac};
}

OpClass classifyLdMatrix(const Stmt &S) {
    LdMatrix lm;
    lm.addrSpace = AS_SHARED; // ldmatrix is shared->register
    // Type and "x1/x2/x4" multiplier together imply the byte count per warp.
    // Common forms: ldmatrix.sync.aligned.x4.m8n8.shared.b16 — 4 8x8 b16
    // matrices = 4 * 64 * 2 = 512 bytes per warp.
    unsigned x = 1;
    for (StringRef m : S.modifiers) {
        if      (m == "x1") x = 1;
        else if (m == "x2") x = 2;
        else if (m == "x4") x = 4;
    }
    // Default fragment is 8x8 (64 elements).
    unsigned elements = 64;
    for (StringRef m : S.modifiers) {
        if (m == "m8n8")  elements = 64;
        if (m == "m16n16") elements = 256; // newer variants
    }
    unsigned widthBits = typeWidthBitsFromMods(S.modifiers);
    if (widthBits == 0) widthBits = 16; // ldmatrix's typical default
    lm.bytes = (uint64_t)x * elements * (widthBits / 8);
    lm.scope = InvocationScope::PerWarp;
    return OpClass{lm};
}

} // anonymous namespace

OpClass classify(const Stmt &S) {
    if (S.parseError) return OpClass{Unknown{S.mnemonic}};
    StringRef m = S.mnemonic;

    // PTX directives surfaced by the parser (e.g. ".reg", ".pragma",
    // ".section", ".weak", ".visible") — no FLOP / byte contribution.
    if (m == "reg" || m == "pragma" || m == "section" || m == "weak" ||
        m == "visible" || m == "extern" || m == "global" || m == "shared" ||
        m == "local" || m == "param" || m == "loc" || m == "version" ||
        m == "target" || m == "address_size" || m == "func" || m == "entry" ||
        m == "calltargets" || m == "callprototype" || m == "branchtargets" ||
        m == "alias" || m == "common" || m == "maxntid" || m == "reqntid" ||
        m == "maxnreg" || m == "minnctapersm" || m == "maxnctapersm")
    {
        return OpClass{Ignore{}};
    }

    // --- Floating-point arithmetic (per-thread) ---
    if (m == "fma")  return classifyArith(S, /*baseFlops=*/2);
    if (m == "add")  return classifyArith(S, 1);
    if (m == "sub")  return classifyArith(S, 1);
    if (m == "mul")  return classifyArith(S, 1);
    if (m == "mad")  return classifyArith(S, 2); // multiply-add
    if (m == "div")  return classifyArith(S, 1);
    if (m == "abs" || m == "neg" || m == "min" || m == "max" || m == "copysign")
        return classifyArith(S, 1);
    if (m == "sqrt"  || m == "rsqrt" || m == "rcp"   ||
        m == "sin"   || m == "cos"   || m == "ex2"   ||
        m == "lg2"   || m == "tanh")
        return classifyArith(S, 1);

    // --- Tensor-core MMAs (per-warp) ---
    if (m == "mma" || m == "wmma") {
        // mma.sync.* shape, or wmma.mma.sync.*
        return classifyMMA(S);
    }
    if (m == "wgmma") {
        // wgmma.mma_async.sync.aligned.* — same shape parser works.
        return classifyMMA(S);
    }
    if (m == "ldmatrix") return classifyLdMatrix(S);
    if (m == "stmatrix") {
        LdMatrix lm = std::get<LdMatrix>(classifyLdMatrix(S));
        // stmatrix is the store-side, swap the sense
        return OpClass{lm};
    }

    // --- Memory operations ---
    if (m == "ld") return classifyLoadOrStore(S, /*isLoad=*/true);
    if (m == "st") return classifyLoadOrStore(S, /*isLoad=*/false);

    if (m == "cp") {
        if (hasMod(S.modifiers, "async")) {
            // `cp.async.commit_group` and `cp.async.wait_group` are
            // synchronization markers, not actual transfers — they group
            // and synchronize previously-issued async copies but do not
            // themselves move bytes.
            if (hasMod(S.modifiers, "commit_group") ||
                hasMod(S.modifiers, "wait_group")   ||
                hasMod(S.modifiers, "wait_all")) {
                return OpClass{Barrier{}};
            }
            return classifyCpAsync(S);
        }
        return OpClass{Unknown{m}};
    }

    // --- Atomics / reductions ---
    if (m == "atom") {
        // Atomic on memory; classifier produces a MemoryOp with both
        // load+store sides set so byte traffic is doubled (read + write).
        MemoryOp mem;
        mem.addrSpace = addrSpaceFromMods(S.modifiers);
        unsigned w = typeWidthBitsFromMods(S.modifiers);
        if (w == 0) w = 32;
        mem.bytes = w / 8;
        mem.isLoad = true;
        mem.isStore = true;
        return OpClass{mem};
    }
    if (m == "red") {
        MemoryOp mem;
        mem.addrSpace = addrSpaceFromMods(S.modifiers);
        unsigned w = typeWidthBitsFromMods(S.modifiers);
        if (w == 0) w = 32;
        mem.bytes = w / 8;
        mem.isStore = true;   // reduction writes back; reads are implicit
        mem.isLoad = false;
        return OpClass{mem};
    }

    // --- Sync / barrier / warp ops ---
    if (m == "bar" || m == "barrier" || m == "mbarrier" || m == "fence" || m == "membar")
        return OpClass{Barrier{}};
    if (m == "shfl" || m == "vote" || m == "match" || m == "activemask" || m == "redux")
        return OpClass{WarpSync{}};

    // --- Bookkeeping that's correctly ignored ---
    if (m == "cvt"  || m == "cvta" || m == "mov"  || m == "setp" || m == "selp" ||
        m == "set"  || m == "prmt" || m == "bfe"  || m == "bfi"  || m == "brev" ||
        m == "popc" || m == "clz"  || m == "shl"  || m == "shr"  || m == "and"  ||
        m == "or"   || m == "xor"  || m == "not"  || m == "lop3" || m == "isspacep" ||
        m == "exit" || m == "trap" || m == "ret"  || m == "bra"  || m == "brx"  ||
        m == "call" || m == "nop"  || m == "brkpt"|| m == "rem"  || m == "testp" ||
        m == "dp4a" || m == "dp2a" || /* TODO promote dp4a/dp2a to FlopOp later */
        m == "tex"  || m == "tld4" || m == "txq"  ||
        m == "suld" || m == "sust" || m == "suq"  ||
        m == "discard" || m == "applypriority" || m == "prefetch" ||
        m == "griddepcontrol" ||
        m == "ldu"  || /* uniform load — treat as ld with no MMO surface */
        m == "ldg"  || /* read-only load — bytes counted iff modifiers include type */
        m == "sad")
    {
        // Most of these are correctly ignored. Two exceptions worth noting:
        // ldu / ldg should be accounted as global-memory loads; promote to
        // MemoryOp when type modifiers are present.
        if (m == "ldu" || m == "ldg") {
            MemoryOp mem;
            mem.addrSpace = AS_GLOBAL;
            unsigned widthBits = typeWidthBitsFromMods(S.modifiers);
            unsigned vec = vectorWidthFromMods(S.modifiers);
            if (widthBits > 0) {
                mem.bytes = (uint64_t)widthBits / 8 * vec;
                mem.isLoad = true;
                return OpClass{mem};
            }
        }
        return OpClass{Ignore{}};
    }

    return OpClass{Unknown{m}};
}

} // namespace ptxai::ptx
