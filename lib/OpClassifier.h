//===- OpClassifier.h - NVPTX opcode classification ----------*- C++ -*-===//
//
// Classifies NVPTX MachineInstr opcodes for arithmetic-intensity / roofline
// analysis. Memory accounting is handled separately via MachineMemOperand
// bucketing; this module is for opcodes that perform computation.
//
//===---------------------------------------------------------------------===//

#ifndef PTXAI_OP_CLASSIFIER_H
#define PTXAI_OP_CLASSIFIER_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace ptxai {

enum class FpPrecision : unsigned char {
    F16,
    BF16,
    F32,
    F64,
    Other,
};

enum class OpKind : unsigned char {
    None,           // not a classified compute op (memory, control flow, sreg
                    // read, integer ALU, sync, debug, …) — caller routes
                    // these elsewhere.
    ScalarFLOP,     // FADD/FSUB/FMUL/FMA, per-thread, scalar FP.
    // Reserved for future expansion. The pass should treat these as None
    // until the classifier implements them, but the enum lives here so the
    // API is stable across the build-out.
    MMA,            // mma.sync / wmma / wgmma — warp-level cooperative.
    SpecialMath,    // div, sqrt, rsqrt, sin, cos, exp2, log2, rcp.
    AsyncCopy,      // cp.async, cp.async.bulk, TMA family.
    LdMatrix,       // ldmatrix.* — warp-level shared->register transfer.
    WarpOp,         // shfl, vote, match (no FLOPs but warp-scope).
};

// Where the FLOP count is denominated. Critical for MMAs: one mma.sync
// instruction in MIR represents a warp-cooperative tile-matmul whose FLOPs
// must be counted once per warp invocation, NOT once per thread. The
// aggregator multiplies by the number of warp invocations or thread
// invocations as appropriate.
enum class InvocationScope : unsigned char {
    PerThread,      // every lane in the warp executes — scalar FP, scalar
                    // special math.
    PerWarp,        // one cooperative invocation produces the work — MMAs
                    // and warp-level LdMatrix / WarpOps.
    PerCTA,         // block-scope (rare; mostly mbarrier and cluster ops).
};

struct OpClass {
    OpKind kind = OpKind::None;
    FpPrecision precision = FpPrecision::Other;
    // FLOPs produced by one *invocation* (in the scope below). For scalar
    // FMA: 2. For mma.m16n8k16: 2 * 16 * 8 * 16 = 4096 (per-warp invocation).
    uint64_t flopsPerInvocation = 0;
    InvocationScope scope = InvocationScope::PerThread;

    bool isFlopProducer() const {
        return flopsPerInvocation > 0 &&
               (kind == OpKind::ScalarFLOP || kind == OpKind::MMA ||
                kind == OpKind::SpecialMath);
    }
};

// Classify an NVPTX MIR opcode by its mnemonic name. Pure function over the
// name string; takes no MIR context. Designed so future tests can exercise
// the classifier without spinning up an LLVM pipeline.
OpClass classify(llvm::StringRef Name);

// Detect the FP precision encoded in an NVPTX opcode mnemonic. Handles both
// LLVM 18 (FMA32rrr, bare width) and newer (FMA_F32rrr) naming. Public so
// the caller can re-derive precision in cases where it doesn't go through
// classify() — though normal usage is to read OpClass::precision.
FpPrecision detectPrecision(llvm::StringRef Name);

// ===========================================================================
// Memory-opcode-name parsing (for opcodes that lack MachineMemOperands)
// ===========================================================================
//
// Some NVPTX backend opcodes are loads or stores but emit without a
// MachineMemOperand attached, so the analyzer's MMO-walking path silently
// misses their byte traffic. The most consequential family is the LDG
// path (read-only / non-coherent global loads), triggered when CUDA code
// uses `__restrict__` on input pointers. Empirically this opcode shows up
// as the 9th-most-common opcode in our CUTLASS corpus.
//
// parseMemoryOpcodeName recovers the implicit address space and byte width
// from the opcode name itself, so the analyzer can fall back when the MMO
// is absent. The function is pure — it takes no MIR context — and is fully
// unit-testable.
//
// Coverage (matches NVPTX LLVM 23 TableGen):
//   LD_GLOBAL_NC_*, LDU_GLOBAL_*           — global, load
//   LD_GLOBAL_*, LDG_*                     — global, load (defensive)
//   LD_SHARED_*, LDS_*                     — shared, load (defensive)
//   LD_LOCAL_*                             — local, load
//   LD_CONST_*, LDC_*                      — const, load
//   LD_PARAM_*                             — param, load
//   ST_GLOBAL_*, ST_SHARED_*, ST_LOCAL_*   — store-side mirrors
//   Vector forms: ".._v2.." / ".._v4.." / ".._v8.." (LDV-style)
//                 ".._v2i32" / ".._v4i64" (LD_GLOBAL_NC-style, no separator)
struct OpcodeNameMemInfo {
    unsigned addrSpace = 0;     // matches NVPTXAS values
    uint64_t bytes = 0;
    bool isLoad = false;
    bool isStore = false;
};

// Returns parsed memory-op info if the opcode name matches a known NVPTX
// load/store family, otherwise std::nullopt. Pure function on the name.
std::optional<OpcodeNameMemInfo>
parseMemoryOpcodeName(llvm::StringRef Name);

} // namespace ptxai

#endif // PTXAI_OP_CLASSIFIER_H
