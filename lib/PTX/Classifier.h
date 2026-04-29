//===- PTX/Classifier.h - Semantic classification of PTX statements ===---===//
//
// Maps a parsed Stmt to an OpClass — the analysis-relevant semantic
// summary. This is where mnemonic+modifiers turn into FLOP counts, byte
// counts, MMA shapes, and scope (per-thread / per-warp / per-CTA).
//
// OpClass is a sum type, distinct per category:
//
//   FlopOp     — scalar/packed FP arithmetic; PerThread.
//   MMAOp      — mma / wmma / wgmma; PerWarp; FLOPs = 2·M·N·K.
//   MemoryOp   — ld / st with explicit address space and width.
//   AsyncCopy  — cp.async family; bytes optional (TMA descriptors aren't
//                statically derivable from the asm alone).
//   LdMatrix   — ldmatrix.* — warp-level shared->register transfer.
//   WarpSync   — shfl / vote / match — no FLOPs, no bytes, but warp-scope.
//   Barrier    — bar.* / mbarrier.* / fence.* — no FLOPs, no bytes.
//   Ignore     — recognized bookkeeping with zero contribution: cvta,
//                exit, trap, prmt, bfind, bmsk, get_sreg, etc.
//   Unknown    — not in any handler; aggregator logs and undercounts
//                conservatively (zero FLOPs, zero bytes).
//
// The aggregator dispatches via std::visit; new arms added here force a
// compile error at every dispatch site — the exhaustiveness check.
//
//===---------------------------------------------------------------------===//

#ifndef PTXAI_PTX_CLASSIFIER_H
#define PTXAI_PTX_CLASSIFIER_H

#include "OpClassifier.h"   // shared FpPrecision / InvocationScope from MIR side
#include "PTX/Parser.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>
#include <variant>

namespace ptxai::ptx {

struct FlopOp {
    FpPrecision precision = FpPrecision::Other;
    uint64_t flops = 0;
    InvocationScope scope = InvocationScope::PerThread;
};

struct MMAOp {
    uint16_t M = 0, N = 0, K = 0;
    FpPrecision inputPrecision = FpPrecision::Other;
    FpPrecision accumPrecision = FpPrecision::Other;
    uint64_t flops = 0;                              // per warp invocation
    InvocationScope scope = InvocationScope::PerWarp;
};

struct MemoryOp {
    unsigned addrSpace = 0;
    uint64_t bytes = 0;
    bool isLoad = false;
    bool isStore = false;
};

struct AsyncCopy {
    unsigned dstAddrSpace = 0;
    unsigned srcAddrSpace = 0;
    std::optional<uint64_t> bytes;     // unset for TMA when not derivable
};

struct LdMatrix {
    unsigned addrSpace = 0;
    uint64_t bytes = 0;
    InvocationScope scope = InvocationScope::PerWarp;
};

struct WarpSync {};
struct Barrier {};
struct Ignore {};

struct Unknown {
    llvm::StringRef mnemonic;          // for diagnostic logging
};

using OpClass = std::variant<FlopOp, MMAOp, MemoryOp, AsyncCopy, LdMatrix,
                              WarpSync, Barrier, Ignore, Unknown>;

// Map a parsed PTX statement to its semantic classification. Pure
// function — no MIR context needed. Failure mode is to return Unknown,
// never to misclassify.
OpClass classify(const Stmt &S);

} // namespace ptxai::ptx

#endif // PTXAI_PTX_CLASSIFIER_H
