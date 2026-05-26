//===- Stats.cpp - Query helper implementation -------------------------===//

#include "Stats.h"

#include "llvm/Support/NVPTXAddrSpace.h"

#include <cmath>
#include <cstdio>

namespace ptxai {

namespace {

// Match policy for Flop measurements: only precision and scope are
// consulted. Memory-only fields are silently ignored (see Filter
// docstring in Stats.h).
bool matchesFlop(const Filter &F, const Measurement &M) {
    if (F.precision && *F.precision != M.precision) return false;
    if (F.scope     && *F.scope     != M.scope)     return false;
    return true;
}

// Match policy for Memory measurements: scope, addrSpace, isLoad,
// isStore are consulted. Precision is silently ignored.
bool matchesMemory(const Filter &F, const Measurement &M) {
    if (F.scope     && *F.scope     != M.scope)     return false;
    if (F.addrSpace && *F.addrSpace != M.addrSpace) return false;
    if (F.isLoad    && *F.isLoad    != M.isLoad)    return false;
    if (F.isStore   && *F.isStore   != M.isStore)   return false;
    return true;
}

} // anonymous namespace

uint64_t Stats::flops(const Filter &F) const {
    uint64_t total = 0;
    for (const Measurement &M : Ms) {
        if (M.kind != Measurement::Kind::Flop) continue;
        if (!matchesFlop(F, M)) continue;
        total += M.count;
    }
    return total;
}

uint64_t Stats::bytes(const Filter &F) const {
    uint64_t total = 0;
    for (const Measurement &M : Ms) {
        if (M.kind != Measurement::Kind::Memory) continue;
        if (!matchesMemory(F, M)) continue;
        total += M.count;
    }
    return total;
}

double Stats::ai(const Filter &flopF, const Filter &byteF) const {
    uint64_t b = bytes(byteF);
    if (b == 0) return std::nan("");
    uint64_t f = flops(flopF);
    return static_cast<double>(f) / static_cast<double>(b);
}

void printFlopsAndBytes(llvm::raw_ostream &OS, uint64_t Instrs,
                        const Stats &S) {
    using namespace llvm;
    using namespace NVPTXAS;

    // Per-thread flops total + per-precision breakdown. The choice of
    // PerThread-only here matches the BlockStats.Flops / .FlopsF*
    // correspondence verified by PR 4's parity assertion: those fields
    // were only ever bumped from the PerThread branch of applyToBlockStats.
    Filter pt; pt.scope = InvocationScope::PerThread;
    auto ptOf = [&](FpPrecision p) {
        Filter f; f.scope = InvocationScope::PerThread; f.precision = p;
        return f;
    };
    uint64_t flopsTotal = S.flops(pt);
    uint64_t flopsF16   = S.flops(ptOf(FpPrecision::F16));
    uint64_t flopsBF16  = S.flops(ptOf(FpPrecision::BF16));
    uint64_t flopsF32   = S.flops(ptOf(FpPrecision::F32));
    uint64_t flopsF64   = S.flops(ptOf(FpPrecision::F64));

    // flops_other carries the PR-0 double duty: per-thread Other +
    // all PerWarp + all PerCTA (the latter two come from MMAOp routing
    // through applyToBlockStats). The PR 4 parity assertion verifies
    // this sum matches BlockStats.FlopsOther exactly.
    Filter pw; pw.scope = InvocationScope::PerWarp;
    Filter pc; pc.scope = InvocationScope::PerCTA;
    uint64_t flopsOther = S.flops(ptOf(FpPrecision::Other))
                        + S.flops(pw) + S.flops(pc);

    // global_bytes / local_bytes are per-AS aggregates regardless of
    // direction. Both scopes contribute; the existing BlockStats
    // accounting collapses scope, and PR 4 asserts the equivalence
    // (with the SHARED / SHARED_CLUSTER aliasing handled for memory).
    Filter g; g.addrSpace = ADDRESS_SPACE_GLOBAL;
    Filter l; l.addrSpace = ADDRESS_SPACE_LOCAL;
    uint64_t globalBytes = S.bytes(g);
    uint64_t localBytes  = S.bytes(l);

    OS << " instrs=" << Instrs
       << " flops=" << flopsTotal
       << " flops_f16=" << flopsF16
       << " flops_bf16=" << flopsBF16
       << " flops_f32=" << flopsF32
       << " flops_f64=" << flopsF64
       << " flops_other=" << flopsOther
       << " global_bytes=" << globalBytes
       << " local_bytes=" << localBytes
       << " ai=";
    if (globalBytes == 0) {
        OS << "n/a";
    } else {
        // snprintf rather than llvm::format(): format_object_base has a
        // virtual destructor and brings in RTTI symbols that LLVM
        // doesn't export from its (-fno-rtti) builds. Using format()
        // here would force the entire ptxai_ptx OBJECT library to also
        // build with -fno-rtti, which is a broader project-config
        // change than this PR warrants.
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6f",
                      static_cast<double>(flopsTotal) /
                          static_cast<double>(globalBytes));
        OS << buf;
    }
}

} // namespace ptxai
