//===- Stats.h - Query helper over a Measurement stream -------*- C++ -*-===//
//
// Lightweight non-owning view over an ArrayRef<Measurement> that answers
// the questions the analyzer's consumers actually ask: how many FLOPs of
// precision P at scope S; how many bytes at address space A; what's the
// arithmetic intensity for a given (flop subset, byte subset) pair.
//
// Has no production consumer yet — PR 3 introduces the API + unit tests
// only; PR 4 wires it into the pass; PR 5-6 migrate the printers to use
// it; PR 7 deletes BlockStats.
//
// See docs/measurement-refactor.md §3.3 for the design rationale.
//
//===---------------------------------------------------------------------===//

#ifndef PTXAI_STATS_H
#define PTXAI_STATS_H

#include "Measurement.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

namespace ptxai {

// Filter for Measurement queries. Empty filter = match all. Set fields
// are ANDed. Field applicability is kind-dependent:
//
//   For flops(): only `precision` and `scope` are consulted.
//                Memory-only fields (addrSpace, isLoad, isStore) are
//                silently ignored — a flop has no address space, so
//                a query like flops({addrSpace=AS_GLOBAL}) returns
//                "all flops" rather than zero.
//
//   For bytes(): only `scope`, `addrSpace`, `isLoad`, `isStore` are
//                consulted. The precision field is silently ignored —
//                a memory access has no FP precision in our model.
//
// This is documented "soft" filtering: filter fields that don't apply
// to a measurement kind are no-ops. Trade-off: easier composition at
// the cost of one possible surprise (irrelevant filter field = ignored,
// not "no match"). The alternative — two filter types — is type-safer
// but adds API surface. See §3.3 of the design doc for the discussion.
struct Filter {
    std::optional<FpPrecision>     precision;
    std::optional<InvocationScope> scope;
    std::optional<unsigned>        addrSpace;
    std::optional<bool>            isLoad;
    std::optional<bool>            isStore;
};

class Stats {
public:
    explicit Stats(llvm::ArrayRef<Measurement> Ms) : Ms(Ms) {}

    // Sum of count over Flop measurements matching the filter.
    uint64_t flops(const Filter &F = {}) const;

    // Sum of count over Memory measurements matching the filter.
    uint64_t bytes(const Filter &F = {}) const;

    // Arithmetic intensity: flops(flopF) / bytes(byteF). Returns NaN
    // when the byte subset is empty — matches the printer's "n/a"
    // presentation. Consumers can detect with std::isnan.
    double ai(const Filter &flopF, const Filter &byteF) const;

private:
    llvm::ArrayRef<Measurement> Ms;
};

// Print the one-line "instrs=N flops=N ... ai=X" summary derived from
// a Measurement stream. Used by the pass for both per-BB and kernel-
// summary output; the format is the same.
//
// Format (single line, leading space):
//   " instrs=N flops=N flops_f16=N flops_bf16=N flops_f32=N flops_f64=N"
//   " flops_other=N global_bytes=N local_bytes=N ai=X"
//
// Caller writes any preceding text (e.g. "bb.0.entry loop_depth=…") and
// a trailing newline. AI uses global bytes only as the denominator;
// "n/a" is emitted when global bytes == 0. local_bytes is a diagnostic
// field, NOT folded into AI.
//
// Lives in Stats.{h,cpp} for now — it's a thin Stats consumer that
// fits with the query API. PR 8 will move it into a Reporter class
// once a second output format (JSON) needs the same data path.
void printFlopsAndBytes(llvm::raw_ostream &OS, uint64_t Instrs,
                        const Stats &S);

// Print the per-direction, per-address-space memory line:
//   "    memory: global_load=N global_store=N shared_load=N shared_store=N"
//   " local_load=N local_store=N const_load=N const_store=N param_load=N"
//   " param_store=N unknown_bytes=N unknown_accesses=N\n"
//
// The per-AS / per-direction fields come from Stats queries. The two
// "unknown_*" trailing fields are passed in separately: they're
// diagnostic bumps fired by paths that don't produce a Measurement
// (size-unknown MMO; mayLoad/mayStore both false; opaque PTX). PR 4's
// parity assertion deliberately excludes these for the same reason.
//
// SHARED and SHARED_CLUSTER are summed into the shared_* fields (the
// existing BlockStats accounting collapses them; the Measurement stream
// preserves them honestly so future per-cluster queries are possible).
//
// Trailing newline IS emitted (unlike printFlopsAndBytes, which leaves
// newline placement to the caller — historical asymmetry preserved).
void printMemoryStats(llvm::raw_ostream &OS, const Stats &S,
                      uint64_t UnknownBytes, uint64_t UnknownAccesses);

} // namespace ptxai

#endif // PTXAI_STATS_H
