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

} // namespace ptxai

#endif // PTXAI_STATS_H
