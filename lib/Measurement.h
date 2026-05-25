//===- Measurement.h - Tagged value for analyzer measurements -*- C++ -*-===//
//
// A single measurement record — one FLOP count, one memory access, etc.
// Pure data; consumed by aggregators (today: a thin BlockStats updater
// in the pass; future: a Stats query helper).
//
// See docs/measurement-refactor.md for the architectural intent. PR 1
// introduces this type and routes the PTX-side dispatcher through it
// without changing behaviour observable in the FileCheck tests.
//
//===---------------------------------------------------------------------===//

#ifndef PTXAI_MEASUREMENT_H
#define PTXAI_MEASUREMENT_H

#include "OpClassifier.h"     // FpPrecision, InvocationScope
#include "PTX/Classifier.h"   // ptx::OpClass

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace ptxai {

struct Measurement {
    enum class Kind : uint8_t { Flop, Memory };
    Kind            kind;
    InvocationScope scope     = InvocationScope::PerThread;
    FpPrecision     precision = FpPrecision::Other;   // Flop only
    unsigned        addrSpace = 0;                    // Memory only
    bool            isLoad    = false;                // Memory only
    bool            isStore   = false;                // Memory only
    uint64_t        count     = 0;                    // FLOPs or bytes
};

// Convert a parsed-PTX OpClass into 0..2 Measurements. Pure function.
// Mapping (preserves PR-0 aggregator behaviour byte-for-byte; scopes
// that the aggregator currently flattens are still emitted honestly so
// future PRs only need to change the aggregator, not the converter):
//
//   FlopOp{PerThread}      → 1 Flop, precision preserved
//   FlopOp{!PerThread}     → 0 measurements (dropped today; the MIR-side
//                            aggregator asserts the same)
//   MMAOp                  → 1 Flop, scope=PerWarp, precision=Other
//                            (matches current FlopsOther routing)
//   MemoryOp               → 1 Memory, scope=PerThread
//   AsyncCopy{bytes set}   → 2 Memory: global load + shared store
//   AsyncCopy{bytes unset} → 0 measurements
//   LdMatrix               → 1 Memory, scope=PerWarp, AS_SHARED, load
//   WarpSync / Barrier
//     / Ignore / Unknown   → 0 measurements
//
// Unknown's diagnostic UnknownAccesses bump is handled at the call site
// — Measurement deliberately has no "no-count, bump-counter" form.
llvm::SmallVector<Measurement, 2>
toMeasurements(const ptx::OpClass &PtxOp);

// MIR-side overload. Converts a classified MIR opcode into 0..1
// Measurements:
//
//   ScalarFLOP, PerThread, flops>0 → 1 Flop measurement
//   ScalarFLOP, !PerThread         → assertion failure (preserves the
//                                    tripwire from the old addFlops; when
//                                    Phase 3 MMA support lands on the MIR
//                                    side, this branch grows scope-aware
//                                    emission, not silent acceptance)
//   anything else (None, MMA, …)   → 0 measurements (today's MIR
//                                    classifier only emits ScalarFLOP;
//                                    other OpKind values are forward
//                                    declarations)
llvm::SmallVector<Measurement, 1>
toMeasurements(const OpClass &MirOp);

} // namespace ptxai

#endif // PTXAI_MEASUREMENT_H
