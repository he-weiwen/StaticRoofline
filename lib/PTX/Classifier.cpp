//===- PTX/Classifier.cpp - Stub implementation ------------------------===//
//
// SCAFFOLD ONLY: every statement classifies as Unknown. Per-family
// dispatch lands incrementally — see the build-out plan in the design
// doc. Order: FP arith → ld/st → cp.async → MMA family → tcgen05 →
// sync/bookkeeping → bit-ops/sreg.
//
//===---------------------------------------------------------------------===//

#include "PTX/Classifier.h"

namespace ptxai::ptx {

OpClass classify(const Stmt &S) {
    return Unknown{S.mnemonic};
}

} // namespace ptxai::ptx
