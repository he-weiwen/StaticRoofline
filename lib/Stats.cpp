//===- Stats.cpp - Query helper implementation -------------------------===//

#include "Stats.h"

#include <cmath>

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

} // namespace ptxai
