//===- Measurement.cpp - Converter from ptx::OpClass to Measurement ----===//

#include "Measurement.h"

#include "llvm/Support/NVPTXAddrSpace.h"

#include <type_traits>
#include <variant>

using namespace llvm;

namespace ptxai {

SmallVector<Measurement, 2>
toMeasurements(const ptx::OpClass &PtxOp) {
    using namespace NVPTXAS;
    SmallVector<Measurement, 2> Out;
    std::visit([&](const auto &Op) {
        using T = std::decay_t<decltype(Op)>;
        if constexpr (std::is_same_v<T, ptx::FlopOp>) {
            if (Op.scope != InvocationScope::PerThread) return;
            Out.push_back({Measurement::Kind::Flop, Op.scope, Op.precision,
                           /*addrSpace=*/0u,
                           /*isLoad=*/false, /*isStore=*/false, Op.flops});
        } else if constexpr (std::is_same_v<T, ptx::MMAOp>) {
            Out.push_back({Measurement::Kind::Flop, Op.scope,
                           FpPrecision::Other,
                           /*addrSpace=*/0u,
                           /*isLoad=*/false, /*isStore=*/false, Op.flops});
        } else if constexpr (std::is_same_v<T, ptx::MemoryOp>) {
            Out.push_back({Measurement::Kind::Memory,
                           InvocationScope::PerThread, FpPrecision::Other,
                           Op.addrSpace, Op.isLoad, Op.isStore, Op.bytes});
        } else if constexpr (std::is_same_v<T, ptx::AsyncCopy>) {
            if (Op.bytes) {
                Out.push_back({Measurement::Kind::Memory,
                               InvocationScope::PerThread, FpPrecision::Other,
                               ADDRESS_SPACE_GLOBAL,
                               /*isLoad=*/true, /*isStore=*/false, *Op.bytes});
                Out.push_back({Measurement::Kind::Memory,
                               InvocationScope::PerThread, FpPrecision::Other,
                               ADDRESS_SPACE_SHARED,
                               /*isLoad=*/false, /*isStore=*/true, *Op.bytes});
            }
        } else if constexpr (std::is_same_v<T, ptx::LdMatrix>) {
            Out.push_back({Measurement::Kind::Memory, Op.scope,
                           FpPrecision::Other, ADDRESS_SPACE_SHARED,
                           /*isLoad=*/true, /*isStore=*/false, Op.bytes});
        }
        // WarpSync / Barrier / Ignore / Unknown: 0 measurements.
    }, PtxOp);
    return Out;
}

} // namespace ptxai
