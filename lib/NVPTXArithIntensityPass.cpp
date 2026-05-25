#include "OpClassifier.h"
#include "Measurement.h"
#include "PTX/Classifier.h"
#include "PTX/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Pass.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/NVPTXAddrSpace.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <utility>

using namespace llvm;

namespace {
    struct MemStats {
        uint64_t GlobalLoadBytes = 0;
        uint64_t GlobalStoreBytes = 0;
        uint64_t SharedLoadBytes = 0;
        uint64_t SharedStoreBytes = 0;
        uint64_t LocalLoadBytes = 0;
        uint64_t LocalStoreBytes = 0;
        uint64_t ConstLoadBytes = 0;
        uint64_t ConstStoreBytes = 0;
        uint64_t ParamLoadBytes = 0;
        uint64_t ParamStoreBytes = 0;
        uint64_t UnknownBytes = 0;
        uint64_t UnknownAccesses = 0;
    };

    struct BlockStats {
        uint64_t Instrs = 0;
        uint64_t Flops = 0;
        uint64_t FlopsF16 = 0;
        uint64_t FlopsBF16 = 0;
        uint64_t FlopsF32 = 0;
        uint64_t FlopsF64 = 0;
        uint64_t FlopsOther = 0;
        DenseMap<unsigned, uint64_t> OpcodeCounts;
        MemStats Mem;
    };

    // Symbolic trip count for one machine loop. At MIR level this is only a
    // stable loop identity; a later IR/SCEV integration can map it to a source
    // expression such as K, ceil(K / 32), or a constant tile size.
    struct LoopTripCountSymbol {
        unsigned HeaderBlockNumber = 0;
    };

    // Symbolic execution count for one machine basic block, relative to one
    // kernel invocation. It is represented as the product of enclosing machine
    // loop trip-count symbols:
    //   {}          => 1
    //   {L3}        => L3
    //   {L2, L10}   => L2 * L10
    //
    // This deliberately does not model conditional branch probability yet.
    struct BasicBlockExecutionCount {
        SmallVector<LoopTripCountSymbol, 4> Factors;

        bool isOne() const { return Factors.empty(); }
        unsigned getLoopDepth() const {
            return static_cast<unsigned>(Factors.size());
        }
    };

    static MemStats &operator+=(MemStats &LHS, const MemStats &RHS) {
        LHS.GlobalLoadBytes += RHS.GlobalLoadBytes;
        LHS.GlobalStoreBytes += RHS.GlobalStoreBytes;
        LHS.SharedLoadBytes += RHS.SharedLoadBytes;
        LHS.SharedStoreBytes += RHS.SharedStoreBytes;
        LHS.LocalLoadBytes += RHS.LocalLoadBytes;
        LHS.LocalStoreBytes += RHS.LocalStoreBytes;
        LHS.ConstLoadBytes += RHS.ConstLoadBytes;
        LHS.ConstStoreBytes += RHS.ConstStoreBytes;
        LHS.ParamLoadBytes += RHS.ParamLoadBytes;
        LHS.ParamStoreBytes += RHS.ParamStoreBytes;
        LHS.UnknownBytes += RHS.UnknownBytes;
        LHS.UnknownAccesses += RHS.UnknownAccesses;
        return LHS;
    }

    static BlockStats &operator+=(BlockStats &LHS, const BlockStats &RHS) {
        LHS.Instrs += RHS.Instrs;
        LHS.Flops += RHS.Flops;
        LHS.FlopsF16 += RHS.FlopsF16;
        LHS.FlopsBF16 += RHS.FlopsBF16;
        LHS.FlopsF32 += RHS.FlopsF32;
        LHS.FlopsF64 += RHS.FlopsF64;
        LHS.FlopsOther += RHS.FlopsOther;
        LHS.Mem += RHS.Mem;
        for (const auto &Entry : RHS.OpcodeCounts)
            LHS.OpcodeCounts[Entry.first] += Entry.second;
        return LHS;
    }

    static uint64_t getGlobalBytes(const MemStats &Mem) {
        return Mem.GlobalLoadBytes + Mem.GlobalStoreBytes;
    }

    static uint64_t getLocalBytes(const MemStats &Mem) {
        return Mem.LocalLoadBytes + Mem.LocalStoreBytes;
    }

    // Route a Measurement into the existing per-thread BlockStats buckets.
    // PR 1: preserves PR-0 behaviour byte-for-byte. Today's mapping
    // intentionally flattens PerWarp Memory into per-thread shared/global
    // buckets (matches the existing LdMatrix / AsyncCopy convention) and
    // routes PerWarp Flops into FlopsOther (matches MMAOp). When per-scope
    // buckets land in a later PR, this function grows scope-aware
    // dispatch; the converter (toMeasurements) already emits honest
    // scopes, so no caller-side change will be needed.
    static void applyToBlockStats(const ptxai::Measurement &M,
                                  BlockStats &Stats) {
        using namespace NVPTXAS;
        if (M.kind == ptxai::Measurement::Kind::Flop) {
            if (M.scope != ptxai::InvocationScope::PerThread) {
                Stats.FlopsOther += M.count;
                return;
            }
            Stats.Flops += M.count;
            switch (M.precision) {
            case ptxai::FpPrecision::F16:   Stats.FlopsF16   += M.count; break;
            case ptxai::FpPrecision::BF16:  Stats.FlopsBF16  += M.count; break;
            case ptxai::FpPrecision::F32:   Stats.FlopsF32   += M.count; break;
            case ptxai::FpPrecision::F64:   Stats.FlopsF64   += M.count; break;
            case ptxai::FpPrecision::Other: Stats.FlopsOther += M.count; break;
            }
            return;
        }
        auto bucket = [&](uint64_t &lo, uint64_t &st) {
            if (M.isLoad)  lo += M.count;
            if (M.isStore) st += M.count;
        };
        switch (M.addrSpace) {
        case ADDRESS_SPACE_GLOBAL:
            bucket(Stats.Mem.GlobalLoadBytes, Stats.Mem.GlobalStoreBytes); break;
        case ADDRESS_SPACE_SHARED:
        case ADDRESS_SPACE_SHARED_CLUSTER:
            bucket(Stats.Mem.SharedLoadBytes, Stats.Mem.SharedStoreBytes); break;
        case ADDRESS_SPACE_LOCAL:
            bucket(Stats.Mem.LocalLoadBytes,  Stats.Mem.LocalStoreBytes);  break;
        case ADDRESS_SPACE_CONST:
            bucket(Stats.Mem.ConstLoadBytes,  Stats.Mem.ConstStoreBytes);  break;
        case ADDRESS_SPACE_ENTRY_PARAM:
            bucket(Stats.Mem.ParamLoadBytes,  Stats.Mem.ParamStoreBytes);  break;
        default:
            Stats.Mem.UnknownBytes += M.count;
            ++Stats.Mem.UnknownAccesses;
            break;
        }
    }

    // Dispatch a parsed inline-PTX OpClass into the existing BlockStats
    // counters via the Measurement value type. Exhaustiveness check on
    // the variant lives inside toMeasurements (see Measurement.cpp).
    static void applyInlinePtxOpClass(BlockStats &Stats,
                                      const ptxai::ptx::OpClass &PtxOp) {
        for (const ptxai::Measurement &M : ptxai::toMeasurements(PtxOp))
            applyToBlockStats(M, Stats);
        // Unknown emits no Measurement by design; preserve the diagnostic
        // counter bump here. The "we encountered something opaque" signal
        // is load-bearing for diff-ing against canonical opcode tables.
        if (std::holds_alternative<ptxai::ptx::Unknown>(PtxOp))
            ++Stats.Mem.UnknownAccesses;
    }

    static void addFlops(BlockStats &Stats, const ptxai::OpClass &Op) {
        // Today only PerThread sources exist in the classifier. When MMA
        // (PerWarp) lands, this aggregator MUST grow per-scope buckets:
        // collapsing PerWarp FLOPs into the per-thread total understates by
        // ~32× per warp-cooperative instruction. Assert keeps that mistake
        // visible the moment a non-PerThread classifier output is added.
        assert(Op.scope == ptxai::InvocationScope::PerThread &&
               "non-PerThread FLOP source needs scope-aware aggregation");
        Stats.Flops += Op.flopsPerInvocation;
        switch (Op.precision) {
        case ptxai::FpPrecision::F16:   Stats.FlopsF16   += Op.flopsPerInvocation; break;
        case ptxai::FpPrecision::BF16:  Stats.FlopsBF16  += Op.flopsPerInvocation; break;
        case ptxai::FpPrecision::F32:   Stats.FlopsF32   += Op.flopsPerInvocation; break;
        case ptxai::FpPrecision::F64:   Stats.FlopsF64   += Op.flopsPerInvocation; break;
        case ptxai::FpPrecision::Other: Stats.FlopsOther += Op.flopsPerInvocation; break;
        }
    }

    static void printDensity(uint64_t FLOPs, uint64_t Bytes) {
        if (Bytes == 0) {
            errs() << "n/a";
            return;
        }
        double Density = static_cast<double>(FLOPs) / static_cast<double>(Bytes);
        errs() << format("%.6f", Density);
    }

    static void addLoadStoreBytes(uint64_t &LoadBytes, uint64_t &StoreBytes,
                                  uint64_t Bytes, const MachineInstr &MI) {
        if (MI.mayLoad())
            LoadBytes += Bytes;
        if (MI.mayStore())
            StoreBytes += Bytes;
    }

    // Bucket `Bytes` into Stats.Mem according to `AddrSpace`. Used by both
    // the MMO-driven path and the opcode-name-driven fallback.
    static void bucketBytesByAddrSpace(BlockStats &Stats, unsigned AddrSpace,
                                       uint64_t Bytes, bool IsLoad,
                                       bool IsStore) {
        auto add = [&](uint64_t &lo, uint64_t &st) {
            if (IsLoad)  lo += Bytes;
            if (IsStore) st += Bytes;
        };
        switch (AddrSpace) {
        case NVPTXAS::ADDRESS_SPACE_GLOBAL:
            add(Stats.Mem.GlobalLoadBytes, Stats.Mem.GlobalStoreBytes); break;
        case NVPTXAS::ADDRESS_SPACE_SHARED:
        case NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER:
            add(Stats.Mem.SharedLoadBytes, Stats.Mem.SharedStoreBytes); break;
        case NVPTXAS::ADDRESS_SPACE_LOCAL:
            add(Stats.Mem.LocalLoadBytes,  Stats.Mem.LocalStoreBytes);  break;
        case NVPTXAS::ADDRESS_SPACE_CONST:
            add(Stats.Mem.ConstLoadBytes,  Stats.Mem.ConstStoreBytes);  break;
        case NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM:
            add(Stats.Mem.ParamLoadBytes,  Stats.Mem.ParamStoreBytes);  break;
        default:
            Stats.Mem.UnknownBytes += Bytes;
            ++Stats.Mem.UnknownAccesses;
            break;
        }
    }

    static void recordMemory(BlockStats &Stats, const MachineInstr &MI,
                             const TargetInstrInfo &TII) {
        bool sawAnyMMO = false;
        for (MachineMemOperand *MMO : MI.memoperands()) {
            sawAnyMMO = true;
            LocationSize Size = MMO->getSize();
            if (!Size.hasValue() || Size.isScalable()) {
                ++Stats.Mem.UnknownAccesses;
                continue;
            }

            uint64_t Bytes = Size.getValue().getFixedValue();
            if (!MI.mayLoad() && !MI.mayStore()) {
                Stats.Mem.UnknownBytes += Bytes;
                ++Stats.Mem.UnknownAccesses;
                continue;
            }
            bucketBytesByAddrSpace(Stats, MMO->getAddrSpace(), Bytes,
                                    MI.mayLoad(), MI.mayStore());
        }

        // Opcode-name-driven fallback for loads/stores that lack MMOs.
        // The dominant case is the LDG family (LD_GLOBAL_NC_*, LDU_GLOBAL_*),
        // emitted when CUDA code uses __restrict__ on input pointers — the
        // NVPTX backend doesn't attach MMOs to these.
        //
        // Note: we deliberately do NOT gate on MI.mayLoad()/mayStore().
        // NVPTX's LD_GLOBAL_NC_* family is tagged with UnmodeledSideEffects
        // instead of MayLoad in the auto-generated MCInstrDesc, so those
        // queries return false. The opcode-name pattern itself is the
        // authoritative signal — `parseMemoryOpcodeName` returns nullopt
        // for anything that doesn't match a known load/store family.
        if (sawAnyMMO) return;
        if (auto info = ptxai::parseMemoryOpcodeName(TII.getName(MI.getOpcode())))
            bucketBytesByAddrSpace(Stats, info->addrSpace, info->bytes,
                                    info->isLoad, info->isStore);
    }

    static void printMemoryStats(const MemStats &Mem) {
        errs() << "    memory:"
               << " global_load=" << Mem.GlobalLoadBytes
               << " global_store=" << Mem.GlobalStoreBytes
               << " shared_load=" << Mem.SharedLoadBytes
               << " shared_store=" << Mem.SharedStoreBytes
               << " local_load=" << Mem.LocalLoadBytes
               << " local_store=" << Mem.LocalStoreBytes
               << " const_load=" << Mem.ConstLoadBytes
               << " const_store=" << Mem.ConstStoreBytes
               << " param_load=" << Mem.ParamLoadBytes
               << " param_store=" << Mem.ParamStoreBytes
               << " unknown_bytes=" << Mem.UnknownBytes
               << " unknown_accesses=" << Mem.UnknownAccesses << "\n";
    }

    static void printFlopsAndBytes(const BlockStats &Stats) {
        // AI denominator is global bytes only — matches conventional roofline.
        // local_bytes is reported as a diagnostic; it's nonzero in spilling /
        // un-promoted-alloca cases and worth flagging, but folding it into AI
        // hides the global-memory signal in the common case where local=0.
        uint64_t GlobalBytes = getGlobalBytes(Stats.Mem);
        errs() << " instrs=" << Stats.Instrs
               << " flops=" << Stats.Flops
               << " flops_f16=" << Stats.FlopsF16
               << " flops_bf16=" << Stats.FlopsBF16
               << " flops_f32=" << Stats.FlopsF32
               << " flops_f64=" << Stats.FlopsF64
               << " flops_other=" << Stats.FlopsOther
               << " global_bytes=" << GlobalBytes
               << " local_bytes=" << getLocalBytes(Stats.Mem)
               << " ai=";
        printDensity(Stats.Flops, GlobalBytes);
    }

    static BasicBlockExecutionCount
    getExecutionCountForBlock(const MachineBasicBlock &MBB,
                              const MachineLoopInfo &MLI) {
        SmallVector<const MachineLoop *, 4> LoopNest;
        for (const MachineLoop *Loop = MLI.getLoopFor(&MBB); Loop;
             Loop = Loop->getParentLoop())
            LoopNest.push_back(Loop);

        BasicBlockExecutionCount Count;
        for (const MachineLoop *Loop : llvm::reverse(LoopNest))
            Count.Factors.push_back(
                {static_cast<unsigned>(Loop->getHeader()->getNumber())});
        return Count;
    }

    static void printExecutionCount(raw_ostream &OS,
                                    const BasicBlockExecutionCount &Count) {
        if (Count.isOne()) {
            OS << "1";
            return;
        }

        bool First = true;
        for (const LoopTripCountSymbol &Factor : Count.Factors) {
            if (!First)
                OS << "*";
            First = false;
            OS << "L" << Factor.HeaderBlockNumber;
        }
    }

    static void printBlockStats(const MachineBasicBlock &MBB,
                                const BlockStats &Stats,
                                const TargetInstrInfo &TII,
                                const MachineLoopInfo &MLI) {
        errs() << "  bb." << MBB.getNumber();
        if (const BasicBlock *BB = MBB.getBasicBlock()) {
            if (BB->hasName())
                errs() << "." << BB->getName();
        }
        BasicBlockExecutionCount Count = getExecutionCountForBlock(MBB, MLI);
        errs() << " loop_depth=" << Count.getLoopDepth() << " exec_count=";
        printExecutionCount(errs(), Count);
        printFlopsAndBytes(Stats);
        errs() << "\n";

        SmallVector<std::pair<unsigned, uint64_t>, 32> Opcodes;
        for (const auto &Entry : Stats.OpcodeCounts)
            Opcodes.push_back({Entry.first, Entry.second});

        llvm::sort(Opcodes, [&TII](const auto &LHS, const auto &RHS) {
            return TII.getName(LHS.first) < TII.getName(RHS.first);
        });

        for (const auto &[Opcode, Count] : Opcodes)
            errs() << "    " << TII.getName(Opcode) << ": " << Count << "\n";

        printMemoryStats(Stats.Mem);
    }

    struct NVPTXArithIntensityPass : public MachineFunctionPass {
        static char ID;
        NVPTXArithIntensityPass() : MachineFunctionPass(ID) {}

        StringRef getPassName() const override { return "NVPTX Arithmetic Intensity"; }

        void getAnalysisUsage(AnalysisUsage &AU) const override {
            AU.addRequired<MachineLoopInfoWrapperPass>();
            AU.setPreservesAll();
            MachineFunctionPass::getAnalysisUsage(AU);
        }

        bool runOnMachineFunction(MachineFunction &MF) override {
            uint64_t Blocks = 0;
            BlockStats Total;
            const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
            const MachineLoopInfo &MLI =
                getAnalysis<MachineLoopInfoWrapperPass>().getLI();

            errs() << "kernel " << MF.getName() << "\n";

            for (auto &MBB : MF) {
                ++Blocks;
                BlockStats Stats;

                for (auto &MI : MBB) {
                    if (MI.isDebugInstr())
                        continue;

                    ++Stats.Instrs;
                    ++Stats.OpcodeCounts[MI.getOpcode()];

                    // INLINEASM: extract the asm body and route each
                    // parsed PTX statement through the inline-PTX
                    // classifier. Operand 0 of an INLINEASM MI is the
                    // asm string (per llvm/IR/InlineAsm.h: MIOp_AsmString = 0).
                    if (MI.isInlineAsm()) {
                        const char *Asm = MI.getOperand(0).getSymbolName();
                        if (Asm && *Asm) {
                            for (const ptxai::ptx::Stmt &S :
                                 ptxai::ptx::parse(StringRef(Asm))) {
                                applyInlinePtxOpClass(Stats,
                                                       ptxai::ptx::classify(S));
                            }
                        }
                        // Still record any MMOs LLVM attached to the
                        // INLINEASM (rare but possible on some atomic
                        // intrinsics).
                        recordMemory(Stats, MI, *TII);
                        continue;
                    }

                    ptxai::OpClass Op =
                        ptxai::classify(TII->getName(MI.getOpcode()));
                    if (Op.isFlopProducer())
                        addFlops(Stats, Op);
                    recordMemory(Stats, MI, *TII);
                }

                printBlockStats(MBB, Stats, *TII, MLI);
                Total += Stats;
            }

            errs() << "summary: " << MF.getName()
                   << " blocks=" << Blocks;
            printFlopsAndBytes(Total);
            errs() << "\n";

            return false;
        }
    };
}

char  NVPTXArithIntensityPass::ID = 0;
static RegisterPass<NVPTXArithIntensityPass>
    X("ptx-ai",                     // command-line pass name
      "NVPTX Arithmetic Intensity", // desc
      false,                        // does not only inspect CFG
      true                          // is an analysis pass
    );
