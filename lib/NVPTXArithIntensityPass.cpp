#include "OpClassifier.h"
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

    static void recordMemory(BlockStats &Stats, const MachineInstr &MI) {
        for (MachineMemOperand *MMO : MI.memoperands()) {
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

            switch (MMO->getAddrSpace()) {
            case NVPTXAS::ADDRESS_SPACE_GLOBAL:
                addLoadStoreBytes(Stats.Mem.GlobalLoadBytes,
                                  Stats.Mem.GlobalStoreBytes, Bytes, MI);
                break;
            case NVPTXAS::ADDRESS_SPACE_SHARED:
            case NVPTXAS::ADDRESS_SPACE_SHARED_CLUSTER:
                addLoadStoreBytes(Stats.Mem.SharedLoadBytes,
                                  Stats.Mem.SharedStoreBytes, Bytes, MI);
                break;
            case NVPTXAS::ADDRESS_SPACE_LOCAL:
                addLoadStoreBytes(Stats.Mem.LocalLoadBytes,
                                  Stats.Mem.LocalStoreBytes, Bytes, MI);
                break;
            case NVPTXAS::ADDRESS_SPACE_CONST:
                addLoadStoreBytes(Stats.Mem.ConstLoadBytes,
                                  Stats.Mem.ConstStoreBytes, Bytes, MI);
                break;
            case NVPTXAS::ADDRESS_SPACE_ENTRY_PARAM:
                addLoadStoreBytes(Stats.Mem.ParamLoadBytes,
                                  Stats.Mem.ParamStoreBytes, Bytes, MI);
                break;
            default:
                Stats.Mem.UnknownBytes += Bytes;
                ++Stats.Mem.UnknownAccesses;
                break;
            }
        }
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
               << " global_bytes=" << GlobalBytes
               << " local_bytes=" << getLocalBytes(Stats.Mem)
               << " ai=";
        printDensity(Stats.Flops, GlobalBytes);
    }

    static void printBlockStats(const MachineBasicBlock &MBB,
                                const BlockStats &Stats,
                                const TargetInstrInfo &TII) {
        errs() << "  bb." << MBB.getNumber();
        if (const BasicBlock *BB = MBB.getBasicBlock()) {
            if (BB->hasName())
                errs() << "." << BB->getName();
        }
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
            AU.setPreservesAll();
            MachineFunctionPass::getAnalysisUsage(AU);
        }

        bool runOnMachineFunction(MachineFunction &MF) override {
            uint64_t Blocks = 0;
            BlockStats Total;
            const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

            errs() << "kernel " << MF.getName() << "\n";

            for (auto &MBB : MF) {
                ++Blocks;
                BlockStats Stats;

                for (auto &MI : MBB) {
                    if (MI.isDebugInstr())
                        continue;

                    ++Stats.Instrs;
                    ++Stats.OpcodeCounts[MI.getOpcode()];

                    // INLINEASM integration point. Once PTX/Classifier.cpp
                    // is implemented, this branch parses the asm body and
                    // dispatches each statement's OpClass into the same
                    // BlockStats counters. For now the parse() call returns
                    // empty, so inline asm contributes only via whatever
                    // MMOs LLVM happened to attach (typically none).
                    if (MI.isInlineAsm()) {
                        // TODO: extract asm string from MI operand 0,
                        // call ptx::parse + ptx::classify per stmt,
                        // route variant arms into addFlops/recordMemory.
                        recordMemory(Stats, MI);
                        continue;
                    }

                    ptxai::OpClass Op =
                        ptxai::classify(TII->getName(MI.getOpcode()));
                    if (Op.isFlopProducer())
                        addFlops(Stats, Op);
                    recordMemory(Stats, MI);
                }

                printBlockStats(MBB, Stats, *TII);
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
