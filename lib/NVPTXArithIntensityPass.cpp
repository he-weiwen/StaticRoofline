#include "OpClassifier.h"
#include "Measurement.h"
#include "Stats.h"
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
    // Per-BB accumulator. After PR 7, this is the only per-block state:
    //   - Ms              : the canonical Measurement stream; consumed by
    //                       Stats queries for all numeric output.
    //   - Instrs          : non-debug instruction count; pass-level
    //                       counter, not a Measurement.
    //   - OpcodeCounts    : per-opcode histogram; diagnostic, printed
    //                       verbatim, not aggregated by Stats.
    //   - UnknownBytes /  : diagnostic counters bumped by paths that
    //     UnknownAccesses   don't produce a Measurement (size-unknown
    //                       MMO; mayLoad/mayStore both false; opaque
    //                       PTX). PR 4's parity assertion deliberately
    //                       excluded these for the same reason; the
    //                       memory printer takes them as explicit args.
    // The previous BlockStats / MemStats god-object (~19 fields) is
    // gone — its writes were verified equivalent to the Measurement
    // stream by PR 4's parity assertion across every BB of every test
    // kernel.
    struct BBRecord {
        SmallVector<ptxai::Measurement, 32> Ms;
        DenseMap<unsigned, uint64_t> OpcodeCounts;
        uint64_t Instrs = 0;
        uint64_t UnknownBytes = 0;
        uint64_t UnknownAccesses = 0;
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

    // Dispatch a parsed inline-PTX OpClass into the per-BB Measurement
    // stream. Two-step semantic: emit the measurements; if the variant
    // was Unknown, also bump the diagnostic UnknownAccesses counter
    // (Unknown emits zero measurements by design; the counter bump is
    // the "we encountered something opaque" signal load-bearing for
    // diff-ing against canonical opcode tables).
    static void applyInlinePtxOpClass(BBRecord &R,
                                      const ptxai::ptx::OpClass &PtxOp) {
        for (const ptxai::Measurement &M : ptxai::toMeasurements(PtxOp))
            R.Ms.push_back(M);
        if (std::holds_alternative<ptxai::ptx::Unknown>(PtxOp))
            ++R.UnknownAccesses;
    }

    // Build the Measurement that represents one MMO's byte traffic, or
    // nullopt for diagnostic-only outcomes (size unknown / mayLoad &
    // mayStore both false). The diagnostic bumps stay inline in
    // recordMemory; they're not Measurements — they're "we saw something
    // we couldn't quantify."
    static std::optional<ptxai::Measurement>
    measurementFromMMO(const MachineMemOperand &MMO, bool IsLoad, bool IsStore) {
        ptxai::Measurement M;
        M.kind = ptxai::Measurement::Kind::Memory;
        M.scope = ptxai::InvocationScope::PerThread;
        M.addrSpace = MMO.getAddrSpace();
        M.isLoad = IsLoad;
        M.isStore = IsStore;
        M.count = MMO.getSize().getValue().getFixedValue();
        return M;
    }

    static void recordMemory(BBRecord &R, const MachineInstr &MI,
                             const TargetInstrInfo &TII) {
        bool sawAnyMMO = false;
        for (MachineMemOperand *MMO : MI.memoperands()) {
            sawAnyMMO = true;
            LocationSize Size = MMO->getSize();
            if (!Size.hasValue() || Size.isScalable()) {
                ++R.UnknownAccesses;
                continue;
            }

            uint64_t Bytes = Size.getValue().getFixedValue();
            if (!MI.mayLoad() && !MI.mayStore()) {
                R.UnknownBytes += Bytes;
                ++R.UnknownAccesses;
                continue;
            }
            if (auto M = measurementFromMMO(*MMO, MI.mayLoad(), MI.mayStore()))
                R.Ms.push_back(*M);
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
        if (auto info = ptxai::parseMemoryOpcodeName(TII.getName(MI.getOpcode()))) {
            R.Ms.push_back({ptxai::Measurement::Kind::Memory,
                            ptxai::InvocationScope::PerThread,
                            ptxai::FpPrecision::Other,
                            info->addrSpace, info->isLoad, info->isStore,
                            info->bytes});
        }
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
                                const BBRecord &R,
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
        ptxai::printFlopsAndBytes(errs(), R.Instrs, ptxai::Stats(R.Ms));
        errs() << "\n";

        SmallVector<std::pair<unsigned, uint64_t>, 32> Opcodes;
        for (const auto &Entry : R.OpcodeCounts)
            Opcodes.push_back({Entry.first, Entry.second});

        llvm::sort(Opcodes, [&TII](const auto &LHS, const auto &RHS) {
            return TII.getName(LHS.first) < TII.getName(RHS.first);
        });

        for (const auto &[Opcode, Count] : Opcodes)
            errs() << "    " << TII.getName(Opcode) << ": " << Count << "\n";

        ptxai::printMemoryStats(errs(), ptxai::Stats(R.Ms),
                                R.UnknownBytes, R.UnknownAccesses);
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
            // Kernel-wide accumulators for the summary line. The summary
            // only needs Instrs and the Measurement stream — opcode
            // histogram and the unknown-* counters are per-BB diagnostics
            // and don't appear in the summary's printFlopsAndBytes call.
            uint64_t TotalInstrs = 0;
            SmallVector<ptxai::Measurement, 0> TotalMs;
            const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
            const MachineLoopInfo &MLI =
                getAnalysis<MachineLoopInfoWrapperPass>().getLI();

            errs() << "kernel " << MF.getName() << "\n";

            for (auto &MBB : MF) {
                ++Blocks;
                BBRecord R;

                for (auto &MI : MBB) {
                    if (MI.isDebugInstr())
                        continue;

                    ++R.Instrs;
                    ++R.OpcodeCounts[MI.getOpcode()];

                    // INLINEASM: extract the asm body and route each
                    // parsed PTX statement through the inline-PTX
                    // classifier. Operand 0 of an INLINEASM MI is the
                    // asm string (per llvm/IR/InlineAsm.h: MIOp_AsmString = 0).
                    if (MI.isInlineAsm()) {
                        const char *Asm = MI.getOperand(0).getSymbolName();
                        if (Asm && *Asm) {
                            for (const ptxai::ptx::Stmt &S :
                                 ptxai::ptx::parse(StringRef(Asm))) {
                                applyInlinePtxOpClass(R, ptxai::ptx::classify(S));
                            }
                        }
                        // Still record any MMOs LLVM attached to the
                        // INLINEASM (rare but possible on some atomic
                        // intrinsics).
                        recordMemory(R, MI, *TII);
                        continue;
                    }

                    ptxai::OpClass Op =
                        ptxai::classify(TII->getName(MI.getOpcode()));
                    for (const ptxai::Measurement &M : ptxai::toMeasurements(Op))
                        R.Ms.push_back(M);
                    recordMemory(R, MI, *TII);
                }

                printBlockStats(MBB, R, *TII, MLI);
                TotalInstrs += R.Instrs;
                TotalMs.append(R.Ms.begin(), R.Ms.end());
            }

            errs() << "summary: " << MF.getName()
                   << " blocks=" << Blocks;
            ptxai::printFlopsAndBytes(errs(), TotalInstrs, ptxai::Stats(TotalMs));
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
