//===- ptx_unit_tests.cpp - Self-contained tests for lib/PTX/ ----------===//
//
// Small assertion-based test binary for the inline-PTX
// Tokenizer / Parser / Classifier pipeline. Linked against the same
// objects as the analyzer plugin via $<TARGET_OBJECTS:ptxai_ptx>.
//
// Run via the `test-ptx-unit` CMake target (see test/CMakeLists.txt).
//
// Each test prints a single line on entry; a failure aborts via assert
// or via a printed message + nonzero exit. Add new tests by appending
// to the registry at the bottom.
//
//===---------------------------------------------------------------------===//

#include "PTX/Classifier.h"
#include "PTX/Parser.h"
#include "PTX/Tokenizer.h"

#include "llvm/ADT/StringRef.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <variant>
#include <vector>

// Don't `using namespace ptxai;` — there's both ptxai::OpClass (struct, MIR
// side) and ptxai::ptx::OpClass (variant, PTX side). Use the ptx:: alias.
using namespace ptxai::ptx;
using ptxai::FpPrecision;
using ptxai::InvocationScope;
using ptxai::OpcodeNameMemInfo;
using ptxai::parseMemoryOpcodeName;
using llvm::StringRef;

// ---------------------------------------------------------- assertion utility

static int g_failures = 0;
static const char *g_current_test = "<none>";

#define EXPECT(cond)                                                           \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "  FAIL [%s:%d] in %s: %s\n",                 \
                         __FILE__, __LINE__, g_current_test, #cond);           \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

#define EXPECT_EQ(actual, expected)                                            \
    do {                                                                       \
        auto _a = (actual);                                                    \
        auto _e = (expected);                                                  \
        if (!(_a == _e)) {                                                     \
            std::fprintf(stderr, "  FAIL [%s:%d] in %s: %s != %s\n",           \
                         __FILE__, __LINE__, g_current_test, #actual, #expected); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------- tokenizer tests

static void test_tokenizer_basic() {
    auto toks = tokenize("mov.u32 %0, %%tid.x;");
    // Expected: mov(Ident) .(Dot) u32(Ident) %0(OpRef) ,(Comma) %%tid.x(Reg) ;(Semi) EOF
    EXPECT_EQ((int)toks.size(), 8);
    EXPECT_EQ(toks[0].kind, TokenKind::Identifier);
    EXPECT(toks[0].text == "mov");
    EXPECT_EQ(toks[1].kind, TokenKind::Dot);
    EXPECT_EQ(toks[2].kind, TokenKind::Identifier);
    EXPECT(toks[2].text == "u32");
    EXPECT_EQ(toks[3].kind, TokenKind::OperandRef);
    EXPECT(toks[3].text == "%0");
    EXPECT_EQ(toks[4].kind, TokenKind::Comma);
    EXPECT_EQ(toks[5].kind, TokenKind::Register);
    EXPECT(toks[5].text == "%%tid.x");
    EXPECT_EQ(toks[6].kind, TokenKind::Semicolon);
    EXPECT_EQ(toks[7].kind, TokenKind::EndOfFile);
}

static void test_tokenizer_scope_qualified() {
    // "shared::cluster" must be a single identifier token.
    auto toks = tokenize("cp.async.bulk.shared::cluster.global");
    bool found = false;
    for (auto &t : toks) {
        if (t.kind == TokenKind::Identifier && t.text == "shared::cluster") {
            found = true;
            break;
        }
    }
    EXPECT(found);
}

static void test_tokenizer_numbers() {
    auto toks = tokenize("123 0xff 3.14 -16 1e5");
    EXPECT(toks.size() >= 6); // 5 numbers + EOF
    int numCount = 0;
    for (auto &t : toks)
        if (t.kind == TokenKind::Number) ++numCount;
    EXPECT_EQ(numCount, 5);
}

static void test_tokenizer_brace_and_brackets() {
    auto toks = tokenize("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];");
    int lb = 0, rb = 0, lbk = 0, rbk = 0;
    for (auto &t : toks) {
        if (t.kind == TokenKind::LBrace)   ++lb;
        if (t.kind == TokenKind::RBrace)   ++rb;
        if (t.kind == TokenKind::LBracket) ++lbk;
        if (t.kind == TokenKind::RBracket) ++rbk;
    }
    EXPECT_EQ(lb, 1); EXPECT_EQ(rb, 1);
    EXPECT_EQ(lbk, 1); EXPECT_EQ(rbk, 1);
}

static void test_tokenizer_dollar_operand_ref() {
    // LLVM MIR inline-asm bodies use $N (not %N) for operand placeholders.
    // Both must lex to OperandRef.
    auto a = tokenize("add.f32 %0, %1, %2;");
    auto b = tokenize("add.f32 $0, $1, $2;");
    int aRefs = 0, bRefs = 0;
    for (auto &t : a) if (t.kind == TokenKind::OperandRef) ++aRefs;
    for (auto &t : b) if (t.kind == TokenKind::OperandRef) ++bRefs;
    EXPECT_EQ(aRefs, 3);
    EXPECT_EQ(bRefs, 3);
}

static void test_tokenizer_predicate() {
    auto toks = tokenize("@!%p1 bra LBB;");
    EXPECT_EQ(toks[0].kind, TokenKind::At);
    EXPECT_EQ(toks[1].kind, TokenKind::Bang);
    EXPECT_EQ(toks[2].kind, TokenKind::Register);
    EXPECT(toks[2].text == "%p1");
}

static void test_tokenizer_comments() {
    auto toks = tokenize("// line comment\n add.f32 /*inline*/ %0, %1, %2;");
    int idents = 0;
    for (auto &t : toks)
        if (t.kind == TokenKind::Identifier) ++idents;
    EXPECT_EQ(idents, 2); // "add" and "f32"
}

// ----------------------------------------------------------- parser tests

static void test_parser_single_stmt() {
    auto stmts = parse("add.f32 %0, %1, %2;");
    EXPECT_EQ((int)stmts.size(), 1);
    EXPECT(stmts[0].mnemonic == "add");
    EXPECT_EQ((int)stmts[0].modifiers.size(), 1);
    EXPECT(stmts[0].modifiers[0] == "f32");
    EXPECT_EQ((int)stmts[0].operands.size(), 3);
    EXPECT(!stmts[0].parseError);
}

static void test_parser_multi_stmt() {
    auto stmts = parse("mov.u32 %0, 1; add.f32 %1, %2, %3;");
    EXPECT_EQ((int)stmts.size(), 2);
    EXPECT(stmts[0].mnemonic == "mov");
    EXPECT(stmts[1].mnemonic == "add");
}

static void test_parser_predicate() {
    auto stmts = parse("@!%p1 bra LBB;");
    EXPECT_EQ((int)stmts.size(), 1);
    EXPECT(stmts[0].predicateNegated);
    EXPECT(stmts[0].predicate == "%p1");
    EXPECT(stmts[0].mnemonic == "bra");
}

static void test_parser_memory_operand() {
    auto stmts = parse("ld.global.f32 %0, [%1];");
    EXPECT_EQ((int)stmts.size(), 1);
    EXPECT(stmts[0].mnemonic == "ld");
    EXPECT_EQ((int)stmts[0].operands.size(), 2);
    auto *mem = std::get_if<Memory>(&stmts[0].operands[1]);
    EXPECT(mem != nullptr);
}

static void test_parser_brace_list() {
    auto stmts = parse("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];");
    EXPECT_EQ((int)stmts.size(), 1);
    EXPECT_EQ((int)stmts[0].operands.size(), 2);
    auto *bl = std::get_if<BraceList>(&stmts[0].operands[0]);
    EXPECT(bl != nullptr);
    if (bl) EXPECT_EQ((int)bl->children.size(), 4);
}

static void test_parser_error_recovery() {
    // The "BAD" token is an identifier where a mnemonic is expected on the
    // SECOND statement, but the first should parse OK and the parser should
    // continue past the second's semicolon.
    auto stmts = parse("add.f32 %0, %1, %2; @ ; mul.f32 %3, %4, %5;");
    // Statements: 1 valid add, 1 malformed (predicate without target), 1
    // valid mul. The middle one should be flagged but not stop parsing.
    EXPECT(stmts.size() >= 2);
    EXPECT(stmts.front().mnemonic == "add" && !stmts.front().parseError);
    EXPECT(stmts.back().mnemonic == "mul" && !stmts.back().parseError);
}

// ------------------------------------------------------- classifier tests

static const FlopOp *expectFlop(const OpClass &op) { return std::get_if<FlopOp>(&op); }
static const MMAOp  *expectMMA(const OpClass &op)  { return std::get_if<MMAOp>(&op); }
static const MemoryOp *expectMem(const OpClass &op) { return std::get_if<MemoryOp>(&op); }
static const AsyncCopy *expectAC(const OpClass &op) { return std::get_if<AsyncCopy>(&op); }
static const LdMatrix *expectLM(const OpClass &op) { return std::get_if<LdMatrix>(&op); }
static const Barrier *expectBar(const OpClass &op) { return std::get_if<Barrier>(&op); }
static const WarpSync *expectWarp(const OpClass &op) { return std::get_if<WarpSync>(&op); }
static const Ignore *expectIgnore(const OpClass &op) { return std::get_if<Ignore>(&op); }

static OpClass classifyOnly(StringRef src) {
    auto stmts = parse(src);
    if (stmts.empty()) return OpClass{Unknown{StringRef()}};
    return classify(stmts[0]);
}

static void test_classify_fma_f32() {
    auto op = classifyOnly("fma.rn.f32 %0, %1, %2, %3;");
    auto *f = expectFlop(op);
    EXPECT(f != nullptr);
    if (f) {
        EXPECT_EQ((int)f->flops, 2);
        EXPECT(f->precision == FpPrecision::F32);
        EXPECT(f->scope == InvocationScope::PerThread);
    }
}

static void test_classify_fma_f16x2() {
    auto op = classifyOnly("fma.rn.f16x2 %0, %1, %2, %3;");
    auto *f = expectFlop(op);
    EXPECT(f != nullptr);
    if (f) {
        EXPECT_EQ((int)f->flops, 4);  // 2 (FMA) × 2 (lane)
        EXPECT(f->precision == FpPrecision::F16);
    }
}

static void test_classify_fma_bf16x2() {
    auto op = classifyOnly("fma.rn.bf16x2 %0, %1, %2, %3;");
    auto *f = expectFlop(op);
    EXPECT(f != nullptr);
    if (f) {
        EXPECT_EQ((int)f->flops, 4);
        EXPECT(f->precision == FpPrecision::BF16);
    }
}

static void test_classify_add_f32() {
    auto op = classifyOnly("add.f32 %0, %1, %2;");
    auto *f = expectFlop(op);
    EXPECT(f != nullptr);
    if (f) {
        EXPECT_EQ((int)f->flops, 1);
        EXPECT(f->precision == FpPrecision::F32);
    }
}

static void test_classify_div_sqrt() {
    auto sqop = classifyOnly("sqrt.approx.f32 %0, %1;");
    EXPECT(expectFlop(sqop) != nullptr);
    auto divop = classifyOnly("div.rn.f64 %0, %1, %2;");
    auto *f = expectFlop(divop);
    EXPECT(f != nullptr);
    if (f) EXPECT(f->precision == FpPrecision::F64);
}

static void test_classify_mma_m16n8k16() {
    auto op = classifyOnly(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};");
    auto *m = expectMMA(op);
    EXPECT(m != nullptr);
    if (m) {
        EXPECT_EQ((int)m->M, 16);
        EXPECT_EQ((int)m->N, 8);
        EXPECT_EQ((int)m->K, 16);
        EXPECT_EQ((int)m->flops, 2 * 16 * 8 * 16);
        EXPECT(m->scope == InvocationScope::PerWarp);
    }
}

static void test_classify_wgmma_m64n128k16() {
    auto op = classifyOnly(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3}, %4, %5, 1, 1, 1, 0, 0;");
    auto *m = expectMMA(op);
    EXPECT(m != nullptr);
    if (m) {
        EXPECT_EQ((int)m->M, 64);
        EXPECT_EQ((int)m->N, 128);
        EXPECT_EQ((int)m->K, 16);
        EXPECT_EQ((int)m->flops, 2 * 64 * 128 * 16);
    }
}

static void test_classify_ld_global_f32() {
    auto op = classifyOnly("ld.global.f32 %0, [%1];");
    auto *m = expectMem(op);
    EXPECT(m != nullptr);
    if (m) {
        EXPECT_EQ((int)m->bytes, 4);
        EXPECT(m->isLoad && !m->isStore);
        EXPECT_EQ((int)m->addrSpace, 1); // global
    }
}

static void test_classify_ld_global_v4_f32() {
    auto op = classifyOnly("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];");
    auto *m = expectMem(op);
    EXPECT(m != nullptr);
    if (m) EXPECT_EQ((int)m->bytes, 16); // 4 lanes × 4 bytes
}

static void test_classify_st_shared_b32() {
    auto op = classifyOnly("st.shared.b32 [%0], %1;");
    auto *m = expectMem(op);
    EXPECT(m != nullptr);
    if (m) {
        EXPECT_EQ((int)m->bytes, 4);
        EXPECT(!m->isLoad && m->isStore);
        EXPECT_EQ((int)m->addrSpace, 3); // shared
    }
}

static void test_classify_cp_async_with_bytes() {
    auto op = classifyOnly("cp.async.cg.shared.global [%0], [%1], 16;");
    auto *ac = expectAC(op);
    EXPECT(ac != nullptr);
    if (ac) {
        EXPECT(ac->bytes.has_value());
        if (ac->bytes) EXPECT_EQ((int)*ac->bytes, 16);
    }
}

static void test_classify_ldmatrix_x4() {
    auto op = classifyOnly(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];");
    auto *l = expectLM(op);
    EXPECT(l != nullptr);
    if (l) {
        // x4 × 8x8 × 2 bytes (b16) = 4 × 64 × 2 = 512 bytes per warp.
        EXPECT_EQ((int)l->bytes, 512);
        EXPECT(l->scope == InvocationScope::PerWarp);
    }
}

static void test_classify_barrier_warp() {
    EXPECT(expectBar(classifyOnly("bar.sync 0;")) != nullptr);
    EXPECT(expectBar(classifyOnly("mbarrier.arrive.shared::cta [%0];")) != nullptr);
    EXPECT(expectWarp(classifyOnly("shfl.sync.idx.b32 %0, %1, %2, %3, %4;")) != nullptr);
    EXPECT(expectWarp(classifyOnly("vote.sync.all.pred %0, %1, %2;")) != nullptr);
}

static void test_classify_atomic_add_f32() {
    auto op = classifyOnly("atom.global.add.f32 %0, [%1], %2;");
    auto *m = expectMem(op);
    EXPECT(m != nullptr);
    if (m) {
        EXPECT(m->isLoad && m->isStore);  // RMW: counted as both
        EXPECT_EQ((int)m->bytes, 4);
    }
}

static void test_classify_ignore_set() {
    EXPECT(expectIgnore(classifyOnly("cvt.u32.s32 %0, %1;")) != nullptr);
    EXPECT(expectIgnore(classifyOnly("mov.u32 %0, %%tid.x;")) != nullptr);
    EXPECT(expectIgnore(classifyOnly("setp.eq.s32 %p1, %0, %1;")) != nullptr);
    EXPECT(expectIgnore(classifyOnly("selp.b32 %0, %1, %2, %3;")) != nullptr);
}

static void test_classify_unknown() {
    auto op = classifyOnly("totally.fake.opcode %0, %1;");
    auto *u = std::get_if<Unknown>(&op);
    EXPECT(u != nullptr);
    if (u) EXPECT(u->mnemonic == "totally");
}

// --------------------------------------------------- realistic excerpts

// Realistic excerpt from cuda_fp16.h's __hadd2 expansion.
static void test_classify_cuda_fp16_hadd2() {
    auto op = classifyOnly("{add.f16x2 %0,%1,%2;\n}");
    // Note the surrounding "{ ... }" — these are PTX scope braces. Our
    // tokenizer emits LBrace/RBrace; the parser treats them as start of a
    // BraceList operand, but we don't currently allow braces at statement
    // start. Verify graceful handling: either parses to "add" with the
    // brace as an operand, or marks parseError. Either way the FLOP count
    // shouldn't lie.
    if (auto *f = expectFlop(op)) {
        EXPECT(f->precision == FpPrecision::F16);
        EXPECT_EQ((int)f->flops, 2); // add × 2 lanes
    }
}

// Realistic excerpt from cuda_fp16.h's __hfma2 expansion.
static void test_classify_cuda_fp16_hfma2() {
    auto op = classifyOnly("fma.rn.f16x2 %0,%1,%2,%3;");
    auto *f = expectFlop(op);
    EXPECT(f != nullptr);
    if (f) {
        EXPECT_EQ((int)f->flops, 4); // FMA × 2 lanes
        EXPECT(f->precision == FpPrecision::F16);
    }
}

// Realistic CCCL get_sreg pattern.
static void test_classify_get_sreg() {
    auto op = classifyOnly("mov.u32 %0, %%tid.x;");
    EXPECT(expectIgnore(op) != nullptr);
}

// Realistic CUTLASS pattern: integer MMA with sat finite.
static void test_classify_mma_int8_sat() {
    auto op = classifyOnly(
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 "
        "{$0,$1}, {$2}, {$3}, {$4,$5};");
    auto *m = std::get_if<MMAOp>(&op);
    EXPECT(m != nullptr);
    if (m) {
        EXPECT_EQ((int)m->M, 8);
        EXPECT_EQ((int)m->N, 8);
        EXPECT_EQ((int)m->K, 16);
        EXPECT_EQ((int)m->flops, 2 * 8 * 8 * 16);
    }
}

// CUTLASS predicated store inside inline asm: PTX-directive `.reg` should
// be classified as Ignore (not parseError).
static void test_classify_predicated_store_block() {
    auto stmts = parse(
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; "
        "@p st.global.v4.u32 [$0], {$1,$2,$3,$4}; }");
    EXPECT(stmts.size() >= 3);
    int errs = 0;
    for (auto &s : stmts) if (s.parseError) ++errs;
    EXPECT_EQ(errs, 0);  // .reg/setp/@p st should ALL parse without errors
}

// cp.async.commit_group is a sync marker, not a transfer.
static void test_classify_cp_async_commit_group() {
    auto op = classifyOnly("cp.async.commit_group;");
    EXPECT(std::get_if<Barrier>(&op) != nullptr);
}

// cp.async.cg.shared.global.L2::128B passes byte-count via OperandRef
// (runtime register), so AsyncCopy::bytes should be unset.
static void test_classify_cp_async_runtime_bytes() {
    auto op = classifyOnly(
        "cp.async.cg.shared.global.L2::128B [$0], [$1], $2, $3;");
    auto *ac = std::get_if<AsyncCopy>(&op);
    EXPECT(ac != nullptr);
    if (ac) {
        EXPECT(!ac->bytes.has_value());  // operand 2 is %2, not an immediate
    }
}

// Empty body: classifier should produce nothing harmful.
static void test_parse_empty_body() {
    auto stmts = parse("");
    EXPECT(stmts.empty());
    auto stmts2 = parse("   \n  /* nothing */  ");
    EXPECT(stmts2.empty());
}

// Realistic CCCL cp.async.bulk.tensor pattern.
static void test_classify_cp_async_bulk_tensor() {
    auto op = classifyOnly(
        "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
        "[%0], [%1, {%2,%3,%4,%5,%6}];");
    auto *ac = expectAC(op);
    EXPECT(ac != nullptr);
    // Byte count is descriptor-defined; not statically derivable from asm.
    // Verify we surface as AsyncCopy rather than Unknown.
}

// =============================================================
// parseMemoryOpcodeName — opcode-name fallback for LDG/LDU/etc.
// =============================================================
//
// Tests cover: every prefix family, every recognized type-width, both
// vector-form notations (LDV-style "_v2_<type>" and LD_GLOBAL_NC-style
// joined "v2<type>"), plus negative cases.

// Helpers for assertions on the OpcodeNameMemInfo result
static void expectMem(const std::optional<OpcodeNameMemInfo> &info,
                      unsigned addrSpace, uint64_t bytes,
                      bool isLoad, bool isStore) {
    EXPECT(info.has_value());
    if (!info) return;
    EXPECT_EQ((int)info->addrSpace, (int)addrSpace);
    EXPECT_EQ((int)info->bytes, (int)bytes);
    EXPECT_EQ(info->isLoad, isLoad);
    EXPECT_EQ(info->isStore, isStore);
}

// --- LDG family: read-only / non-coherent global loads -------------------

static void test_mem_ld_global_nc_i8()  { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i8"),  1, 1, true, false); }
static void test_mem_ld_global_nc_i16() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i16"), 1, 2, true, false); }
static void test_mem_ld_global_nc_i32() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i32"), 1, 4, true, false); }
static void test_mem_ld_global_nc_i64() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i64"), 1, 8, true, false); }

// Vector LDG: NVPTX writes "v2i32" / "v4i64" without separator.
static void test_mem_ld_global_nc_v2i16() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_v2i16"), 1,  4, true, false); }
static void test_mem_ld_global_nc_v2i32() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_v2i32"), 1,  8, true, false); }
static void test_mem_ld_global_nc_v2i64() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_v2i64"), 1, 16, true, false); }
static void test_mem_ld_global_nc_v4i16() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_v4i16"), 1,  8, true, false); }
static void test_mem_ld_global_nc_v4i32() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_v4i32"), 1, 16, true, false); }
static void test_mem_ld_global_nc_v4i64() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_v4i64"), 1, 32, true, false); }
static void test_mem_ld_global_nc_v8i32() { expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_v8i32"), 1, 32, true, false); }

// --- LDU_GLOBAL family: uniform global loads -----------------------------

static void test_mem_ldu_global_i32()  { expectMem(parseMemoryOpcodeName("LDU_GLOBAL_i32"),   1, 4, true, false); }
static void test_mem_ldu_global_i64()  { expectMem(parseMemoryOpcodeName("LDU_GLOBAL_i64"),   1, 8, true, false); }
static void test_mem_ldu_global_v4i32(){ expectMem(parseMemoryOpcodeName("LDU_GLOBAL_v4i32"), 1, 16, true, false); }

// --- Defensive: alternative naming we may see in older/newer LLVM --------

static void test_mem_ldg_b32() { expectMem(parseMemoryOpcodeName("LDG_b32"), 1, 4, true, false); }
static void test_mem_ldu_f64() { expectMem(parseMemoryOpcodeName("LDU_f64"), 1, 8, true, false); }
static void test_mem_ld_global_f32() {
    // Plain LD_GLOBAL_<type> (without _NC) — defensive coverage in case
    // future lowering moves into this naming.
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_f32"), 1, 4, true, false);
}

// --- Other address spaces (defensive) -----------------------------------

static void test_mem_ld_shared_b16() { expectMem(parseMemoryOpcodeName("LD_SHARED_b16"), 3, 2, true, false); }
static void test_mem_lds_b32()       { expectMem(parseMemoryOpcodeName("LDS_b32"),        3, 4, true, false); }
static void test_mem_ld_local_i32()  { expectMem(parseMemoryOpcodeName("LD_LOCAL_i32"),   5, 4, true, false); }
static void test_mem_ld_const_i64()  { expectMem(parseMemoryOpcodeName("LD_CONST_i64"),   4, 8, true, false); }
static void test_mem_ldc_f32()       { expectMem(parseMemoryOpcodeName("LDC_f32"),        4, 4, true, false); }
static void test_mem_ld_param_b64()  { expectMem(parseMemoryOpcodeName("LD_PARAM_b64"), 101, 8, true, false); }

// --- Stores (defensive — no current empty-MMO store family observed) ----

static void test_mem_st_global_i32()  { expectMem(parseMemoryOpcodeName("ST_GLOBAL_i32"),  1, 4, false, true); }
static void test_mem_st_shared_i64()  { expectMem(parseMemoryOpcodeName("ST_SHARED_i64"),  3, 8, false, true); }
static void test_mem_st_local_b16()   { expectMem(parseMemoryOpcodeName("ST_LOCAL_b16"),   5, 2, false, true); }
static void test_mem_st_global_v4i32(){ expectMem(parseMemoryOpcodeName("ST_GLOBAL_v4i32"), 1, 16, false, true); }

// --- All recognized type widths -----------------------------------------

static void test_mem_widths_8()   {
    // 1-byte: i8 / u8 / s8 / b8
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i8"), 1, 1, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_u8"), 1, 1, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_s8"), 1, 1, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_b8"), 1, 1, true, false);
}
static void test_mem_widths_16()  {
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i16"),  1, 2, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_u16"),  1, 2, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_s16"),  1, 2, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_b16"),  1, 2, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_f16"),  1, 2, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_bf16"), 1, 2, true, false);
}
static void test_mem_widths_32()  {
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i32"), 1, 4, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_u32"), 1, 4, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_s32"), 1, 4, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_b32"), 1, 4, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_f32"), 1, 4, true, false);
}
static void test_mem_widths_64()  {
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i64"), 1, 8, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_u64"), 1, 8, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_s64"), 1, 8, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_b64"), 1, 8, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_f64"), 1, 8, true, false);
}
static void test_mem_widths_128() {
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_i128"), 1, 16, true, false);
    expectMem(parseMemoryOpcodeName("LD_GLOBAL_NC_b128"), 1, 16, true, false);
}

// --- Negative cases: must return std::nullopt ---------------------------

static void test_mem_neg_empty()        { EXPECT(!parseMemoryOpcodeName("").has_value()); }
static void test_mem_neg_unknown_pfx()  { EXPECT(!parseMemoryOpcodeName("FOO_BAR_i32").has_value()); }
static void test_mem_neg_no_width()     { EXPECT(!parseMemoryOpcodeName("LD_GLOBAL_NC_garbage").has_value()); }
static void test_mem_neg_plain_ld() {
    // Plain LD_<type> opcodes without an addrspace prefix carry their
    // address space in the MMO. We must NOT classify them — the MMO path
    // handles them. Returning nullopt here ensures we never silently
    // claim "global" for a generic load whose addrspace is unknown.
    EXPECT(!parseMemoryOpcodeName("LD_i32").has_value());
    EXPECT(!parseMemoryOpcodeName("LD_i64").has_value());
    EXPECT(!parseMemoryOpcodeName("ST_i32").has_value());
}
static void test_mem_neg_just_ldv() {
    // LDV_*/STV_* always carry MMOs (we verified empirically). The
    // existing MMO path handles them; the fallback must not match,
    // otherwise it would risk double-counting. LDV_i32_v4 has no
    // address-space prefix, so classifyPrefix should return nullopt.
    EXPECT(!parseMemoryOpcodeName("LDV_i32_v4").has_value());
    EXPECT(!parseMemoryOpcodeName("STV_i64_v2").has_value());
}
static void test_mem_neg_unrelated()    {
    EXPECT(!parseMemoryOpcodeName("FMA_F32rrr").has_value());
    EXPECT(!parseMemoryOpcodeName("ADD64rr").has_value());
    EXPECT(!parseMemoryOpcodeName("INT_PTX_SREG_TID_x").has_value());
}

// =============================================================
// Measurement + toMeasurements (PR 1)
// =============================================================
//
// Coverage rationale (see docs/measurement-refactor.md §6.3):
//   - One test per variant arm of ptx::OpClass (12 arms / cases)
//   - All five FpPrecision values for the Flop arm
//   - All five named address spaces for the Memory arm
//   - The two "metadata-shaped" cases that distinguish Measurement from
//     the variant: AsyncCopy{bytes unset} → empty; LdMatrix scope
//     preserved as PerWarp.
//   - The lane-multiplier case (FMA on packed fp16x2 → count = 4).
//   - End-to-end integration through parse + classify + toMeasurements
//     for three canonical asm strings.
// applyToBlockStats coverage is intentionally deferred — it's tested
// end-to-end by the FileCheck suite (all 8 green tests verify exact
// output text), and will be deleted in PR 6/7. Direct unit tests would
// require exposing BlockStats publicly, which we're not going to do
// since the struct is going away.

#include "Measurement.h"

using ptxai::Measurement;
using ptxai::toMeasurements;

// ----- Measurement struct sanity -----

static void test_meas_default_init() {
    Measurement m{Measurement::Kind::Flop};
    EXPECT(m.kind == Measurement::Kind::Flop);
    EXPECT(m.scope == ptxai::InvocationScope::PerThread);
    EXPECT(m.precision == ptxai::FpPrecision::Other);
    EXPECT_EQ((int)m.addrSpace, 0);
    EXPECT(!m.isLoad);
    EXPECT(!m.isStore);
    EXPECT_EQ((int)m.count, 0);
}

static void test_meas_size_within_budget() {
    // Guards against accidental field bloat. Design target = 24 bytes.
    EXPECT(sizeof(Measurement) <= 24);
}

// ----- toMeasurements: FlopOp arm × precision -----

static void test_meas_flop_per_thread_f16() {
    auto ms = toMeasurements(ptxai::ptx::OpClass{
        ptxai::ptx::FlopOp{ptxai::FpPrecision::F16, 4,
                            ptxai::InvocationScope::PerThread}});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].kind == Measurement::Kind::Flop);
        EXPECT(ms[0].precision == ptxai::FpPrecision::F16);
        EXPECT(ms[0].scope == ptxai::InvocationScope::PerThread);
        EXPECT_EQ((int)ms[0].count, 4);
    }
}

static void test_meas_flop_per_thread_bf16() {
    auto ms = toMeasurements(ptxai::ptx::OpClass{
        ptxai::ptx::FlopOp{ptxai::FpPrecision::BF16, 2,
                            ptxai::InvocationScope::PerThread}});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].precision == ptxai::FpPrecision::BF16);
}

static void test_meas_flop_per_thread_f32() {
    auto ms = toMeasurements(ptxai::ptx::OpClass{
        ptxai::ptx::FlopOp{ptxai::FpPrecision::F32, 2,
                            ptxai::InvocationScope::PerThread}});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].precision == ptxai::FpPrecision::F32);
}

static void test_meas_flop_per_thread_f64() {
    auto ms = toMeasurements(ptxai::ptx::OpClass{
        ptxai::ptx::FlopOp{ptxai::FpPrecision::F64, 1,
                            ptxai::InvocationScope::PerThread}});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].precision == ptxai::FpPrecision::F64);
}

static void test_meas_flop_per_thread_other() {
    auto ms = toMeasurements(ptxai::ptx::OpClass{
        ptxai::ptx::FlopOp{ptxai::FpPrecision::Other, 1,
                            ptxai::InvocationScope::PerThread}});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].precision == ptxai::FpPrecision::Other);
}

static void test_meas_flop_per_warp_dropped() {
    // Preserves PR-0 behaviour: PerWarp FlopOp is dropped at the converter
    // (no current producer emits one; the MIR-side aggregator asserts).
    auto ms = toMeasurements(ptxai::ptx::OpClass{
        ptxai::ptx::FlopOp{ptxai::FpPrecision::F32, 2,
                            ptxai::InvocationScope::PerWarp}});
    EXPECT_EQ((int)ms.size(), 0);
}

// ----- toMeasurements: MMAOp -----

static void test_meas_mma_per_warp_flop() {
    ptxai::ptx::MMAOp m;
    m.M = 16; m.N = 8; m.K = 16;
    m.flops = 2u * 16 * 8 * 16;
    m.scope = ptxai::InvocationScope::PerWarp;
    auto ms = toMeasurements(ptxai::ptx::OpClass{m});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].kind == Measurement::Kind::Flop);
        EXPECT(ms[0].scope == ptxai::InvocationScope::PerWarp);
        EXPECT(ms[0].precision == ptxai::FpPrecision::Other);
        EXPECT_EQ((int)ms[0].count, 2 * 16 * 8 * 16);
    }
}

// ----- toMeasurements: MemoryOp × addrSpace × direction -----

static void test_meas_memop_global_load() {
    ptxai::ptx::MemoryOp mem;
    mem.addrSpace = 1; mem.bytes = 4; mem.isLoad = true;
    auto ms = toMeasurements(ptxai::ptx::OpClass{mem});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].kind == Measurement::Kind::Memory);
        EXPECT_EQ((int)ms[0].addrSpace, 1);
        EXPECT(ms[0].isLoad && !ms[0].isStore);
        EXPECT_EQ((int)ms[0].count, 4);
    }
}

static void test_meas_memop_global_store() {
    ptxai::ptx::MemoryOp mem;
    mem.addrSpace = 1; mem.bytes = 4; mem.isStore = true;
    auto ms = toMeasurements(ptxai::ptx::OpClass{mem});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(!ms[0].isLoad && ms[0].isStore);
}

static void test_meas_memop_shared_load() {
    ptxai::ptx::MemoryOp mem;
    mem.addrSpace = 3; mem.bytes = 16; mem.isLoad = true;
    auto ms = toMeasurements(ptxai::ptx::OpClass{mem});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT_EQ((int)ms[0].addrSpace, 3);
        EXPECT_EQ((int)ms[0].count, 16);
    }
}

static void test_meas_memop_atomic_rmw() {
    // Atomics produce a single measurement with both flags set.
    ptxai::ptx::MemoryOp mem;
    mem.addrSpace = 1; mem.bytes = 4;
    mem.isLoad = true; mem.isStore = true;
    auto ms = toMeasurements(ptxai::ptx::OpClass{mem});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].isLoad && ms[0].isStore);
}

static void test_meas_memop_local() {
    ptxai::ptx::MemoryOp mem;
    mem.addrSpace = 5; mem.bytes = 8; mem.isLoad = true;
    auto ms = toMeasurements(ptxai::ptx::OpClass{mem});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT_EQ((int)ms[0].addrSpace, 5);
}

static void test_meas_memop_const() {
    ptxai::ptx::MemoryOp mem;
    mem.addrSpace = 4; mem.bytes = 4; mem.isLoad = true;
    auto ms = toMeasurements(ptxai::ptx::OpClass{mem});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT_EQ((int)ms[0].addrSpace, 4);
}

static void test_meas_memop_param() {
    ptxai::ptx::MemoryOp mem;
    mem.addrSpace = 101; mem.bytes = 8; mem.isLoad = true;
    auto ms = toMeasurements(ptxai::ptx::OpClass{mem});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT_EQ((int)ms[0].addrSpace, 101);
}

// ----- toMeasurements: AsyncCopy -----

static void test_meas_async_copy_known_bytes() {
    // cp.async with literal byte count: 2 measurements (global load +
    // shared store), each carrying the same byte count.
    ptxai::ptx::AsyncCopy ac;
    ac.dstAddrSpace = 3; ac.srcAddrSpace = 1; ac.bytes = 16;
    auto ms = toMeasurements(ptxai::ptx::OpClass{ac});
    EXPECT_EQ((int)ms.size(), 2);
    if (ms.size() == 2) {
        EXPECT(ms[0].kind == Measurement::Kind::Memory);
        EXPECT_EQ((int)ms[0].addrSpace, 1);   // AS_GLOBAL
        EXPECT(ms[0].isLoad && !ms[0].isStore);
        EXPECT_EQ((int)ms[0].count, 16);
        EXPECT_EQ((int)ms[1].addrSpace, 3);   // AS_SHARED
        EXPECT(!ms[1].isLoad && ms[1].isStore);
        EXPECT_EQ((int)ms[1].count, 16);
    }
}

static void test_meas_async_copy_unknown_bytes() {
    // cp.async with runtime byte count (operand 2 is a register, not an
    // immediate). bytes is nullopt → zero measurements.
    ptxai::ptx::AsyncCopy ac;
    ac.dstAddrSpace = 3; ac.srcAddrSpace = 1;
    auto ms = toMeasurements(ptxai::ptx::OpClass{ac});
    EXPECT_EQ((int)ms.size(), 0);
}

// ----- toMeasurements: LdMatrix -----

static void test_meas_ldmatrix_per_warp_shared_load() {
    ptxai::ptx::LdMatrix lm;
    lm.addrSpace = 3; lm.bytes = 512;
    lm.scope = ptxai::InvocationScope::PerWarp;
    auto ms = toMeasurements(ptxai::ptx::OpClass{lm});
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].kind == Measurement::Kind::Memory);
        EXPECT(ms[0].scope == ptxai::InvocationScope::PerWarp);
        EXPECT_EQ((int)ms[0].addrSpace, 3);
        EXPECT(ms[0].isLoad);
        EXPECT_EQ((int)ms[0].count, 512);
    }
}

// ----- toMeasurements: zero-measurement arms -----

static void test_meas_warpsync_empty() {
    auto ms = toMeasurements(ptxai::ptx::OpClass{ptxai::ptx::WarpSync{}});
    EXPECT_EQ((int)ms.size(), 0);
}

static void test_meas_barrier_empty() {
    auto ms = toMeasurements(ptxai::ptx::OpClass{ptxai::ptx::Barrier{}});
    EXPECT_EQ((int)ms.size(), 0);
}

static void test_meas_ignore_empty() {
    auto ms = toMeasurements(ptxai::ptx::OpClass{ptxai::ptx::Ignore{}});
    EXPECT_EQ((int)ms.size(), 0);
}

static void test_meas_unknown_empty() {
    // Unknown emits no Measurement; the diagnostic UnknownAccesses bump
    // lives at the call site (the pass shim), not in the converter.
    auto ms = toMeasurements(ptxai::ptx::OpClass{
        ptxai::ptx::Unknown{llvm::StringRef("foo")}});
    EXPECT_EQ((int)ms.size(), 0);
}

// ----- Integration via parse + classify -----

static llvm::SmallVector<Measurement, 2> classifyAndConvert(llvm::StringRef src) {
    auto stmts = parse(src);
    if (stmts.empty()) return {};
    return toMeasurements(classify(stmts[0]));
}

static void test_meas_integration_fma_f16x2() {
    auto ms = classifyAndConvert("fma.rn.f16x2 %0,%1,%2,%3;");
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].kind == Measurement::Kind::Flop);
        EXPECT(ms[0].precision == ptxai::FpPrecision::F16);
        EXPECT_EQ((int)ms[0].count, 4);     // 2 (FMA) × 2 (lane)
    }
}

static void test_meas_integration_ld_global_f32() {
    auto ms = classifyAndConvert("ld.global.f32 %0, [%1];");
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].kind == Measurement::Kind::Memory);
        EXPECT_EQ((int)ms[0].addrSpace, 1);
        EXPECT(ms[0].isLoad && !ms[0].isStore);
        EXPECT_EQ((int)ms[0].count, 4);
    }
}

static void test_meas_integration_cp_async() {
    auto ms = classifyAndConvert("cp.async.cg.shared.global [%0], [%1], 16;");
    EXPECT_EQ((int)ms.size(), 2);
    if (ms.size() == 2) {
        EXPECT(ms[0].isLoad  && ms[0].count == 16 && ms[0].addrSpace == 1);
        EXPECT(ms[1].isStore && ms[1].count == 16 && ms[1].addrSpace == 3);
    }
}

// =============================================================
// MIR-side Measurement (PR 2)
// =============================================================
//
// The toMeasurements(OpClass) overload on the MIR side. Coverage:
//   - One test per FpPrecision for the ScalarFLOP arm (5 tests)
//   - "Already-multiplied" count preserved (lane multiplier baked in by
//     ptxai::classify before reaching us)
//   - Each non-ScalarFLOP OpKind → 0 measurements (forward-declared
//     enum values that the MIR classifier doesn't yet emit)
//   - Integration through ptxai::classify(opcodeName): the canonical
//     LLVM 18 (bare-width) and modern (underscored) FMA opcode forms.
// applyToBlockStats coverage and recordMemory coverage stay end-to-end
// (FileCheck) for the same reason as in PR 1: they need MachineInstr
// fixtures, and applyToBlockStats is going away in PR 6/7.

static void test_mir_meas_flop_f16() {
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::ScalarFLOP;
    op.precision = ptxai::FpPrecision::F16;
    op.flopsPerInvocation = 2;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].kind == Measurement::Kind::Flop);
        EXPECT(ms[0].precision == ptxai::FpPrecision::F16);
        EXPECT(ms[0].scope == ptxai::InvocationScope::PerThread);
        EXPECT_EQ((int)ms[0].count, 2);
    }
}

static void test_mir_meas_flop_bf16() {
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::ScalarFLOP;
    op.precision = ptxai::FpPrecision::BF16;
    op.flopsPerInvocation = 1;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].precision == ptxai::FpPrecision::BF16);
}

static void test_mir_meas_flop_f32() {
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::ScalarFLOP;
    op.precision = ptxai::FpPrecision::F32;
    op.flopsPerInvocation = 2;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].precision == ptxai::FpPrecision::F32);
}

static void test_mir_meas_flop_f64() {
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::ScalarFLOP;
    op.precision = ptxai::FpPrecision::F64;
    op.flopsPerInvocation = 1;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].precision == ptxai::FpPrecision::F64);
}

static void test_mir_meas_flop_other() {
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::ScalarFLOP;
    op.precision = ptxai::FpPrecision::Other;
    op.flopsPerInvocation = 1;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT(ms[0].precision == ptxai::FpPrecision::Other);
}

static void test_mir_meas_flop_lanes_already_multiplied() {
    // The MIR-side ptxai::classify bakes the lane multiplier into
    // flopsPerInvocation before we see it (e.g. FMA_F16x2rrr → 4 = 2·2).
    // toMeasurements must NOT multiply again.
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::ScalarFLOP;
    op.precision = ptxai::FpPrecision::F16;
    op.flopsPerInvocation = 4;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) EXPECT_EQ((int)ms[0].count, 4);
}

static void test_mir_meas_flop_zero_count_empty() {
    // Defensive: classify returns flopsPerInvocation=0 for non-FLOP
    // opcodes routed through OpKind::None, but if a future arm emits
    // ScalarFLOP with count=0 (shouldn't happen), we must not produce
    // a phantom measurement that the aggregator would then route into
    // Stats.Flops with no effect.
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::ScalarFLOP;
    op.precision = ptxai::FpPrecision::F32;
    op.flopsPerInvocation = 0;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 0);
}

static void test_mir_meas_none_kind_empty() {
    // Default OpClass (kind = None) — the bulk of MIR opcodes (memory,
    // control flow, sreg reads, integer ALU).
    ptxai::OpClass op;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 0);
}

static void test_mir_meas_mma_kind_empty_for_now() {
    // OpKind::MMA is reserved but unimplemented on the MIR side; emitting
    // 0 measurements is correct until Phase 3 wires up the shape table.
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::MMA;
    op.flopsPerInvocation = 4096;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 0);
}

static void test_mir_meas_special_math_empty_for_now() {
    ptxai::OpClass op;
    op.kind = ptxai::OpKind::SpecialMath;
    op.flopsPerInvocation = 1;
    auto ms = toMeasurements(op);
    EXPECT_EQ((int)ms.size(), 0);
}

// ----- Integration via ptxai::classify(opcodeName) -----

static void test_mir_meas_integration_classify_fma_underscored() {
    // Modern LLVM emits FMA_F32rrr; classifier marks it ScalarFLOP, F32,
    // count=2 (FMA = 2 flops); toMeasurements emits one Flop measurement.
    auto ms = toMeasurements(ptxai::classify("FMA_F32rrr"));
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].precision == ptxai::FpPrecision::F32);
        EXPECT_EQ((int)ms[0].count, 2);
    }
}

static void test_mir_meas_integration_classify_fma_bare_width() {
    // LLVM 18 bare-width form (FMA32rrr / FMA64rrr). Same FLOP count;
    // precision recovered from the digit suffix.
    auto ms = toMeasurements(ptxai::classify("FMA64rrr"));
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].precision == ptxai::FpPrecision::F64);
        EXPECT_EQ((int)ms[0].count, 2);
    }
}

static void test_mir_meas_integration_classify_fmul_f16x2() {
    // FMUL_F16x2rrr → 1 flop × 2 lanes = 2.
    auto ms = toMeasurements(ptxai::classify("FMUL_F16x2rrr"));
    EXPECT_EQ((int)ms.size(), 1);
    if (ms.size() == 1) {
        EXPECT(ms[0].precision == ptxai::FpPrecision::F16);
        EXPECT_EQ((int)ms[0].count, 2);
    }
}

static void test_mir_meas_integration_classify_non_flop() {
    // Non-FLOP opcode → OpKind::None → 0 measurements.
    auto ms = toMeasurements(ptxai::classify("ADD64rr"));
    EXPECT_EQ((int)ms.size(), 0);
}

// ---------------------------------------------------------- driver

struct Test { const char *name; std::function<void()> fn; };

int main() {
    std::vector<Test> tests = {
        // tokenizer
        {"tokenizer_basic",          test_tokenizer_basic},
        {"tokenizer_scope_qualified",test_tokenizer_scope_qualified},
        {"tokenizer_numbers",        test_tokenizer_numbers},
        {"tokenizer_brace_brackets", test_tokenizer_brace_and_brackets},
        {"tokenizer_dollar_operand_ref", test_tokenizer_dollar_operand_ref},
        {"tokenizer_predicate",      test_tokenizer_predicate},
        {"tokenizer_comments",       test_tokenizer_comments},
        // parser
        {"parser_single_stmt",       test_parser_single_stmt},
        {"parser_multi_stmt",        test_parser_multi_stmt},
        {"parser_predicate",         test_parser_predicate},
        {"parser_memory_operand",    test_parser_memory_operand},
        {"parser_brace_list",        test_parser_brace_list},
        {"parser_error_recovery",    test_parser_error_recovery},
        // classifier
        {"classify_fma_f32",         test_classify_fma_f32},
        {"classify_fma_f16x2",       test_classify_fma_f16x2},
        {"classify_fma_bf16x2",      test_classify_fma_bf16x2},
        {"classify_add_f32",         test_classify_add_f32},
        {"classify_div_sqrt",        test_classify_div_sqrt},
        {"classify_mma_m16n8k16",    test_classify_mma_m16n8k16},
        {"classify_wgmma_m64n128k16",test_classify_wgmma_m64n128k16},
        {"classify_ld_global_f32",   test_classify_ld_global_f32},
        {"classify_ld_global_v4_f32",test_classify_ld_global_v4_f32},
        {"classify_st_shared_b32",   test_classify_st_shared_b32},
        {"classify_cp_async",        test_classify_cp_async_with_bytes},
        {"classify_ldmatrix_x4",     test_classify_ldmatrix_x4},
        {"classify_barrier_warp",    test_classify_barrier_warp},
        {"classify_atomic_add_f32",  test_classify_atomic_add_f32},
        {"classify_ignore_set",      test_classify_ignore_set},
        {"classify_unknown",         test_classify_unknown},
        // realistic excerpts
        {"cuda_fp16_hadd2",          test_classify_cuda_fp16_hadd2},
        {"cuda_fp16_hfma2",          test_classify_cuda_fp16_hfma2},
        {"cccl_get_sreg",            test_classify_get_sreg},
        {"cccl_cp_async_bulk_tensor",test_classify_cp_async_bulk_tensor},
        {"mma_int8_sat",             test_classify_mma_int8_sat},
        {"predicated_store_block",   test_classify_predicated_store_block},
        {"cp_async_commit_group",    test_classify_cp_async_commit_group},
        {"cp_async_runtime_bytes",   test_classify_cp_async_runtime_bytes},
        {"parse_empty_body",         test_parse_empty_body},
        // memory-opcode-name parser (LDG/LDU/etc. fallback)
        {"mem_ld_global_nc_i8",      test_mem_ld_global_nc_i8},
        {"mem_ld_global_nc_i16",     test_mem_ld_global_nc_i16},
        {"mem_ld_global_nc_i32",     test_mem_ld_global_nc_i32},
        {"mem_ld_global_nc_i64",     test_mem_ld_global_nc_i64},
        {"mem_ld_global_nc_v2i16",   test_mem_ld_global_nc_v2i16},
        {"mem_ld_global_nc_v2i32",   test_mem_ld_global_nc_v2i32},
        {"mem_ld_global_nc_v2i64",   test_mem_ld_global_nc_v2i64},
        {"mem_ld_global_nc_v4i16",   test_mem_ld_global_nc_v4i16},
        {"mem_ld_global_nc_v4i32",   test_mem_ld_global_nc_v4i32},
        {"mem_ld_global_nc_v4i64",   test_mem_ld_global_nc_v4i64},
        {"mem_ld_global_nc_v8i32",   test_mem_ld_global_nc_v8i32},
        {"mem_ldu_global_i32",       test_mem_ldu_global_i32},
        {"mem_ldu_global_i64",       test_mem_ldu_global_i64},
        {"mem_ldu_global_v4i32",     test_mem_ldu_global_v4i32},
        {"mem_ldg_b32",              test_mem_ldg_b32},
        {"mem_ldu_f64",              test_mem_ldu_f64},
        {"mem_ld_global_f32",        test_mem_ld_global_f32},
        {"mem_ld_shared_b16",        test_mem_ld_shared_b16},
        {"mem_lds_b32",              test_mem_lds_b32},
        {"mem_ld_local_i32",         test_mem_ld_local_i32},
        {"mem_ld_const_i64",         test_mem_ld_const_i64},
        {"mem_ldc_f32",              test_mem_ldc_f32},
        {"mem_ld_param_b64",         test_mem_ld_param_b64},
        {"mem_st_global_i32",        test_mem_st_global_i32},
        {"mem_st_shared_i64",        test_mem_st_shared_i64},
        {"mem_st_local_b16",         test_mem_st_local_b16},
        {"mem_st_global_v4i32",      test_mem_st_global_v4i32},
        {"mem_widths_8",             test_mem_widths_8},
        {"mem_widths_16",            test_mem_widths_16},
        {"mem_widths_32",            test_mem_widths_32},
        {"mem_widths_64",            test_mem_widths_64},
        {"mem_widths_128",           test_mem_widths_128},
        {"mem_neg_empty",            test_mem_neg_empty},
        {"mem_neg_unknown_pfx",      test_mem_neg_unknown_pfx},
        {"mem_neg_no_width",         test_mem_neg_no_width},
        {"mem_neg_plain_ld",         test_mem_neg_plain_ld},
        {"mem_neg_just_ldv",         test_mem_neg_just_ldv},
        {"mem_neg_unrelated",        test_mem_neg_unrelated},
        // Measurement + toMeasurements (PR 1)
        {"meas_default_init",        test_meas_default_init},
        {"meas_size_within_budget",  test_meas_size_within_budget},
        {"meas_flop_per_thread_f16", test_meas_flop_per_thread_f16},
        {"meas_flop_per_thread_bf16",test_meas_flop_per_thread_bf16},
        {"meas_flop_per_thread_f32", test_meas_flop_per_thread_f32},
        {"meas_flop_per_thread_f64", test_meas_flop_per_thread_f64},
        {"meas_flop_per_thread_other", test_meas_flop_per_thread_other},
        {"meas_flop_per_warp_dropped", test_meas_flop_per_warp_dropped},
        {"meas_mma_per_warp_flop",   test_meas_mma_per_warp_flop},
        {"meas_memop_global_load",   test_meas_memop_global_load},
        {"meas_memop_global_store",  test_meas_memop_global_store},
        {"meas_memop_shared_load",   test_meas_memop_shared_load},
        {"meas_memop_atomic_rmw",    test_meas_memop_atomic_rmw},
        {"meas_memop_local",         test_meas_memop_local},
        {"meas_memop_const",         test_meas_memop_const},
        {"meas_memop_param",         test_meas_memop_param},
        {"meas_async_copy_known_bytes",   test_meas_async_copy_known_bytes},
        {"meas_async_copy_unknown_bytes", test_meas_async_copy_unknown_bytes},
        {"meas_ldmatrix_per_warp_shared_load", test_meas_ldmatrix_per_warp_shared_load},
        {"meas_warpsync_empty",      test_meas_warpsync_empty},
        {"meas_barrier_empty",       test_meas_barrier_empty},
        {"meas_ignore_empty",        test_meas_ignore_empty},
        {"meas_unknown_empty",       test_meas_unknown_empty},
        {"meas_integration_fma_f16x2",    test_meas_integration_fma_f16x2},
        {"meas_integration_ld_global_f32",test_meas_integration_ld_global_f32},
        {"meas_integration_cp_async",     test_meas_integration_cp_async},
        // MIR-side toMeasurements overload (PR 2)
        {"mir_meas_flop_f16",        test_mir_meas_flop_f16},
        {"mir_meas_flop_bf16",       test_mir_meas_flop_bf16},
        {"mir_meas_flop_f32",        test_mir_meas_flop_f32},
        {"mir_meas_flop_f64",        test_mir_meas_flop_f64},
        {"mir_meas_flop_other",      test_mir_meas_flop_other},
        {"mir_meas_flop_lanes_already_multiplied",
                                     test_mir_meas_flop_lanes_already_multiplied},
        {"mir_meas_flop_zero_count_empty",
                                     test_mir_meas_flop_zero_count_empty},
        {"mir_meas_none_kind_empty", test_mir_meas_none_kind_empty},
        {"mir_meas_mma_kind_empty_for_now",
                                     test_mir_meas_mma_kind_empty_for_now},
        {"mir_meas_special_math_empty_for_now",
                                     test_mir_meas_special_math_empty_for_now},
        {"mir_meas_integration_classify_fma_underscored",
                                     test_mir_meas_integration_classify_fma_underscored},
        {"mir_meas_integration_classify_fma_bare_width",
                                     test_mir_meas_integration_classify_fma_bare_width},
        {"mir_meas_integration_classify_fmul_f16x2",
                                     test_mir_meas_integration_classify_fmul_f16x2},
        {"mir_meas_integration_classify_non_flop",
                                     test_mir_meas_integration_classify_non_flop},
    };

    int passes = 0;
    for (const Test &t : tests) {
        g_current_test = t.name;
        int before = g_failures;
        std::printf("[ RUN  ] %s\n", t.name);
        t.fn();
        if (g_failures == before) {
            std::printf("[  OK  ] %s\n", t.name);
            ++passes;
        } else {
            std::printf("[ FAIL ] %s\n", t.name);
        }
    }
    std::printf("\nSummary: %d/%zu passed, %d assertion(s) failed.\n",
                passes, tests.size(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
