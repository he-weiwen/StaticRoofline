//===- PTX/Parser.h - AST + parser for inline-PTX statements --*- C++ -*-===//
//
// Produces a structural AST from a tokenized PTX statement sequence. Does
// not interpret modifiers or operand semantics — that's the classifier's
// job. The parser only knows the grammar shape:
//
//   stmt    := [predicate] mnemonic { '.' modifier } [operand-list] ';'
//   operand := register | operand_ref | memory | immediate | brace_list
//   memory  := '[' (register | operand_ref) [ '+' immediate ] ']'
//
// Operand types are an algebraic sum (std::variant) — adding a new arm
// triggers a compile error at every std::visit site that doesn't handle
// it, which is the exhaustiveness property we want.
//
//===---------------------------------------------------------------------===//

#ifndef PTXAI_PTX_PARSER_H
#define PTXAI_PTX_PARSER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace ptxai::ptx {

// Forward declarations for the recursive Operand variant. Memory operands
// can contain a Register or OperandRef as their base; BraceList contains
// nested operands for MMA fragment lists.
struct Register;
struct OperandRef;
struct Immediate;
struct Memory;
struct BraceList;

using Operand = std::variant<Register, OperandRef, Immediate, Memory, BraceList>;

struct Register {
    llvm::StringRef name;       // "%rd5", "%r0", "%f32", "%p3"
};

struct OperandRef {
    unsigned index = 0;         // 0 for %0, 1 for %1, …; resolves via OperandResolver
};

struct Immediate {
    llvm::StringRef text;       // verbatim — caller decides how to interpret
};

struct Memory {
    std::shared_ptr<Operand> base;     // typically a Register or OperandRef
    int64_t offset = 0;
};

struct BraceList {
    std::vector<Operand> children;
};

struct Stmt {
    llvm::StringRef predicate;          // "" if no @p prefix
    bool predicateNegated = false;
    llvm::StringRef mnemonic;           // "fma", "mma", "ld", "cp", ...
    llvm::SmallVector<llvm::StringRef, 6> modifiers;   // dot-separated; "::" stays inside the modifier text
    llvm::SmallVector<Operand, 8> operands;
    bool parseError = false;            // true if this statement failed to parse cleanly
};

// Parse an inline-asm body into a sequence of statements. On a parse
// error within one statement, that statement is marked parseError=true
// and parsing resumes at the next ';'.
std::vector<Stmt> parse(llvm::StringRef Source);

} // namespace ptxai::ptx

#endif // PTXAI_PTX_PARSER_H
