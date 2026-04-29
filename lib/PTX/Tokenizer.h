//===- PTX/Tokenizer.h - Lexer for inline-PTX strings ---------*- C++ -*-===//
//
// Splits an inline-asm body into PTX tokens. Handles the small grammar
// surface we care about: identifiers (with embedded "::" for scope-
// qualified modifiers like `shared::cluster`), dots, brackets, commas,
// semicolons, register references (`%rd5`), inline-asm operand refs
// (`%0`, `%1`), numbers, predicate prefix.
//
// Pure function over a StringRef. No LLVM dependency beyond StringRef
// and SmallVector — Tokenizer/Parser/Classifier are independently
// testable; only OperandResolver (deferred) pulls in MIR types.
//
//===---------------------------------------------------------------------===//

#ifndef PTXAI_PTX_TOKENIZER_H
#define PTXAI_PTX_TOKENIZER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace ptxai::ptx {

enum class TokenKind : unsigned char {
    Identifier,    // mnemonic, modifier, possibly containing "::"
    Dot,
    Comma,
    Semicolon,
    LBracket,      // [   (memory operand)
    RBracket,      // ]
    LBrace,        // {   (vector / fragment list)
    RBrace,        // }
    Plus,          // +   (offset in memory operands)
    Minus,
    At,            // @   (predicate prefix)
    Bang,          // !   (negated predicate)
    Register,      // %r0, %rd5, %f32, %p3 — alphabetic register class + index
    OperandRef,    // %0, %1, ... — index into the LLVM inline-asm operand list
    Number,        // immediate (decimal, hex, float)
    EndOfFile,
    Error,         // tokenizer hit something unexpected; surfaces as diagnostic
};

struct Token {
    TokenKind kind = TokenKind::EndOfFile;
    llvm::StringRef text;     // verbatim source slice
    unsigned offset = 0;      // byte offset into the original asm body
};

// Tokenize an entire inline-asm body. Stops at end of input. On a lex
// error, emits a single Token{Error, ...} and stops.
llvm::SmallVector<Token, 64> tokenize(llvm::StringRef Source);

} // namespace ptxai::ptx

#endif // PTXAI_PTX_TOKENIZER_H
