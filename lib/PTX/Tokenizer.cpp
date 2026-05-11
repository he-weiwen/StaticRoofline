//===- PTX/Tokenizer.cpp - Lexer for inline-PTX strings -----------------===//
//
// Tokenizes a PTX inline-asm body. Recognizes the small grammar surface our
// classifier needs:
//
//   identifier  := [a-zA-Z_][a-zA-Z0-9_]*  (with embedded "::" allowed
//                                           between identifier-chars,
//                                           e.g. shared::cluster)
//   register    := %name           e.g. %r0, %rd5
//   register    := %%name          e.g. %%tid.x  (system-register read in
//                                                 inline asm, where `%`
//                                                 has been escaped as `%%`)
//   operandref  := %N              e.g. %0, %1, ... (LLVM inline-asm operand)
//   number      := decimal | 0x.. | float | signed
//   punctuation := . , ; [ ] { } + - @ !
//
// Whitespace and `// line` / `/* block */` comments are skipped.
//
//===---------------------------------------------------------------------===//

#include "Tokenizer.h"

#include <cctype>

using namespace llvm;

namespace ptxai::ptx {

namespace {

inline bool isIdentStart(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
inline bool isIdentCont(char c) {
    return isIdentStart(c) || (c >= '0' && c <= '9');
}
inline bool isDecDigit(char c) { return c >= '0' && c <= '9'; }
inline bool isHexDigit(char c) {
    return isDecDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
inline bool isWhitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

class Lexer {
public:
    explicit Lexer(StringRef Src) : Source(Src) {}

    SmallVector<Token, 64> lex() {
        SmallVector<Token, 64> result;
        while (true) {
            skipWhitespaceAndComments();
            if (Pos >= Source.size()) break;
            Token tok = nextToken();
            result.push_back(tok);
            if (tok.kind == TokenKind::Error) break;
        }
        result.push_back(Token{TokenKind::EndOfFile, StringRef(), (unsigned)Pos});
        return result;
    }

private:
    StringRef Source;
    size_t Pos = 0;

    void skipWhitespaceAndComments() {
        while (Pos < Source.size()) {
            char c = Source[Pos];
            if (isWhitespace(c)) { ++Pos; continue; }
            if (c == '/' && Pos + 1 < Source.size()) {
                if (Source[Pos + 1] == '/') {
                    Pos += 2;
                    while (Pos < Source.size() && Source[Pos] != '\n') ++Pos;
                    continue;
                }
                if (Source[Pos + 1] == '*') {
                    Pos += 2;
                    while (Pos + 1 < Source.size() &&
                           !(Source[Pos] == '*' && Source[Pos + 1] == '/'))
                        ++Pos;
                    if (Pos + 1 < Source.size()) Pos += 2;
                    continue;
                }
            }
            break;
        }
    }

    Token make(TokenKind k, size_t start) {
        return Token{k, Source.substr(start, Pos - start), (unsigned)start};
    }

    Token nextToken() {
        size_t start = Pos;
        char c = Source[Pos];

        switch (c) {
            case '.': ++Pos; return make(TokenKind::Dot, start);
            case ',': ++Pos; return make(TokenKind::Comma, start);
            case ';': ++Pos; return make(TokenKind::Semicolon, start);
            case '[': ++Pos; return make(TokenKind::LBracket, start);
            case ']': ++Pos; return make(TokenKind::RBracket, start);
            case '{': ++Pos; return make(TokenKind::LBrace, start);
            case '}': ++Pos; return make(TokenKind::RBrace, start);
            case '+': ++Pos; return make(TokenKind::Plus, start);
            case '!': ++Pos; return make(TokenKind::Bang, start);
            case '@': ++Pos; return make(TokenKind::At, start);
            case '-':
                if (Pos + 1 < Source.size() && isDecDigit(Source[Pos + 1]))
                    return lexNumber();
                ++Pos;
                return make(TokenKind::Minus, start);
        }

        if (isDecDigit(c)) return lexNumber();
        if (isIdentStart(c)) return lexIdentifier();
        if (c == '%') return lexPercent();
        if (c == '$') return lexDollar();

        ++Pos;
        return make(TokenKind::Error, start);
    }

    Token lexNumber() {
        size_t start = Pos;
        if (Source[Pos] == '-') ++Pos;
        bool isHex = false;
        if (Pos + 1 < Source.size() && Source[Pos] == '0' &&
            (Source[Pos + 1] == 'x' || Source[Pos + 1] == 'X')) {
            isHex = true;
            Pos += 2;
            while (Pos < Source.size() && isHexDigit(Source[Pos])) ++Pos;
        } else {
            while (Pos < Source.size() && isDecDigit(Source[Pos])) ++Pos;
            if (Pos < Source.size() && Source[Pos] == '.') {
                ++Pos;
                while (Pos < Source.size() && isDecDigit(Source[Pos])) ++Pos;
            }
            if (Pos < Source.size() && (Source[Pos] == 'e' || Source[Pos] == 'E')) {
                ++Pos;
                if (Pos < Source.size() && (Source[Pos] == '+' || Source[Pos] == '-'))
                    ++Pos;
                while (Pos < Source.size() && isDecDigit(Source[Pos])) ++Pos;
            }
        }

        // PTX modifier names sometimes start with digits and contain letters
        // ("5d", "1d", "64x4"). If we land on alphanumeric continuation after
        // the numeric body, promote the whole run to an Identifier token —
        // the parser doesn't actually distinguish between the two when
        // collecting modifiers.
        bool promoteToIdent = false;
        while (Pos < Source.size() && isIdentCont(Source[Pos])) {
            promoteToIdent = true;
            ++Pos;
        }
        if (promoteToIdent && !isHex) {
            return make(TokenKind::Identifier, start);
        }
        return make(TokenKind::Number, start);
    }

    // Identifier with optional "::" segments. PTX uses scope-qualified
    // modifiers like `shared::cluster`, `complete_tx::bytes`. We treat
    // these as a single identifier token so the parser doesn't need to
    // reassemble them. Allow the segment after "::" to start with either
    // a letter OR a digit — PTX cache hints like `L2::128B` need this.
    Token lexIdentifier() {
        size_t start = Pos;
        while (Pos < Source.size()) {
            if (isIdentCont(Source[Pos])) { ++Pos; continue; }
            if (Source[Pos] == ':' && Pos + 1 < Source.size() &&
                Source[Pos + 1] == ':' && Pos + 2 < Source.size() &&
                isIdentCont(Source[Pos + 2])) {
                Pos += 2; // include "::"
                continue;
            }
            break;
        }
        return make(TokenKind::Identifier, start);
    }

    // LLVM's MIR / IR inline-asm bodies use `$N` (not `%N`) for operand
    // placeholders. `$0` `$1` ... refer to the LLVM constraint operands. We
    // emit OperandRef so the parser/classifier can treat them uniformly with
    // the C-source `%N` form.
    Token lexDollar() {
        size_t start = Pos;
        ++Pos; // consume $
        if (Pos < Source.size() && isDecDigit(Source[Pos])) {
            while (Pos < Source.size() && isDecDigit(Source[Pos])) ++Pos;
            return make(TokenKind::OperandRef, start);
        }
        return make(TokenKind::Error, start);
    }

    Token lexPercent() {
        size_t start = Pos;
        ++Pos; // consume %
        if (Pos < Source.size() && Source[Pos] == '%') {
            // %%name — system register read (e.g. %%tid.x, %%ctaid.y).
            // The C escape `%%` represents one literal `%` in PTX text.
            ++Pos;
            while (Pos < Source.size() &&
                   (isIdentCont(Source[Pos]) || Source[Pos] == '.'))
                ++Pos;
            return make(TokenKind::Register, start);
        }
        if (Pos < Source.size() && isDecDigit(Source[Pos])) {
            // %0, %1, ... — LLVM inline-asm operand placeholder.
            while (Pos < Source.size() && isDecDigit(Source[Pos])) ++Pos;
            return make(TokenKind::OperandRef, start);
        }
        if (Pos < Source.size() && isIdentStart(Source[Pos])) {
            // %r0, %rd5, %f32 — local register reference (rare in inline asm,
            // common in standalone PTX text).
            while (Pos < Source.size() && isIdentCont(Source[Pos])) ++Pos;
            return make(TokenKind::Register, start);
        }
        return make(TokenKind::Error, start);
    }
};

} // anonymous namespace

llvm::SmallVector<Token, 64> tokenize(llvm::StringRef Source) {
    return Lexer(Source).lex();
}

} // namespace ptxai::ptx
