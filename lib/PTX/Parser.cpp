//===- PTX/Parser.cpp - Recursive-descent parser for PTX statements ---===//
//
// Grammar:
//
//   stmt    := [predicate] mnemonic { '.' modifier } [operand_list] ';'
//   mnemonic   := identifier
//   modifier   := identifier
//   predicate  := '@' [ '!' ] register
//   operand    := register | operand_ref | memory | immediate | brace_list
//   memory     := '[' (register | operand_ref) [ '+' immediate ] ']'
//   brace_list := '{' operand { ',' operand } '}'
//
// On a structural error within a statement, that statement is marked
// `parseError = true` and parsing resumes at the next ';'. This keeps
// one malformed instruction from poisoning the whole inline-asm body.
//
//===---------------------------------------------------------------------===//

#include "Parser.h"
#include "Tokenizer.h"

#include "llvm/ADT/ArrayRef.h"

using namespace llvm;

namespace ptxai::ptx {

namespace {

class ParserImpl {
public:
    ParserImpl(ArrayRef<Token> Toks) : Tokens(Toks) {}

    std::vector<Stmt> parseAll() {
        std::vector<Stmt> result;
        while (!atEnd()) {
            // Skip stray semicolons (empty statements) and PTX scope braces.
            // CUDA-header inline asm wraps each body in `{ ... }` for scoping
            // (e.g. `{add.f16x2 %0,%1,%2;\n}`); the braces are not statements
            // and should not produce parseError diagnostics.
            TokenKind k = peek().kind;
            if (k == TokenKind::Semicolon ||
                k == TokenKind::LBrace    ||
                k == TokenKind::RBrace) {
                ++Pos;
                continue;
            }
            // Bail on a stray Error token — let the lexer's diagnostic
            // surface, don't loop forever trying to consume it.
            if (k == TokenKind::Error) {
                ++Pos;
                continue;
            }
            Stmt s = parseStmt();
            result.push_back(std::move(s));
        }
        return result;
    }

private:
    ArrayRef<Token> Tokens;
    size_t Pos = 0;

    const Token &peek(size_t lookahead = 0) const {
        size_t i = Pos + lookahead;
        if (i >= Tokens.size()) return Tokens.back();
        return Tokens[i];
    }
    bool atEnd() const {
        return Pos >= Tokens.size() || peek().kind == TokenKind::EndOfFile;
    }
    bool consume(TokenKind k) {
        if (peek().kind == k) { ++Pos; return true; }
        return false;
    }

    void resyncToSemicolon() {
        while (!atEnd() && peek().kind != TokenKind::Semicolon) ++Pos;
        if (peek().kind == TokenKind::Semicolon) ++Pos;
    }

    Stmt parseStmt() {
        Stmt s;

        // PTX directive (e.g. `.reg .pred p;`, `.pragma "..." ;`) — these
        // declare scratch registers / annotations inside an inline-asm
        // scope. They don't produce FLOPs or bytes; skip them so we don't
        // pollute the parseError/Unknown counts.
        if (peek().kind == TokenKind::Dot &&
            peek(1).kind == TokenKind::Identifier) {
            // Treat the whole rest of the statement as opaque; consume
            // through the next semicolon. Mark mnemonic as the directive
            // name so the classifier surfaces it as Ignore.
            ++Pos;                                // consume '.'
            s.mnemonic = peek().text;
            ++Pos;
            resyncToSemicolon();
            // Force classification as Ignore by giving it a recognised
            // mnemonic shape. The classifier handles unknown identifiers
            // as Unknown, so we set parseError=false here and rely on the
            // dispatcher route below.
            s.parseError = false;
            // We can't return Ignore directly from the parser, so we leave
            // mnemonic = the directive name; the classifier knows to
            // treat directives starting with these names as Ignore.
            return s;
        }

        // Optional predicate: @p, @!p. The predicate name can be either:
        //   - a register-style name (`%p`, `%%p`) — standalone PTX text
        //   - a plain identifier (`p`) — inline asm where .reg declared
        //     the predicate symbol earlier in the same body
        if (peek().kind == TokenKind::At) {
            ++Pos;
            if (peek().kind == TokenKind::Bang) {
                s.predicateNegated = true;
                ++Pos;
            }
            if (peek().kind == TokenKind::Register ||
                peek().kind == TokenKind::Identifier) {
                s.predicate = peek().text;
                ++Pos;
            } else {
                s.parseError = true;
                resyncToSemicolon();
                return s;
            }
        }

        // Mnemonic
        if (peek().kind != TokenKind::Identifier) {
            s.parseError = true;
            resyncToSemicolon();
            return s;
        }
        s.mnemonic = peek().text;
        ++Pos;

        // Modifiers: '.' identifier (or '.' number — some PTX uses numeric
        // modifiers like `.v2`, `.u32` where v2 / u32 are identifiers, but
        // also `cp.async.bulk.tensor.5d` — `5d` is identifier in our lexer
        // since it starts with a digit only when... actually `5d` starts with
        // a digit so it's a number. Handle both.)
        while (peek().kind == TokenKind::Dot) {
            ++Pos;
            if (peek().kind == TokenKind::Identifier ||
                peek().kind == TokenKind::Number) {
                s.modifiers.push_back(peek().text);
                ++Pos;
            } else {
                s.parseError = true;
                resyncToSemicolon();
                return s;
            }
        }

        // Operand list (until ';')
        if (peek().kind != TokenKind::Semicolon && !atEnd()) {
            // First operand
            if (auto op = parseOperand()) {
                s.operands.push_back(std::move(*op));
            } else {
                s.parseError = true;
                resyncToSemicolon();
                return s;
            }
            // Subsequent: ',' operand
            while (peek().kind == TokenKind::Comma) {
                ++Pos;
                if (auto op = parseOperand()) {
                    s.operands.push_back(std::move(*op));
                } else {
                    s.parseError = true;
                    resyncToSemicolon();
                    return s;
                }
            }
        }

        if (!consume(TokenKind::Semicolon)) {
            // Missing semicolon — accept the statement but note the issue.
            s.parseError = true;
        }
        return s;
    }

    std::optional<Operand> parseOperand() {
        const Token &t = peek();
        switch (t.kind) {
            case TokenKind::Register: {
                ++Pos;
                return Operand{Register{t.text}};
            }
            case TokenKind::OperandRef: {
                ++Pos;
                unsigned idx = 0;
                // text is "%N"; skip the '%' and parse digits
                if (t.text.size() > 1)
                    t.text.drop_front(1).getAsInteger(10, idx);
                return Operand{OperandRef{idx}};
            }
            case TokenKind::Number:
            case TokenKind::Minus: {
                ++Pos;
                StringRef text = t.text;
                if (t.kind == TokenKind::Minus) {
                    // Combine "-" + following number into one immediate text.
                    if (peek().kind == TokenKind::Number) {
                        size_t start = (size_t)t.offset;
                        const Token &n = peek();
                        ++Pos;
                        text = StringRef(t.text.data(),
                                         (n.text.data() + n.text.size()) - t.text.data());
                        (void)start;
                    }
                }
                return Operand{Immediate{text}};
            }
            case TokenKind::LBracket: {
                ++Pos; // consume [
                Operand baseOp = Operand{Register{StringRef()}};
                if (peek().kind == TokenKind::Register) {
                    baseOp = Operand{Register{peek().text}};
                    ++Pos;
                } else if (peek().kind == TokenKind::OperandRef) {
                    unsigned idx = 0;
                    StringRef txt = peek().text;
                    if (txt.size() > 1)
                        txt.drop_front(1).getAsInteger(10, idx);
                    baseOp = Operand{OperandRef{idx}};
                    ++Pos;
                } else if (peek().kind == TokenKind::Identifier) {
                    // [global_var] — symbol reference. Treat as a Register
                    // with the symbol name; classifier doesn't currently
                    // distinguish.
                    baseOp = Operand{Register{peek().text}};
                    ++Pos;
                }
                int64_t offset = 0;
                if (peek().kind == TokenKind::Plus) {
                    ++Pos;
                    if (peek().kind == TokenKind::Number) {
                        peek().text.getAsInteger(0, offset);
                        ++Pos;
                    }
                }
                // Tolerant skip: TMA tensor instructions use the form
                // `[base, {dim_list}]`. We don't model the dim list at the
                // memory-operand level — skip tokens (including nested
                // braces/brackets) until the matching `]`. This lets the
                // classifier still recognize `cp.async.bulk.tensor.*`.
                int depth = 0;
                while (!atEnd()) {
                    TokenKind k = peek().kind;
                    if (k == TokenKind::RBracket && depth == 0) break;
                    if (k == TokenKind::LBracket || k == TokenKind::LBrace) ++depth;
                    if (k == TokenKind::RBracket || k == TokenKind::RBrace) --depth;
                    ++Pos;
                }
                if (!consume(TokenKind::RBracket)) return std::nullopt;
                Memory mem;
                mem.base = std::make_shared<Operand>(std::move(baseOp));
                mem.offset = offset;
                return Operand{std::move(mem)};
            }
            case TokenKind::LBrace: {
                ++Pos; // consume {
                BraceList list;
                if (peek().kind != TokenKind::RBrace) {
                    if (auto op = parseOperand())
                        list.children.push_back(std::move(*op));
                    else
                        return std::nullopt;
                    while (peek().kind == TokenKind::Comma) {
                        ++Pos;
                        if (auto op = parseOperand())
                            list.children.push_back(std::move(*op));
                        else
                            return std::nullopt;
                    }
                }
                if (!consume(TokenKind::RBrace)) return std::nullopt;
                return Operand{std::move(list)};
            }
            case TokenKind::Identifier: {
                // Bare identifier as an operand — usually a global symbol
                // reference or a state-space-prefixed operand. Treat as
                // Immediate with the identifier text for now; classifier
                // doesn't need finer detail for the families we handle.
                ++Pos;
                return Operand{Immediate{t.text}};
            }
            default:
                return std::nullopt;
        }
    }
};

} // anonymous namespace

std::vector<Stmt> parse(llvm::StringRef Source) {
    auto tokens = tokenize(Source);
    return ParserImpl(tokens).parseAll();
}

} // namespace ptxai::ptx
