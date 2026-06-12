//! The flat module IR (PLAN.md §2 ground rule, after Sampson's
//! "Flattening ASTs").
//!
//! No pointer structures anywhere: instructions live in per-kernel
//! `Vec<Stmt>`, operands in a module-level arena referenced by spans,
//! nested operands (memory bases, vector lists) via `OperandId` — never
//! `Box`. The IR is built once by the parser and read-only ever after.
//! Only the report tree (PR 12) is ergonomic/owned; these indices never
//! leak into user-facing output — the dumper resolves them.

use super::intern::{Interner, Symbol};

/// Index into [`Module::operands`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperandId(pub u32);

/// Span into one of the module-level pools: `(start, len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span(pub u32, pub u32);

impl Span {
    pub fn range(self) -> std::ops::Range<usize> {
        self.0 as usize..(self.0 + self.1) as usize
    }
}

/// A source location from a `.loc` directive. For the extended form
/// (`.loc f l c, function_name $sym, inlined_at f l c`) this is the
/// `inlined_at` location: attribution wants the user's line, not the
/// header the code was inlined from. `line` 0 is PTX for "no source
/// line" and is skipped at attribution time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLoc {
    pub file: u32,
    pub line: u32,
    pub col: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Predicate {
    pub reg: Symbol,
    pub negated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    /// `%rd5`, `%tid.x` — register or special-register reference.
    Register(Symbol),
    /// Numeric immediate, original text preserved (`-7`, `0f3F800000`).
    Immediate(Symbol),
    /// Bare identifier: branch-target label, global symbol, `WARP_SZ`.
    SymbolRef(Symbol),
    /// `[base]` / `[base+offset]`.
    Memory { base: OperandId, offset: i64 },
    /// `{a, b, ...}` — vector / fragment list; children are a span into
    /// [`Module::operand_lists`].
    VectorList { children: Span },
}

/// One executable instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instr {
    pub mnemonic: Symbol,
    /// Span into [`Module::modifier_pool`].
    pub modifiers: Span,
    /// Span into [`Module::operand_lists`].
    pub operands: Span,
    pub predicate: Option<Predicate>,
    pub loc: Option<SourceLoc>,
    /// Byte offset of the mnemonic in the source — provenance for
    /// diagnostics and the report verifier.
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Instr(Instr),
    Label(Symbol),
    /// `LABEL: .branchtargets L0, L1, ...;` — the jump table for a
    /// preceding `brx.idx`.
    BranchTargets(Vec<Symbol>),
    /// A statement the parser could not understand. Counted by the
    /// corpus-wide parse check; never poisons the kernel.
    Unparsed {
        offset: u32,
    },
}

/// `.param .u32 name` in a kernel signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Param {
    pub ty: Symbol,
    pub name: Symbol,
}

/// `.reg .f32 %f<789>;` — a counted register family (count is the
/// declared `<N>`), or a singular declaration (`.reg .pred p;`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegDecl {
    pub class: Symbol,
    pub prefix: Symbol,
    pub count: Option<u32>,
}

/// In-body `.shared .align A .b8 name[N];` (or `.extern` dynamic form
/// with empty brackets — `size` is `None` there).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedDecl {
    pub name: Symbol,
    pub align: Option<u32>,
    pub ty: Symbol,
    pub size: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: Symbol,
    pub params: Vec<Param>,
    pub reg_decls: Vec<RegDecl>,
    pub shared_decls: Vec<SharedDecl>,
    /// `.maxntid x, y, z` / `.reqntid x, y, z` between signature and body.
    pub maxntid: Option<[u32; 3]>,
    pub reqntid: Option<[u32; 3]>,
    pub stmts: Vec<Stmt>,
}

/// `.file N "path"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileDirective {
    pub index: u32,
    pub path: Symbol,
}

#[derive(Debug)]
pub struct Module {
    pub interner: Interner,
    /// `.version major.minor`.
    pub version: (u32, u32),
    /// `.target` first argument (`sm_80`).
    pub target: Symbol,
    pub address_size: u32,
    pub files: Vec<FileDirective>,
    pub kernels: Vec<Kernel>,
    /// Operand arena; `Operand::Memory::base` points here.
    pub operands: Vec<Operand>,
    /// Pool of operand-id lists; `Instr::operands` and
    /// `Operand::VectorList::children` are spans into it.
    pub operand_lists: Vec<OperandId>,
    /// Pool of modifier symbols; `Instr::modifiers` is a span into it.
    pub modifier_pool: Vec<Symbol>,
}

impl Module {
    pub fn operand(&self, id: OperandId) -> &Operand {
        &self.operands[id.0 as usize]
    }

    pub fn operand_ids(&self, span: Span) -> &[OperandId] {
        &self.operand_lists[span.range()]
    }

    pub fn modifiers(&self, instr: &Instr) -> &[Symbol] {
        &self.modifier_pool[instr.modifiers.range()]
    }

    /// Resolve a `.file` index to its path, if declared.
    pub fn file_path(&self, index: u32) -> Option<&str> {
        self.files
            .iter()
            .find(|f| f.index == index)
            .map(|f| self.interner.resolve(f.path))
    }
}
