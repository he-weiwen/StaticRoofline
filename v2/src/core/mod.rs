//! Core types: the flat module IR, its interner, and the Measurement
//! record. SymExpr joins in PR 10.

pub mod intern;
pub mod ir;
pub mod measurement;

pub use intern::{Interner, Symbol};
pub use ir::{
    FileDirective, Instr, Kernel, Module, Operand, OperandId, Param, Predicate, RegDecl,
    SharedDecl, SourceLoc, Span, Stmt,
};
