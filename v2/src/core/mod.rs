//! Core types: the flat module IR and its interner (PLAN.md §5).
//! SymExpr and Measurement join in their PRs (10, 09).

pub mod intern;
pub mod ir;

pub use intern::{Interner, Symbol};
pub use ir::{
    FileDirective, Instr, Kernel, Module, Operand, OperandId, Param, Predicate, RegDecl,
    SharedDecl, SourceLoc, Span, Stmt,
};
