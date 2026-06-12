//! Core types: the flat module IR, its interner, the Measurement
//! record, and the SymExpr count datatype.

pub mod intern;
pub mod ir;
pub mod measurement;
pub mod symexpr;

pub use intern::{Interner, Symbol};
pub use ir::{
    FileDirective, Instr, Kernel, Module, Operand, OperandId, Param, Predicate, RegDecl,
    SharedDecl, SourceLoc, Span, Stmt,
};
pub use symexpr::SymExpr;
