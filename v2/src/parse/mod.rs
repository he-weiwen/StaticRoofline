//! PTX text frontend: lexer -> parser -> flat module IR, plus the
//! canonical dumper (the debug view snapshots and --dump-ast use).

pub mod ast;
pub mod lexer;
pub mod parser;
