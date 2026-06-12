//! Control-flow structure: graph now (PR 05); dominators and the loop
//! forest are next (PR 06).

pub mod graph;

pub use graph::{Block, BlockId, Cfg, build_cfg};
