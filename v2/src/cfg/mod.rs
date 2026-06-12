//! Control-flow structure: the graph, dominators, and the loop forest.
//! Loop naming (display names, demangling) is next (PR 07).

pub mod dominators;
pub mod graph;
pub mod loops;

pub use dominators::{Dominators, dominators};
pub use graph::{Block, BlockId, Cfg, build_cfg};
pub use loops::{Loop, LoopForest, LoopId, loop_forest};
