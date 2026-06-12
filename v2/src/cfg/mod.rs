//! Control-flow structure: the graph, dominators, the loop forest,
//! and human-readable loop/kernel naming.

pub mod dominators;
pub mod graph;
pub mod loops;
pub mod naming;

pub use dominators::{Dominators, dominators};
pub use graph::{Block, BlockId, Cfg, build_cfg};
pub use loops::{Loop, LoopForest, LoopId, loop_forest};
pub use naming::{LoopName, demangle, loop_names};
