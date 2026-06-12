//! Measurement collection, queries, and the analyze report (text +
//! JSON views over one result tree).

pub mod build;
pub mod collect;
pub mod stats;
pub mod text;
pub mod tree;

pub use build::{AnalyzeError, BindingSpec, analyze, parse_bind};
pub use collect::{BlockMeasurements, ClassCounts, CountQualifier, collect};
pub use stats::{Stats, Tally};
pub use tree::Report;
