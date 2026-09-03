//! The result tree (PLAN.md §6, PR 12): ergonomic, owned, resolved —
//! the only structure that leaves the library. JSON output IS the
//! `Serialize` derivation of these structs; the text report renders
//! the same values, so the two views cannot drift.
//!
//! Schema conventions, pinned by the committed scenario expectations:
//! - every count is `{"expr": string, "at_most": bool}` — symbolic
//!   expressions print via SymExpr's deterministic form, `at_most`
//!   marks upper bounds (rendered `≤` in text);
//! - trips are `{"expr": ...}` or `{"unknown": reason}` — an unknown
//!   is a result, not an error;
//! - the three flop tables (one per pipe) always carry every
//!   precision plus "total", so "0 f16 flops" is assertable (S8);
//! - byte tables always carry global/shared/local; other spaces appear
//!   when touched;
//! - `coverage` is `{metric: {num, den}}` count pairs — the runner
//!   aggregates them corpus-wide (percentages cannot be aggregated).

use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct Report {
    pub input: String,
    /// `--bind` values, echoed (bet 4: inputs are visible).
    pub bindings: Vec<Binding>,
    pub kernels: Vec<KernelReport>,
    pub coverage: BTreeMap<String, Fraction>,
}

#[derive(Debug, Serialize)]
pub struct Binding {
    pub param: usize,
    pub name: String,
    pub value: i64,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq)]
pub struct Fraction {
    pub num: u64,
    pub den: u64,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct Count {
    pub expr: String,
    pub at_most: bool,
}

#[derive(Debug, Serialize)]
pub struct KernelReport {
    pub name: String,
    pub demangled: String,
    pub params: Vec<ParamInfo>,
    /// Shared memory reserved per CTA. `static_bytes` is the sum of the
    /// kernel's `.shared` array declarations — a `[static]` demand
    /// figure that matches ptxas's `bytes smem` and Nsight Compute's
    /// `launch__shared_mem_per_block_static`; driver-reserved shared
    /// memory (NCU's `_driver`) is not included. `dynamic` is set when
    /// the kernel also declares an `.extern .shared` array whose size is
    /// fixed at launch and so is not statically knowable.
    pub shared_memory: SharedMemory,
    /// Instruction-class tallies; the verifier's accounting identity
    /// (`flop + non_flop_arith + memory + sync + communication +
    /// control + ignore + unknown == total`) runs on these.
    pub instruction_classes: InstructionClasses,
    /// The loop with the largest static weight (instructions × trips).
    pub heaviest_loop: Option<String>,
    /// Roofline knees, one per requested (or defaulted) architecture,
    /// for the dominant flop bucket of the deepest heaviest-chain loop
    /// whose per-iteration AI(global) is defined. A reference number
    /// next to that loop's requested AI — not a verdict: requested
    /// bytes are neither DRAM traffic (overfetch) nor a lower bound on
    /// it (cache reuse).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub knees: Vec<Knee>,
    /// Launch configuration, when known (flag or PTX directive).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub launch: Option<LaunchInfo>,
    /// Kernel totals scaled to one CTA (needs `launch`; upper bounds
    /// when the block size is only a maximum).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub totals_per_cta: Option<Aggregates>,
    /// Loops ranked by symbolic weight, heaviest first.
    pub ranking: Vec<RankEntry>,
    /// Top-level loop nodes, in program order.
    pub loops: Vec<LoopNode>,
    pub totals: Aggregates,
    /// Every named hole in the analysis: unclassified instructions,
    /// unquantifiable bytes, unresolved trips, irreducible regions,
    /// call sites. Never silently empty when something was dropped.
    pub unknowns: Vec<UnknownEntry>,
}

#[derive(Debug, Serialize)]
pub struct ParamInfo {
    pub index: usize,
    #[serde(rename = "type")]
    pub ty: String,
    pub name: String,
}

#[derive(Debug, Serialize, Default, Clone, Copy)]
pub struct InstructionClasses {
    pub total: u64,
    pub flop: u64,
    pub non_flop_arith: u64,
    pub memory: u64,
    pub sync: u64,
    pub communication: u64,
    pub control: u64,
    pub ignore: u64,
    pub unknown: u64,
    /// Statements the parser could not read. Not part of `total` (they
    /// are not instructions), so outside the accounting identity; each
    /// is also an entry in `unknowns`.
    pub unparsed: u64,
}

#[derive(Debug, Serialize)]
pub struct Knee {
    pub arch: String,
    /// The concrete part the machine table describes.
    pub machine: String,
    /// Where the table came from: "flag" or "target-directive".
    pub source: String,
    /// The loop whose AI the knee is printed next to.
    #[serde(rename = "loop")]
    pub loop_name: String,
    /// Dominant flop bucket of that loop — pipe ("cuda-core",
    /// "tensor", "sfu") and precision; the knee uses its peak.
    pub pipe: String,
    pub precision: String,
    pub ai_global: Intensity,
    /// `knee = peak_tflops * 1000 / dram_bw_gbps`, both cited in the
    /// machine table.
    pub peak_tflops: f64,
    pub dram_bw_gbps: f64,
    pub knee: f64,
}

#[derive(Debug, Serialize)]
pub struct SharedMemory {
    /// Statically-declared shared memory per CTA, in bytes.
    pub static_bytes: u64,
    /// An `.extern .shared` array is present; its size is set at launch.
    pub dynamic: bool,
}

#[derive(Debug, Serialize)]
pub struct LaunchInfo {
    pub block: [u32; 3],
    pub threads: u64,
    /// "flag", ".reqntid", or ".maxntid".
    pub source: String,
    /// `.maxntid` is a maximum, not the launch: `false` there, and
    /// every per-CTA total is then an upper bound.
    pub exact: bool,
}

#[derive(Debug, Serialize)]
pub struct RankEntry {
    #[serde(rename = "loop")]
    pub loop_name: String,
    /// The weight expression: executed instructions per invocation.
    pub weight: String,
}

#[derive(Debug, Serialize)]
pub struct LoopNode {
    pub name: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    pub depth: u32,
    pub trips: Trips,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unroll: Option<Unroll>,
    pub per_iteration: Aggregates,
    pub loops: Vec<LoopNode>,
}

#[derive(Debug, Serialize)]
pub struct Trips {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unknown: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Unroll {
    pub factor: i64,
    pub remainder: String,
}

#[derive(Debug, Serialize)]
pub struct Aggregates {
    /// CUDA-core flops. Keys: "total" and every precision key
    /// ("f16", "bf16", "tf32", "f32", "f64") — always present.
    pub flops: BTreeMap<String, Count>,
    /// Tensor-core flops (`wmma.mma`, `mma`), same keys.
    pub tensor_flops: BTreeMap<String, Count>,
    /// Special-function-unit flops (`ex2`, `rsqrt`, ...), same keys.
    pub sfu_flops: BTreeMap<String, Count>,
    /// Keys: space names; global/shared/local always present.
    pub bytes: BTreeMap<String, DirectionCounts>,
    pub conversions: Count,
    /// Flops of all three pipes per global byte, when both are
    /// constants, bytes > 0, and at most one side is an upper bound
    /// (a bound over a bound bounds nothing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_global: Option<Intensity>,
    /// Straight-line repeated source lines (fully-unrolled loops):
    /// "file:line" → workload-op copies. Empty = omitted.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub unrolled_source_lines: BTreeMap<String, u64>,
}

/// A flop/byte ratio with the direction it is known in: `exact`,
/// `at_least` (exact flops over bytes that are an upper bound) or
/// `at_most` (flops that are an upper bound over exact bytes).
#[derive(Debug, Serialize, Clone, Copy, PartialEq)]
pub struct Intensity {
    pub value: f64,
    pub bound: Bound,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    Exact,
    AtLeast,
    AtMost,
}

#[derive(Debug, Serialize, Clone, PartialEq)]
pub struct DirectionCounts {
    pub load: Count,
    pub store: Count,
}

#[derive(Debug, Serialize)]
pub struct UnknownEntry {
    pub what: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    pub reason: String,
}
