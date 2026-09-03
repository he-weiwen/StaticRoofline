//! Machine model: per-SM peak/bandwidth tables and the roofline knee
//! (PLAN.md §6, PR 13).
//!
//! Tables live in `data/machine/*.toml` (sources cited inline there)
//! and are embedded at compile time — the binary needs no data
//! directory at runtime; editing a table is a reviewable diff of the
//! TOML, not of code.
//!
//! The knee is `peak flops / DRAM bandwidth` for a given precision:
//! the arithmetic intensity at which a kernel stops being
//! memory-bound on that part. A verdict is a comparison of a loop's
//! AI(global) against the knee — both numbers appear in the report so
//! the comparison is checkable.

use crate::classify::Pipe;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
pub struct Machine {
    /// The concrete part the numbers describe (e.g. "A100-SXM4-40GB");
    /// an SM version does not pin clocks or memory, so the table names
    /// its representative.
    pub name: String,
    pub arch: String,
    pub dram_bw_gbps: f64,
    /// Cuda-core (non-tensor) peaks per precision key ("f16", ...).
    pub peak_tflops: BTreeMap<String, f64>,
    /// Tensor-core peaks per precision key; absent where the part has
    /// none (or the table predates them).
    #[serde(default)]
    pub tensor_peak_tflops: BTreeMap<String, f64>,
}

/// The embedded tables: (arch, TOML source).
const TABLES: &[(&str, &str)] = &[
    ("sm_70", include_str!("../data/machine/sm_70.toml")),
    ("sm_75", include_str!("../data/machine/sm_75.toml")),
    ("sm_80", include_str!("../data/machine/sm_80.toml")),
    ("sm_86", include_str!("../data/machine/sm_86.toml")),
    ("sm_89", include_str!("../data/machine/sm_89.toml")),
    ("sm_90", include_str!("../data/machine/sm_90.toml")),
];

pub fn known_archs() -> Vec<&'static str> {
    TABLES.iter().map(|(a, _)| *a).collect()
}

pub fn arch_table(arch: &str) -> Option<Machine> {
    let (_, toml_src) = TABLES.iter().find(|(a, _)| *a == arch)?;
    let m: Machine =
        toml::from_str(toml_src).expect("embedded machine table parses (compile-time data)");
    debug_assert_eq!(m.arch, arch, "table file disagrees with its registry key");
    Some(m)
}

impl Machine {
    /// Peak TFLOPS for one pipe and precision; `None` when the table
    /// has no such peak — the SFU has none anywhere.
    pub fn peak_tflops(&self, pipe: Pipe, precision: &str) -> Option<f64> {
        match pipe {
            Pipe::CudaCore => self.peak_tflops.get(precision).copied(),
            Pipe::Tensor => self.tensor_peak_tflops.get(precision).copied(),
            Pipe::Sfu => None,
        }
    }

    /// Roofline knee in flop/B for one pipe and precision: TFLOPS·1000
    /// / GB/s (10^12 flop/s over 10^9 B/s).
    pub fn knee_flop_per_byte(&self, pipe: Pipe, precision: &str) -> Option<f64> {
        Some(self.peak_tflops(pipe, precision)? * 1000.0 / self.dram_bw_gbps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_table_loads_and_matches_its_key() {
        for arch in known_archs() {
            let m = arch_table(arch).expect("table loads");
            assert_eq!(m.arch, arch);
            assert!(m.dram_bw_gbps > 0.0);
            assert!(
                m.peak_tflops.contains_key("f32"),
                "{arch}: f32 peak required"
            );
        }
        assert!(arch_table("sm_42").is_none());
    }

    #[test]
    fn knees_match_hand_computation_from_the_cited_specs() {
        // A100: 19.5 TFLOPS / 1555 GB/s = 12.54 flop/B.
        let a100 = arch_table("sm_80").expect("sm_80");
        let knee = a100.knee_flop_per_byte(Pipe::CudaCore, "f32").expect("f32");
        assert!((knee - 12.54).abs() < 0.01, "got {knee}");
        // RTX 3090: 35.58 TFLOPS / 936.2 GB/s = 38.0 flop/B.
        let ga102 = arch_table("sm_86").expect("sm_86");
        let knee = ga102
            .knee_flop_per_byte(Pipe::CudaCore, "f32")
            .expect("f32");
        assert!((knee - 38.0).abs() < 0.05, "got {knee}");
        // The S1 design point sits between the two: AI = 32 is
        // compute-bound on sm_80 and memory-bound on sm_86.
        assert!(32.0 > a100.knee_flop_per_byte(Pipe::CudaCore, "f32").expect("f32"));
        assert!(
            32.0 < ga102
                .knee_flop_per_byte(Pipe::CudaCore, "f32")
                .expect("f32")
        );
        // Missing precision is None, not zero (V100 has no bf16 row).
        let v100 = arch_table("sm_70").expect("sm_70");
        assert!(v100.knee_flop_per_byte(Pipe::CudaCore, "bf16").is_none());
        // A100 tensor f16: 312 TFLOPS / 1555 GB/s = 200.6 flop/B — the
        // k14 design point (AI 64) is memory-bound there but compute-
        // bound on cuda cores.
        let knee = a100
            .knee_flop_per_byte(Pipe::Tensor, "f16")
            .expect("tensor f16");
        assert!((knee - 200.6).abs() < 0.05, "got {knee}");
        // No tensor f64 on GeForce parts; no SFU peak anywhere.
        assert!(ga102.knee_flop_per_byte(Pipe::Tensor, "f64").is_none());
        assert!(a100.knee_flop_per_byte(Pipe::Sfu, "f32").is_none());
    }
}
