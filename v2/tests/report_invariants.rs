//! IR-level verifier identities (PLAN.md §3) that need data the JSON
//! deliberately omits — the runner's JSON checks cover the rest.
//!
//! 1. Every Measurement's provenance index resolves to a real
//!    instruction.
//! 2. Per-block class tallies sum to the kernel's instruction count.
//! 3. Two-path consistency: with every parameter bound, the report's
//!    kernel flop total equals an independently-computed sum
//!    (per-block flat tallies × numerically-evaluated trip chains) —
//!    the check that catches two code paths disagreeing.

use ptxroof::cfg::{build_cfg, loop_forest, loop_names};
use ptxroof::classify::Precision;
use ptxroof::core::Stmt;
use ptxroof::parse::parser::parse;
use ptxroof::report::{AnalyzeOptions, BindingSpec, Stats, analyze, collect};
use ptxroof::trips::trip_counts;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

const FIXTURES: &[&str] = &[
    "k1/k1.sm_80.ptx",
    "k2/k2.sm_80.ptx",
    "k5/k5.sm_80.ptx",
    "micro/single_loop.ptx",
    "micro/branchy.ptx",
    "micro/irreducible.ptx",
    "micro/no_loc.ptx",
    "micro/data_dep.ptx",
];

fn read(fixture: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture);
    fs::read_to_string(&path).expect("fixture readable")
}

#[test]
fn every_measurement_provenance_resolves_to_an_instruction() {
    for fixture in FIXTURES {
        let src = read(fixture);
        let m = parse(&src).expect("fixture parses");
        for k in &m.kernels {
            let cfg = build_cfg(&m, k);
            let f = loop_forest(&cfg);
            for bm in collect(&m, k, &cfg, &f) {
                for meas in &bm.measurements {
                    assert!(
                        matches!(k.stmts.get(meas.provenance), Some(Stmt::Instr(_))),
                        "{fixture}: provenance {} is not an instruction",
                        meas.provenance
                    );
                }
            }
        }
    }
}

#[test]
fn block_class_tallies_sum_to_kernel_instruction_count() {
    for fixture in FIXTURES {
        let src = read(fixture);
        let m = parse(&src).expect("fixture parses");
        for k in &m.kernels {
            let cfg = build_cfg(&m, k);
            let f = loop_forest(&cfg);
            let from_blocks: u32 = collect(&m, k, &cfg, &f)
                .iter()
                .map(|b| b.class_counts.total)
                .sum();
            let from_stmts = k
                .stmts
                .iter()
                .filter(|s| matches!(s, Stmt::Instr(_)))
                .count() as u32;
            assert_eq!(from_blocks, from_stmts, "{fixture}");
        }
    }
}

#[test]
fn bound_flop_totals_agree_with_an_independent_evaluation() {
    // Fixtures with fully resolvable trips and one symbolic parameter.
    let cases: &[(&str, usize)] = &[
        ("k1/k1.sm_80.ptx", 2),
        ("k2/k2.sm_80.ptx", 2),
        ("k5/k5.sm_80.ptx", 2),
        ("micro/single_loop.ptx", 1),
        ("micro/branchy.ptx", 1),
    ];
    for &(fixture, param) in cases {
        let src = read(fixture);
        let value = 4099; // deliberately not a multiple of the unrolls

        // Path 1: the report.
        let opts = AnalyzeOptions {
            bindings: vec![BindingSpec {
                index: Some(param),
                name: "N".into(),
                value,
            }],
            ..Default::default()
        };
        let report = analyze(&src, fixture, &opts).expect("analyzes");
        let reported: i64 = report.kernels[0].totals.flops["total"]
            .expr
            .parse()
            .expect("bound total is numeric");

        // Path 2: flat per-block tallies × numerically evaluated chains.
        let m = parse(&src).expect("parses");
        let k = &m.kernels[0];
        let cfg = build_cfg(&m, k);
        let f = loop_forest(&cfg);
        let names = loop_names(&m, k, &cfg, &f);
        let info = trip_counts(&m, k, &cfg, &f, &names);
        let bind_map: HashMap<String, i64> =
            [(format!("param_{param}"), value)].into_iter().collect();
        let trips_num: Vec<i64> = info
            .trips
            .iter()
            .map(|t| match t {
                Ok(e) => e.bind(&bind_map).as_const().expect("trips fully bound"),
                Err(_) => panic!("{fixture}: unexpected unknown trips"),
            })
            .collect();
        let blocks = collect(&m, k, &cfg, &f);
        let stats = Stats::new(&blocks);
        let mut independent = 0i64;
        for bm in &blocks {
            let mut mult = 1i64;
            let mut cur = f.block_loop[bm.block.0 as usize];
            while let Some(l) = cur {
                mult *= trips_num[l.0 as usize];
                cur = f.get(l).parent;
            }
            let flat = stats.flops(&[bm.block], None).value as i64;
            independent += flat * mult;
        }
        assert_eq!(reported, independent, "{fixture}: two paths disagree");
        // Sanity: the f32 column carries everything in this corpus.
        let f32_only: i64 = report.kernels[0].totals.flops[Precision::F32.key()]
            .expr
            .parse()
            .expect("numeric");
        assert_eq!(f32_only, reported, "{fixture}");
    }
}
