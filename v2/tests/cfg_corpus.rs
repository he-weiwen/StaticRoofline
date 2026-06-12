//! Pinned block/edge counts per fixture (PLAN.md §6, PR 05).
//!
//! The expected values were derived by hand from the committed PTX
//! (k1/k2: entry guard, K<1 guard, unroll guard, main-loop preheader,
//! main loop, remainder guard, remainder preheader, remainder loop,
//! epilogue, ret = 10 blocks; k5: entry guard, preheader, tile loop
//! head, dot loop, outer latch, bra-join, zero-init, epilogue = 8) and
//! double-checked against the implementation. A toolchain upgrade that
//! reshapes a fixture's control flow shows up here as a reviewable
//! diff, not silently.

use ptxroof::cfg::build_cfg;
use ptxroof::parse::parser::parse;
use std::fs;
use std::path::PathBuf;

#[test]
fn block_and_edge_counts_are_pinned() {
    // (fixture, blocks, edges)
    let expected = [
        ("k1/k1.sm_80.ptx", 10, 15),
        ("k2/k2.sm_80.ptx", 10, 15),
        ("k5/k5.sm_80.ptx", 8, 10),
        ("micro/single_loop.ptx", 4, 5),
        ("micro/branchy.ptx", 6, 8),
        ("micro/irreducible.ptx", 4, 5),
        ("micro/no_loc.ptx", 4, 5),
        ("micro/data_dep.ptx", 3, 3),
    ];
    for (fixture, blocks, edges) in expected {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture);
        let src = fs::read_to_string(&path).expect("fixture readable");
        let m = parse(&src).expect("fixture parses");
        assert_eq!(m.kernels.len(), 1, "{fixture}: one kernel per fixture");
        let cfg = build_cfg(&m, &m.kernels[0]);
        let got_edges: usize = cfg.blocks.iter().map(|b| b.succs.len()).sum();
        assert_eq!((cfg.blocks.len(), got_edges), (blocks, edges), "{fixture}");
        assert!(
            cfg.unresolved_branches.is_empty(),
            "{fixture}: unresolved branch targets"
        );
        assert!(
            cfg.call_sites.is_empty(),
            "{fixture}: unexpected call sites"
        );
    }
}

#[test]
fn every_edge_is_bidirectionally_consistent() {
    for fixture in [
        "k1/k1.sm_80.ptx",
        "k2/k2.sm_80.ptx",
        "k5/k5.sm_80.ptx",
        "micro/branchy.ptx",
    ] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(fixture);
        let src = fs::read_to_string(&path).expect("fixture readable");
        let m = parse(&src).expect("fixture parses");
        let cfg = build_cfg(&m, &m.kernels[0]);
        for (i, b) in cfg.blocks.iter().enumerate() {
            for s in &b.succs {
                let back = &cfg.blocks[s.0 as usize].preds;
                assert!(
                    back.iter().any(|p| p.0 as usize == i),
                    "{fixture}: edge {i}->{} missing from preds",
                    s.0
                );
            }
        }
    }
}
