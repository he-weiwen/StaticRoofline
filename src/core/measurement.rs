//! Measurement: the canonical record of one instruction's contribution
//! (PLAN.md §6, PR 09 — v1's Measurement contract, kept and extended).
//!
//! `count` is the magnitude contributed by ONE execution of the
//! instruction by ONE thread (flops for `Flops`, bytes for `Bytes`,
//! 1 for op-counting kinds). Loop trip multiplication happens at
//! report aggregation (PR 12); until then everything is per-execution
//! and the constants stay exact.
//!
//! Honesty is in the kinds: an instruction that moves statically
//! unquantifiable bytes becomes `UnquantifiedBytes` (an op count with
//! a visible hole), an unhandled instruction becomes `UnknownOp` with
//! its mnemonic — the v1 `AsyncCopy{bytes unset} → 0` silent-zero bug
//! class is unrepresentable.

use super::intern::Symbol;
use crate::classify::{ArithKind, Direction, Precision, Space};

/// Which threads an instruction's `count` is per: everything Phase 1
/// classifies is per-thread; warp-cooperative ops (tensor families,
/// Phase 2) are per-warp. Normalization to per-CTA/launch happens with
/// `--launch` in hand (PR 13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    PerThread,
    PerWarp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasureKind {
    Flops {
        precision: Precision,
    },
    Bytes {
        space: Space,
        direction: Direction,
    },
    /// A memory op whose byte count is statically unknowable: counted
    /// as an op, surfaced in the unquantified counter — never zero.
    UnquantifiedBytes {
        space: Space,
        direction: Direction,
    },
    /// `cvt` ops — the precision-conversion overhead (S8).
    Conversions,
    /// Integer/predicate/move bookkeeping ops.
    NonFlopOps {
        kind: ArithKind,
    },
    SyncOps,
    ControlOps,
    /// An instruction the classifier does not handle: counted by
    /// mnemonic, reported by name.
    UnknownOps {
        mnemonic: Symbol,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Measurement {
    pub kind: MeasureKind,
    pub scope: Scope,
    pub count: u64,
    /// The instruction itself is `@%p`-guarded: its count is an upper
    /// bound regardless of where its block sits.
    pub predicated: bool,
    /// Statement index into the owning kernel's `stmts` — provenance
    /// for diagnostics and the report verifier.
    pub provenance: usize,
}
