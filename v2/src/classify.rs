//! Instruction → semantic record (PLAN.md §6, PR 08).
//!
//! Transcribed from v1's `lib/PTX/Classifier.cpp` and narrowed to what
//! the Phase 1 corpus contains: cuda-core flops, non-flop arithmetic
//! (conversions counted separately — they are the precision-audit
//! overhead S8 reports), memory, sync, control, ignore, unknown. The
//! Phase 2 families (tensor/wmma, cp.async, atomics, SFU
//! transcendentals, tex/surf) classify as `Unknown` until their item
//! lands: visible, counted, and policed by the corpus coverage check —
//! never silently zero.
//!
//! Two deliberate divergences from the v1 reference, both documented
//! at the match arm:
//! - an arithmetic mnemonic is a flop only when it carries an FP type
//!   modifier; `mad.lo.s32`/`add.s64` address arithmetic is non-flop
//!   (v1 gave integer `mad` 2 flops of precision "Other");
//! - `div`/`sqrt`/`rcp`/... are the SFU family, a Phase 2 item with
//!   its own documented FLOP policy; until then they are Unknown, not
//!   silently 1-flop.
//!
//! FLOPs convention (Williams roofline accounting, as v1): fma/mad = 2
//! per invocation, add/sub/mul/min/max/abs/neg/copysign = 1, all
//! multiplied by packed lanes (`f16x2` = ×2).

use crate::core::{Instr, Module};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Precision {
    F16,
    BF16,
    F32,
    F64,
}

impl Precision {
    pub fn key(self) -> &'static str {
        match self {
            Precision::F16 => "f16",
            Precision::BF16 => "bf16",
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Space {
    Global,
    Shared,
    /// `.shared::cluster` — distinct from CTA-local shared (`::cta` is
    /// the ISA default and folds into `Shared`).
    SharedCluster,
    Local,
    Const,
    Param,
    /// `ld`/`st` with no state space: PTX generic addressing. Its own
    /// honest bucket — unclassifiable to a concrete space without
    /// provenance (anti-scope until a fixture emits one).
    Generic,
}

impl Space {
    pub fn key(self) -> &'static str {
        match self {
            Space::Global => "global",
            Space::Shared => "shared",
            Space::SharedCluster => "shared::cluster",
            Space::Local => "local",
            Space::Const => "const",
            Space::Param => "param",
            Space::Generic => "generic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Load,
    Store,
}

/// Non-flop arithmetic, split because conversions are first-class in a
/// precision audit (S8: 8 `cvt` per k2 main-loop iteration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithKind {
    /// `cvt` — precision/width conversion.
    Conversion,
    /// Integer/bit arithmetic: address math, masks, shifts.
    Integer,
    /// Predicate computation and selection: `setp`, `selp`, `set`...
    Predicate,
    /// Register/address bookkeeping: `mov`, `cvta`.
    Move,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// Cuda-core floating-point work. `flops` already includes the
    /// base convention and packed lanes.
    Flop {
        precision: Precision,
        flops: u32,
    },
    NonFlopArith {
        kind: ArithKind,
    },
    /// `bytes` is per-thread per-execution; `None` = statically
    /// unquantifiable (surfaces in the unquantified counter, never 0).
    Memory {
        space: Space,
        direction: Direction,
        bytes: Option<u32>,
    },
    /// Barriers, fences, and warp-collective communication.
    Sync,
    /// Branches, returns, calls.
    Control,
    /// Correctly contributes nothing (`nop`, cache hints).
    Ignore,
    /// Not yet handled — counted and named, never dropped.
    Unknown,
}

pub fn classify(module: &Module, instr: &Instr) -> OpClass {
    let interner = &module.interner;
    let mnemonic = interner.resolve(instr.mnemonic);
    let mods: Vec<&str> = module
        .modifiers(instr)
        .iter()
        .map(|&m| interner.resolve(m))
        .collect();

    let fp = fp_precision_and_lanes(&mods);

    match mnemonic {
        // -- cuda-core flops (FP type modifier required) -----------------
        "fma" | "mad" => match fp {
            Some((p, lanes)) => OpClass::Flop {
                precision: p,
                flops: 2 * lanes,
            },
            // mad.lo.s32 and friends are address arithmetic, not flops
            // (divergence from v1, which counted 2 "Other" flops here).
            None => OpClass::NonFlopArith {
                kind: ArithKind::Integer,
            },
        },
        "add" | "sub" | "mul" | "min" | "max" | "abs" | "neg" | "copysign" => match fp {
            Some((p, lanes)) => OpClass::Flop {
                precision: p,
                flops: lanes,
            },
            None => OpClass::NonFlopArith {
                kind: ArithKind::Integer,
            },
        },

        // -- non-flop arithmetic ------------------------------------------
        "cvt" => OpClass::NonFlopArith {
            kind: ArithKind::Conversion,
        },
        "mov" | "cvta" => OpClass::NonFlopArith {
            kind: ArithKind::Move,
        },
        "setp" | "selp" | "set" | "slct" | "testp" | "isspacep" => OpClass::NonFlopArith {
            kind: ArithKind::Predicate,
        },
        "and" | "or" | "xor" | "not" | "shl" | "shr" | "lop3" | "bfe" | "bfi" | "brev" | "popc"
        | "clz" | "bfind" | "bmsk" | "szext" | "prmt" | "rem" | "sad" | "dp4a" | "dp2a"
        | "mul24" | "mad24" => {
            if mods.contains(&"pred") {
                OpClass::NonFlopArith {
                    kind: ArithKind::Predicate,
                }
            } else {
                OpClass::NonFlopArith {
                    kind: ArithKind::Integer,
                }
            }
        }

        // -- memory -----------------------------------------------------------
        "ld" | "st" => {
            let direction = if mnemonic == "ld" {
                Direction::Load
            } else {
                Direction::Store
            };
            OpClass::Memory {
                space: space_of(&mods),
                direction,
                bytes: bytes_of(&mods),
            }
        }
        // Read-only / uniform global loads (v1 rule: count as global
        // loads when a width is present).
        "ldu" | "ldg" => OpClass::Memory {
            space: Space::Global,
            direction: Direction::Load,
            bytes: bytes_of(&mods),
        },

        // -- sync / warp collectives ---------------------------------------
        "bar" | "barrier" | "mbarrier" | "membar" | "fence" => OpClass::Sync,
        "shfl" | "vote" | "match" | "activemask" | "redux" => OpClass::Sync,

        // -- control -----------------------------------------------------------
        "bra" | "brx" | "ret" | "exit" | "call" | "trap" | "brkpt" => OpClass::Control,

        // -- correctly ignored -------------------------------------------------
        "nop" | "prefetch" | "prefetchu" | "discard" | "applypriority" | "griddepcontrol" => {
            OpClass::Ignore
        }

        // -- everything else: Phase 2 families and genuine novelty ------
        // (tensor/wmma/mma/ldmatrix, cp.async, atom/red, SFU
        // transcendentals incl. div/sqrt/rcp/sin/cos/ex2/lg2/tanh/rsqrt,
        // tex/surf). Counted and named by the coverage check.
        _ => OpClass::Unknown,
    }
}

/// FP precision + packed-lane multiplier from type modifiers.
/// `bf16` is checked before `f16` deliberately (v1 had to dodge the
/// substring trap; exact matching keeps the order only for clarity).
fn fp_precision_and_lanes(mods: &[&str]) -> Option<(Precision, u32)> {
    for &m in mods {
        let hit = match m {
            "bf16" => Some((Precision::BF16, 1)),
            "bf16x2" => Some((Precision::BF16, 2)),
            "f16" => Some((Precision::F16, 1)),
            "f16x2" => Some((Precision::F16, 2)),
            "f32" => Some((Precision::F32, 1)),
            "f32x2" => Some((Precision::F32, 2)),
            "f64" => Some((Precision::F64, 1)),
            _ => None,
        };
        if hit.is_some() {
            return hit;
        }
    }
    None
}

fn space_of(mods: &[&str]) -> Space {
    for &m in mods {
        match m {
            "global" => return Space::Global,
            "shared" | "shared::cta" => return Space::Shared,
            "shared::cluster" => return Space::SharedCluster,
            "local" => return Space::Local,
            "const" => return Space::Const,
            "param" => return Space::Param,
            _ => {}
        }
    }
    Space::Generic
}

/// Per-execution byte count: type width × vector multiplier.
fn bytes_of(mods: &[&str]) -> Option<u32> {
    let mut width: Option<u32> = None;
    let mut vec = 1u32;
    for &m in mods {
        match m {
            "b8" | "u8" | "s8" | "e4m3" | "e5m2" => width = Some(1),
            "b16" | "u16" | "s16" | "f16" | "bf16" => width = Some(2),
            "b32" | "u32" | "s32" | "f32" | "f16x2" | "bf16x2" => width = Some(4),
            "b64" | "u64" | "s64" | "f64" | "f32x2" => width = Some(8),
            "b128" => width = Some(16),
            "v2" => vec = 2,
            "v4" => vec = 4,
            "v8" => vec = 8,
            _ => {}
        }
    }
    width.map(|w| w * vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Stmt;
    use crate::parse::parser::parse;

    /// Classify the single instruction in `text`.
    fn class_of(text: &str) -> OpClass {
        let src = format!(
            ".version 8.7\n.target sm_80\n.address_size 64\n\
             .visible .entry k()\n{{\n{text}\n}}\n"
        );
        let m = parse(&src).expect("snippet parses");
        let instr = m.kernels[0]
            .stmts
            .iter()
            .find_map(|s| match s {
                Stmt::Instr(i) => Some(*i),
                _ => None,
            })
            .expect("one instruction");
        classify(&m, &instr)
    }

    #[test]
    fn flops_follow_the_fma_2_convention_and_packed_lanes() {
        assert_eq!(
            class_of("fma.rn.f32 %f1, %f2, %f3, %f4;"),
            OpClass::Flop {
                precision: Precision::F32,
                flops: 2
            }
        );
        assert_eq!(
            class_of("fma.rn.f16x2 %r1, %r2, %r3, %r4;"),
            OpClass::Flop {
                precision: Precision::F16,
                flops: 4
            }
        );
        assert_eq!(
            class_of("add.f32 %f1, %f2, %f3;"),
            OpClass::Flop {
                precision: Precision::F32,
                flops: 1
            }
        );
        assert_eq!(
            class_of("mul.bf16x2 %r1, %r2, %r3;"),
            OpClass::Flop {
                precision: Precision::BF16,
                flops: 2
            }
        );
        assert_eq!(
            class_of("max.f64 %fd1, %fd2, %fd3;"),
            OpClass::Flop {
                precision: Precision::F64,
                flops: 1
            }
        );
    }

    #[test]
    fn integer_arithmetic_is_not_a_flop() {
        // The divergence-from-v1 case: integer mad is address math.
        assert_eq!(
            class_of("mad.lo.s32 %r1, %r2, %r3, %r4;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
        assert_eq!(
            class_of("add.s64 %rd1, %rd2, 8;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
        assert_eq!(
            class_of("mul.wide.s32 %rd1, %r1, 2;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
    }

    #[test]
    fn conversions_are_their_own_kind() {
        assert_eq!(
            class_of("cvt.f32.f16 %f1, %rs1;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Conversion
            }
        );
        assert_eq!(
            class_of("cvt.rn.f16.f32 %rs1, %f1;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Conversion
            }
        );
        // cvta is an address-space cast, not a value conversion.
        assert_eq!(
            class_of("cvta.to.global.u64 %rd1, %rd2;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Move
            }
        );
    }

    #[test]
    fn predicate_ops_including_or_pred() {
        assert_eq!(
            class_of("setp.lt.s32 %p1, %r1, %r2;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Predicate
            }
        );
        assert_eq!(
            class_of("or.pred %p3, %p1, %p2;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Predicate
            }
        );
        assert_eq!(
            class_of("and.b32 %r1, %r2, 3;"),
            OpClass::NonFlopArith {
                kind: ArithKind::Integer
            }
        );
    }

    #[test]
    fn memory_width_space_and_direction() {
        assert_eq!(
            class_of("ld.global.u16 %rs1, [%rd1];"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(2)
            }
        );
        assert_eq!(
            class_of("st.shared.u16 [%r1], %rs1;"),
            OpClass::Memory {
                space: Space::Shared,
                direction: Direction::Store,
                bytes: Some(2)
            }
        );
        assert_eq!(
            class_of("ld.param.u64 %rd1, [k_param_4];"),
            OpClass::Memory {
                space: Space::Param,
                direction: Direction::Load,
                bytes: Some(8)
            }
        );
        assert_eq!(
            class_of("ld.global.v2.f32 {%f1, %f2}, [%rd1];"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(8)
            }
        );
        assert_eq!(
            class_of("ld.global.nc.L2::128B.b128 {%rd1, %rd2}, [%rd3];"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(16)
            }
        );
        assert_eq!(
            class_of("ldg.global.f32 %f1, [%rd1];"),
            OpClass::Memory {
                space: Space::Global,
                direction: Direction::Load,
                bytes: Some(4)
            }
        );
    }

    #[test]
    fn generic_and_cluster_spaces_are_distinct_buckets() {
        assert_eq!(
            class_of("ld.f32 %f1, [%rd1];"),
            OpClass::Memory {
                space: Space::Generic,
                direction: Direction::Load,
                bytes: Some(4)
            }
        );
        assert_eq!(
            class_of("st.shared::cluster.b32 [%r1], %r2;"),
            OpClass::Memory {
                space: Space::SharedCluster,
                direction: Direction::Store,
                bytes: Some(4)
            }
        );
        assert_eq!(
            class_of("ld.shared::cta.b32 %r1, [%r2];"),
            OpClass::Memory {
                space: Space::Shared,
                direction: Direction::Load,
                bytes: Some(4)
            }
        );
    }

    #[test]
    fn sync_control_ignore() {
        assert_eq!(class_of("bar.sync 0;"), OpClass::Sync);
        assert_eq!(class_of("membar.gl;"), OpClass::Sync);
        assert_eq!(
            class_of("shfl.sync.bfly.b32 %r1, %r2, %r3, %r4, %r5;"),
            OpClass::Sync
        );
        assert_eq!(class_of("@%p1 bra $L__X;"), OpClass::Control);
        assert_eq!(class_of("ret;"), OpClass::Control);
        assert_eq!(class_of("nop;"), OpClass::Ignore);
    }

    #[test]
    fn phase_2_families_are_unknown_not_zero() {
        for text in [
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1}, {%r1}, {%r2}, {%f2};",
            "cp.async.cg.shared.global [%r1], [%rd1], 16;",
            "atom.global.add.u32 %r1, [%rd1], %r2;",
            "sqrt.rn.f32 %f1, %f2;",
            "div.rn.f32 %f1, %f2, %f3;",
            "tex.2d.v4.f32.f32 {%f1, %f2, %f3, %f4}, [t, {%f5, %f6}];",
        ] {
            assert_eq!(class_of(text), OpClass::Unknown, "{text}");
        }
    }
}
