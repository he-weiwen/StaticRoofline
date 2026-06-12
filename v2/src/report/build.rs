//! Build the result tree for one module (PLAN.md §6, PR 12).
//!
//! Aggregation model: a block's contribution to an aggregate is its
//! per-execution tally times the product of the trip counts of the
//! loops between it and the aggregation root (exclusive). A loop with
//! unresolved trips contributes through a *named opaque symbol*
//! `trips(<loop>)` — totals stay symbolic, never silently zero (S9.1).
//!
//! `at_most` propagation: a tally is an upper bound if any contributing
//! block is conditional within its innermost scope (PR 09 rule), any
//! contributing instruction is predicated, or any loop on the
//! multiplier chain is *conditionally entered* (its header does not
//! dominate the enclosing scope's latch/exits — e.g. the whole k2 body
//! behind the bounds guard). Per-iteration views of a loop only look
//! below that loop, which is why a guarded kernel still has exact
//! per-iteration numbers — the altitude where the verdict lives.

use crate::cfg::loops::{LoopForest, LoopId};
use crate::cfg::naming::{LoopName, demangle, loop_names};
use crate::cfg::{BlockId, Cfg, build_cfg, loop_forest};
use crate::classify::{Direction, Precision, Space};
use crate::core::measurement::MeasureKind;
use crate::core::symexpr::SymExpr;
use crate::core::{Kernel, Module, Stmt};
use crate::parse::parser::{ParseError, parse};
use crate::report::collect::{BlockMeasurements, CountQualifier, collect};
use crate::report::tree::*;
use crate::trips::{TripInfo, trip_counts};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, thiserror::Error)]
pub enum AnalyzeError {
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("bad --bind: {0}")]
    Binding(String),
}

/// One `--bind` argument: `name=value` or `idx:name=value`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingSpec {
    pub index: Option<usize>,
    pub name: String,
    pub value: i64,
}

pub fn parse_bind(text: &str) -> Result<BindingSpec, String> {
    let (lhs, value) = text
        .split_once('=')
        .ok_or_else(|| format!("`{text}`: expected name=value or idx:name=value"))?;
    let value: i64 = value
        .parse()
        .map_err(|_| format!("`{text}`: value `{value}` is not an integer"))?;
    let (index, name) = match lhs.split_once(':') {
        Some((idx, name)) => {
            let idx = idx
                .parse()
                .map_err(|_| format!("`{text}`: index `{idx}` is not a number"))?;
            (Some(idx), name)
        }
        None => (None, lhs),
    };
    if name.is_empty() {
        return Err(format!("`{text}`: empty parameter name"));
    }
    Ok(BindingSpec {
        index,
        name: name.to_owned(),
        value,
    })
}

pub fn analyze(
    source: &str,
    input_name: &str,
    binds: &[BindingSpec],
) -> Result<Report, AnalyzeError> {
    let module = parse(source)?;
    let mut kernels = Vec::new();
    let mut classified = Fraction { num: 0, den: 0 };
    let mut trips_resolved = Fraction { num: 0, den: 0 };
    let mut bindings_echo = Vec::new();

    for kernel in &module.kernels {
        let bind_map = resolve_bindings(&module, kernel, binds, &mut bindings_echo)?;
        let k = KernelBuilder::new(&module, kernel, &bind_map).build();
        classified.num += k.instruction_classes.total - k.instruction_classes.unknown;
        classified.den += k.instruction_classes.total;
        let mut count_loops = |nodes: &[LoopNode]| {
            fn walk(nodes: &[LoopNode], f: &mut Fraction) {
                for n in nodes {
                    f.den += 1;
                    f.num += u64::from(n.trips.expr.is_some());
                    walk(&n.loops, f);
                }
            }
            walk(nodes, &mut trips_resolved);
        };
        count_loops(&k.loops);
        kernels.push(k);
    }

    let mut coverage = BTreeMap::new();
    coverage.insert("instructions_classified".to_owned(), classified);
    coverage.insert("loop_trips_resolved".to_owned(), trips_resolved);

    Ok(Report {
        input: input_name.to_owned(),
        bindings: bindings_echo,
        kernels,
        coverage,
    })
}

fn resolve_bindings(
    module: &Module,
    kernel: &Kernel,
    binds: &[BindingSpec],
    echo: &mut Vec<Binding>,
) -> Result<HashMap<String, i64>, AnalyzeError> {
    let mut map = HashMap::new();
    for spec in binds {
        let index = match spec.index {
            Some(i) => i,
            None => {
                // Name-only form: positional `param_N`, or the PTX
                // parameter name itself.
                if let Some(n) = spec
                    .name
                    .strip_prefix("param_")
                    .and_then(|s| s.parse().ok())
                {
                    n
                } else if let Some(i) = kernel
                    .params
                    .iter()
                    .position(|p| module.interner.resolve(p.name) == spec.name)
                {
                    i
                } else {
                    return Err(AnalyzeError::Binding(format!(
                        "`{}` names no parameter; use idx:name=value (params are positional)",
                        spec.name
                    )));
                }
            }
        };
        if index >= kernel.params.len() {
            return Err(AnalyzeError::Binding(format!(
                "param index {index} out of range ({} params)",
                kernel.params.len()
            )));
        }
        map.insert(format!("param_{index}"), spec.value);
        if !echo.iter().any(|b: &Binding| b.param == index) {
            echo.push(Binding {
                param: index,
                name: spec.name.clone(),
                value: spec.value,
            });
        }
    }
    Ok(map)
}

/// Symbolic sum that collects like terms: contributions are grouped
/// by their constant-free multiplier, so two blocks under the same
/// trip chain merge into one term ("4 * param_1", never
/// "2 * param_1 + 2 * param_1"). `at_most` is the OR over
/// contributions — an upper bound on any term makes the sum one.
#[derive(Default)]
struct TermSum {
    groups: Vec<(SymExpr, i64)>,
    at_most: bool,
    touched: bool,
}

impl TermSum {
    fn add(&mut self, count: i64, mult: &SymExpr, at_most: bool) {
        let (coeff, rest) = split_const(mult.clone());
        match self.groups.iter_mut().find(|(m, _)| *m == rest) {
            Some((_, n)) => *n += coeff * count,
            None => self.groups.push((rest, coeff * count)),
        }
        self.at_most |= at_most;
        self.touched = true;
    }

    fn expr(&self) -> SymExpr {
        self.groups
            .iter()
            .fold(SymExpr::Const(0), |acc, (mult, n)| {
                SymExpr::add(acc, SymExpr::mul(SymExpr::Const(*n), mult.clone()))
            })
    }

    fn count(&self) -> Count {
        Count {
            expr: self.expr().to_string(),
            at_most: self.at_most,
        }
    }
}

/// `(constant coefficient, the rest)` of a product.
fn split_const(e: SymExpr) -> (i64, SymExpr) {
    match e {
        SymExpr::Const(c) => (c, SymExpr::Const(1)),
        SymExpr::Prod(fs) => match fs.first() {
            Some(&SymExpr::Const(c)) => {
                let rest = fs[1..]
                    .iter()
                    .cloned()
                    .reduce(SymExpr::mul)
                    .unwrap_or(SymExpr::Const(1));
                (c, rest)
            }
            _ => (1, SymExpr::Prod(fs)),
        },
        other => (1, other),
    }
}

struct KernelBuilder<'a> {
    module: &'a Module,
    kernel: &'a Kernel,
    cfg: Cfg,
    forest: LoopForest,
    names: Vec<LoopName>,
    trip_info: TripInfo,
    blocks: Vec<BlockMeasurements>,
    bind_map: &'a HashMap<String, i64>,
    /// Trip expression per loop for aggregation (opaque symbol when
    /// unresolved), already bound.
    trip_exprs: Vec<SymExpr>,
    /// Loop is conditionally entered within its parent scope.
    cond_entry: Vec<bool>,
    /// Display names with the remainder suffix applied.
    display: Vec<String>,
}

impl<'a> KernelBuilder<'a> {
    fn new(module: &'a Module, kernel: &'a Kernel, bind_map: &'a HashMap<String, i64>) -> Self {
        let cfg = build_cfg(module, kernel);
        let forest = loop_forest(&cfg);
        let names = loop_names(module, kernel, &cfg, &forest);
        let trip_info = trip_counts(module, kernel, &cfg, &forest, &names);
        let blocks = collect(module, kernel, &cfg, &forest);

        let mut display: Vec<String> = names.iter().map(|n| n.display.clone()).collect();
        for pair in &trip_info.unroll_pairs {
            let r = pair.remainder.0 as usize;
            display[r] = format!("{} (remainder)", display[r]);
        }

        let trip_exprs: Vec<SymExpr> = trip_info
            .trips
            .iter()
            .enumerate()
            .map(|(i, t)| match t {
                Ok(e) => e.bind(bind_map),
                Err(_) => SymExpr::sym(format!("trips({})", display[i])),
            })
            .collect();

        let exit_blocks: Vec<BlockId> = (0..cfg.blocks.len() as u32)
            .map(BlockId)
            .filter(|&b| cfg.block(b).succs.is_empty())
            .collect();
        let cond_entry: Vec<bool> = (0..forest.loops.len())
            .map(|i| {
                let l = &forest.loops[i];
                let targets: &[BlockId] = match l.parent {
                    Some(p) => &forest.get(p).latches,
                    None => &exit_blocks,
                };
                targets.is_empty() || !targets.iter().all(|&t| forest.doms.dominates(l.header, t))
            })
            .collect();

        KernelBuilder {
            module,
            kernel,
            cfg,
            forest,
            names,
            trip_info,
            blocks,
            bind_map,
            trip_exprs,
            cond_entry,
            display,
        }
    }

    /// Loop chain of a block, innermost first.
    fn chain(&self, b: BlockId) -> Vec<LoopId> {
        let mut out = Vec::new();
        let mut cur = self.forest.block_loop[b.0 as usize];
        while let Some(l) = cur {
            out.push(l);
            cur = self.forest.get(l).parent;
        }
        out
    }

    /// Aggregate over blocks; `below` = aggregate within this loop
    /// (multipliers stop there), `None` = whole kernel.
    fn aggregates(&self, below: Option<LoopId>) -> Aggregates {
        let mut flops: BTreeMap<&'static str, TermSum> = BTreeMap::new();
        let mut flops_total = TermSum::default();
        let mut bytes: BTreeMap<&'static str, (TermSum, TermSum)> = BTreeMap::new();
        let mut conversions = TermSum::default();
        for p in [
            Precision::F16,
            Precision::BF16,
            Precision::F32,
            Precision::F64,
        ] {
            flops.insert(p.key(), TermSum::default());
        }
        for s in ["global", "shared", "local"] {
            bytes.insert(s, Default::default());
        }

        for bm in &self.blocks {
            let chain = self.chain(bm.block);
            // Inside `below`? (kernel root: always.)
            let cut = match below {
                Some(l) => {
                    let Some(pos) = chain.iter().position(|&x| x == l) else {
                        continue;
                    };
                    pos
                }
                None => chain.len(),
            };
            let mut mult = SymExpr::Const(1);
            let mut chain_at_most = false;
            for &l in &chain[..cut] {
                mult = SymExpr::mul(mult, self.trip_exprs[l.0 as usize].clone());
                chain_at_most |= self.cond_entry[l.0 as usize];
            }
            for m in &bm.measurements {
                let at_most =
                    bm.qualifier == CountQualifier::AtMost || m.predicated || chain_at_most;
                let n = m.count as i64;
                match m.kind {
                    MeasureKind::Flops { precision } => {
                        flops
                            .get_mut(precision.key())
                            .expect("all precisions pre-inserted")
                            .add(n, &mult, at_most);
                        flops_total.add(n, &mult, at_most);
                    }
                    MeasureKind::Bytes { space, direction } => {
                        let entry = bytes.entry(space.key()).or_default();
                        match direction {
                            Direction::Load => entry.0.add(n, &mult, at_most),
                            Direction::Store => entry.1.add(n, &mult, at_most),
                        }
                    }
                    MeasureKind::Conversions => conversions.add(n, &mult, at_most),
                    // Op-count kinds appear in unknowns/classes, not in
                    // the workload aggregates.
                    MeasureKind::UnquantifiedBytes { .. }
                    | MeasureKind::NonFlopOps { .. }
                    | MeasureKind::SyncOps
                    | MeasureKind::ControlOps
                    | MeasureKind::UnknownOps { .. } => {}
                }
            }
        }

        let mut flops_out: BTreeMap<String, Count> = flops
            .iter()
            .map(|(k, v)| (k.to_string(), v.count()))
            .collect();
        flops_out.insert("total".to_owned(), flops_total.count());
        let bytes_out: BTreeMap<String, DirectionCounts> = bytes
            .iter()
            .map(|(k, (l, s))| {
                (
                    k.to_string(),
                    DirectionCounts {
                        load: l.count(),
                        store: s.count(),
                    },
                )
            })
            .collect();

        // AI(global): defined when flops and global bytes are constants.
        let ai_global = match (flops_total.expr().as_const(), bytes.get("global")) {
            (Some(f), Some((l, s))) => match (l.expr().as_const(), s.expr().as_const()) {
                (Some(lb), Some(sb)) if lb + sb > 0 => Some(f as f64 / (lb + sb) as f64),
                _ => None,
            },
            _ => None,
        };

        Aggregates {
            flops: flops_out,
            bytes: bytes_out,
            conversions: conversions.count(),
            ai_global,
            unrolled_source_lines: self.unrolled_lines(below),
        }
    }

    /// Workload ops (flops + memory) per effective source line over the
    /// blocks DIRECTLY in this scope — the line-aggregation view that
    /// recovers fully-unrolled source loops. Entries with ≥ 2 copies.
    fn unrolled_lines(&self, scope: Option<LoopId>) -> BTreeMap<String, u64> {
        let mut per_line: BTreeMap<String, u64> = BTreeMap::new();
        for bm in &self.blocks {
            if self.forest.block_loop[bm.block.0 as usize] != scope {
                continue;
            }
            let blk = self.cfg.block(bm.block);
            for stmt in &self.kernel.stmts[blk.start..blk.end] {
                let Stmt::Instr(instr) = stmt else { continue };
                let Some(loc) = instr.loc.filter(|l| l.line != 0) else {
                    continue;
                };
                let is_workload = matches!(
                    crate::classify::classify(self.module, instr),
                    crate::classify::OpClass::Flop { .. } | crate::classify::OpClass::Memory { .. }
                );
                if !is_workload {
                    continue;
                }
                let file = self
                    .module
                    .file_path(loc.file)
                    .map(|p| p.rsplit('/').next().unwrap_or(p).to_owned())
                    .unwrap_or_else(|| "<unknown file>".to_owned());
                *per_line.entry(format!("{file}:{}", loc.line)).or_default() += 1;
            }
        }
        per_line.retain(|_, &mut v| v >= 2);
        per_line
    }

    fn loop_node(&self, id: LoopId) -> LoopNode {
        let name = &self.names[id.0 as usize];
        let trips = match &self.trip_info.trips[id.0 as usize] {
            Ok(e) => {
                let bound = e.bind(self.bind_map);
                Trips {
                    expr: Some(bound.to_string()),
                    unknown: None,
                }
            }
            Err(reason) => Trips {
                expr: None,
                unknown: Some(reason.clone()),
            },
        };
        let unroll = self
            .trip_info
            .unroll_pairs
            .iter()
            .find(|p| p.main == id)
            .map(|p| Unroll {
                factor: p.factor,
                remainder: self.display[p.remainder.0 as usize].clone(),
            });
        LoopNode {
            name: self.display[id.0 as usize].clone(),
            label: name.label.clone(),
            line: name.line,
            depth: self.forest.get(id).depth,
            trips,
            unroll,
            per_iteration: self.aggregates(Some(id)),
            loops: self
                .forest
                .children_of(id)
                .into_iter()
                .map(|c| self.loop_node(c))
                .collect(),
        }
    }

    /// Weight = executed instructions per kernel invocation; ranking
    /// compares weights at all-symbols = 2^20 + 3 (large, with nonzero
    /// residues mod small powers of two so remainder loops keep their
    /// share). Deliberately a comparison heuristic, not a claim.
    fn ranking(&self) -> Vec<RankEntry> {
        let mut entries: Vec<(String, SymExpr, i64)> = (0..self.forest.loops.len())
            .map(|i| {
                let l = &self.forest.loops[i];
                let mut weight = TermSum::default();
                for &b in &l.blocks {
                    let instrs = self.blocks[b.0 as usize].class_counts.total as i64;
                    let mut mult = SymExpr::Const(1);
                    for &cl in &self.chain(b) {
                        mult = SymExpr::mul(mult, self.trip_exprs[cl.0 as usize].clone());
                    }
                    weight.add(instrs, &mult, false);
                }
                let weight = weight.expr();
                let approx: HashMap<String, i64> = weight
                    .symbols()
                    .into_iter()
                    .map(|s| (s, (1 << 20) + 3))
                    .collect();
                let key = weight.bind(&approx).as_const().unwrap_or(i64::MAX);
                (self.display[i].clone(), weight, key)
            })
            .collect();
        entries.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
        entries
            .into_iter()
            .map(|(name, w, _)| RankEntry {
                loop_name: name,
                weight: w.to_string(),
            })
            .collect()
    }

    fn unknowns(&self) -> Vec<UnknownEntry> {
        let mut out = Vec::new();
        let mut unknown_ops: BTreeMap<String, u64> = BTreeMap::new();
        let mut unquantified: BTreeMap<String, u64> = BTreeMap::new();
        for bm in &self.blocks {
            for m in &bm.measurements {
                match m.kind {
                    MeasureKind::UnknownOps { mnemonic } => {
                        *unknown_ops
                            .entry(self.module.interner.resolve(mnemonic).to_owned())
                            .or_default() += m.count;
                    }
                    MeasureKind::UnquantifiedBytes { space, direction } => {
                        let dir = match direction {
                            Direction::Load => "load",
                            Direction::Store => "store",
                        };
                        *unquantified
                            .entry(format!("{} {dir}", Space::key(space)))
                            .or_default() += m.count;
                    }
                    _ => {}
                }
            }
        }
        for (mnemonic, count) in unknown_ops {
            out.push(UnknownEntry {
                what: format!("instruction `{mnemonic}`"),
                count: Some(count),
                reason: "not classified — its flops/bytes are not counted".to_owned(),
            });
        }
        for (what, count) in unquantified {
            out.push(UnknownEntry {
                what: format!("{what} with statically unknown byte count"),
                count: Some(count),
                reason: "counted as an op; bytes missing from every byte total".to_owned(),
            });
        }
        for (i, t) in self.trip_info.trips.iter().enumerate() {
            if let Err(reason) = t {
                out.push(UnknownEntry {
                    what: format!("loop {}", self.display[i]),
                    count: None,
                    reason: reason.clone(),
                });
            }
        }
        for (src, dst) in &self.forest.irreducible_edges {
            let label = |b: BlockId| {
                self.cfg
                    .block(b)
                    .label
                    .map(|s| self.module.interner.resolve(s).to_owned())
                    .unwrap_or_else(|| format!("<block {}>", b.0))
            };
            out.push(UnknownEntry {
                what: format!(
                    "irreducible control flow {} -> {}",
                    label(*src),
                    label(*dst)
                ),
                count: None,
                reason: "cycle with multiple entries — execution multiplicity unknown".to_owned(),
            });
        }
        if !self.cfg.call_sites.is_empty() {
            out.push(UnknownEntry {
                what: "call".to_owned(),
                count: Some(self.cfg.call_sites.len() as u64),
                reason: "non-inlined callee — its cost is not included".to_owned(),
            });
        }
        out
    }

    fn build(self) -> KernelReport {
        let name = self.module.interner.resolve(self.kernel.name).to_owned();
        let mut classes = InstructionClasses::default();
        for bm in &self.blocks {
            let c = bm.class_counts;
            classes.total += c.total as u64;
            classes.flop += c.flop as u64;
            classes.non_flop_arith += c.non_flop_arith as u64;
            classes.memory += c.memory as u64;
            classes.sync += c.sync as u64;
            classes.control += c.control as u64;
            classes.ignore += c.ignore as u64;
            classes.unknown += c.unknown as u64;
        }
        let ranking = self.ranking();
        KernelReport {
            demangled: demangle(&name),
            name,
            params: self
                .kernel
                .params
                .iter()
                .enumerate()
                .map(|(i, p)| ParamInfo {
                    index: i,
                    ty: self.module.interner.resolve(p.ty).to_owned(),
                    name: self.module.interner.resolve(p.name).to_owned(),
                })
                .collect(),
            instruction_classes: classes,
            hot_loop: ranking.first().map(|r| r.loop_name.clone()),
            ranking,
            loops: self
                .forest
                .top_level()
                .into_iter()
                .map(|t| self.loop_node(t))
                .collect(),
            totals: self.aggregates(None),
            unknowns: self.unknowns(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_parsing_grammar() {
        assert_eq!(
            parse_bind("K=4096"),
            Ok(BindingSpec {
                index: None,
                name: "K".into(),
                value: 4096
            })
        );
        assert_eq!(
            parse_bind("2:K=4096"),
            Ok(BindingSpec {
                index: Some(2),
                name: "K".into(),
                value: 4096
            })
        );
        assert!(parse_bind("K").is_err());
        assert!(parse_bind("K=x").is_err());
        assert!(parse_bind("=4").is_err());
        assert!(parse_bind("a:K=4").is_err());
    }
}
