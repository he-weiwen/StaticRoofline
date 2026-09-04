//! Text view of the result tree (PLAN.md §6, PR 12). Renders the same
//! structs `--json` serializes — the two views cannot disagree.
//!
//! Conventions: every static quantity is labeled `[static]` once per
//! section header (bet 3: a lone static number without its provenance
//! label is a half-truth); `at_most` counts render with a `<=` prefix;
//! zero rows are skipped in flop/byte tables but unknowns are always
//! printed, even (especially) when present.

use super::tree::*;
use std::collections::BTreeMap;
use std::fmt::Write;

pub fn render(report: &Report) -> String {
    let mut out = String::new();
    let w = &mut out;
    let _ = writeln!(w, "ptxroof analyze [static] — {}", report.input);
    let _ = writeln!(
        w,
        "counts are static, per thread, as requested by the PTX: not measured, and not what the memory system moves"
    );
    if !report.bindings.is_empty() {
        let binds: Vec<String> = report
            .bindings
            .iter()
            .map(|b| format!("param {} ({}) = {}", b.param, b.name, b.value))
            .collect();
        let _ = writeln!(w, "bindings: {}", binds.join(", "));
    }

    for k in &report.kernels {
        let _ = writeln!(w);
        let _ = writeln!(w, "kernel {}", k.demangled);
        if k.demangled != k.name {
            let _ = writeln!(w, "  mangled: {}", k.name);
        }
        let params: Vec<String> = k
            .params
            .iter()
            .map(|p| format!("{}:{}", p.index, p.ty))
            .collect();
        let _ = writeln!(w, "  params: {}", params.join(" "));
        if let Some(heaviest) = &k.heaviest_loop {
            let _ = writeln!(w, "  heaviest loop (static weight): {heaviest}");
        }
        for mp in &k.machine_peaks {
            let unit = if mp.pipe == "cuda-core" {
                String::new()
            } else {
                format!(" {}", mp.pipe)
            };
            let _ = writeln!(
                w,
                "  machine @ {} ({}, from {}): {}{unit} peak {} TFLOPS / {} GB/s DRAM = {:.1} flop/B; loop {} AI(global) {} flop/B",
                mp.arch,
                mp.machine,
                mp.source,
                mp.precision,
                mp.peak_tflops,
                mp.dram_bw_gbps,
                mp.peak_flop_per_byte,
                mp.loop_name,
                intensity(&mp.ai_global)
            );
        }
        if let Some(l) = &k.launch {
            let (bound, note) = if l.exact {
                ("", "")
            } else {
                ("<= ", " — a maximum, not the launch")
            };
            let _ = writeln!(
                w,
                "  block size: {bound}{} threads ({}x{}x{} from {}{note})",
                l.threads, l.block[0], l.block[1], l.block[2], l.source
            );
        }
        let sm = &k.shared_memory;
        if sm.static_bytes > 0 || sm.dynamic {
            let dyn_note = if sm.dynamic {
                " + dynamic (set at launch)"
            } else {
                ""
            };
            let _ = writeln!(
                w,
                "  shared memory [static]: {} B per CTA{dyn_note}",
                sm.static_bytes
            );
        }
        if k.ranking.len() > 1 {
            let _ = writeln!(w, "  loops by static weight:");
            for (i, r) in k.ranking.iter().enumerate() {
                let _ = writeln!(
                    w,
                    "    {}. {}  ({} instructions)",
                    i + 1,
                    r.loop_name,
                    r.weight
                );
            }
        }

        for l in &k.loops {
            render_loop(w, l, 1);
        }

        let _ = writeln!(w, "  totals [static]:");
        render_aggregates(w, &k.totals, "    ");
        if let Some(per_cta) = &k.totals_per_cta {
            let _ = writeln!(w, "  totals per CTA [static]:");
            render_aggregates(w, per_cta, "    ");
        }

        if k.unknowns.is_empty() {
            let _ = writeln!(w, "  unknowns: none");
        } else {
            let _ = writeln!(w, "  unknowns:");
            for u in &k.unknowns {
                let count = u.count.map(|c| format!(" x{c}")).unwrap_or_default();
                let _ = writeln!(w, "    {}{count} — {}", u.what, u.reason);
            }
        }
    }

    let _ = writeln!(w);
    for (metric, f) in &report.coverage {
        let pct = if f.den == 0 {
            100.0
        } else {
            100.0 * f.num as f64 / f.den as f64
        };
        let _ = writeln!(w, "coverage: {metric} {pct:.1}% ({}/{})", f.num, f.den);
    }
    out
}

fn render_loop(w: &mut String, l: &LoopNode, depth: usize) {
    let pad = "  ".repeat(depth);
    let unroll = match &l.unroll {
        Some(u) => format!("  [unrolled x{}, remainder: {}]", u.factor, u.remainder),
        None => String::new(),
    };
    let _ = writeln!(w, "{pad}loop {} ({}){unroll}", l.name, l.label);
    match (&l.trips.expr, &l.trips.unknown) {
        (Some(e), _) => {
            let _ = writeln!(w, "{pad}  trips = {e}");
        }
        (None, Some(reason)) => {
            let _ = writeln!(w, "{pad}  trips = unknown: {reason}");
        }
        _ => {}
    }
    let _ = writeln!(w, "{pad}  per iteration:");
    render_aggregates(w, &l.per_iteration, &format!("{pad}    "));
    for child in &l.loops {
        render_loop(w, child, depth + 1);
    }
}

fn intensity(ai: &Intensity) -> String {
    let sign = match ai.bound {
        Bound::Exact => "=",
        Bound::AtLeast => ">=",
        Bound::AtMost => "<=",
    };
    format!("{sign} {}", ai.value)
}

fn both_sides_are_bounds(a: &Aggregates) -> bool {
    let flops = [&a.flops, &a.tensor_flops, &a.sfu_flops]
        .iter()
        .any(|t| t["total"].at_most);
    let global = &a.bytes["global"];
    flops && (global.load.at_most || global.store.at_most)
}

fn count(c: &Count) -> String {
    if c.at_most {
        format!("<= {}", c.expr)
    } else {
        c.expr.clone()
    }
}

fn render_flops(w: &mut String, pad: &str, label: &str, table: &BTreeMap<String, Count>) {
    let total = &table["total"];
    if total.expr == "0" {
        return;
    }
    let by_precision: Vec<String> = table
        .iter()
        .filter(|(k, v)| k.as_str() != "total" && v.expr != "0")
        .map(|(k, v)| format!("{k} {}", count(v)))
        .collect();
    let _ = writeln!(
        w,
        "{pad}{label} = {}  ({})",
        count(total),
        by_precision.join(", ")
    );
}

fn instruction_group(kind: &str) -> (u8, &'static str) {
    let workload = [
        "tensor ",
        "cuda-core ",
        "sfu ",
        " load ",
        " store ",
        " copy ",
        " atomic ",
    ];
    if workload
        .iter()
        .any(|w| kind.starts_with(w) || kind.contains(w))
    {
        (0, "workload")
    } else if kind == "warp communication" {
        (1, "warp communication")
    } else if kind == "register move" {
        (3, "register moves (mostly removed by ptxas)")
    } else if kind == "hint / no-op" || kind == "unknown" {
        (4, "other")
    } else {
        (2, "bookkeeping")
    }
}

fn render_instructions(w: &mut String, pad: &str, i: &InstructionCounts) {
    if i.total.expr == "0" {
        return;
    }
    let _ = writeln!(w, "{pad}instructions = {}", count(&i.total));
    let mut groups: BTreeMap<u8, (&str, Vec<String>)> = BTreeMap::new();
    for (kind, n) in &i.by_kind {
        let (order, title) = instruction_group(kind);
        let entry = groups.entry(order).or_insert((title, Vec::new()));
        entry.1.push(if matches!(order, 1 | 3) {
            count(n)
        } else {
            format!("{kind} {}", count(n))
        });
    }
    for (title, rows) in groups.values() {
        let _ = writeln!(w, "{pad}  {title}: {}", rows.join(", "));
    }
}

fn render_aggregates(w: &mut String, a: &Aggregates, pad: &str) {
    render_instructions(w, pad, &a.instructions);
    render_flops(w, pad, "flops", &a.flops);
    render_flops(w, pad, "tensor flops", &a.tensor_flops);
    render_flops(w, pad, "sfu flops", &a.sfu_flops);
    for (space, d) in &a.bytes {
        if d.load.expr == "0" && d.store.expr == "0" {
            continue;
        }
        let _ = writeln!(
            w,
            "{pad}{space} bytes: load {} B, store {} B",
            count(&d.load),
            count(&d.store)
        );
    }
    if a.conversions.expr != "0" {
        let _ = writeln!(w, "{pad}conversions = {}", count(&a.conversions));
    }
    match a.ai_global {
        Some(ai) => {
            let _ = writeln!(w, "{pad}AI(global) {} flop/B", intensity(&ai));
        }
        None if both_sides_are_bounds(a) => {
            let _ = writeln!(
                w,
                "{pad}AI(global): not bounded (flops and global bytes are both upper bounds)"
            );
        }
        None => {}
    }
    if !a.unrolled_source_lines.is_empty() {
        let lines: Vec<String> = a
            .unrolled_source_lines
            .iter()
            .map(|(l, n)| format!("{l} x{n}"))
            .collect();
        let _ = writeln!(w, "{pad}unrolled source lines: {}", lines.join(", "));
    }
}
