//! Text view of the result tree (PLAN.md §6, PR 12). Renders the same
//! structs `--json` serializes — the two views cannot disagree.
//!
//! Conventions: every static quantity is labeled `[static]` once per
//! section header (bet 3: a lone static number without its provenance
//! label is a half-truth); `at_most` counts render with a `<=` prefix;
//! zero rows are skipped in flop/byte tables but unknowns are always
//! printed, even (especially) when present.

use super::tree::*;
use std::fmt::Write;

pub fn render(report: &Report) -> String {
    let mut out = String::new();
    let w = &mut out;
    let _ = writeln!(w, "ptxroof analyze [static] — {}", report.input);
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
        if let Some(hot) = &k.hot_loop {
            let _ = writeln!(w, "  hot loop: {hot}");
        }
        if k.ranking.len() > 1 {
            let _ = writeln!(w, "  loops by weight:");
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

fn count(c: &Count) -> String {
    if c.at_most {
        format!("<= {}", c.expr)
    } else {
        c.expr.clone()
    }
}

fn render_aggregates(w: &mut String, a: &Aggregates, pad: &str) {
    let total = &a.flops["total"];
    if total.expr != "0" {
        let by_precision: Vec<String> = a
            .flops
            .iter()
            .filter(|(k, v)| k.as_str() != "total" && v.expr != "0")
            .map(|(k, v)| format!("{k} {}", count(v)))
            .collect();
        let _ = writeln!(
            w,
            "{pad}flops = {}  ({})",
            count(total),
            by_precision.join(", ")
        );
    }
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
    if let Some(ai) = a.ai_global {
        let _ = writeln!(w, "{pad}AI(global) = {ai} flop/B");
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
