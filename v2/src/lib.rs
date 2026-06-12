//! ptxroof — static roofline analysis for PTX kernels (v2 of
//! nvptx_analyzer; see PLAN.md).
//!
//! Modules land per the Phase 1 sequence in PLAN.md: `parse/` holds the
//! text frontend (lexer now; parser next, PR 04). The library/binary
//! split is architectural: everything analyzable lives here behind a
//! `Result`-returning API; `main.rs` only parses arguments and renders
//! errors.

pub mod core;
pub mod parse;

/// Tool version, baked in from Cargo.toml at compile time.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn version_is_nonempty() {
        assert!(!super::version().is_empty());
    }
}
