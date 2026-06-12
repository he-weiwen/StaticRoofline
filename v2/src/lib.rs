//! ptxroof — static roofline analysis for PTX kernels (v2 of
//! nvptx_analyzer; see PLAN.md).
//!
//! PR 01 scaffold: the library is intentionally empty of analysis code.
//! Modules land per the PR train (parse/ at PR 03, cfg/ at PR 09, …).
//! The library/binary split is architectural: everything analyzable
//! lives here behind a `Result`-returning API; `main.rs` only parses
//! arguments and renders errors.

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
