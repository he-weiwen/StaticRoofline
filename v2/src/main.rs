//! Thin CLI shell over the `ptxroof` library.
//!
//! PR 01: `analyze` is a stub — it echoes what it was asked to do,
//! states that it is not implemented, and exits with code 2, so the
//! CLI test runner and the scenario status file can already drive the
//! real binary and the acceptance scenario xfails honestly instead of
//! being skipped. Further verbs join the `Command` enum additively as
//! their features land.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

/// Exit code for operational errors (unreadable input, not-implemented).
/// Mirrors the diff(1)/grep(1) convention: 0 = success, 1 = reserved
/// for "ran fine and the answer is negative", 2 = could not do the job.
/// clap itself exits 2 on usage errors, the same class of problem.
const EXIT_ERROR: u8 = 2;

#[derive(Parser)]
#[command(
    name = "ptxroof",
    version,
    about = "Static roofline analysis for PTX kernels"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Analyze a PTX file: per-loop flops/bytes/AI and verdicts
    Analyze { input: PathBuf },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let Command::Analyze { input } = cli.command;
    eprintln!(
        "ptxroof: not implemented yet (PR 01 stub): analyze {}",
        input.display()
    );
    ExitCode::from(EXIT_ERROR)
}
