//! Thin CLI shell over the `ptxroof` library.
//!
//! PR 01: all five subcommands are stubs. Each echoes what it was asked
//! to do, states that it is not implemented, and exits with code 2 — so
//! the golden runner and the acceptance manifest can already drive the
//! real binary, and every scenario xfails honestly instead of being
//! skipped. Argument shapes below are the real ones from PLAN.md §2/§6;
//! implementations replace the stub bodies without changing the CLI.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

/// Exit code for operational errors (unreadable input, malformed rules
/// file, not-implemented). Distinct from exit 1, which `check` (PR 18)
/// reserves for "rules evaluated and failed" — the code a CI gate keys
/// on. clap itself exits 2 on usage errors, the same class of problem.
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
    /// Per-loop delta between two analyses (joined on stable loop IDs)
    Diff { before: PathBuf, after: PathBuf },
    /// Evaluate TOML rules against an analysis; exit 1 on rule failure
    Check {
        input: PathBuf,
        /// Rules file (TOML)
        #[arg(long)]
        rules: PathBuf,
    },
    /// Source listing annotated with per-line costs
    Annotate { input: PathBuf },
    /// Derived capability report: classified / unknown-by-policy / unhandled
    Capabilities,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let what = match cli.command {
        Command::Analyze { input } => format!("analyze {}", input.display()),
        Command::Diff { before, after } => {
            format!("diff {} {}", before.display(), after.display())
        }
        Command::Check { input, rules } => {
            format!("check {} --rules {}", input.display(), rules.display())
        }
        Command::Annotate { input } => format!("annotate {}", input.display()),
        Command::Capabilities => "capabilities".to_string(),
    };
    eprintln!("ptxroof: not implemented yet (PR 01 stub): {what}");
    ExitCode::from(EXIT_ERROR)
}
