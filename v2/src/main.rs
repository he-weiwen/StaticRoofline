//! Thin CLI shell over the `ptxroof` library.
//!
//! `analyze` is still a stub (exit 2) until the report lands in PR 12;
//! `analyze --dump-ast` already works — it is the canonical human/debug
//! view of the parsed flat IR. Further verbs join the `Command` enum
//! additively as their features land.

use anyhow::Context;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

/// Exit code for operational errors (unreadable input, parse error,
/// not-implemented). Mirrors the diff(1)/grep(1) convention: 0 =
/// success, 1 = reserved for "ran fine and the answer is negative",
/// 2 = could not do the job. clap itself exits 2 on usage errors, the
/// same class of problem.
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
    Analyze {
        input: PathBuf,
        /// Print the parsed module in canonical PTX form and exit
        #[arg(long)]
        dump_ast: bool,
    },
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("ptxroof: {err:#}");
            ExitCode::from(EXIT_ERROR)
        }
    }
}

fn run() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let Command::Analyze { input, dump_ast } = cli.command;
    if dump_ast {
        let source = std::fs::read_to_string(&input)
            .with_context(|| format!("reading {}", input.display()))?;
        let module = ptxroof::parse::parser::parse(&source)
            .with_context(|| format!("parsing {}", input.display()))?;
        print!("{}", ptxroof::parse::ast::dump(&module));
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!(
        "ptxroof: not implemented yet (lands PR 12): analyze {}",
        input.display()
    );
    Ok(ExitCode::from(EXIT_ERROR))
}
