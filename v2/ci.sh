#!/usr/bin/env bash
# PR-blocking CI (PLAN.md §2): T1 via cargo, then T2/T3 via the golden
# runner. Hermetic — needs only a rustup-managed toolchain (pinned by
# rust-toolchain.toml) and python3 >= 3.11. No CUDA, no LLVM, no C++.
set -euo pipefail
cd "$(dirname "$0")"

# --locked: refuse to run if Cargo.lock disagrees with Cargo.toml, so the
# committed lockfile is always the one actually being built.
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --locked # the golden runner drives target/debug/ptxroof

python3 tests/golden/run.py --self-test
python3 tests/golden/run.py

echo "ci.sh: all green"
