# v2 (`ptxroof`) — Implementation Plan (Rust)

A ground-up rewrite of the analyzer with PTX text as the input substrate,
implemented in Rust. Lives entirely under `v2/`; v1 (`lib/`, `test/`)
stays untouched and working until the differential harness (PR 21) proves
parity, after which v1 goes into maintenance mode.

Companion to `STATUS.md` (v1 state) and `docs/measurement-refactor.md`
(v1's value-stream refactor, whose Measurement contract this design keeps
and extends). This revision supersedes the earlier C++17 draft of this
plan: with the LLVM dependency gone, every boundary of the tool is text
(PTX/SASS/CSV in, JSON/HTML out, compilers as subprocesses), so the
implementation language has zero interop surface — and the design's
central types are sum types, which Rust expresses natively. The v1
`lib/PTX/` C++ code (~900 LOC) is kept as a **reference specification**
for transcription, pinned by shared golden gates, not ported.

This document is the execution plan: step by step, PR by PR, every PR
with its tests and exit criteria. The checklist at the bottom is ticked
as PRs land.

---

## 1. The four bets (decisions, not options)

1. **PTX text in. No LLVM dependency.** Input is `.ptx` from any producer
   (nvcc, clang, Triton, XLA, `cuobjdump --dump-ptx` on built binaries).
   Kills the MIR/inline-asm dual path, opcode-name churn against LLVM
   trunk, and the clang-only audience restriction.
2. **The loop is the unit of analysis; counts are symbolic natively.**
   A PTX file is a recipe, not a trace: a hot loop's body appears once
   in the text but executes `K/4` times, and `K` is a kernel argument
   that exists nowhere in the file. Counting instructions as written
   weights the epilogue (runs once) the same as the inner loop (runs
   millions of times) — a number that describes no execution for any
   `K`. So the count datatype is an expression over trip-count symbols
   (e.g. `8·(K − K mod 4)/4`), not a `u64`; integers are merely the
   fully-bound special case. What this buys: (a) the report is a loop
   tree whose headline is the steady-state inner iteration — the only
   altitude where execution concentrates and a verdict names source
   lines the user can change; (b) dividing two symbolic quantities
   yields shape-independent facts ("AI = 0.5 flop/B for all K") that
   no concrete-count tool can state, because you cannot divide numbers
   you don't have; (c) `diff` and `check` get stable subjects ("loop
   gemm.cu:84: flops/iter ≥ 8") that survive recompiles, problem-size
   changes, and compiler unrolling — the expression *encodes* the
   unroll factor instead of being broken by it; (d) uncertainty lives
   in the type, not in prose: an underivable multiplier degrades to a
   *named* symbol with a reason, and counts under in-loop branches
   carry an `at_most` marker that propagates into every aggregate.
3. **Static counts are one column of three.** PTX is two stages removed
   from reality: closed-source ptxas compiles it to SASS — assigning
   the registers PTX pretends are infinite, and inserting spill traffic
   (`LDL`/`STL`) the PTX never shows (verified here: identical PTX, two
   ptxas flags → 0 vs 5.3 KB of spill bytes) — then the memory system
   moves more bytes than requested (uncoalesced sector overfetch) or
   fewer (cache reuse). A static number presented alone is therefore an
   unfalsifiable half-truth. The report prints three labeled columns —
   **requested** (ours: static, exact, always available — the
   algorithm's demand), **transferred** (NCU-measured DRAM bytes),
   **achieved** (NCU-measured rates) — plus the SASS sidecar's
   post-ptxas facts (registers, spill bytes per loop). The diagnosis
   lives in the *ratios*: transferred ≪ requested → cache reuse is
   working; transferred ≫ requested → coalescing pathology; achieved
   under both roofs → latency/occupancy, consult the SASS column.
   Implementation consequences: columns 2–3 are text parsers plus a
   kernel-name join, never profilers — CI stays GPU-free; measured
   columns are `Option` (absent renders blank, never zero); hardware
   counters are per-launch, so measured columns attach only to the
   kernel node, never to loops — the schema enforces the granularity;
   and every value prints its provenance (`[static]`, `[ptxas -v]`,
   `[ncu: file.csv]`) so a bound is never mistaken for a measurement.
4. **Launch config and architecture are inputs.** `--arch`, `--launch`,
   `--bind` are required for numeric output; symbolic output never blocks
   on them. Scope normalization (per-thread/warp/CTA → per-CTA/launch) is
   only coherent with blockDim in hand.

**Audience boundary** (stated in README): the target is regular/tiled
kernels — GEMM, conv, stencils, attention — from mainstream producers
(nvcc, clang, Triton). Sparse/irregular kernels whose loop bounds are
data-dependent (CSR SpMV being the canonical case) get honest, labeled
unknowns; that is designed behavior, not a gap awaiting a feature.

**Anti-scope** — the maintenance budget. Each item trades "more useful
in rare cases" against "permanently more surface," and under this
project's goals the trade always resolves the same way. Revisiting any
item means editing this list in the same PR that implements it:
- cache-reuse modeling, divergence, bank conflicts, occupancy/latency
  analysis — NCU owns those; we point at it;
- branch-probability modeling — conditional blocks carry `≤` bounds
  (PR 09), never probabilities;
- SCEV-equivalent generality — trip counts come from a fixed shape
  catalog, grown demand-driven via the coverage floors (§3);
- symbolic series for triangular nests — a note, never a solver;
- guard-implication analysis for loop versioning — variants are
  reported side by side (PR 11), never auto-resolved;
- resolution of data-dependent bounds;
- SASS instruction semantics beyond line-join + `LDL`/`STL`/resource
  counting;
- `cvta`-provenance refinement of generic addressing — deferred until a
  fixture actually emits generic loads (none do today, verified).

## 2. Ground rules

- **Language**: Rust, stable toolchain pinned via `rust-toolchain.toml`,
  edition 2024. v1's classifier comment — "new arms force a compile error
  at every dispatch site" — is native `match` exhaustiveness here; the
  core types (`OpClass`, `Measurement`, `SymExpr`, the affine domain) are
  enums dispatched by pattern matching, with `#[non_exhaustive]` nowhere
  in our own analysis types (exhaustiveness IS the feature).
- **Dependency policy** (allowlist; anything else is justified in its PR):
  `clap` (CLI), `serde`/`serde_json`/`toml`/`csv` (all
  structured I/O), `cpp_demangle` (kernel names), `thiserror` (lib
  errors), `anyhow` (bin only). Dev/tooling: `insta` (unit-level
  snapshots), `proptest` (SymExpr properties; lands with PR 10),
  `cargo-llvm-cov` (coverage), `cargo-fuzz` (PR 21; needs nightly, so
  it lives in non-blocking T4 — the stable pin holds for PR-blocking CI).
  Deliberately NOT used: YAML anywhere (`serde_yaml` is archived/
  unmaintained since 2024 and its forks are worse; every config surface —
  check rules, acceptance manifest, machine tables — is TOML, which the
  Python runner also reads dependency-free via stdlib `tomllib`),
  `petgraph`/graph crates (the CFG uses index
  arenas and hand-rolled textbook algorithms — ~300 LOC we want full
  control over, esp. irreducibility flagging), template engines (the HTML
  emitter is plain `format!`), lexer generators (the lexer is a
  transcription of the C++ reference).
- **The analysis IR is flat, interned, and index-addressed** (in the
  sense of Sampson's "Flattening ASTs", cs.cornell.edu/~asampson/blog/
  flattening.html). No pointer structures anywhere: graphs are index
  arenas (`BlockId(u32)`, `Vec<Block>`, edges as ids), instructions
  live in per-kernel `Vec<Instr>`, operands in a module-level
  `Vec<Operand>` referenced by spans (this retires, by construction,
  v1's documented `shared_ptr<Operand>` recursive-variant wart), and
  all mnemonics/modifiers/identifiers are interned to `Symbol(u32)` in
  a module-owned, hand-rolled interner (~50 LOC, no new dependency) —
  the classifier matches on integers. All indices are newtypes, never
  raw `u32`. The fit is ideal: the IR is built once by the parser and
  read-only ever after (no transformation passes), so flattening's
  costs (mutation, arena sizing) don't apply, while its benefits do —
  trivial ownership (the borrow-checker friction disappears as a
  design invariant, not a discovered fix), uniform index-based
  provenance end-to-end, and cheap alloc/drop for fuzz throughput.
  Only the **report tree** is ergonomic/owned (resolved strings,
  nested structs) — it is what `--json` serializes, so flat-IR indices
  never leak into user-facing output. The AST/interner own their
  strings (PTX inputs are ≤ a few MB; no lifetime threading).
- **Error policy**: the library never panics on malformed input — all
  frontend paths return `Result`; the fuzz gate (PR 21) enforces this.
  Per the honesty principle below, recoverable analysis failures are not
  errors; they are *named unknowns* in the result.
- **Build/test**: cargo only. `ci.sh` = `cargo fmt --check`, `cargo
  clippy -- -D warnings`, `cargo test`, then `python3 tests/golden/run.py`
  (golden + acceptance tiers). The Python runner is kept deliberately —
  it is implementation-language-neutral, so the acceptance suite would
  survive even another rewrite. FileCheck is NOT used.
- **CLI**: one binary `ptxroof` (thin `main.rs` over the library crate),
  subcommands `analyze | diff | check | annotate | capabilities`. JSON
  output (`--json`)
  from PR 12 onward; text and JSON are views over the same result struct
  (`serde::Serialize` on the result tree is the JSON emitter).
- **Honesty principle** (inherited from v1): every analysis either
  succeeds verifiably or degrades to a *named, visible* unknown. No
  silent zero-measurement paths — the v1 `AsyncCopy{bytes unset} → 0
  measurements` bug class is structurally outlawed: every dropped or
  unquantifiable item increments a reported counter with a reason.
- **PR conventions**: each PR lands green (`./ci.sh`), ticks its
  checklist entry here, and updates the acceptance manifest if it flips a
  scenario. Target size 150–600 LOC excluding fixtures.
- **Review protocol (artifact-only)**: the human review surface of a PR
  is its *data diffs* — manifest flips, coverage-floor deltas,
  classification-table/allowlist diffs, golden diffs — never the code.
  Complement: a PR that claims to be a refactor must show a **zero
  golden delta**, which is a machine-checked proof of behavior
  preservation. This is the honesty principle applied to the project
  itself: every capability gained or lacking must be visible in a
  derived, diffable artifact; anything reviewable only by reading code
  gets restructured until it isn't.

## 3. Testing strategy

Four tiers; the first three run hermetically (no compiler, no GPU, no
CUDA) and are PR-blocking.

**T1 — unit (`cargo test`).** Inline `#[cfg(test)]` module tests for
algorithms in isolation: lexer token tables, dominators on textbook
graphs, trip-count matcher on the canonical loop shapes, affine evaluator
on 5-line snippets, rules engine on in-memory results. Cross-module
integration tests live in `tests/*.rs`. Where a structure is easier to
review as a snapshot (AST dumps, loop trees), use `insta` — snapshot
review is `cargo insta review`, and snapshots are committed.

**T2 — golden (Python runner).** Run the `ptxroof` binary on committed
fixtures, assert on output. Two assertion forms:
- `expected.json` — **subset matcher**: only asserted fields are compared
  (additions never break goldens; behavioral changes always do).
- `expected.greps` — ordered substring assertions on the text report, for
  human-facing lines (verdicts, warnings).

**T3 — acceptance (the five scenarios).** Committed up front in PR 02 as
an executable spec, tracked in `tests/acceptance/manifest.toml` with
status `xfail` or `pass`. The runner enforces both directions: an `xfail`
test that passes is an error (forces the flip), a `pass` test that fails
blocks the PR. Development progress *is* the sequence of xfail→pass flips.

**T4 — live matrix (scheduled, non-blocking; PR 21).** Recompiles the
fixture sources with whatever toolchains are present (host nvcc 13.2,
clang/LLVM trunk), runs relaxed invariant rules (the `check` DSL with
tolerant bounds), runs the v1↔v2 differential harness, and runs the
cargo-fuzz targets for a fixed wall-clock budget. Catches toolchain
drift; never blocks a PR.

**Coverage floors (cross-cutting).** The analyzer emits per-run coverage
stats in the result JSON — % of instructions classified non-Unknown, %
of loops with resolved trip counts, % of global accesses with a known
pattern. The golden runner aggregates these across the whole corpus and
enforces floors recorded in `tests/acceptance/manifest.toml`; each floor
activates at the PR that lands its analysis (classification at PR 08,
trip counts at PR 11, access patterns at PR 17). This makes "useful in
the majority of cases" a tested number rather than an aspiration, and
it closes the demand-driven maintenance loop: when a future toolchain
changes an idiom, the regen diff shows the floor metric dropping and
names exactly which new shape has earned a place in the catalog —
before any user files a bug. Floors are **ratchets**: after each PR the
floor auto-rises to the achieved value (visible as a manifest diff), so
corpus coverage can only go up — a regression is a CI failure, never
something a human must notice.

**Conservation gates (cross-cutting).** The runner enforces accounting
identities on every fixture's `--json` output — no fixture-specific
expectations, no human review: `classified + allowlisted-unknown =
total instructions` (every instruction accounted exactly once — the
check v1's AsyncCopy silent-zero bug would have failed); `Σ per-block =
kernel totals`; `per-loop per-iteration × trip expr = flat count`
whenever trips are fully bound; every Measurement's provenance index
resolves to a real instruction. Each identity registers as its analysis
lands (PR 08, 09, 12). This is the mechanism that catches
*implementation inconsistency* — two code paths that disagree — without
anyone reading the code.

**Fixture policy.**
- Every fixture has a provenance header (source file + git rev, compiler
  version string, exact flags, date) and a `regen.sh`. Regeneration is a
  reviewable event: when a toolkit upgrade changes fixture content, that
  diff is information, not noise.
- **Doctored fixtures are first-class.** PTX/SASS are text; negative
  cases are manufactured by hand-editing committed copies (strip `.loc`,
  swap `mma.sync` for `fma.rn` sequences, perturb a stride constant,
  inject `STL`/`LDL` lines into SASS). Doctored files carry a
  `// DOCTORED:` header stating base fixture and edit.
- **Transcription fidelity gate.** PRs 03/04/08 transcribe the v1 C++
  lexer/parser/classifier. The C++ files are the reference spec; the
  corpus-wide gates (lex-all, parse-all, classify-all) plus the v1
  differential (PR 21) pin the transcription. The C++ reference is not
  deleted until PR 22.
- Environment pins for initial generation (recorded per fixture):
  CUDA 13.2 (`V13.2.78`), ptxas/nvdisasm/cuobjdump from same, clang
  against local LLVM trunk, Triton 3.5.1, GPU RTX 4090 (sm_89) for NCU
  captures. PTX/SASS fixtures target sm_80 and sm_90 (cross-compilation
  needs no matching GPU; only NCU capture runs on the 4090).
- **Template instantiation**: the ladder kernels (k5/k11/k12/k14) are
  C++ templates; each fixture wrapper `.cu` explicitly instantiates one
  configuration, recorded in provenance. (Verified: including the bare
  header yields PTX with no kernel body at all.)
- **PTX ISA version span is deliberate**: nvcc 13.2 emits
  `.version 9.2`, clang trunk `.version 8.8`, Triton 3.5.1
  `.version 8.7` (all verified locally). The frontend corpus gates run
  across all three producers.
- **Official references**: the PTX ISA manual and CUDA Binary Utilities
  doc are fetched by `tools/fetch-manuals.sh` into an untracked `refs/`
  directory (NVIDIA copyright — cite, don't commit). Grammar and
  programming-model claims below were checked against PTX ISA 9.2 and
  local experiments on 2026-06-12; load-bearing ones carry "verified"
  notes.

## 4. Acceptance suite (the five scenarios)

| ID | Scenario | Fixtures | Key assertions | Flips at |
|----|----------|----------|----------------|----------|
| S1 | Blocktiling design verification | `k5` (= `test/5_2d_blocktiling.cuh`) sm_80 PTX | nested loop tree w/ source lines; trips `K/8`; unroll detected; per-iter flops/bytes; AI(global)=32.0; per-arch knee verdicts (sm_80 compute-bound, sm_86 memory-bound) | PR 13 |
| S2 | Spill regression diff | `k12` PTX + two SASS builds of the *same* PTX: default vs `ptxas --maxrregcount=32` (real spills, not doctored) | static columns identical; SASS column: reg + spill-bytes delta; per-loop attribution of `STL`/`LDL` | PR 15 |
| S3 | Precision/pipe audit | `k2` (= `test/2_coalesced.cuh`) sm_80 PTX | pipe table: f32 cuda-core only, 0 f16; `cvt` overhead counted; load A uniform/broadcast, load B coalesced @ 2 B width w/ transaction note | PR 17 |
| S4 | Black-box Triton kernel | Triton 3.5.1 matmul w/ one strided operand (generator script committed), no-`.loc` doctored variant | parses; structural loop naming without `.loc`; trips honestly `unknown`; strided access flagged with stride expr | PR 17 |
| S5 | CI check / toolchain regression | `k14` (= `test/14_ldmatrix_mma.cuh`) sm_90 PTX + doctored copy with `mma.sync` removed + `rules.toml` | original passes; doctored fails exit-code 1 with message naming loop + `tensor_flops_per_iter: got 0` | PR 18 |

Table values are design intent; goldens are authored from fixture
reality at PR 02. Verified against real nvcc 13.2 output so far: k2's
K-loop is unrolled ×4 with a `.pragma "nounroll"` remainder loop, so S3
sees main-loop trips `(K − K mod 4)/4` (4 `fma` + 8 `cvt` + 8 loads per
iteration) plus a remainder loop trips `K mod 4` — not a single
trips-K loop. S3's access-pattern claims were hand-verified in the
emitted PTX: A's address derives from `2·(K·row)` (lane-invariant
product, non-affine), B's carries tid.x coefficient 2 B. k5's inner dot
loop is partially unrolled; k11/k12/k14 carry `.maxntid` from
`__launch_bounds__`. wmma role discrimination by first modifier
(`wmma.load.a` / `wmma.store.d` / `wmma.mma`) and the
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` /
`ldmatrix...x2.trans.shared.b16` grammars are confirmed in k11/k14
output.

## 5. Repository layout

```
v2/
├── PLAN.md                  # this file; checklist ticked per PR
├── Cargo.toml               # lib `ptxroof` + bin `ptxroof`
├── rust-toolchain.toml
├── ci.sh                    # fmt, clippy -D warnings, test, golden runner
├── src/
│   ├── main.rs              # thin: clap dispatch into the library
│   ├── lib.rs
│   ├── core/                # flat IR: Module/Kernel/Block/Instr + operand
│   │                        #   arena, interner/Symbol, newtype ids, SourceLoc,
│   │                        #   SymExpr, Measurement, Result tree (Serialize)
│   ├── parse/               # lexer.rs, ast.rs, parser.rs (transcribed from lib/PTX)
│   ├── cfg/                 # graph.rs (index arenas), dominators.rs, loops.rs,
│   │                        #   identity.rs (stable IDs, demangling)
│   ├── classify.rs          # instruction → semantic record (enum, exhaustive match)
│   ├── affine/              # eval.rs, trips.rs, coalesce.rs
│   ├── machine.rs           # loader for data/machine/*.toml
│   ├── sass.rs              # nvdisasm / resource-usage readers, line join
│   ├── ncu.rs               # CSV import, three-column join
│   ├── report/              # collect.rs, stats.rs, text.rs, html.rs
│   ├── check.rs             # rules TOML engine (`toml` crate)
│   └── diff.rs              # per-loop delta on stable IDs
├── data/machine/            # per-SM peak/BW tables (TOML, sources cited inline)
├── data/ptx_opcodes_*.txt   # canonical instruction inventory per ISA version (PR 02)
├── fuzz/                    # cargo-fuzz targets (PR 21): fuzz_lexer, fuzz_parser
├── tests/                   # cargo integration tests (*.rs) — and, as plain data:
│   ├── fixtures/<name>/     # {src ref, *.ptx, *.sass, *.csv, regen.sh, DOCTORED}
│   ├── golden/              # run.py + per-case expected.json / expected.greps
│   └── acceptance/          # manifest.toml + scenario test defs (S1–S5)
└── tools/
    ├── regen-fixtures.sh    # drives all fixture regen.sh, container-pinnable
    ├── extract-opcodes.py   # ISA manual → data/ptx_opcodes_*.txt inventory
    └── triton_fixture.py    # Triton kernel generator for S4
```

---

## 6. The PR train

Each PR: **Goal / Contents / Tests / Done when**. Phases group PRs;
scenario flips are marked ★.

### Phase 0 — Foundations

**PR 01 — Scaffold and test harness.** (~200 LOC)
- Contents: cargo package (lib + bin), `rust-toolchain.toml`, `ci.sh`
  (fmt --check, clippy -D warnings, test, golden runner); `clap` skeleton
  with the five subcommands stubbed; `tests/golden/run.py` with the
  JSON-subset matcher, greps, xfail-manifest support, the
  coverage-floor mechanism (reads floors from the acceptance manifest;
  each floor activates when its analysis lands, then ratchets — §3),
  and the conservation-gate hook (the §3 accounting identities, run on
  every fixture's JSON; identities register as their analyses land) —
  all tested against the stub binary.
- Tests: one trivial unit test; runner self-tests (subset matcher
  semantics: missing field fails, extra field passes; xfail-that-passes
  errors); clippy clean.
- Done when: `./ci.sh` green from a clean checkout with only rustup +
  python3 ≥ 3.11 (stdlib `tomllib` reads the manifest — the runner has
  zero third-party deps) — no CUDA, no LLVM, no C++ toolchain.

**PR 02 — Fixture corpus + acceptance spec.** (scripts ~150 LOC + fixtures)
- Contents (language-neutral): `regen.sh` per fixture with provenance
  headers and explicit template-instantiation wrapper `.cu` files;
  committed PTX for `k1, k2, k5, k11, k12, k14` at sm_80/sm_90 (nvcc
  13.2 `-ptx -lineinfo`, plus clang-trunk variants — verified working
  against the CUDA 13.2 headers with only a `-Wunknown-cuda-version`
  warning); `tools/triton_fixture.py` + committed Triton PTX (verified:
  Triton 3.5.1 + torch 2.9.1 on the 4090 produces PTX with `cp.async` +
  `mma.sync` idioms, param `.ptr .global` attributes, and `.loc` mapping
  to the *Python* source); `tools/fetch-manuals.sh` +
  `tools/extract-opcodes.py` → `data/ptx_opcodes_<ver>.txt`, the
  **canonical instruction inventory** per ISA version, derived from the
  manual's instruction-set chapters and committed with provenance — a
  future PTX-version bump becomes a one-screen list diff naming every
  new instruction, each of which must then be assigned a bucket (PR 08's
  cross-check enforces this); hand-written micro
  fixtures (`micro/` — single-loop, branchy, irreducible, no-loc);
  S1–S5 acceptance defs, all `xfail` in `manifest.toml`, plus the
  coverage-floor table (entries inactive until their analyses land).
- Tests: fixture lint (a runner check): every fixture has provenance
  header + regen.sh; manifest loads; opcode inventory non-empty and
  regenerates to a no-op diff; all five scenarios runnable-and-
  xfailing against the stub.
- Done when: corpus committed, regen reproducible (`regen.sh` rerun is a
  no-op diff modulo date), S1–S5 visible as xfail in the runner output.

### Phase 1 — Frontend

**PR 03 — Lexer.** (~350 LOC; transcription of `lib/PTX/Tokenizer.cpp`)
- Contents: hand-rolled lexer (no lexer-generator dep) extended for full
  programs: the complete directive set from PTX ISA §11 (`.version
  .target .address_size .entry .func .param .reg .loc .file .pragma
  .branchtargets .calltargets .callprototype .section .maxntid .reqntid
  .maxnreg .minnctapersm .visible .extern .weak .common .global .shared
  .local .const .align .noreturn` …), `::`-bearing identifiers
  (`shared::cluster`, `mbarrier::complete_tx`), vector/brace operands,
  immediates (hex, float bit-patterns `0f3F800000`, the `WARP_SZ` named
  constant), labels, predicates (`@%p`, `@!%p`). Token carries byte-span
  into the owned source buffer.
- Tests: T1 table-driven token cases (each token kind; the `::` and
  bit-pattern-float cases pinned); T2 gate: *every* fixture lexes with
  zero error tokens.
- Done when: lex-all-fixtures gate green.

**PR 04 — Parser → Module AST.** (~500 LOC; transcription of
`lib/PTX/Parser.cpp`, extended to full programs)
- Contents: recursive descent: module directives, kernel signatures +
  param tables (name/type/offset, **including pointer attributes**
  `.ptr .global .align N` — Triton emits these and they carry free
  state-space facts), `.reg` decls, `.extern .shared .b8 name[]`
  (dynamic smem, empty brackets), body statements (mnemonic + modifier
  list + operands), `{ … }` statement blocks (inline-asm expansions and
  scoped `.reg` decls — verified present in the very first nvcc
  fixture), labels, **extended `.loc`** (`.loc f l c` and `.loc f l c,
  function_name $sym, inlined_at f l c` — both forms appear in k2),
  `.pragma "nounroll"` in body position, `.section .debug_* { .b8 … }`
  data sections (parsed-and-skipped), trailing `//` comments anywhere,
  version tolerance across `.version 8.7–9.x`, per-statement error
  recovery (a bad statement yields an `Stmt::Unparsed` node, never a
  panic and never poisons the kernel). **Flat representation per the §2
  ground rule**: per-kernel `Vec<Instr>`, module-level operand arena
  referenced by spans (vector/address operands nest via `OperandId`,
  not `Box`), string interner (`Symbol(u32)`) for mnemonics, modifiers,
  and identifiers. Canonical dumper (`--dump-ast`) resolves symbols —
  it is the human/debug view of the flat IR.
- Tests: T1 snippet units incl. error recovery, plus interner units
  (idempotent intern, resolve round-trip); `insta` snapshots of the
  AST for two micro fixtures — snapshots go through the dumper, so
  goldens stay readable, no raw indices; T2: parse-all-fixtures gate
  (zero `Unparsed` statements, or listed in `parse-allowlist.txt`);
  dump→reparse→dump idempotence on all fixtures.
- Done when: both gates green; param tables for `k2`/`k5` match the known
  signatures in golden files.

### Phase 2 — Structure

**PR 05 — CFG.** (~250 LOC)
- Contents: index-arena graph (`BlockId(u32)`, `Vec<Block>`); leaders/
  edges from labels, `bra`, `@%p bra` (target + fallthrough), `brx.idx`
  via `.branchtargets`, `ret`/`exit` terminators; `call` noted
  (intra-procedural CFG; non-inlined callees flagged as a visible unknown
  per ground rules).
- Tests: T1 on `micro/` snippets (fallthrough into label, cond-branch
  diamond, brx table, unreachable block); T2: pinned block/edge counts
  per ladder fixture.
- Done when: goldens pinned and green.

**PR 06 — Dominators + loop forest.** (~300 LOC)
- Contents: Cooper-Harvey-Kennedy dominators over the index arena; back
  edges; natural loops; nesting forest; **irreducible-region detection →
  flagged `unknown-multiplicity`, never guessed**. Hand-rolled (no graph
  crate) — we want line-item control of the irreducibility path.
- Tests: T1 textbook graphs (incl. the classic irreducible two-entry
  loop); T2: golden loop trees for the ladder (`k1`: 1 loop; `k5`: 2
  nested @ known lines; `k12`: pipeline loop) and for `micro/irreducible`
  asserting the honest flag; `insta` snapshot of the `k5` loop tree.
- Done when: ladder loop trees match goldens.

**PR 07 — Identity and naming.** (~200 LOC)
- Contents: stable loop IDs with resolution order source-line (`.loc`) →
  structural path (`kernel/loop[i]/loop[j]`) → raw label; kernel-name
  demangling via `cpp_demangle`; param-table printout (the `--bind` UX
  foundation).
- Tests: T1 ID resolution incl. ties; demangle cases for the ladder's
  mangled names; T2: `k5` loops named by source line; doctored `k5-noloc`
  falls back to structural path (this doctored fixture also serves S4's
  no-`.loc` assertion later).
- Done when: same logical loop gets the same ID across the
  `k5`/`k5-noloc` pair (structural path agrees).

### Phase 3 — Semantics

**PR 08 — Instruction classifier.** (~500 LOC; transcription + major
extension of `lib/PTX/Classifier.cpp`)
- Contents: `enum OpClass` with arms for {cuda-core flop, **SFU op**
  (`rcp/sqrt/div/ex2/lg2/sin/cos` — distinct pipe with an explicit,
  documented FLOP policy; the softmax fixture exercises it), tensor op,
  non-flop arith (incl. `cvt` — 8 per main-loop iteration in k2, so it
  materially affects instruction-mix reporting), memory, async-copy,
  **atomic** (`atom`/`red`, counted load+store per v1 policy), sync,
  control, ignore, unknown}, carrying {precision, packed-lanes, space,
  direction, scope, bytes-or-unknown}; **`Space::Generic` for `ld`/`st`
  with no state space** — PTX ISA: "If no state space is given, perform
  the load using Generic Addressing" — reported as its own honest
  bucket, refinable later via `cvta` provenance; `.shared` sub-qualifier
  handling (`::cta` default per ISA, `::cluster` distinct); every
  consumer dispatches by exhaustive `match`; `wmma` **role split by
  first modifier** (load/store/mma — verified grammar in k11; fixes
  v1's 8192-phantom-FLOP bug by construction); `mma/wgmma` shape tables
  (verified: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`);
  `cp.async` with explicit size operand (bytes statically present —
  verified in Triton output), `commit_group`/`wait_group` as sync arms,
  bulk/TMA with optional bytes → *visible* unquantified counter;
  predicated non-branch instructions get a `predicated` qualifier.
- Tests: T1 table-driven per family (one case per modifier axis that
  changes the answer); the v1 wmma bug as a pinned regression test
  (`wmma.load.*` ⇒ memory, 0 FLOPs); T1 **inventory cross-check**:
  every mnemonic in the classifier tables and in
  `classify-allowlist.txt` exists in the canonical inventory from
  PR 02 — a typo'd table entry is a CI failure, not a match arm that
  silently never fires — and the same join derives the three-way
  partition (classified / unknown-by-policy with reason /
  not-yet-handled) that PR 12's capability report renders; T2 **corpus
  coverage gate**: every
  instruction in every fixture classifies non-Unknown or appears in
  `classify-allowlist.txt` — additions to the allowlist require review.
  Classification coverage floor activates in the manifest; the
  `classified + allowlisted-unknown = total` conservation gate registers.
- Done when: coverage gate green over the full corpus incl. Triton
  fixture.

**PR 09 — Measurement v2 + collection + Stats.** (~300 LOC)
- Contents: `Measurement{kind, pipe, precision, scope, space, direction,
  count: SymExpr(=const for now), provenance: instr index}` (serde-
  derived); per-block collection; `Stats` filter queries (v1 design kept;
  filters gain `pipe`); unquantified/unknown counters carried with
  reasons. Execution counts carry an **`exact` vs `at_most` qualifier**:
  a conditional block inside a loop body executes a data-dependent
  fraction of iterations, so its count is an upper bound, and every
  aggregate derived from it inherits the marker (rendered `≤` in
  reports). One bit, propagated mechanically — the alternative, branch
  probabilities, is anti-scope.
- Tests: T1 Stats filter semantics (incl. the documented soft-filter
  rule) and qualifier propagation (conditional-in-loop block ⇒ `at_most`
  on every aggregate it touches; loop-only nesting stays `exact`); T2:
  per-block flop/byte goldens for `k1`/`k2`; cross-check totals for `k2`
  against hand-computed values in the golden file comment.
- Done when: `k2` per-block numbers match hand calculation.

### Phase 4 — Symbolic counts → first useful report

**PR 10 — SymExpr mini-library.** (~250 LOC)
- Contents: `enum SymExpr` — symbols (params, opaque loop-trip symbols),
  products, sums/differences, integer ceil-/floor-div, **mod-by-power-
  of-two** (PTX lowers `K mod 4` as `and.b32 r, K, 3` — verified in k2's
  unroll bounds; real trip counts are shapes like `(K − K mod 4)/4`, so
  the pure `c·Π(sym)` form from the first draft is provably too weak),
  scalar multiply, substitution/binding, ordered printing. Still
  deliberately not a CAS — exactly the forms the trip matcher emits.
- Tests: T1 only — algebra, binding, printing stability, div edge cases;
  plus `proptest` properties (the dev-dep lands here): for random
  expressions and bindings, simplification/binding preserve the evaluated
  value, and printing is deterministic — example-based tests under-cover
  an algebraic domain.
  Coverage measured with `cargo llvm-cov`; this module is small enough to
  insist on 100% branch coverage.
- Done when: coverage target met; printing is deterministic.

**PR 11 — Trip-count matcher.** (~400 LOC)
- Contents: per-loop: latch-condition extraction (`setp`+`@%p bra`),
  induction recognition (single in-loop def `add r, r, const`), and a
  **scalar affine tracer** shared with PR 16 (lives in `affine/`):
  real nvcc latches compare *derived* registers, not the IV — k2's main
  loop exits on `setp.ne.s32 %p6, %r29, 0` where `%r29 = %r7 + %r35`,
  `%r7 = (K&3) − K`, `%r35` is the IV — so the matcher must normalize
  the latch condition as an affine expression of the IV and loop
  invariants, tracing through `add/sub/and-mask/mul.wide/mad.lo/shl` to
  `ld.param`/constants. Recognizes the **unroll main+remainder pair**
  idiom as one logical loop (main `(K − K mod 4)/4`, remainder
  `K mod 4`), reported linked. Result `SymExpr` or named opaque symbol
  with reason. **Loop-variant detection, minimal form**: sibling loops
  sharing a `.loc` source line are reported as variants — the unroll
  main+remainder pair is the summable special case (linked); anything
  else is flagged `variants — not summed` and excluded from kernel
  totals. No guard-implication analysis (anti-scope): we surface the
  ambiguity instead of resolving it. Per-register tracer state is a
  dense `Vec` indexed by a `RegId` newtype — PTX declares counted
  register families (`.reg .b32 %r<38>;`, verified in k2), so state
  arena sizes come straight from the source; no string-keyed maps
  anywhere in the dataflow.
- Tests: T1 on the canonical shapes, one fixture snippet each: rotated
  do-while; `i<N`; `i!=N`; countdown; stride>1; 64-bit IV; derived-
  register latch (the verified k2 shape above, pinned verbatim);
  **pointer-induction latch** (`setp.lt.u64 %p, %ptr, %end` ⇒ trips
  `(end − base)/stride` — clang/LLVM's common strength-reduced form, so
  Triton output hits it); unroll main+remainder pair; multi-exit →
  unknown; data-dependent → unknown; triangular (inner bound = outer
  IV) → symbolic w/ note; doctored versioned-loop fixture (two cloned
  loops, same `.loc`, mutually exclusive guards) → reported as
  variants, totals exclude double-counting. T2: `k2` main loop trips
  `(K − K mod 4)/4` with `K = param 2`, remainder `K mod 4`; `k5` outer
  tile loop trips `K/8`; Triton fixture K-loop. Trip-count coverage
  floor activates in the manifest.
- Done when: ladder trip counts match goldens; every unknown carries a
  reason string.

**PR 12 — `analyze` subcommand.** (~450 LOC)
- Contents: loop-tree report (text + JSON via `Serialize` on the result
  tree): per-loop per-iteration steady-state aggregation (Measurement
  counts × enclosing trip exprs, `≤`-rendering for `at_most` counts,
  variant loops shown side by side and excluded from totals);
  pipe/precision flop table, per-space byte table, unknown/unquantified
  blocks with fix-it hooks; **source-line aggregation for straight-line
  code** — recovers a per-source-iteration view when the compiler fully
  unrolled a loop (every instruction carries the same `.loc`; verified
  on k5's inner BK loop), the universal fallback whenever loop
  structure was transformed away; **small-trip-count warning** when
  bindings make a loop's trips small (boundary code dominates — the
  steady-state headline is redirected); per-run coverage stats in the
  result JSON (consumed by the runner's floors); a **ranked unknown
  histogram** (per file or aggregated over a corpus, weighted by
  symbolic execution count — an unknown at loop depth 2 outranks one in
  the epilogue): the "what to add next" queue is data, not judgment,
  and "your numbers look off on my kernel" triages mechanically;
  `ptxroof capabilities` — the **derived capability report** (PR 08's
  three-way partition + corpus coverage numbers), committed as a golden
  so drift fails CI. The anti-STATUS.md rule: hand-maintained
  capability docs rot (v1's did), so the only inventory of what the
  tool handles is the one the tool generates; `--bind name=value` /
  `--bind idx:name=value`, `--assume 'loop trips=expr'`, assumptions
  echoed in report header; `--kernel` glob.
- Tests: T2 goldens for `k2`, `k5` (symbolic and `--bind K=4096` numeric
  columns); k5 unrolled-inner-loop line-aggregation golden; small-trips
  warning golden (`--bind 2:K=8`); T1 for bind/assume parsing;
  assumption-echo grep; capability-report and unknown-histogram goldens
  (the committed capability report is itself a golden); the remaining
  conservation gates register (Σ per-block = kernel; per-loop × bound
  trips = flat count); S1 partially satisfied but stays xfail (needs
  verdicts).
- Done when: `k5` report shows the S1 numbers except roofline verdicts.

### Phase 5 — Machine model

**PR 13 — Machine model + normalization + verdicts. ★ S1 flips.** (~300 LOC)
- Contents: `data/machine/*.toml` per-SM tables (sm_70/75/80/86/89/90:
  peak FLOPs per pipe×precision, DRAM BW; sources cited in comments),
  serde-loaded; knee computation; `--arch` + `--launch x,y,z` →
  per-CTA/per-launch normalization of mixed-scope measurements; verdict
  lines. **Defaults from the PTX itself where present**: `--arch` from
  the `.target` directive, launch dims from `.reqntid`/`.maxntid`
  (verified: k11/k12/k14 carry `.maxntid` via `__launch_bounds__`);
  explicit flags override, and the report states which source was used.
- Tests: T1 knee math vs hand-computed values from the cited specs; T1
  scope normalization (per-warp byte × blockDim cases); T2: S1 golden
  incl. the two-arch verdict pair (sm_80 compute-bound / sm_86
  memory-bound at AI=32) — **S1 manifest flips to pass**.
- Done when: S1 green; mixed-scope kernel (`k11`) normalizes without the
  v1 flatten-by-32× error (pinned golden).

### Phase 6 — SASS sidecar

**PR 14 — SASS + resource readers.** (~350 LOC + fixtures)
- Contents: `nvdisasm -g` reader — verified format: `//## File "…",
  line N` comment lines interleaved with SASS in the code section
  (leading `.debug_*` sections are raw bytes, skipped); `cuobjdump
  --dump-resource-usage` reader (REG/STACK/SHARED — **verified: it
  reports `LOCAL:0` even when spilling**; spills live in STACK, and
  explicit spill-byte numbers exist only in `ptxas -v` output, so
  `regen.sh` captures `ptxas -v` stderr as a committed `.res.txt`
  alongside the SASS); spill counters from SASS (`LDL`/`STL` widths —
  verified: 1326 in the forced-spill k5 build vs 0 default),
  join-on-(kernel,line) → per-loop SASS metrics; fixtures: SASS twice
  from the *same PTX* — default ptxas vs `--maxrregcount=32` (verified
  on k5: 124 reg/0 spill → 32 reg/2772 B spill stores + 2560 B spill
  loads) — plus a doctored injected-`STL` variant.
- Tests: T1 reader units on committed SASS; doctored-injection test
  (spill counter must move by exactly the injected bytes); T2: per-loop
  spill attribution golden for the maxrregcount build.
- Done when: spill bytes and register counts for both `k12` builds match
  goldens; line join places ≥95% of SASS instructions into a loop
  (remainder reported, not dropped).

**PR 15 — `diff` subcommand. ★ S2 flips.** (~300 LOC)
- Contents: two analysis results → per-loop delta keyed on stable IDs
  (PR 07); added/removed loops reported as such; `--kernel-map old=new`
  for renames; text + JSON.
- Tests: T1 ID-matching edge cases (loop added/removed/renamed); T2: S2
  golden — `k12` default vs maxrregcount build: static columns
  byte-identical, SASS column shows reg/spill delta — **S2 flips**;
  doctored static-change diff (perturbed unroll fixture) as the inverse
  case.
- Done when: S2 green.

### Phase 7 — Affine analysis

**PR 16 — Affine evaluator.** (~450 LOC)
- Contents: extends PR 11's scalar tracer to memory addresses, with a
  **three-tier domain** — this is a verified necessity, not a
  refinement: in k2, A's address is `base + 2·(K·row + k)` where `K·row`
  is a *product of two symbols* (param × ctaid/tid.y expression). A
  two-tier affine-or-⊤ domain (the first draft) classifies that as
  unknown; the correct answer is "uniform across the warp." Tiers:
  1. **affine in lane-varying symbols** (`c₀ + Σ cᵢ·sᵢ`) — yields the
     tid.x stride coefficient;
  2. **lane-invariant, non-affine** — closed under arbitrary arithmetic
     on lane-invariant inputs (param×param, K·row, …);
  3. **⊤ unknown** (data-dependent, `add.cc/addc` carry chains, register
     reuse across joins — all verified-or-ISA-documented forms).
  Handled ops: `mov/add/sub/shl/mul.wide/mul.lo/mad.lo/and-mask/cvt/
  ld.param` + sreg reads (all forms observed in the k2/k5 fixtures);
  merge = equality else demote tier; loop bodies use PR 11's IV facts.
- Tests: T1 snippet battery incl. the S3 pair as it actually appears in
  k2's PTX (A: lane-invariant product → uniform; B: tid.x coeff 2 B) and
  deliberate ⊤ cases (data-dependent index, carry-chain arithmetic,
  register reuse across paths).
- Done when: snippet battery green; evaluator never claims tier 1 or 2
  for the ⊤ cases (asserted).

**PR 17 — Coalescing classification. ★ S3, S4 flip.** (~250 LOC)
- Contents: per global access: lane-stride coefficient ⇒ uniform /
  coalesced / strided(w) / unknown; **lane decomposition is launch-
  config-dependent** (PTX ISA: warps are consecutive linearized thread
  IDs, x-major — when `ntid.x < 32` a warp spans multiple `tid.y`
  values, so the lane coordinate is derived from `--launch` or the
  kernel's `.reqntid`/`.maxntid`, and degrades to an honest unknown
  without them); **32-byte-sector** estimate per warp access
  (`min(32, max(1, ceil(32·coeff/32)))`, adjusted for vector width —
  sectors, not 128 B transactions, matching NCU's accounting); report +
  JSON wiring; `all_coalesced` summary fact (for PR 18's DSL).
- Tests: T2: S3 golden on `k2` (A uniform, B coalesced @2 B with
  transaction note; pipe table from PR 08/12 asserted in the same
  scenario) — **S3 flips**; S4 golden on Triton fixture (strided flag w/
  stride expr, structural naming, unknown trips) — **S4 flips**;
  doctored stride-perturbed `k2` must flip B's verdict (mutation test);
  access-pattern coverage floor activates in the manifest.
- Done when: S3+S4 green; mutation test green; floor green.

### Phase 8 — Check, NCU, annotate

**PR 18 — `check` subcommand. ★ S5 flips.** (~300 LOC)
- Contents: rules TOML via the `toml` crate already on the allowlist —
  no YAML, see §2 (kernel/loop selectors by stable
  ID; numeric comparators over any Stats metric; `all_coalesced`;
  instruction-class exists/absent); exit codes 0/1/2
  (pass/fail/usage-error); failure messages name loop, metric, expected
  vs got.
- Tests: T1 evaluator per assertion type incl. threshold boundaries; T2:
  S5 — `k14` passes, doctored mma-removed copy fails exit 1 with pinned
  message — **S5 flips**; rules-file error cases (unknown loop ID names
  candidates, exit 2; malformed TOML, exit 2).
- Done when: S5 green; all five scenarios now `pass` in the manifest.

**PR 19 — NCU import + three-column report.** (~250 LOC + fixture)
- Contents: NCU CSV reader via the `csv` crate (kernel-name join,
  demangling-tolerant); transferred/requested ratio per kernel;
  `achieved` column when FLOP metrics present; missing-kernel and
  schema-drift handling (versioned header check, fail loudly on
  mismatch). Fixture: real capture on the RTX 4090 for `k1`+`k5`
  (`ncu --csv` with dram/L2/FLOP metrics), committed with capture
  provenance. **Verified blocker on this machine**: the driver
  restricts counters (`ERR_NVGPUCTRPERM`) — capture requires `sudo ncu`
  or `NVreg_RestrictProfilingToAdminUsers=0`; `regen.sh` documents
  both, and a hand-authored CSV in the pinned schema is the committed
  fallback until a privileged capture is run (provenance header says
  which).
- Tests: T1 reader on committed CSV + truncated/reordered-column
  variants; T2: three-column golden for `k5` (requested from static,
  transferred from CSV, ratio line); unmatched-kernel warning golden.
- Done when: three-column report golden green from committed artifacts
  only (no GPU in CI).

**PR 20 — `annotate` HTML.** (~300 LOC)
- Contents: source↔PTX interleave via `.loc`, per-instruction badges
  (loop depth, class, measurement contribution), uncertainty highlights
  with fix-it hooks; self-contained HTML (inline CSS/JS, plain
  `format!` templating — no engine dependency, no CDN).
- Tests: T2 structural assertions via the Python runner (parse HTML:
  every instruction row badged; every unknown highlighted; anchors per
  loop ID) — not byte-exact.
- Done when: `k5` and Triton-fixture pages render with zero unbadged
  instructions.

### Phase 9 — Hardening and switchover

**PR 21 — Tier-4 live matrix + v1 differential + fuzz.** (~300 LOC scripts
+ fuzz crate)
- Contents: `tools/live_matrix.sh` — recompile ladder sources with host
  toolchains (nvcc 13.2, clang/trunk), run relaxed invariant rules
  (`rules-invariant.toml`: spill==0 where designed, tensor pipe engaged,
  all_coalesced where designed, trips symbolic in K); **v1↔v2
  differential**: run v1 (`llc -load NVPTXArithIntensity.so`) and v2 on
  the same compiles of `smoke/fsub/fma/add_f64/local`, assert exact
  agreement on flops, precision buckets, and global/local bytes;
  `cargo-fuzz` targets `fuzz_lexer` + `fuzz_parser` (mutational, corpus
  seeded from fixtures; smoke-length run wired into T4, longer runs
  scripted) — gate: no panics, ever, on arbitrary bytes. The flat IR's
  bump-allocate / drop-as-one-block lifecycle keeps per-iteration cost
  low (allocator churn was 38% of runtime in the flattening post's
  measurement), which translates directly into fuzz throughput and
  therefore coverage.
- Tests: the PR *is* tests; runner gains `live` and `differential` labels
  excluded from default runs.
- Done when: differential green on all five shared cases; one full live
  run + a 1-hour fuzz run recorded in the PR description.

**PR 22 — Docs + v1 maintenance mode.** (~doc-only)
- Contents: `v2/README.md` (install via `cargo install --path`, the four
  verbs, fixture/regen policy, limitations section mirroring the §1
  audience boundary and anti-scope list verbatim — users read the same
  honesty contract the code enforces); user guide with the five
  scenarios as worked examples
  (they're real now — outputs pasted from the acceptance goldens);
  `cargo doc` builds clean with `#![warn(missing_docs)]` on the library;
  top-level `STATUS.md` note: v1 frozen, v2 canonical; the C++ reference
  files under `lib/PTX/` annotated as superseded.
- Tests: doc lint (links resolve; CLI examples in docs are executed by a
  runner case and must exit 0 — docs can't rot silently).
- Done when: a newcomer can go from `git clone` to S1's report using only
  the README.

---

## 7. Dependency graph (what can proceed in parallel)

```
01 → 02 → 03 → 04 → 05 → 06 → 07 ─┬→ 08 → 09 ─┬→ 12 → 13★S1
                                   │           │
                                   │   10 → 11 ┘      14 → 15★S2
                                   │                  (14 needs 07,09)
                                   └→ 16 → 17★S3,S4   18★S5 (needs 12)
                                                       19 (needs 12)
                                                       20 (needs 12,17)
                                                       21 (needs all★)
```
PRs 10–11 (symbolics) can be developed in parallel with 08–09
(semantics); 16 (affine) extends the scalar tracer that PR 11
introduces (they share the `affine/` module — the k2 experiments showed
even basic trip counts need affine tracing, so the tracer cannot wait
for Phase 7), and its address/lane analysis can overlap Phase 5–6 work.

## 8. Risks and open questions

- **Transcription fidelity** (C++ → Rust for lexer/parser/classifier):
  mitigated by the corpus-wide gates landing in the same PRs and the v1
  differential in PR 21; the C++ reference stays in-tree until PR 22.
- **Rust idiom frictions are resolved by the flat-IR ground rule (§2)**,
  not discovered mid-build: no pointer structures means no graph/
  lifetime fights with the borrow checker. The residual cost is index
  opacity in tests and debuggers — mitigated by newtype ids everywhere
  and by routing all snapshots/goldens through the symbol-resolving
  dumper. If contributor Rust fluency is still ramping, budget Phases
  1–2 at reduced velocity — the frontend is the gentlest terrain to
  learn on, and the gates catch semantic drift regardless.
- **Param naming is positional** (`_param_2`) without debug info. Bound
  via `--bind 2:K=...`; the param-table printout (PR 07) is the
  mitigation. Revisit if `.loc`-adjacent DWARF gives real names.
- **Triton PTX idioms** (heavily predicated masked loads, `v4` ops) may
  stress PR 08's families; the corpus gate will say so precisely. Budget
  one allowlist-review cycle in PR 08.
- **wgmma/TMA bytes** are descriptor-driven and statically unknowable;
  the design answer is the visible unquantified counter + `--bind-bytes`
  (add to PR 12's `--assume` family when first needed; not speculatively).
- **ptxas version skew** between fixture SASS and user SASS is real;
  that's why SASS fixtures are committed with provenance and tier-4
  re-derives them live.
- **NCU schema drift** across versions: PR 19 pins the captured header
  and fails loudly on mismatch rather than guessing column meaning.
- **NCU counter permission** (verified failing on this machine):
  `ERR_NVGPUCTRPERM` unless run privileged or the driver is configured
  with `NVreg_RestrictProfilingToAdminUsers=0`. Capture is a manual,
  documented step; CI never needs it (committed CSV).
- **Generic addressing** (`ld`/`st` with no state space) is legal PTX
  and unclassifiable to a concrete space without provenance; it gets
  its own honest bucket from day one. None of the current fixtures emit
  it (verified) — the risk is producer-dependent, so the corpus gate
  will flag when it first appears.
- **Loop ID stability under heavy unrolling** (loop disappears): `diff`
  reports removed-loop honestly; acceptance has no case where we pretend
  to match.
- **Dependency creep**: the allowlist in §2 is the budget; serde-family +
  clap + cpp_demangle is already the bulk of compile time. Any proposed
  addition states what it replaces and why hand-rolling is worse.

## 9. Checklist

- [x] PR 01 — scaffold + harness (cargo, clap stub, golden runner)
- [ ] PR 02 — fixtures + acceptance spec (S1–S5 xfail)
- [ ] PR 03 — lexer
- [ ] PR 04 — parser/AST
- [ ] PR 05 — CFG
- [ ] PR 06 — dominators + loop forest
- [ ] PR 07 — identity/naming
- [ ] PR 08 — classifier (+ corpus gate)
- [ ] PR 09 — Measurement v2 + Stats
- [ ] PR 10 — SymExpr
- [ ] PR 11 — trip counts
- [ ] PR 12 — `analyze`
- [ ] PR 13 — machine model + verdicts ★S1
- [ ] PR 14 — SASS readers
- [ ] PR 15 — `diff` ★S2
- [ ] PR 16 — affine evaluator
- [ ] PR 17 — coalescing ★S3 ★S4
- [ ] PR 18 — `check` ★S5
- [ ] PR 19 — NCU import
- [ ] PR 20 — `annotate`
- [ ] PR 21 — live matrix + v1 differential + fuzz ★(no-panic gate)
- [ ] PR 22 — docs + switchover
