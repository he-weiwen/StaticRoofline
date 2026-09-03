# v2 (`ptxroof`) — Implementation Plan (Rust)

A ground-up rewrite of the analyzer with PTX text as the input substrate,
implemented in Rust. Lives entirely under `v2/`. v1 — the LLVM
`MachineFunctionPass` under `lib/` and `test/`, with its `STATUS.md`,
`TUTORIAL.md`, `docs/` and the CUTLASS MIR corpus — was removed once
Phase 1 shipped (last present at commit 690d81d; the kernel ladder it
held moved to `tests/fixtures/src/`). Anything below that still says
"v1" refers to that history.

The plan has two phases. **Phase 1 (PRs 01–13)** builds the minimal
*genuinely useful* analyzer: `ptxroof analyze` on nvcc PTX, producing a
loop tree with per-iteration flops/bytes/AI, honest named unknowns, and
a compute-vs-memory verdict — the five Phase 1 acceptance
scenarios of §4, headlined by S1. **Phase 2** is a
demand-driven backlog: every other capability, each held until a real
occasion triggers it, never built speculatively. The source tree
contains only what Phase 1 needs; extensibility lives in the
architecture (enums, the verb-subcommand CLI, the registry hooks), not
in dormant code.

Originally a companion to v1's `STATUS.md` and
`docs/measurement-refactor.md` (v1's value-stream refactor, whose
Measurement contract this design keeps and extends); both are in git
history at 690d81d. This revision supersedes the earlier C++17 draft of this
plan: with the LLVM dependency gone, every boundary of the tool is text
(PTX/SASS/CSV in, JSON/HTML out, compilers as subprocesses), so the
implementation language has zero interop surface — and the design's
central types are sum types, which Rust expresses natively. The v1
`lib/PTX/` C++ code (~900 LOC) served as the **reference specification**
for transcription, pinned by shared expected-output tests, not ported.

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
   Phase 1 ships only the requested column — *with* its `[static]`
   label, because the labeling is what keeps a lone static number
   honest; the SASS and NCU columns are Phase 2 items.
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
  catalog, grown demand-driven via the minimum-coverage thresholds (§3);
- symbolic series for triangular nests — a note, never a solver;
- guard-implication analysis for loop versioning — variants are
  reported side by side (PR 11), never auto-resolved;
- resolution of data-dependent bounds;
- SASS instruction semantics beyond line-join + `LDL`/`STL`/resource
  counting;
- `cvta`-provenance refinement of generic addressing — deferred until a
  fixture actually emits generic loads (none do today, verified).
- by-value aggregate kernel parameters — a struct/array param lowered to
  `.param .align N .b8 name[size]`, whose scalar fields the body reads via
  `ld.param [name+offset]`. The param-table layout (size/align/field
  offsets) is dropped at parse (PR 04 keeps name + type only), and the
  trip matcher keys on the base param name with the offset ignored
  (`trips.rs`, `Operand::Memory { base, .. }`) — so a field load would be
  mis-bound to the aggregate's positional index, not flagged. Out of the
  audience above (the corpus passes scalar dims + pointers as separate
  named params) and with no effect on bytes/AI (`.param` loads are not
  global traffic); deferred until a fixture emits one (none today,
  verified). The fix when triggered is local: gate the `ld.param` match on
  offset == 0, emit a named unknown otherwise.

## 2. Ground rules

- **Language**: Rust, stable toolchain pinned via `rust-toolchain.toml`,
  edition 2024. v1's classifier comment — "new arms force a compile error
  at every dispatch site" — is native `match` exhaustiveness here; the
  core types (`OpClass`, `Measurement`, `SymExpr`, the affine domain) are
  enums dispatched by pattern matching, with `#[non_exhaustive]` nowhere
  in our own analysis types (exhaustiveness IS the feature).
- **Dependency policy** (allowlist; anything else is justified in its
  PR; each dependency joins `Cargo.toml` at the PR that first uses it).
  Phase 1 runtime: `clap` (CLI), `serde`/`serde_json` (the result tree
  derives `Serialize`; JSON output *is* that derivation), `toml`
  (machine tables), `cpp_demangle` (kernel names), `thiserror` (lib
  errors), `anyhow` (bin only). Phase 1 dev/tooling: `insta`
  (unit-level snapshots), `cargo-llvm-cov` (coverage). Reserved for
  Phase 2 items: `csv` (NCU import), `proptest` (SymExpr properties),
  `cargo-fuzz` (needs nightly, so fuzzing stays outside PR-blocking
  CI — the stable pin holds).
  Deliberately NOT used: YAML anywhere (`serde_yaml` is archived/
  unmaintained since 2024 and its forks are worse; every config surface —
  check rules, the scenario status file, machine tables — is TOML, which the
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
  frontend paths return `Result`; a Phase 2 fuzz pass enforces this
  mechanically, until then it is a review rule backed by error-path
  unit tests.
  Per the honesty principle below, recoverable analysis failures are not
  errors; they are *named unknowns* in the result.
- **Build/test**: cargo only. `ci.sh` = `cargo fmt --check`, `cargo
  clippy -- -D warnings`, `cargo test`, then `python3 tests/run.py`
  (the CLI-test and acceptance tiers). The Python runner is kept
  deliberately — it is implementation-language-neutral, so the
  acceptance suite would survive even another rewrite. FileCheck itself
  is NOT used (the runner's CHECK lines borrow only its matching
  semantics).
- **CLI**: one binary `ptxroof` (thin `main.rs` over the library
  crate). Phase 1 ships one verb: `analyze`, with JSON output (`--json`)
  from PR 12 onward — text and JSON are views over the same result
  struct. Further verbs (`diff`, `check`, `annotate`, `capabilities`)
  are Phase 2 items; the clap subcommand enum makes each an additive
  change.
- **Honesty principle** (inherited from v1): every analysis either
  succeeds verifiably or degrades to a *named, visible* unknown. No
  silent zero-measurement paths — the v1 `AsyncCopy{bytes unset} → 0
  measurements` bug class is structurally outlawed: every dropped or
  unquantifiable item increments a reported counter with a reason.
- **PR conventions**: each PR lands green (`./ci.sh`), ticks its
  checklist entry here, and updates the scenario status file
  (`status.toml`) if it changes a scenario's status. Target size 150–600
  LOC excluding fixtures.
- **Review protocol (artifact-only)**: the human review surface of a PR
  is its *data diffs* — scenario-status changes (xfail → pass),
  minimum-coverage raises, classification-table/allowlist diffs,
  expected-output diffs — never the code. Complement: a PR that claims
  to be a refactor must show **zero expected-output changes**, which is
  a machine-checked proof of behavior preservation. This is the honesty principle applied to the project
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

**T2 — CLI tests (Python runner).** Run the built `ptxroof` binary on
committed inputs, compare against committed expected outputs. Two forms:
- `expected.json` — **partial comparison**: only the fields present in
  the expected file are compared (new report fields never break old
  tests; behavioral changes always do).
- `expected.checks` — **CHECK lines**: substrings that must appear in
  the text report in order — the matching semantics of LLVM FileCheck's
  `CHECK:` directives — for human-facing lines (verdicts, warnings).

**T3 — acceptance scenarios.** Tracked in
`tests/acceptance/status.toml` with status `xfail` ("expected failure",
the LLVM lit / pytest term) or `pass`. The Phase 1 scenarios (S1, S6–S9) are committed in PR 02 as an
executable spec; S2–S5 (§4) are committed with the Phase 2 items that
implement them — a scenario def lands when its work starts, not
before. The runner enforces both directions: an `xfail` test that
passes is an error (forces the status change), a `pass` test that fails
blocks the PR. Development progress *is* the sequence of xfail→pass
status changes in that file.

**T4 — live matrix (scheduled, non-blocking; Phase 2).** Recompiles the
fixture sources with whatever toolchains are present (host nvcc 13.2,
clang/LLVM trunk), runs relaxed invariant rules (the `check` DSL with
tolerant bounds), and runs the cargo-fuzz targets for a fixed
wall-clock budget. Catches toolchain
drift; never blocks a PR.

**Minimum-coverage thresholds (cross-cutting).** The analyzer emits
per-run coverage stats in the result JSON — % of instructions classified
non-Unknown, % of loops with resolved trip counts, % of global accesses
with a known pattern. The test runner aggregates these across the whole
corpus and enforces the minimums recorded in
`tests/acceptance/status.toml`. Runner-side enforcement needs JSON
reports, which exist from PR 12 — so PR 08/11 land their corpus
coverage as cargo-level checks (classification allowlist, trip-shape
units) and both status.toml entries flip `enforced = true` at PR 12;
Phase 2 analyses add their own entries, e.g. access patterns. This makes
"useful in the majority of cases"
a tested number rather than an aspiration, and it closes the
demand-driven maintenance loop: when a future toolchain changes an
idiom, the regen diff shows the coverage metric dropping and names
exactly which new shape has earned a place in the catalog — before any
user files a bug. Minimums **may only rise**: after each PR the runner
records the achieved value as the new minimum (`--raise-min`, visible
as a status.toml diff), so corpus coverage can only go up — a
regression is a CI failure, never something a human must notice.

**Report verifier (cross-cutting).** In the spirit of LLVM's IR
verifier — invariants any correct implementation must satisfy, checked
mechanically — the runner enforces accounting identities on every
fixture's `--json` output, with no fixture-specific expectations and no
human review: `classified + allowlisted-unknown = total instructions`
(every instruction accounted exactly once — the check v1's AsyncCopy
silent-zero bug would have failed); `Σ per-block = kernel totals`;
`per-loop per-iteration × trip expr = flat count` whenever trips are
fully bound; every Measurement's provenance index resolves to a real
instruction. Each check registers as its analysis lands (PR 08, 09,
12). This is the mechanism that catches *implementation inconsistency*
— two code paths that disagree — without anyone reading the code.

**Fixture policy.**
- Every fixture has a provenance header (source file + git rev, compiler
  version string, exact flags, date) and a `regen.sh`. Regeneration is a
  reviewable event: when a toolkit upgrade changes fixture content, that
  diff is information, not noise.
- **Hand-edited fixtures are first-class.** PTX/SASS are text; negative
  cases are manufactured by hand-editing committed copies (strip `.loc`,
  swap `mma.sync` for `fma.rn` sequences, perturb a stride constant,
  inject `STL`/`LDL` lines into SASS). Hand-edited files carry a
  `// HAND-EDITED:` header stating base fixture and edit.
- **Transcription fidelity.** PRs 03/04/08 transcribe the v1 C++
  lexer/parser/classifier. The C++ files were the reference spec; the
  corpus-wide checks (lex-all, parse-all, classify-all) pin the
  transcription. The C++ reference was removed with the rest of v1
  after Phase 1 (git history, 690d81d); the planned v1↔v2 differential
  is retired with it — the fixtures' committed expected outputs, which
  v1 agreed with at PR 08, are the surviving pin.
- Environment pins for initial generation (recorded per fixture):
  CUDA 13.2 (`V13.2.78`), fixtures target sm_80 (cross-compilation
  needs no matching GPU; since PR 14, k1/k5 are also committed at
  sm_89 — same toolchain, byte-identical bodies, only `.target`
  differs). nvcc output is NOT byte-deterministic by
  default — `-lineinfo`'s `.debug_str` embeds an `_INTERNAL_` module
  token derived from a per-invocation random number (verified: two
  identical compiles differ). Every regen.sh therefore passes
  `-frandom-seed=<fixture-name>`, nvcc's documented fix, which makes
  rerun output byte-identical (verified). Phase 2 producers carry their own verified
  pins: clang against local LLVM trunk, Triton 3.5.1, ptxas/nvdisasm/
  cuobjdump from CUDA 13.2, RTX 4090 (sm_89) for NCU captures.
- **Template instantiation**: the ladder kernels (k5/k11/k12/k14) are
  C++ templates; each fixture wrapper `.cu` explicitly instantiates one
  configuration, recorded in provenance. (Verified: including the bare
  header yields PTX with no kernel body at all.)
- **PTX ISA version span is known**: nvcc 13.2 emits `.version 9.2`
  (the Phase 1 corpus), clang trunk `.version 8.8`, Triton 3.5.1
  `.version 8.7` (all verified locally). The clang and Triton producers
  join the corpus-wide checks with their Phase 2 item.
- **Official references**: the PTX ISA manual and CUDA Binary Utilities
  doc are fetched by `tools/fetch-manuals.sh` into an untracked `refs/`
  directory (NVIDIA copyright — cite, don't commit). Grammar and
  programming-model claims below were checked against PTX ISA 9.2 and
  local experiments on 2026-06-12; load-bearing ones carry "verified"
  notes.

## 4. Acceptance scenarios

Each scenario is a user question turned into an executable spec.

**Phase 1 — `analyze` only; all committed (xfail) in PR 02; all
fixtures are already in the Phase 1 corpus.** One case dir = one
binary invocation, so S1, S7 and S9 fan out into dotted scenario ids
(S1.1/S1.2, S7.1/S7.2, S9.1–S9.3) — nine status.toml entries for five
user questions. S6–S9 flip together at PR 12 (the signal that the
report is real); S1 closes Phase 1 at PR 13 by adding the verdicts;
S1.2 (the sm_89 fan-out) follows in PR 14.

| ID | User question | Fixtures | Key assertions | Lands |
|----|---------------|----------|----------------|-------|
| S1.1 / S1.2 | Is this kernel's design point what I computed on paper? (and: does the verdict follow the part?) | `k5` (= `tests/fixtures/src/5_2d_blocktiling.cuh`, BM=64 BN=64 BK=8 TM=8 TN=8) sm_80 + sm_89 PTX | nested loop tree w/ source lines; trips `ceildiv(K, 8)` (the latch is `setp.lt.u32`, so the general form is ceil-div); fully-unrolled register-tile loops recovered by line aggregation; per-iter flops/bytes; AI(global)=32.0; per-arch knee verdicts (sm_80 compute-bound, sm_86 memory-bound). S1.2: the same kernel at `.target sm_89`, no `--arch` flag — the verdict defaults to the target directive (RTX 4090 table) and lands memory-bound (AI 32 < f32 knee 81.9) | PR 13 / PR 14 |
| S6 | Where does the work go? | `k2` | loops ranked by symbolic weight: main loop `K`-dependent, remainder `K mod 4`; headline names the main loop's source line; unroll main+remainder pair linked as one logical loop (the bet-2 ranking claim, tested) | PR 12 |
| S7.1 / S7.2 | Did tiling pay off? | `k1`, `k5` | two independent runs, no `diff` verb: AI(k1 main loop) = 0.5 flop/B vs AI(k5 tile loop) = 32 flop/B, both shape-independent (no `--bind`) — the contrast is two comparable numbers. (0.5, not the 0.25 the design table first guessed: 8 flops / 16 B per unrolled iteration under the same fma=2 convention that makes k5 = 32.) | PR 12 |
| S8 | Am I on the precision path I think? | `k2` | flop table: f32 cuda-core only, **0 f16 flops** despite `__half` data (compute is converted to f32); 8 `cvt` per main-loop iteration counted as conversion overhead; 2 B loads ×8/iter; one guarded 2 B store in the epilogue (`at_most` — the bounds guard makes kernel totals upper bounds) | PR 12 |
| S9.1 / S9.2 / S9.3 | Does the tool admit what it can't see? | `micro/data_dep`, `micro/branchy`, `micro/no_loc` | data-dependent latch → trips = *named* unknown with reason, totals stay symbolic (never silent zero), trip-coverage stat visibly < 100%; in-loop conditional → `≤` markers propagate to every aggregate; no `.loc` → loops named by label, report complete; exit 0 in all three (unknowns are results, not errors) | PR 12 |

**Phase 2 — each lands with, and is the acceptance test of, its
backlog item; "matters when" is that item's trigger.** Fully designed,
key claims verified against real fixtures; defs are committed only
when the work starts.

| ID | User question | Fixtures | Key assertions | Matters when → lands with |
|----|---------------|----------|----------------|---------------------------|
| S2 | Did this build regress spills? | `k12` PTX + two SASS builds of the *same* PTX: default vs `ptxas --maxrregcount=32` (real spills, not hand-edited) | static columns identical; SASS column: reg + spill-bytes delta; per-loop attribution of `STL`/`LDL` | first spill-regression hunt → `diff` + SASS sidecar |
| S3 | Are my global accesses coalesced? | `k2` sm_80 PTX | load A uniform/broadcast, load B coalesced @ 2 B width w/ transaction note (the precision/pipe assertions this row once carried moved to Phase 1's S8) | first uncoalesced-access suspicion → affine + coalescing |
| S4 | Can it handle a black-box Triton kernel? | Triton 3.5.1 matmul w/ one strided operand (generator script committed), no-`.loc` hand-edited variant | parses; structural loop naming without `.loc`; trips honestly `unknown`; strided access flagged with stride expr | first non-nvcc kernel → Triton producers + coalescing |
| S5 | Can CI gate a kernel property? | `k14` (= `tests/fixtures/src/14_ldmatrix_mma.cuh`) sm_90 PTX + hand-edited copy with `mma.sync` removed + `rules.toml` | original passes; the edited copy fails exit-code 1 with message naming loop + `tensor_flops_per_iter: got 0` | first CI gate on a kernel → `check` verb |

Table values are design intent; expected outputs are authored from
fixture reality when each scenario's def is committed (PR 02 for all
of Phase 1's). Verified against real nvcc 13.2 output so far: k2's
K-loop is unrolled ×4 with a `.pragma "nounroll"` remainder loop, so
S6 and S8 see main-loop trips `(K − K mod 4)/4` (4 `fma` + 8 `cvt` + 8 loads per
iteration) plus a remainder loop trips `K mod 4` — not a single
trips-K loop. k5 (BM=64 BN=64 BK=8 TM=8 TN=8): the outer tile loop's
latch is `setp.lt.u32` stepping by 8 → trips `ceildiv(K, 8)`; the
inner dot loop survives un-unrolled (trips 8, 16 `ld.shared` + 64
`fma` per iteration); the register-tile and epilogue loops are fully
unrolled into straight-line code (what PR 12's line aggregation
recovers — including `.loc 1 0 …` "no source line" markers, which
attribution must skip). S3's access-pattern claims were hand-verified in the
emitted PTX: A's address derives from `2·(K·row)` (lane-invariant
product, non-affine), B's carries tid.x coefficient 2 B. k5's inner dot
loop is partially unrolled; k11/k12/k14 carry `.maxntid` from
`__launch_bounds__`. wmma role discrimination by first modifier
(`wmma.load.a` / `wmma.store.d` / `wmma.mma`) and the
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` /
`ldmatrix...x2.trans.shared.b16` grammars are confirmed in k11/k14
output.

## 5. Repository layout

Phase 1 only — Phase 2 items add their own surface when they land
(src/affine/, src/sass.rs, src/ncu.rs, src/check.rs, src/diff.rs,
report/html.rs, fuzz/, data/ptx_opcodes_*.txt, tools/extract-opcodes.py,
tools/triton_fixture.py).

```
v2/
├── PLAN.md                  # this file; checklist ticked per PR
├── Cargo.toml               # lib `ptxroof` + bin `ptxroof`
├── rust-toolchain.toml
├── ci.sh                    # fmt, clippy -D warnings, test, CLI test runner
├── src/
│   ├── main.rs              # thin: clap dispatch into the library
│   ├── lib.rs
│   ├── core/                # flat IR: Module/Kernel/Block/Instr + operand
│   │                        #   arena, interner/Symbol, newtype ids, SourceLoc,
│   │                        #   SymExpr, Measurement, result tree (Serialize)
│   ├── parse/               # lexer.rs, ast.rs, parser.rs (transcribed from lib/PTX)
│   ├── cfg/                 # graph.rs (index arenas), dominators.rs, loops.rs,
│   │                        #   naming.rs (display names, demangling)
│   ├── classify.rs          # instruction → semantic record (enum, exhaustive match)
│   ├── trips.rs             # trip-count matcher + scalar affine tracer
│   ├── machine.rs           # loader for data/machine/*.toml
│   └── report/              # collect.rs, stats.rs, text.rs (JSON = Serialize)
├── data/machine/            # per-SM peak/BW tables (TOML, sources cited inline)
├── tests/                   # cargo integration tests (*.rs) — and, as plain data:
│   ├── run.py               # CLI test runner (T2 + T3)
│   ├── fixtures/src/        # the CUDA kernel ladder (1_naive … 14_ldmatrix_mma .cuh)
│   ├── fixtures/<name>/     # {src ref, *.ptx, regen.sh, HAND-EDITED}
│   ├── cli/<case>/          # CLI tests: case.toml + expected.json / expected.checks
│   └── acceptance/          # status.toml + scenario test defs
└── tools/
    ├── regen-fixtures.sh    # drives all fixture regen.sh, container-pinnable
    └── fetch-manuals.sh     # PTX ISA manual → untracked refs/ (cite, don't commit)
```

---

## 6. The PR sequence

Each PR: **Goal / Contents / Tests / Done when**. ★ marks the PR where
a scenario's status changes from xfail to pass.

### Phase 1 — the minimal useful analyzer (PRs 01–13)

Goal: the five Phase 1 acceptance scenarios (§4) — `ptxroof analyze`
on nvcc PTX producing a named loop tree with per-iteration
flops/bytes/AI, honest named unknowns, and per-arch roofline verdicts. Every PR is on the critical path; nothing
outside this list is built until Phase 1 ships.

#### Foundations

**PR 01 — Scaffold and test harness.** (~500 LOC as built)
- Contents: cargo package (lib + bin), `rust-toolchain.toml`, `ci.sh`
  (fmt --check, clippy -D warnings, test --locked, CLI test runner);
  `clap` skeleton with the `analyze` verb stubbed; `tests/run.py`
  with partial JSON comparison (only fields present in `expected.json`
  are compared), CHECK lines (ordered substring matching, FileCheck
  semantics), scenario-status support (pass | xfail, both directions
  enforced), the minimum-coverage mechanism (reads minimums from
  `status.toml`; each becomes enforced when its analysis lands, then
  may only rise — §3), and the report-verifier hook (the §3 accounting
  identities, run on every JSON report; checks register as their
  analyses land) — all tested against the stub binary.
- Tests: one trivial unit test; runner self-tests (partial-comparison
  semantics: missing field fails, extra field passes; xfail-that-passes
  errors); clippy clean.
- Done when: `./ci.sh` green from a clean checkout with only rustup +
  python3 ≥ 3.11 (stdlib `tomllib` reads `status.toml` — the runner has
  zero third-party deps) — no CUDA, no LLVM, no C++ toolchain.

**PR 02 — nvcc fixture corpus + Phase 1 scenario specs.** (scripts ~100 LOC + fixtures)
- Contents (language-neutral): `regen.sh` per fixture with provenance
  headers and explicit template-instantiation wrapper `.cu` files;
  committed sm_80 PTX for `k1`, `k2`, `k5` (nvcc 13.2 `-ptx
  -lineinfo`); hand-written micro fixtures (`micro/` — single-loop,
  branchy, irreducible, no-loc, data-dep); `tools/fetch-manuals.sh` (PTX ISA
  manual into untracked `refs/` — the grammar reference PRs 03–04
  transcribe against); the Phase 1 acceptance defs (S1, S6, S7.1/S7.2,
  S8, S9.1–S9.3 — §4), all `xfail` in `status.toml`, plus the
  classification and trip-count min_coverage entries (unenforced until
  PR 12).
- Tests: fixture lint (a runner check): every fixture declares its
  origin — Provenance header + regen.sh, or HAND-WRITTEN, or
  HAND-EDITED; the status file loads; all Phase 1 scenarios
  runnable-and-xfailing against the stub.
- Done when: corpus committed, regen reproducible (`regen.sh` rerun is
  a no-op diff modulo date), S1 + S6–S9 visible as xfail in the
  runner output.

#### Frontend

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
  bit-pattern-float cases pinned); corpus-wide T2 check: *every* fixture
  lexes with zero error tokens.
- Done when: the lex-all-fixtures check is green.

**PR 04 — Parser → Module AST.** (~500 LOC; transcription of
`lib/PTX/Parser.cpp`, extended to full programs)
- Contents: recursive descent: module directives, kernel signatures +
  param tables (name + type; `.align` and `[size]` are parsed and dropped,
  so by-value aggregate params are not field-resolvable — §1 anti-scope),
  `.reg` decls, `.extern .shared .b8 name[]`
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
  they stay readable, no raw indices; T2: parse-all-fixtures check
  (zero `Unparsed` statements, or listed in `parse-allowlist.txt`);
  dump→reparse→dump idempotence on all fixtures.
- Done when: both corpus checks green; param tables for `k2`/`k5` match
  the committed expected outputs.

#### Structure

**PR 05 — CFG.** (~250 LOC)
- Contents: index-arena graph (`BlockId(u32)`, `Vec<Block>`); leaders/
  edges from labels, `bra`, `@%p bra` (target + fallthrough), `brx.idx`
  via `.branchtargets`, `ret`/`exit` terminators; `call` noted
  (intra-procedural CFG; non-inlined callees flagged as a visible unknown
  per ground rules).
- Tests: T1 on `micro/` snippets (fallthrough into label, cond-branch
  diamond, brx table, unreachable block); T2: pinned block/edge counts
  per ladder fixture.
- Done when: expected outputs pinned and green.

**PR 06 — Dominators + loop forest.** (~300 LOC)
- Contents: Cooper-Harvey-Kennedy dominators over the index arena; back
  edges; natural loops; nesting forest; **irreducible-region detection →
  flagged `unknown-multiplicity`, never guessed**. Hand-rolled (no graph
  crate) — we want line-item control of the irreducibility path.
- Tests: T1 textbook graphs (incl. the classic irreducible two-entry
  loop); T2: expected loop trees for the ladder (`k1`: 1 loop; `k5`: 2
  nested @ known lines; `k12`: pipeline loop) and for `micro/irreducible`
  asserting the honest flag; `insta` snapshot of the `k5` loop tree.
- Done when: ladder loop trees match the committed expected outputs.

**PR 07 — Loop naming.** (~150 LOC)
- Contents: loop display names — source line (`.loc`) when present,
  else the raw label; kernel-name demangling via `cpp_demangle`
  (fallible by type — unmangled names pass through unchanged);
  param-table printout (the `--bind` UX foundation). Stable structural
  IDs for cross-build joins belong to Phase 2's `diff` item.
- Tests: T1 naming incl. the no-`.loc` fallback; demangle cases for the
  ladder's mangled names; T2: `k5` loops named by source line,
  `micro/no-loc` loops named by label.
- Done when: every loop in the Phase 1 corpus has a human-readable name.

#### Semantics

**PR 08 — Instruction classifier.** (~400 LOC; transcription + extension
of `lib/PTX/Classifier.cpp`)
- Contents: `enum OpClass` with arms for what the Phase 1 corpus
  contains: {cuda-core flop, non-flop arith (incl. `cvt` — 8 per
  main-loop iteration in k2, so it materially affects instruction-mix
  reporting), memory, sync, control, ignore, unknown}, carrying
  {precision, packed-lanes, space, direction, scope, bytes-or-unknown};
  **`Space::Generic` for `ld`/`st` with no state space** — PTX ISA: "If
  no state space is given, perform the load using Generic Addressing" —
  reported as its own honest bucket; `.shared` sub-qualifier handling
  (`::cta` default per ISA, `::cluster` distinct); predicated non-branch
  instructions get a `predicated` qualifier; every consumer dispatches
  by exhaustive `match`, so each Phase 2 family (tensor, async-copy,
  atomics, SFU) is an additive arm the compiler then forces every
  dispatch site to handle.
- Tests: T1 table-driven per family (one case per modifier axis that
  changes the answer); T2 **corpus coverage check**: every instruction
  in every fixture classifies non-Unknown or appears in
  `classify-allowlist.txt` — additions to the allowlist require review.
  (The status.toml min_coverage entry flips enforced at PR 12, when the
  JSON reports the runner aggregates first exist; the
  `classified + allowlisted-unknown = total` verifier check registers
  there too.)
- Done when: the coverage check is green over the Phase 1 corpus.

**PR 09 — Measurement v2 + collection + Stats.** (~300 LOC)
- Contents: `Measurement{kind, precision, scope, space, direction,
  count: SymExpr(=const for now), provenance: instr index}` (serde-
  derived); per-block collection; `Stats` filter queries (v1 design
  kept); unquantified/unknown counters carried with reasons. (A `pipe`
  axis — SFU/tensor vs cuda-core — joins with Phase 2's instruction
  families.) Execution counts carry an **`exact` vs `at_most` qualifier**:
  a conditional block inside a loop body executes a data-dependent
  fraction of iterations, so its count is an upper bound, and every
  aggregate derived from it inherits the marker (rendered `≤` in
  reports). One bit, propagated mechanically — the alternative, branch
  probabilities, is anti-scope.
- Tests: T1 Stats filter semantics (incl. the documented soft-filter
  rule) and qualifier propagation (conditional-in-loop block ⇒ `at_most`
  on every aggregate it touches; loop-only nesting stays `exact`); T2:
  per-block flop/byte expected outputs for `k1`/`k2`; cross-check totals
  for `k2` against hand-computed values kept as a comment in the
  expected-output file.
- Done when: `k2` per-block numbers match hand calculation.

#### Symbolic counts → first useful report

**PR 10 — SymExpr mini-library.** (~250 LOC)
- Contents: `enum SymExpr` — symbols (params, opaque loop-trip symbols),
  products, sums/differences, integer ceil-/floor-div, **mod-by-power-
  of-two** (PTX lowers `K mod 4` as `and.b32 r, K, 3` — verified in k2's
  unroll bounds; real trip counts are shapes like `(K − K mod 4)/4`, so
  the pure `c·Π(sym)` form from the first draft is provably too weak),
  scalar multiply, substitution/binding, ordered printing. Still
  deliberately not a CAS — exactly the forms the trip matcher emits.
- Tests: T1 only — algebra, binding, printing stability, div edge
  cases. Coverage measured with `cargo llvm-cov`; this module is small
  enough to insist on 100% branch coverage. (`proptest` properties join
  with Phase 2's self-auditing item.)
- Done when: coverage target met; printing is deterministic.

**PR 11 — Trip-count matcher (nvcc shapes).** (~350 LOC)
- Contents: per-loop: latch-condition extraction (`setp`+`@%p bra`),
  induction recognition (single in-loop def `add r, r, const`), and a
  **scalar affine tracer** (lives with the matcher; Phase 2's affine
  item extends it to addresses): real nvcc latches compare *derived*
  registers, not the IV — k2's main loop exits on `setp.ne.s32 %p6,
  %r29, 0` where `%r29 = %r7 + %r35`, `%r7 = (K&3) − K`, `%r35` is the
  IV — so the matcher must normalize the latch condition as an affine
  expression of the IV and loop invariants, tracing through
  `add/sub/and-mask/mul.wide/mad.lo/shl` to `ld.param`/constants.
  Recognized shapes — exactly what the Phase 1 corpus contains:
  up-counting `i<N`/`i!=N` with constant stride, countdown, the
  derived-register latch, and the **unroll main+remainder pair** linked
  as one logical loop (main `(K − K mod 4)/4`, remainder `K mod 4`).
  Anything else — multi-exit, data-dependent, pointer-induction
  (clang/Triton's strength-reduced form, Phase 2), triangular —
  degrades to a named opaque symbol with a reason. Sibling loops
  sharing a `.loc` beyond the summable unroll pair are flagged
  `variants — not summed` and excluded from kernel totals (no
  guard-implication analysis: anti-scope). Per-register tracer state is
  a dense `Vec` indexed by a `RegId` newtype — PTX declares counted
  register families (`.reg .b32 %r<38>;`, verified in k2), so state
  arena sizes come straight from the source; no string-keyed maps
  anywhere in the dataflow.
- Tests: T1 on the recognized shapes plus the honest failures, one
  fixture snippet each: rotated do-while; `i<N`; `i!=N`; countdown;
  stride>1; 64-bit IV; derived-register latch (the verified k2 shape,
  pinned verbatim); unroll main+remainder pair; multi-exit → unknown;
  data-dependent → unknown. T2: `k2` main loop trips `(K − K mod 4)/4`
  with `K = param 2`, remainder `K mod 4`; `k5` outer tile loop trips
  `ceildiv(K, 8)`. (The status.toml min_coverage entry flips enforced
  at PR 12.)
- Done when: ladder trip counts match the committed expected outputs;
  every unknown carries a reason string.

**PR 12 — `analyze` report. ★ S6–S9 → pass.** (~350 LOC)
- Contents: loop-tree report (text + JSON via `Serialize` on the result
  tree): per-loop per-iteration steady-state aggregation (Measurement
  counts × enclosing trip exprs, `≤`-rendering for `at_most` counts,
  variant loops shown side by side and excluded from totals); precision
  flop table, per-space byte table, unknown/unquantified blocks with
  reasons; **source-line aggregation for straight-line code** —
  recovers a per-source-iteration view when the compiler fully unrolled
  a loop (every instruction carries the same `.loc`; verified on k5's
  inner BK loop), the universal fallback whenever loop structure was
  transformed away; per-run coverage stats in the result JSON (consumed
  by the runner's minimum-coverage thresholds); `--bind name=value` /
  `--bind idx:name=value` for numeric columns, bindings echoed in the
  report header.
- Tests: T2 expected outputs for `k1`, `k2`, `k5` (symbolic and
  `--bind K=4096` numeric columns); k5 unrolled-inner-loop
  line-aggregation test; T1 for bind parsing; binding-echo CHECK line;
  the remaining verifier checks register (Σ per-block = kernel;
  per-loop × bound trips = flat count); **S6–S9's statuses change to
  pass** — loop ranking on k2 (S6), the k1-vs-k5 AI contrast (S7), the
  k2 precision/cvt audit (S8), the three micro honesty cases (S9); S1
  partially satisfied but stays xfail (needs verdicts).
- Done when: S6–S9 green; `k5` report shows the S1 numbers except
  roofline verdicts.

#### Machine model

**PR 13 — Machine model + normalization + verdicts. ★ S1 → pass.** (~300 LOC)
- Contents: `data/machine/*.toml` per-SM tables (sm_70/75/80/86/89/90:
  peak FLOPs per precision, DRAM BW; sources cited in comments),
  serde-loaded; knee computation; `--arch` + `--launch x,y,z` →
  per-CTA/per-launch normalization of mixed-scope measurements; verdict
  lines. **Defaults from the PTX itself where present**: `--arch` from
  the `.target` directive, launch dims from `.reqntid`/`.maxntid`
  when present (several ladder kernels carry `.maxntid` via
  `__launch_bounds__`, verified);
  explicit flags override, and the report states which source was used.
  Plus a minimal `README.md`: install (`cargo install --path`),
  `analyze` usage, and the §1 audience boundary + anti-scope list
  verbatim — users read the same honesty contract the code enforces.
- Tests: T1 knee math vs hand-computed values from the cited specs; T1
  scope-normalization math (per-warp byte × blockDim cases — the v1
  flatten-by-32× error class pinned at the unit level); T2: the S1
  expected output incl. the two-arch verdict pair (sm_80 compute-bound /
  sm_86 memory-bound at AI=32) — **S1's status changes to pass**.
- Done when: S1 green — all five Phase 1 scenarios now `pass` in
  status.toml; a newcomer can go from `git clone` to the `k5` report
  using only the README. **Phase 1 ends here.**

**PR 14 — sm_89 fixtures + S1.2 (the verdict follows the part) + NCU
validation.** (fixtures + 1 case; post-Phase-1, same conventions)
- Contents: k1/k5 `regen.sh` loop over `sm_80 sm_89`; committed
  `k1.sm_89.ptx` / `k5.sm_89.ptx` (verified at generation: the nvcc
  13.2 PTX bodies are byte-identical to sm_80 — only `.target`
  differs, so no new instruction families or trip shapes enter the
  corpus); acceptance case `s1-blocktiling-sm89` (scenario S1.2) with
  no `--arch` flag — first coverage of the `.target`-directive default
  on a non-sm_80 module: the S1 design point (AI 32) lands
  memory-bound on the RTX 4090 table (f32 knee 81.9) where sm_80 said
  compute-bound.
- Hardware validation (one-off on the local RTX 4090; harness and NCU
  runs not committed — NCU import stays a Phase 2 trigger). Every
  static *demand* count was achieved exactly at the test shapes, where
  the analysis's `≤` qualifiers predict tightness (all guards pass):
  - k5 @ 4096³ (grid 64×64, 64 threads/CTA): measured FFMA
    68,736,253,952 and FMUL 16,777,216 ⇒ flops 137,489,285,120 = the
    static bound to the digit. k1 @ 1024³ (grid 32×32, 1024
    threads/CTA): FFMA 1,074,790,400, FMUL 1,048,576 ⇒ 2,150,629,376,
    again exact; k1 L1 global-store bytes = the requested 2,097,152
    exactly.
  - L1 byte counters are sector-granular and deviate from requested
    bytes exactly as the access geometry predicts: a sector model
    derived from the static counts reproduces both measurements to the
    byte — k1 ld 3,223,322,624 B (= requested − half of A's traffic:
    warp-broadcast loads, one 32 B sector serves a 64 B request) and
    k5 ld 6,710,886,400 B (tile rows are 16 B in 32 B sectors ⇒ ×1.5;
    the epilogue's 2 B accesses at 16 B stride fetch ×8). k5 st ×8 for
    the same epilogue-stride reason.
  - DRAM traffic is compulsory bytes only (k5 read 100,687,744 B vs
    A+B+C = 100,663,296; k1 6,295,424 vs 6,291,456): the 72 MB L2
    absorbs all re-reads at these shapes, so the no-reuse DRAM
    roofline does not bite — k5 measured 37.0 TFLOP/s, *above* the
    static AI×BW ceiling of 32.3, with DRAM at 3.2% of peak (NCU
    speed-of-light: SM 73.7%, L1 66%). k1: 4.5 TFLOP/s, L1-bound
    (L1 90.5%, DRAM 1.4%). This quantifies the README's static-vs-
    measured boundary on real hardware: the demand side (this tool's
    output) is exact; what the memory hierarchy does with the demand
    (NCU's side) moved k5 across the verdict line at this problem
    size — at working sets ≫ L2 the bound returns.
- Done when: S1.2 green alongside S1.1; regen of all four kN PTX files
  is a no-op diff; trip-coverage ratchet raised (16/17 = 94.12%).

**PR 15 — Static shared memory per CTA.** (~40 LOC; post-Phase-1, same
conventions)
- Contents: `KernelReport.shared_memory = {static_bytes, dynamic}`,
  rendered in both views. `static_bytes` sums `element_count ×
  element_width` over the kernel's `.shared` array declarations (width
  reuses the classifier's PTX type table — no new table); `dynamic`
  flags an `.extern .shared` array whose size is fixed at launch and so
  is not statically known and not in the static total. A `[static]`
  demand figure: it is the kernel-declared shared memory ptxas reports
  as `bytes smem` and NCU as `launch__shared_mem_per_block_static`, not
  the driver-reserved portion (NCU's `_driver`). Static smem is the one
  on-chip resource ptxas does not reallocate, so the source figure is
  authoritative — unlike registers, whose PTX `.reg` count is virtual,
  not the physical per-thread count occupancy depends on (that stays
  NCU's, per the §1 anti-scope).
- Tests: T1 unit (the k5 two-array shape → 2048; typed-array and
  b64/b128 widths; the no-smem k1 shape → 0; `.extern` dynamic flagged;
  mixed static+dynamic). T2: the S7 pair gains a committed assertion —
  naive 0 B/CTA vs tiled 2048 — and the k5 text CLI check gains the
  rendered line. The 2048 was confirmed against `ptxas -v` on both the
  sm_80 and sm_89 fixtures (`2048 bytes smem`, arch-independent).
- Anti-scope held: reports requested bytes, never an occupancy ceiling
  (that needs a per-arch smem-per-SM table and the runtime carveout —
  NCU's job). Two pieces deferred to their own triggers: module-scope
  `.shared` globals (the corpus has only nvcc's in-body "demoted" form;
  a top-level `.shared` parser arm + fixture lands when a producer emits
  one) and the dynamic byte count itself (knowable only at launch — the
  validator there is NCU's `launch__shared_mem_per_block_dynamic`).
- Done when: unit + S7.1/S7.2 + k5 text checks green; no occupancy
  claim enters the report.

**PR 16 — Typed index arenas.** (~200 LOC; post-Phase-1 hardening of
the §2 flat-IR ground rule, no behavior change)
- Contents: `core/arena.rs` — `IndexVec<I, T>` and `IdxRange<I>`,
  a hand-rolled subset of rustc's `rustc_index` (`Idx` trait,
  `newtype_idx!`), honoring the no-dependencies rule. `OperandId` and
  `BlockId` become `newtype_idx!` handles; `Module::operands` and
  `Cfg::blocks` become `IndexVec`s; the untyped `Span` shared by the
  operand-list and modifier pools splits into two `IdxRange` types, so
  crossing the pools is a compile error instead of a silent read of the
  wrong pool. The `.0` field stays public, so `Foo(0)` construction
  still works.
- Also: the parser accepts module-scope `.shared` declarations
  (nvcc's `.extern .shared .align A .b8 name[];` for `extern
  __shared__`), parse-and-discard — a module-scope dynamic decl is 0
  static bytes, so the PR 15 per-CTA report is unaffected. Triggered
  by a tensor-core demo kernel that failed to parse.
- Done when: `ci.sh` green with no expected-output change.

### Phase 2 — the demand-driven backlog

Nothing here is scheduled. An item starts only when its trigger fires;
it then becomes a short PR sequence under the same conventions (tests,
status.toml updates, expected outputs). Design details and locally
verified facts from the planning sessions are kept with each item so
the work starts warm — but no code, fixture, or CLI surface for any of
them exists in the tree until then.

**Tensor-core / async / atomic / SFU instruction families** (extends
PR 08; brings fixtures `k11`, `k12`, `k14` at sm_80/sm_90). Trigger:
the first tensor-core or async-pipeline kernel analyzed. OpClass arms
for tensor ops with `wmma` **role split by first modifier**
(load/store/mma — verified grammar in k11; lands with v1's
8192-phantom-FLOP wmma bug as a pinned regression test: `wmma.load.*`
⇒ memory, 0 FLOPs); `mma/wgmma` shape tables (verified:
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, and
`ldmatrix...x2.trans.shared.b16` in k14); `cp.async` with explicit
size operand (verified statically present), `commit_group`/
`wait_group` as sync arms, bulk/TMA with optional bytes → visible
unquantified counter; atomics (`atom`/`red`, counted load+store per v1
policy); the SFU pipe (`rcp/sqrt/div/ex2/lg2/sin/cos`) with an
explicit, documented FLOP policy; a `pipe` axis on `Measurement`.
Exhaustive `match` makes every new arm a compile-enforced extension.

**Triton + clang producers** (extends PR 02/03/04/11). Trigger: the
first non-nvcc kernel. `tools/triton_fixture.py` + committed Triton
PTX (verified: Triton 3.5.1 + torch 2.9.1 emits `cp.async` +
`mma.sync` idioms, param `.ptr .global` attributes — the parser grows
pointer-attribute support here — and `.loc` mapping to the *Python*
source); clang-trunk fixture variants (verified working against the
CUDA 13.2 headers with only a `-Wunknown-cuda-version` warning); the
**pointer-induction latch** trip shape (`setp.lt.u64 %p, %ptr, %end` ⇒
trips `(end − base)/stride`, clang/LLVM's strength-reduced form);
`.version` 8.7/8.8 join the corpus-wide checks.

**`diff` + SASS sidecar — S2.** Trigger: the first "did my change
help?" comparison or spill-regression hunt. Stable structural loop IDs
(`kernel/loop[i]/loop[j]`, extending PR 07 naming) so loops join
across builds; `nvdisasm -g` reader (verified format: `//## File "…",
line N` comment lines interleaved in the code section); `cuobjdump
--dump-resource-usage` reader (verified: reports `LOCAL:0` even when
spilling — explicit spill bytes exist only in `ptxas -v` stderr, so
regen captures it as a committed `.res.txt`); `LDL`/`STL` spill
counters from SASS (verified: 1326 in the forced-spill build vs 0
default; k5: 124 reg/0 spill → `--maxrregcount=32` ⇒ 32 reg/2772 B
spill stores + 2560 B spill loads — same PTX, real spills, not
hand-edited); join-on-(kernel,line) → per-loop SASS metrics; the
`diff` verb: per-loop delta on stable IDs, added/removed loops
reported as such.

**Affine address analysis + coalescing — S3, S4.** Trigger: the first
uncoalesced-access suspicion. Extends PR 11's scalar tracer to memory
addresses with a **three-tier domain** (verified necessity in k2: A's
address contains `K·row`, a product of two symbols — lane-invariant
but non-affine, so a two-tier affine-or-⊤ domain wrongly reports
unknown where the right answer is "uniform across the warp"):
(1) affine in lane-varying symbols → the tid.x stride coefficient;
(2) lane-invariant non-affine; (3) ⊤ unknown (data-dependent, carry
chains, register reuse across joins). Lane decomposition is
launch-config-dependent (warps are consecutive linearized thread IDs,
x-major — derived from `--launch` or `.reqntid`/`.maxntid`, else an
honest unknown); 32-byte-sector estimates per warp access (sectors,
not 128 B transactions, matching NCU's accounting);
uniform/coalesced/strided(w)/unknown verdicts; the access-pattern
min_coverage entry; `all_coalesced` summary fact.

**`check` verb — S5.** Trigger: the first CI gate on a kernel. Rules
TOML (the `toml` crate already on the allowlist): kernel/loop
selectors by stable ID, numeric comparators over Stats metrics,
instruction-class exists/absent; exit 0/1/2 = pass/fail/usage-error;
failure messages name loop, metric, expected vs got.

**NCU import — the three-column report.** Trigger: the first
measured-vs-static question. `csv` crate joins the allowlist;
kernel-name join (demangling-tolerant); transferred/requested ratio
per kernel; versioned header check, fail loudly on schema drift.
Verified blocker on this machine: `ERR_NVGPUCTRPERM` — capture needs
`sudo ncu` or `NVreg_RestrictProfilingToAdminUsers=0` (even
`--query-metrics` is gated; sudo also needs ncu's absolute path,
`/usr/local/cuda/bin/ncu`); capture is a manual, documented step and
CI only ever reads committed CSV. Verified at PR 14 (ncu 2026.1.1
`--csv`, local RTX 4090): the demand-side counters that correspond to
this tool's static columns are
`smsp__sass_thread_inst_executed_op_{ffma,fmul,fadd}_pred_on.sum`
(flops = 2·ffma + fmul + fadd) and
`l1tex__t_bytes_pipe_lsu_mem_global_op_{ld,st}.sum` (sector-granular —
expect geometry-dependent deviation from requested bytes, exact-match
only for full-sector access patterns); `dram__bytes_{read,write}.sum`
and the `SpeedOfLight` section carry the measured side.

**`annotate` HTML.** Trigger: wanting the per-line view. Source↔PTX
interleave via `.loc`, per-instruction badges (loop depth, class,
contribution), uncertainty highlights; self-contained HTML via plain
`format!` — no template engine, no CDN.

**Self-auditing extras** (extend PR 02/08/12). Trigger: corpus growth,
a PTX-version bump, or external users. `tools/extract-opcodes.py` →
`data/ptx_opcodes_<ver>.txt`, the canonical per-version instruction
inventory derived from the ISA manual (a version bump becomes a
one-screen diff naming every new instruction; classifier-table typos
become CI failures via an inventory cross-check); the `capabilities`
verb — the derived capability report (classified / unknown-by-policy
with reason / not-yet-handled, plus corpus coverage), committed as an
expected output so drift fails CI (the anti-STATUS.md rule: the only
inventory of what the tool handles is the one the tool generates); a
**ranked unknown histogram** weighted by symbolic execution count
("what to add next" as data, not judgment); `proptest` properties for
SymExpr.

**Hardening + switchover.** Trigger: external users or toolchain
drift. `cargo-fuzz` lexer/parser targets (no panics, ever, on
arbitrary bytes; mutational, corpus-seeded; the flat IR's
drop-as-one-block lifecycle keeps fuzz throughput high);
`tools/live_matrix.sh` (T4: recompile fixture sources with host
toolchains, relaxed invariant rules); the full user
guide with scenarios as worked examples; `cargo doc` clean with
`#![warn(missing_docs)]`; doc lint (CLI examples in docs are executed
by a runner case — docs can't rot silently).

---

## 7. Dependency graph (what can proceed in parallel)

Phase 1:

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 ─┬→ 12 → 13 ★S1
                                    10 → 11 ─┘
```

PRs 10–11 (symbolics) can be developed in parallel with 08–09
(semantics); PR 11's scalar affine tracer is why trip counts don't
wait on any later affine work. Phase 2 items each name the Phase 1 PR
they extend; no backlog item blocks another.

## 8. Risks and open questions

- **Transcription fidelity** (C++ → Rust for lexer/parser/classifier):
  mitigated by the corpus-wide checks landing in the same PRs. The
  planned v1 differential was retired when v1 was removed after
  Phase 1 (§3); the C++ reference is in git history at 690d81d.
- **Rust idiom frictions are resolved by the flat-IR ground rule (§2)**,
  not discovered mid-build: no pointer structures means no graph/
  lifetime fights with the borrow checker. The residual cost is index
  opacity in tests and debuggers — mitigated by newtype ids everywhere
  and by routing all snapshots and expected outputs through the
  symbol-resolving dumper. If contributor Rust fluency is still ramping,
  budget Phases 1–2 at reduced velocity — the frontend is the gentlest
  terrain to learn on, and the corpus checks catch semantic drift
  regardless.
- **Param naming is positional** (`_param_2`) without debug info. Bound
  via `--bind 2:K=...`; the param-table printout (PR 07) is the
  mitigation. Revisit if `.loc`-adjacent DWARF gives real names.
- **Triton PTX idioms** (heavily predicated masked loads, `v4` ops) may
  stress the classifier families when the Triton fixtures join
  (Phase 2); the corpus check will say so precisely. Budget one
  allowlist-review cycle in that item.
- **wgmma/TMA bytes** are descriptor-driven and statically unknowable;
  the design answer is the visible unquantified counter + a
  `--bind-bytes`-style flag (Phase 2, when first needed; not
  speculatively).
- **ptxas version skew** between fixture SASS and user SASS is real;
  that's why SASS fixtures are committed with provenance and the
  Phase 2 live matrix re-derives them.
- **NCU schema drift** across versions: the NCU import item (Phase 2)
  pins the captured header and fails loudly on mismatch rather than
  guessing column meaning.
- **NCU counter permission** (verified failing on this machine):
  `ERR_NVGPUCTRPERM` unless run privileged or the driver is configured
  with `NVreg_RestrictProfilingToAdminUsers=0`. Capture is a manual,
  documented step; CI never needs it (committed CSV).
- **Generic addressing** (`ld`/`st` with no state space) is legal PTX
  and unclassifiable to a concrete space without provenance; it gets
  its own honest bucket from day one. None of the current fixtures emit
  it (verified) — the risk is producer-dependent, so the corpus check
  will flag when it first appears.
- **Loop ID stability under heavy unrolling** (loop disappears):
  `diff` (Phase 2) reports removed-loop honestly; acceptance has no
  case where we pretend to match.
- **Unresolved branch targets** (a `bra` matching no label captured in
  the kernel) are surfaced as a report unknown — the dropped edge is
  reported, not hidden, so the "never silent" rule holds for control
  flow too. The message states only what happened and that it is
  unexpected; it does not blame the input, because on compiler-produced
  PTX (no dangling branches) the likelier cause is a label or branch
  form the parser did not register. Empty on the whole committed corpus.
- **Dependency creep**: the allowlist in §2 is the budget; serde-family +
  clap + cpp_demangle is already the bulk of compile time. Any proposed
  addition states what it replaces and why hand-rolling is worse.

## 9. Checklist

Phase 1:
- [x] PR 01 — scaffold + harness (cargo, clap `analyze` stub, CLI test runner)
- [x] PR 02 — nvcc fixture corpus (k1/k2/k5 + micro) + scenario specs S1, S6–S9
- [x] PR 03 — lexer
- [x] PR 04 — parser/AST
- [x] PR 05 — CFG
- [x] PR 06 — dominators + loop forest
- [x] PR 07 — loop naming + demangling
- [x] PR 08 — classifier (+ corpus coverage check)
- [x] PR 09 — Measurement + Stats
- [x] PR 10 — SymExpr
- [x] PR 11 — trip counts (nvcc shapes)
- [x] PR 12 — `analyze` report ★S6 ★S7 ★S8 ★S9
- [x] PR 13 — machine model + verdicts + README ★S1 — **Phase 1 done**
- [x] PR 14 — sm_89 fixtures (k1/k5) ★S1.2 + NCU validation on the local RTX 4090
- [x] PR 15 — static shared memory per CTA (★S7.1/S7.2 assertions + `ptxas -v` cross-check)
- [x] PR 16 — typed index arenas (`IndexVec`/`IdxRange`) + module-scope `.shared` decls

Phase 2 (backlog — tick when triggered and executed):
- [ ] tensor/async/atomic/SFU families (+ k11/k12/k14 fixtures)
- [ ] Triton + clang producers
- [ ] `diff` + SASS sidecar ★S2
- [ ] affine + coalescing ★S3 ★S4
- [ ] `check` ★S5
- [ ] NCU import (three-column report)
- [ ] `annotate` HTML
- [ ] self-auditing extras (opcode inventory, `capabilities`, histogram, proptest)
- [ ] hardening (fuzz, live matrix, docs) — the v1 differential/switchover is retired: v1 was removed after Phase 1
