# Project instructions

## Keep the coverage audit and the plan current — on every commit

Before finishing any commit in this repo, check whether it makes
either of these stale, and fold the update into the same commit:

- `docs/ptx-instruction-coverage.md` — the per-instruction PTX ISA
  coverage audit. Any change to instruction handling
  (`src/classify.rs`, the parser's instruction surface, the model axes
  in `src/core/measurement.rs`) must update the affected rows, the
  warts and tier sections, and the pinned commit hash in the doc's
  header. If the pinned PTX ISA version changes, re-derive the
  instruction inventory from the manual before editing rows.
- `PLAN.md` — the roadmap. Mark PRs the commit completes, and record
  any Phase 2 item the commit starts, re-scopes, or retires.

The same-commit rule is deliberate, matching the repo's existing
convention (README's anti-scope list is edited in the same change that
revisits an item): updating these in the commit that causes the drift
is what keeps them trustworthy. When the Phase 2 `capabilities` verb
lands, the audit doc's tables become generated output and this
instruction shrinks to covering the assessment sections only.
