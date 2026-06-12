#!/usr/bin/env python3
"""Golden + acceptance runner — tiers T2 and T3 of PLAN.md §3.

Deliberately implementation-language-neutral and dependency-free:
stdlib only, Python >= 3.11 (for tomllib). If ptxroof were ever
rewritten again, this harness and its fixture corpus would survive
unchanged.

Pipeline, in order:

  1. Golden cases (T2). Every directory under tests/golden/cases/ with
     a case.toml is run unconditionally and must pass. A case declares:
       args     = ["analyze", "{case}/in.ptx"]   argv after the binary;
                  "{case}" expands to the case dir, "{fixtures}" to
                  tests/fixtures
       exit     = 0          expected exit code (default 0)
       greps_on = "stdout"   stream expected.greps applies to (or "stderr")
       json     = false      force-parse stdout as JSON even without
                             expected.json (so conservation gates run)
     plus optional files:
       expected.json   subset-matched against stdout parsed as JSON
       expected.greps  ordered substring assertions, one per line
                       ('#'-prefixed lines and blank lines are comments)

  2. Conservation gates. Accounting identities run on EVERY case's
     parsed JSON output — no per-fixture expectations, no human review.
     The registry below is empty in PR 01; identities register here as
     their analyses land (PR 08, 09, 12 — PLAN.md §3).

  3. Acceptance scenarios (T3). manifest.toml [[scenario]] entries each
     reference a case by name and carry status pass|xfail. Enforced in
     both directions: pass-that-fails blocks, xfail-that-passes is ALSO
     an error — it forces the manifest flip that records progress.

  4. Coverage floors. manifest.toml [[floor]] entries. Each JSON report
     may carry a "coverage" object of {metric: {"num": N, "den": D}}
     fraction pairs (counts, not percentages — percentages cannot be
     aggregated without weights). The runner sums num/den across the
     corpus; for each active floor, achieved% < floor fails, and
     achieved% > floor is also an error until --ratchet rewrites the
     floor up to the achieved value — a visible manifest diff. Floors
     only ever rise.

Subset-matcher semantics (the T2 contract):
  - objects: every expected key must exist in actual and match; EXTRA
    actual keys are ignored (additions never break goldens).
  - arrays: same length, element-wise match (arrays are ordered data;
    "subset" applies to object keys only).
  - scalars: exact equality, including floats — analysis is exact
    rational arithmetic on static counts, so goldens store exact values.

Usage:
  run.py                  run everything (binary: target/debug/ptxroof)
  run.py --bin PATH       use a different binary
  run.py --ratchet        also rewrite beaten floors in the manifest
  run.py --self-test      run the runner's own unit tests (no binary)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("run.py needs python3 >= 3.11 (stdlib tomllib)")

import tomllib

V2_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = V2_ROOT / "tests" / "golden" / "cases"
ACCEPTANCE_DIR = V2_ROOT / "tests" / "acceptance"
FIXTURES_DIR = V2_ROOT / "tests" / "fixtures"
MANIFEST = ACCEPTANCE_DIR / "manifest.toml"
DEFAULT_BIN = V2_ROOT / "target" / "debug" / "ptxroof"
CASE_TIMEOUT_S = 120

# --------------------------------------------------------------------------
# Conservation gates (PLAN.md §3): (name, fn(report) -> [violation, ...]).
# Each fn receives one case's parsed JSON report and returns human-readable
# violation strings. Registered as analyses land:
#   PR 08  classified + allowlisted-unknown == total instructions
#   PR 09  sum of per-block counts == kernel totals
#   PR 12  per-loop per-iteration x bound trip expr == flat count;
#          every Measurement provenance index resolves to an instruction
CONSERVATION_GATES = []


def run_gates(report):
    violations = []
    for name, fn in CONSERVATION_GATES:
        violations += [f"gate '{name}': {v}" for v in fn(report)]
    return violations


# --------------------------------------------------------------------------
# JSON subset matcher (T2)


def subset_match(expected, actual, path="$"):
    """Return a list of mismatch strings; empty list means match."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected object, got {type(actual).__name__}"]
        errs = []
        for key, val in expected.items():
            if key not in actual:
                errs.append(f"{path}.{key}: missing from actual output")
            else:
                errs += subset_match(val, actual[key], f"{path}.{key}")
        return errs
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected array, got {type(actual).__name__}"]
        if len(expected) != len(actual):
            return [f"{path}: expected {len(expected)} elements, got {len(actual)}"]
        errs = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            errs += subset_match(e, a, f"{path}[{i}]")
        return errs
    # Scalar. bool is an int subclass in Python; distinguish explicitly so
    # expected `true` never matches actual `1`.
    if isinstance(expected, bool) != isinstance(actual, bool) or expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}"]
    return []


# --------------------------------------------------------------------------
# Ordered greps (T2)


def parse_greps(text):
    needles = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        needles.append(line)
    return needles


def grep_ordered(needles, text):
    """Each needle must occur, after the previous needle's match."""
    errs = []
    pos = 0
    for needle in needles:
        i = text.find(needle, pos)
        if i >= 0:
            pos = i + len(needle)
        elif text.find(needle) >= 0:
            errs.append(f"grep out of order: {needle!r}")
        else:
            errs.append(f"grep not found: {needle!r}")
    return errs


# --------------------------------------------------------------------------
# Coverage floors (T3 cross-cutting)


def aggregate_coverage(reports):
    """Sum {"coverage": {metric: {"num": N, "den": D}}} across reports."""
    totals = {}
    for report in reports:
        for metric, frac in report.get("coverage", {}).items():
            num, den = totals.get(metric, (0, 0))
            totals[metric] = (num + frac["num"], den + frac["den"])
    return totals


def check_floors(floors, totals):
    """Return (violations, ratchets) where ratchets maps metric -> new
    floor percent for floors beaten by the corpus."""
    violations, ratchets = [], {}
    for floor in floors:
        if not floor.get("active", False):
            continue
        metric, threshold = floor["metric"], floor["floor"]
        if metric not in totals:
            violations.append(
                f"floor '{metric}' is active but no case emitted coverage for it"
            )
            continue
        num, den = totals[metric]
        if den == 0:
            violations.append(f"floor '{metric}': denominator 0 across corpus")
            continue
        achieved = round(100.0 * num / den, 2)
        if achieved < threshold:
            violations.append(
                f"floor '{metric}': achieved {achieved}% < floor {threshold}%"
            )
        elif achieved > threshold:
            ratchets[metric] = achieved
    return violations, ratchets


def ratchet_manifest_text(text, ratchets):
    """Rewrite `floor = X` lines to the new achieved values. Relies on the
    documented manifest format contract: within a [[floor]] block the
    `metric` line precedes its `floor` line."""
    out, current_metric = [], None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[[floor]]"):
            current_metric = None
        elif stripped.startswith("metric"):
            current_metric = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("floor") and current_metric in ratchets:
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}floor = {ratchets[current_metric]}\n"
        out.append(line)
    return "".join(out)


# --------------------------------------------------------------------------
# Scenario status (T3)


def evaluate_scenario(status, case_passed):
    """Both directions enforced; returns an error string or None."""
    if status == "pass" and not case_passed:
        return "regressed: status is 'pass' but the case failed"
    if status == "xfail" and case_passed:
        return "passes now: flip status to 'pass' in manifest.toml"
    if status not in ("pass", "xfail"):
        return f"unknown status {status!r} (want 'pass' or 'xfail')"
    return None


# --------------------------------------------------------------------------
# Case execution


def find_case_dir(name):
    for base in (ACCEPTANCE_DIR, CASES_DIR):
        d = base / name
        if (d / "case.toml").is_file():
            return d
    return None


def run_case(binary, case_dir):
    """Run one case; return (passed, details, report_json_or_None)."""
    spec = tomllib.loads((case_dir / "case.toml").read_text())
    args = [
        a.replace("{case}", str(case_dir)).replace("{fixtures}", str(FIXTURES_DIR))
        for a in spec["args"]
    ]
    try:
        proc = subprocess.run(
            [str(binary), *args],
            capture_output=True,
            text=True,
            timeout=CASE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return False, [f"timed out after {CASE_TIMEOUT_S}s"], None

    details = []
    expected_exit = spec.get("exit", 0)
    if proc.returncode != expected_exit:
        details.append(f"exit code: expected {expected_exit}, got {proc.returncode}")

    expected_json = case_dir / "expected.json"
    report = None
    if expected_json.is_file() or spec.get("json", False):
        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            details.append(f"stdout is not valid JSON: {e}")
    if expected_json.is_file() and report is not None:
        details += subset_match(json.loads(expected_json.read_text()), report)

    expected_greps = case_dir / "expected.greps"
    if expected_greps.is_file():
        stream = proc.stderr if spec.get("greps_on", "stdout") == "stderr" else proc.stdout
        details += grep_ordered(parse_greps(expected_greps.read_text()), stream)

    if report is not None:
        details += run_gates(report)

    return not details, details, report


# --------------------------------------------------------------------------
# Driver


def load_manifest():
    if not MANIFEST.is_file():
        return {}
    return tomllib.loads(MANIFEST.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--ratchet", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    opts = ap.parse_args()

    if opts.self_test:
        return self_test()

    if not opts.bin.is_file():
        sys.exit(f"run.py: binary {opts.bin} not found — run `cargo build` first")

    manifest = load_manifest()
    failures = 0
    reports = []

    # 1+2: golden cases (+ gates inside run_case)
    case_dirs = (
        sorted(d for d in CASES_DIR.iterdir() if (d / "case.toml").is_file())
        if CASES_DIR.is_dir()
        else []
    )
    for case_dir in case_dirs:
        passed, details, report = run_case(opts.bin, case_dir)
        if report is not None:
            reports.append(report)
        print(f"{'PASS' if passed else 'FAIL'}  golden  {case_dir.name}")
        for d in details:
            print(f"        {d}")
        failures += 0 if passed else 1

    # 3: acceptance scenarios
    for scenario in manifest.get("scenario", []):
        sid, status = scenario["id"], scenario["status"]
        case_dir = find_case_dir(scenario["case"])
        if case_dir is None:
            print(f"FAIL  {sid:6}  case '{scenario['case']}' not found")
            failures += 1
            continue
        passed, details, report = run_case(opts.bin, case_dir)
        if report is not None:
            reports.append(report)
        err = evaluate_scenario(status, passed)
        label = "FAIL" if err else ("XFAIL" if status == "xfail" else "PASS")
        print(f"{label:5} {sid:6}  {scenario['case']}")
        if err:
            print(f"        {err}")
            for d in details:
                print(f"        {d}")
            failures += 1

    # 4: coverage floors
    floors = manifest.get("floor", [])
    violations, ratchets = check_floors(floors, aggregate_coverage(reports))
    for v in violations:
        print(f"FAIL  floor   {v}")
        failures += 1
    if ratchets:
        if opts.ratchet:
            MANIFEST.write_text(ratchet_manifest_text(MANIFEST.read_text(), ratchets))
            for metric, new in ratchets.items():
                print(f"RATCHET floor '{metric}' -> {new}% (commit the manifest diff)")
        else:
            for metric, new in ratchets.items():
                print(
                    f"FAIL  floor   '{metric}' beaten: achieved {new}% — "
                    "rerun with --ratchet and commit the manifest diff"
                )
                failures += 1

    n_cases = len(case_dirs) + len(manifest.get("scenario", []))
    print(f"run.py: {n_cases} case(s), {failures} failure(s)")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# Self-tests (PR 01): the runner's own logic, no binary needed.


def self_test():
    n = 0

    def ok(cond, what):
        nonlocal n
        n += 1
        assert cond, f"self-test failed: {what}"

    # subset matcher: the T2 contract
    ok(subset_match({"a": 1}, {"a": 1, "b": 2}) == [], "extra actual field passes")
    ok(subset_match({"a": 1, "b": 2}, {"a": 1}) != [], "missing field fails")
    ok(subset_match({"a": {"b": 3}}, {"a": {"b": 3, "c": 4}}) == [], "nested subset")
    ok(subset_match({"a": 1}, {"a": 2}) != [], "wrong scalar fails")
    ok(subset_match({"a": True}, {"a": 1}) != [], "bool is not int")
    ok(subset_match([1, 2], [1, 2]) == [], "array exact match")
    ok(subset_match([1], [1, 2]) != [], "array length mismatch fails")
    ok(subset_match({"a": [1]}, {"a": {"x": 1}}) != [], "type mismatch fails")

    # ordered greps
    ok(grep_ordered(["aa", "bb"], "xx aa yy bb zz") == [], "greps in order")
    ok(grep_ordered(["bb", "aa"], "xx aa yy bb zz") != [], "greps out of order fail")
    ok(any("out of order" in e for e in grep_ordered(["bb", "aa"], "aa bb")),
       "ordering violation named")
    ok(any("not found" in e for e in grep_ordered(["cc"], "aa bb")),
       "absence named")
    ok(parse_greps("# comment\n\nneedle\n") == ["needle"], "comments stripped")

    # scenario status: both directions
    ok(evaluate_scenario("pass", True) is None, "pass+passes ok")
    ok(evaluate_scenario("pass", False) is not None, "pass+fails errors")
    ok(evaluate_scenario("xfail", False) is None, "xfail+fails ok")
    ok(evaluate_scenario("xfail", True) is not None, "xfail+passes errors")
    ok(evaluate_scenario("skip", True) is not None, "unknown status errors")

    # coverage aggregation + floors + ratchet
    totals = aggregate_coverage(
        [
            {"coverage": {"m": {"num": 9, "den": 10}}},
            {"coverage": {"m": {"num": 0, "den": 10}}},
            {"other": 1},
        ]
    )
    ok(totals == {"m": (9, 20)}, "coverage sums counts across corpus")
    floors = [{"metric": "m", "floor": 50.0, "active": True}]
    viol, rat = check_floors(floors, {"m": (9, 20)})
    ok(viol != [] and rat == {}, "below floor violates")
    viol, rat = check_floors(floors, {"m": (10, 20)})
    ok(viol == [] and rat == {}, "exactly at floor passes")
    viol, rat = check_floors(floors, {"m": (15, 20)})
    ok(viol == [] and rat == {"m": 75.0}, "beaten floor demands ratchet")
    viol, rat = check_floors([{"metric": "m", "floor": 99.0, "active": False}],
                             {"m": (0, 20)})
    ok(viol == [] and rat == {}, "inactive floor ignored")
    viol, _ = check_floors(floors, {})
    ok(viol != [], "active floor with no data violates")

    manifest_text = (
        '[[floor]]\nmetric = "m"\nfloor = 50.0\nactive = true\n'
        '[[floor]]\nmetric = "n"\nfloor = 10.0\nactive = true\n'
    )
    new_text = ratchet_manifest_text(manifest_text, {"m": 75.0})
    ok("floor = 75.0" in new_text and "floor = 10.0" in new_text,
       "ratchet rewrites only the beaten floor")
    ok(tomllib.loads(new_text)["floor"][0]["floor"] == 75.0,
       "ratcheted manifest still parses")

    # conservation-gate hook
    CONSERVATION_GATES.append(
        ("totals", lambda r: [] if r.get("a") == r.get("b") else ["a != b"])
    )
    try:
        ok(run_gates({"a": 1, "b": 1}) == [], "holding identity is silent")
        got = run_gates({"a": 1, "b": 2})
        ok(got == ["gate 'totals': a != b"], "violated identity is named")
    finally:
        CONSERVATION_GATES.pop()

    print(f"run.py --self-test: {n} assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
