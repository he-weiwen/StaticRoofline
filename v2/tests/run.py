#!/usr/bin/env python3
"""CLI test runner — tiers T2 and T3 of PLAN.md §3.

Deliberately implementation-language-neutral and dependency-free:
stdlib only, Python >= 3.11 (for tomllib). If ptxroof were ever
rewritten again, this harness and its committed test inputs would
survive unchanged.

Pipeline, in order:

  1. CLI tests (T2). Every directory under tests/cli/ with a case.toml
     is run unconditionally and must pass. A case declares:
       args         = ["analyze", "{case}/in.ptx"]  argv after the
                      binary; "{case}" expands to the case dir,
                      "{fixtures}" to tests/fixtures
       exit         = 0         expected exit code (default 0)
       check_stream = "stdout"  stream expected.checks applies to
                                (or "stderr")
       json         = false     force-parse stdout as JSON even without
                                expected.json (so the report verifier
                                runs)
     plus optional committed expected-output files:
       expected.json    compared field-by-field against stdout parsed
                        as JSON (partial comparison, see below)
       expected.checks  CHECK lines: substrings that must appear in
                        order — the matching semantics of LLVM
                        FileCheck's CHECK: directives. '#'-prefixed
                        and blank lines are comments.

  2. Report verifier. Internal-consistency checks run on EVERY case's
     parsed JSON output — in the spirit of LLVM's IR verifier:
     identities that must hold for any correct implementation, with no
     per-case expectations and no human review. The registry below is
     empty in PR 01; checks register here as their analyses land
     (PR 08, 09, 12 — PLAN.md §3).

  3. Acceptance scenarios (T3). tests/acceptance/status.toml lists the
     acceptance scenarios, each referencing a case by name with status
     pass | xfail ("expected failure" — the LLVM lit / pytest term).
     Enforced in both directions: a pass scenario that fails blocks CI,
     and an xfail scenario that passes is ALSO an error — it forces the
     status.toml edit that records progress.

  4. Minimum-coverage thresholds. status.toml [[min_coverage]] entries.
     Each JSON report may carry a "coverage" object of
     {metric: {"num": N, "den": D}} fraction pairs (counts, not
     percentages — percentages cannot be aggregated without weights).
     The runner sums num/den across all reports; for each enforced
     entry, achieved% below the recorded minimum fails, and achieved%
     above it is also an error until --raise-min rewrites the minimum
     up to the achieved value — a visible status.toml diff. Minimums
     may only rise (a ratchet: it turns one way).

Partial-comparison semantics for expected.json (the T2 contract):
  - objects: every expected key must exist in actual and match; EXTRA
    actual keys are ignored (new report fields never break old tests).
  - arrays: same length, element-wise match (arrays are ordered data;
    partial comparison applies to object keys only).
  - scalars: exact equality, including floats — analysis is exact
    arithmetic on static counts, so expected outputs store exact values.

Usage:
  run.py                  run everything (binary: target/debug/ptxroof)
  run.py --bin PATH       use a different binary
  run.py --raise-min      also record beaten coverage minimums in
                          status.toml
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

V2_ROOT = Path(__file__).resolve().parents[1]
CLI_TESTS_DIR = V2_ROOT / "tests" / "cli"
ACCEPTANCE_DIR = V2_ROOT / "tests" / "acceptance"
FIXTURES_DIR = V2_ROOT / "tests" / "fixtures"
STATUS_FILE = ACCEPTANCE_DIR / "status.toml"
DEFAULT_BIN = V2_ROOT / "target" / "debug" / "ptxroof"
CASE_TIMEOUT_S = 120

# --------------------------------------------------------------------------
# Report verifier (PLAN.md §3): (name, fn(report) -> [violation, ...]).
# Each fn receives one case's parsed JSON report and returns human-readable
# violation strings. Checks register as analyses land:
#   PR 08  classified + allowlisted-unknown == total instructions
#   PR 09  sum of per-block counts == kernel totals
#   PR 12  per-loop per-iteration x bound trip expr == flat count;
#          every Measurement provenance index resolves to an instruction
VERIFIER_CHECKS = []


def verify_report(report):
    violations = []
    for name, fn in VERIFIER_CHECKS:
        violations += [f"verifier '{name}': {v}" for v in fn(report)]
    return violations


# --------------------------------------------------------------------------
# Partial JSON comparison (T2)


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
# CHECK lines (T2) — ordered substring matching, the semantics of LLVM
# FileCheck's CHECK: directives (without the prefix syntax).


def parse_check_lines(text):
    needles = []
    for line in text.splitlines():
        line = line.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        needles.append(line)
    return needles


def match_check_lines(needles, text):
    """Each CHECK line must occur, after the previous line's match."""
    errs = []
    pos = 0
    for needle in needles:
        i = text.find(needle, pos)
        if i >= 0:
            pos = i + len(needle)
        elif text.find(needle) >= 0:
            errs.append(f"CHECK out of order: {needle!r}")
        else:
            errs.append(f"CHECK not found: {needle!r}")
    return errs


# --------------------------------------------------------------------------
# Minimum-coverage thresholds (T3 cross-cutting)


def aggregate_coverage(reports):
    """Sum {"coverage": {metric: {"num": N, "den": D}}} across reports."""
    totals = {}
    for report in reports:
        for metric, frac in report.get("coverage", {}).items():
            num, den = totals.get(metric, (0, 0))
            totals[metric] = (num + frac["num"], den + frac["den"])
    return totals


def check_min_coverage(minimums, totals):
    """Return (violations, raises) where raises maps metric -> new minimum
    percent for entries the corpus now exceeds."""
    violations, raises = [], {}
    for entry in minimums:
        if not entry.get("enforced", False):
            continue
        metric, minimum = entry["metric"], entry["percent"]
        if metric not in totals:
            violations.append(
                f"min_coverage '{metric}' is enforced but no case emitted "
                "coverage for it"
            )
            continue
        num, den = totals[metric]
        if den == 0:
            violations.append(f"min_coverage '{metric}': denominator 0 across corpus")
            continue
        achieved = round(100.0 * num / den, 2)
        if achieved < minimum:
            violations.append(
                f"min_coverage '{metric}': achieved {achieved}% < minimum {minimum}%"
            )
        elif achieved > minimum:
            raises[metric] = achieved
    return violations, raises


def raise_min_text(text, raises):
    """Rewrite `percent = X` lines to the new achieved values. Relies on the
    documented status.toml format contract: within a [[min_coverage]] block
    the `metric` line precedes its `percent` line."""
    out, current_metric = [], None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[[min_coverage]]"):
            current_metric = None
        elif stripped.startswith("metric"):
            current_metric = stripped.split("=", 1)[1].strip().strip('"')
        elif stripped.startswith("percent") and current_metric in raises:
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}percent = {raises[current_metric]}\n"
        out.append(line)
    return "".join(out)


# --------------------------------------------------------------------------
# Scenario status (T3)


def evaluate_scenario(status, case_passed):
    """Both directions enforced; returns an error string or None."""
    if status == "pass" and not case_passed:
        return "regressed: status is 'pass' but the case failed"
    if status == "xfail" and case_passed:
        return "passes now: flip status to 'pass' in status.toml"
    if status not in ("pass", "xfail"):
        return f"unknown status {status!r} (want 'pass' or 'xfail')"
    return None


# --------------------------------------------------------------------------
# Case execution


def find_case_dir(name):
    for base in (ACCEPTANCE_DIR, CLI_TESTS_DIR):
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

    expected_checks = case_dir / "expected.checks"
    if expected_checks.is_file():
        stream = (
            proc.stderr
            if spec.get("check_stream", "stdout") == "stderr"
            else proc.stdout
        )
        details += match_check_lines(
            parse_check_lines(expected_checks.read_text()), stream
        )

    if report is not None:
        details += verify_report(report)

    return not details, details, report


# --------------------------------------------------------------------------
# Driver


def load_status():
    if not STATUS_FILE.is_file():
        return {}
    return tomllib.loads(STATUS_FILE.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    ap.add_argument("--raise-min", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    opts = ap.parse_args()

    if opts.self_test:
        return self_test()

    if not opts.bin.is_file():
        sys.exit(f"run.py: binary {opts.bin} not found — run `cargo build` first")

    status = load_status()
    failures = 0
    reports = []

    # 1+2: CLI tests (+ report verifier inside run_case)
    case_dirs = (
        sorted(d for d in CLI_TESTS_DIR.iterdir() if (d / "case.toml").is_file())
        if CLI_TESTS_DIR.is_dir()
        else []
    )
    for case_dir in case_dirs:
        passed, details, report = run_case(opts.bin, case_dir)
        if report is not None:
            reports.append(report)
        print(f"{'PASS' if passed else 'FAIL'}  cli     {case_dir.name}")
        for d in details:
            print(f"        {d}")
        failures += 0 if passed else 1

    # 3: acceptance scenarios
    for scenario in status.get("scenario", []):
        sid, sc_status = scenario["id"], scenario["status"]
        case_dir = find_case_dir(scenario["case"])
        if case_dir is None:
            print(f"FAIL  {sid:6}  case '{scenario['case']}' not found")
            failures += 1
            continue
        passed, details, report = run_case(opts.bin, case_dir)
        if report is not None:
            reports.append(report)
        err = evaluate_scenario(sc_status, passed)
        label = "FAIL" if err else ("XFAIL" if sc_status == "xfail" else "PASS")
        print(f"{label:5} {sid:6}  {scenario['case']}")
        if err:
            print(f"        {err}")
            for d in details:
                print(f"        {d}")
            failures += 1

    # 4: minimum-coverage thresholds
    minimums = status.get("min_coverage", [])
    violations, raises = check_min_coverage(minimums, aggregate_coverage(reports))
    for v in violations:
        print(f"FAIL  mincov  {v}")
        failures += 1
    if raises:
        if opts.raise_min:
            STATUS_FILE.write_text(raise_min_text(STATUS_FILE.read_text(), raises))
            for metric, new in raises.items():
                print(
                    f"RAISED min_coverage '{metric}' -> {new}% "
                    "(commit the status.toml diff)"
                )
        else:
            for metric, new in raises.items():
                print(
                    f"FAIL  mincov  '{metric}' exceeded: achieved {new}% — "
                    "rerun with --raise-min and commit the status.toml diff"
                )
                failures += 1

    n_cases = len(case_dirs) + len(status.get("scenario", []))
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

    # partial JSON comparison: the T2 contract
    ok(subset_match({"a": 1}, {"a": 1, "b": 2}) == [], "extra actual field passes")
    ok(subset_match({"a": 1, "b": 2}, {"a": 1}) != [], "missing field fails")
    ok(subset_match({"a": {"b": 3}}, {"a": {"b": 3, "c": 4}}) == [], "nested partial")
    ok(subset_match({"a": 1}, {"a": 2}) != [], "wrong scalar fails")
    ok(subset_match({"a": True}, {"a": 1}) != [], "bool is not int")
    ok(subset_match([1, 2], [1, 2]) == [], "array exact match")
    ok(subset_match([1], [1, 2]) != [], "array length mismatch fails")
    ok(subset_match({"a": [1]}, {"a": {"x": 1}}) != [], "type mismatch fails")

    # CHECK lines
    ok(match_check_lines(["aa", "bb"], "xx aa yy bb zz") == [], "CHECKs in order")
    ok(match_check_lines(["bb", "aa"], "xx aa yy bb zz") != [],
       "CHECKs out of order fail")
    ok(any("out of order" in e for e in match_check_lines(["bb", "aa"], "aa bb")),
       "ordering violation named")
    ok(any("not found" in e for e in match_check_lines(["cc"], "aa bb")),
       "absence named")
    ok(parse_check_lines("# comment\n\nneedle\n") == ["needle"], "comments stripped")

    # scenario status: both directions
    ok(evaluate_scenario("pass", True) is None, "pass+passes ok")
    ok(evaluate_scenario("pass", False) is not None, "pass+fails errors")
    ok(evaluate_scenario("xfail", False) is None, "xfail+fails ok")
    ok(evaluate_scenario("xfail", True) is not None, "xfail+passes errors")
    ok(evaluate_scenario("skip", True) is not None, "unknown status errors")

    # coverage aggregation + minimums + raising
    totals = aggregate_coverage(
        [
            {"coverage": {"m": {"num": 9, "den": 10}}},
            {"coverage": {"m": {"num": 0, "den": 10}}},
            {"other": 1},
        ]
    )
    ok(totals == {"m": (9, 20)}, "coverage sums counts across corpus")
    minimums = [{"metric": "m", "percent": 50.0, "enforced": True}]
    viol, rai = check_min_coverage(minimums, {"m": (9, 20)})
    ok(viol != [] and rai == {}, "below minimum violates")
    viol, rai = check_min_coverage(minimums, {"m": (10, 20)})
    ok(viol == [] and rai == {}, "exactly at minimum passes")
    viol, rai = check_min_coverage(minimums, {"m": (15, 20)})
    ok(viol == [] and rai == {"m": 75.0}, "beaten minimum demands --raise-min")
    viol, rai = check_min_coverage(
        [{"metric": "m", "percent": 99.0, "enforced": False}], {"m": (0, 20)}
    )
    ok(viol == [] and rai == {}, "unenforced minimum ignored")
    viol, _ = check_min_coverage(minimums, {})
    ok(viol != [], "enforced minimum with no data violates")

    status_text = (
        '[[min_coverage]]\nmetric = "m"\npercent = 50.0\nenforced = true\n'
        '[[min_coverage]]\nmetric = "n"\npercent = 10.0\nenforced = true\n'
    )
    new_text = raise_min_text(status_text, {"m": 75.0})
    ok("percent = 75.0" in new_text and "percent = 10.0" in new_text,
       "raise rewrites only the beaten minimum")
    ok(tomllib.loads(new_text)["min_coverage"][0]["percent"] == 75.0,
       "rewritten status file still parses")

    # report-verifier hook
    VERIFIER_CHECKS.append(
        ("totals", lambda r: [] if r.get("a") == r.get("b") else ["a != b"])
    )
    try:
        ok(verify_report({"a": 1, "b": 1}) == [], "holding identity is silent")
        got = verify_report({"a": 1, "b": 2})
        ok(got == ["verifier 'totals': a != b"], "violated identity is named")
    finally:
        VERIFIER_CHECKS.pop()

    print(f"run.py --self-test: {n} assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
