#!/usr/bin/env python3
"""Coverage release gate: compare a pytest JUnit XML against ci/expected_tests.json.

    pytest --junitxml=junit.xml ... && python tools/check_test_coverage.py junit.xml ci/expected_tests.json

Fails (exit 1) when the lane did not actually run what it claims: fewer tests collected than expected, a required
file executed fewer tests than its floor (e.g. the Postgres suite skipped for want of a database), a required test did
not pass, or a skip occurred that is not on the allow-list. Prints collection counts and the identity of every problem.
Standard library only."""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def load(junit_path: str):
    root = ET.parse(junit_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = []
    for tc in suite.iter("testcase"):
        classname, name = tc.get("classname", ""), tc.get("name", "")
        skipped = tc.find("skipped")
        failed = tc.find("failure") is not None or tc.find("error") is not None
        cases.append({"file": classname.replace(".", "/") + ".py" if classname else "", "name": name,
                      "node": f"{classname}::{name}",
                      "status": "skipped" if skipped is not None else "failed" if failed else "passed",
                      "reason": (skipped.get("message") or "") if skipped is not None else ""})
    return cases


def check(cases: list[dict], manifest: dict) -> tuple[list[str], dict]:
    problems: list[str] = []
    total = len(cases)
    st = Counter(c["status"] for c in cases)
    per_file = Counter(c["file"] for c in cases if c["status"] == "passed")
    if total < manifest.get("min_collected", 0):
        problems.append(f"collected {total} < expected minimum {manifest['min_collected']}")
    if st["failed"]:
        problems.append(f"{st['failed']} test(s) failed")
    allowed = manifest.get("allowed_skips", [])
    skips = [c for c in cases if c["status"] == "skipped"]
    unexpected = [c for c in skips if not any(a in c["node"] or a in c["reason"] for a in allowed)]
    if unexpected:
        shown = "; ".join(f"{c['node']} ({c['reason'][:60]})" for c in unexpected[:5])
        problems.append(f"{len(unexpected)} skipped test(s) not on the allow-list: {shown}")
    if len(skips) > manifest.get("max_skipped", 0):
        problems.append(f"{len(skips)} skipped > max_skipped {manifest.get('max_skipped', 0)}")
    for f, floor in manifest.get("required_files", {}).items():
        if per_file[f] < floor:
            problems.append(f"{f}: {per_file[f]} passed < required {floor}")
    passed_names = {c["name"].split("[", 1)[0] for c in cases if c["status"] == "passed"}
    for t in manifest.get("required_tests", []):
        if t not in passed_names:
            problems.append(f"required test did not pass (missing or skipped): {t}")
    return problems, {"collected": total, **dict(st), "per_file_passed": dict(per_file)}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__); return 2
    cases = load(argv[1])
    manifest = json.load(open(argv[2]))
    problems, summary = check(cases, manifest)
    print(f"lane={manifest.get('lane')} collected={summary['collected']} passed={summary.get('passed', 0)} "
          f"failed={summary.get('failed', 0)} skipped={summary.get('skipped', 0)}")
    for f, n in sorted(summary["per_file_passed"].items()):
        print(f"  {n:4d}  {f}")
    if problems:
        print("\nCOVERAGE GATE FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("coverage gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
