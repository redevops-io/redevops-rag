"""The coverage gate must fail for the exact failure modes it exists to catch."""
import importlib.util
import json
import pathlib

spec = importlib.util.spec_from_file_location("check_test_coverage", pathlib.Path(__file__).parent.parent / "tools" / "check_test_coverage.py")
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)


def case(file, name, status="passed", reason=""):
    return {"file": file, "name": name, "node": f"{file[:-3].replace('/', '.')}::{name}", "status": status, "reason": reason}


MANIFEST = {"min_collected": 3, "max_skipped": 0, "allowed_skips": [], "required_files": {"tests/test_pg_store.py": 2},
            "required_tests": ["test_tenant_filter"]}
GOOD = [case("tests/test_pg_store.py", "test_a"), case("tests/test_pg_store.py", "test_tenant_filter[param]"), case("tests/x.py", "test_b")]


def test_a_complete_run_passes():
    assert gate.check(GOOD, MANIFEST)[0] == []


def test_a_skipped_postgres_suite_fails_the_gate_even_though_pytest_would_exit_zero():
    skipped = [case("tests/test_pg_store.py", c["name"], "skipped", "set REDEVOPS_RAG_PG_TEST_URL") if c["file"].endswith("pg_store.py") else c
               for c in GOOD]
    problems, _ = gate.check(skipped, MANIFEST)
    assert any("not on the allow-list" in p for p in problems) and any("test_pg_store.py: 0 passed < required 2" in p for p in problems)
    assert any("required test did not pass" in p for p in problems)


def test_a_dropped_test_a_failure_and_a_lower_count_are_each_reported():
    assert any("collected 2 < expected" in p for p in gate.check(GOOD[:2], MANIFEST)[0])
    assert any("failed" in p for p in gate.check(GOOD[:2] + [case("tests/x.py", "t", "failed")], MANIFEST)[0])
    assert any("required test did not pass" in p for p in gate.check([c for c in GOOD if "tenant" not in c["name"]] * 2, MANIFEST)[0])


def test_an_allow_listed_skip_is_tolerated():
    m = {**MANIFEST, "max_skipped": 1, "allowed_skips": ["windows"]}
    run = GOOD + [case("tests/y.py", "test_win", "skipped", "windows only")]
    assert gate.check(run, m)[0] == []
    assert gate.check(GOOD + [case("tests/y.py", "test_other", "skipped", "no gpu")], m)[0] != []


def test_the_committed_manifest_is_well_formed():
    m = json.loads((pathlib.Path(__file__).parent.parent / "ci" / "expected_tests.json").read_text())
    assert m["max_skipped"] == 0 and m["allowed_skips"] == [] and m["min_collected"] >= sum(m["required_files"].values()) - 5
    for t in m["required_tests"]:
        assert t.startswith("test_")
