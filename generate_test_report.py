"""Combine all test results into a single comprehensive report."""

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "benchmarks" / "results"
UNIT_XML = RESULTS_DIR / "unit_test_results.xml"
INTEGRATION_JSON = RESULTS_DIR / "integration_result.json"
REPORT_FILE = RESULTS_DIR / "full_test_report.json"


def parse_junit(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    testsuite = root[0] if root.tag == "testsuites" else root
    cases = []
    for tc in testsuite.iter("testcase"):
        name = tc.get("name", "?")
        classname = tc.get("classname", "?")
        file = tc.get("file", "")
        time_s = float(tc.get("time", 0))
        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        status = "passed"
        detail = None
        if failure is not None:
            status = "failed"
            detail = failure.get("message", "")
        elif error is not None:
            status = "error"
            detail = error.get("message", "")
        elif skipped is not None:
            status = "skipped"
        cases.append({
            "name": name,
            "suite": classname.split(".")[-1] if "." in classname else classname,
            "file": file,
            "time_s": round(time_s, 3),
            "status": status,
            "detail": detail,
        })
    attrs = testsuite.attrib
    return {
        "suite": attrs.get("name", "pytest"),
        "total": int(attrs.get("tests", len(cases))),
        "passed": int(attrs.get("passed", sum(1 for c in cases if c["status"] == "passed"))),
        "failed": int(attrs.get("failures", 0)),
        "errors": int(attrs.get("errors", 0)),
        "skipped": int(attrs.get("skipped", 0)),
        "time_s": float(attrs.get("time", 0)),
        "timestamp": attrs.get("timestamp", ""),
        "cases": cases,
    }


def load_integration(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_report() -> dict:
    unit = parse_junit(UNIT_XML)
    integ = load_integration(INTEGRATION_JSON)
    combined_cases = []
    for c in unit["cases"]:
        combined_cases.append({
            "category": "unit",
            "backend": "mock",
            "name": "{}::{}".format(c["suite"], c["name"]),
            "file": c["file"],
            "status": c["status"],
            "time_s": c["time_s"],
        })
    for c in integ["cases"]:
        combined_cases.append({
            "category": "integration",
            "backend": "llama.cpp",
            "name": c["name"],
            "status": "passed" if c.get("passed") else "failed",
            "time_s": round((c.get("duration_ms", 0) or 0) / 1000, 3),
            "detail": {k: c[k] for k in ["status", "response", "collected_text",
                                          "chunks_count", "duration_ms",
                                          "delta_completed", "delta_tokens", "delta_batches"]
                       if k in c},
        })
    total_unit = unit["total"]
    passed_unit = unit["passed"]
    total_integ = integ["summary"]["total"]
    passed_integ = integ["summary"]["passed"]
    summary = {
        "unit_tests": {
            "total": total_unit, "passed": passed_unit,
            "failed": unit["failed"], "errors": unit["errors"],
            "skipped": unit["skipped"], "time_s": unit["time_s"],
        },
        "integration_tests": {
            "total": total_integ, "passed": passed_integ,
            "failed": total_integ - passed_integ,
        },
        "combined": {
            "total": total_unit + total_integ,
            "passed": passed_unit + passed_integ,
            "failed": (total_unit - passed_unit) + (total_integ - passed_integ),
        },
    }
    report = {
        "report_name": "LLM Serving Runtime ¡ª Full Test Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_tested": integ.get("model", "mock-only"),
        "summary": summary,
        "cases": combined_cases,
    }
    return report


if __name__ == "__main__":
    report = build_report()
    REPORT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    print("=" * 58)
    print("  LLM Serving Runtime ¡ª Full Test Report")
    print("=" * 58)
    print("  Generated: {}".format(report["generated_at"]))
    if report.get("model_tested"):
        model_name = report["model_tested"].rsplit("\\", 1)[-1]
        print("  Model:     {}".format(model_name))
    print()
    print("  {:25s} {:>6s} {:>6s} {:>6s}".format("Category", "Total", "Passed", "Failed"))
    print("  " + "-" * 25 + " " + "-" * 6 + " " + "-" * 6 + " " + "-" * 6)
    print("  {:25s} {:>6d} {:>6d} {:>6d}".format(
        "Unit (MockBackend)", s["unit_tests"]["total"],
        s["unit_tests"]["passed"], s["unit_tests"]["failed"]))
    print("  {:25s} {:>6d} {:>6d} {:>6d}".format(
        "Integration (llama.cpp)", s["integration_tests"]["total"],
        s["integration_tests"]["passed"], s["integration_tests"]["failed"]))
    print("  " + "-" * 25 + " " + "-" * 6 + " " + "-" * 6 + " " + "-" * 6)
    print("  {:25s} {:>6d} {:>6d} {:>6d}".format(
        "TOTAL", s["combined"]["total"],
        s["combined"]["passed"], s["combined"]["failed"]))
    print()
    print("  Report file: {}".format(REPORT_FILE))
    print()
    for cat, label in [("unit", "Unit Test Results"), ("integration", "Integration Test Results")]:
        print("  [{}]".format(label))
        for case in report["cases"]:
            if case["category"] != cat:
                continue
            status = "PASS" if case["status"] == "passed" else "FAIL"
            print("    [{}] {}".format(status, case["name"]))
    print()
    print("  All tests: {}/{} passed".format(s["combined"]["passed"], s["combined"]["total"]))
    print()
