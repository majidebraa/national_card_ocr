#!/usr/bin/env python3
"""Fail CI when benchmark quality regresses. Usage: check_benchmark.py REPORT.json PROFILE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Thresholds sit a few points below what the synthetic benchmark measures, so they catch real
# regressions without flaking on tiny differences between Tesseract builds.
PROFILES: dict[str, dict[str, float]] = {
    "easy": {"classification": 95, "national_id": 90, "birth_date": 85, "silent": 2.0},
    "medium": {"classification": 90, "national_id": 70, "birth_date": 60, "silent": 3.0},
}


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[2] not in PROFILES:
        print(f"usage: {argv[0]} REPORT.json {{{'|'.join(PROFILES)}}}", file=sys.stderr)
        return 2
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    limits = PROFILES[argv[2]]
    fields = report["fields"]
    checks = {
        "classification accuracy": (
            report["classification_accuracy_pct"],
            ">=",
            limits["classification"],
        ),
        "national_id exact match": (
            fields["national_id"]["exact_match_pct"],
            ">=",
            limits["national_id"],
        ),
        "birth_date exact match": (
            fields["birth_date"]["exact_match_pct"],
            ">=",
            limits["birth_date"],
        ),
        "overall silent error rate": (report["overall_silent_error_pct"], "<=", limits["silent"]),
    }
    failed = False
    for name, (value, op, limit) in checks.items():
        ok = value >= limit if op == ">=" else value <= limit
        failed |= not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {value} (needs {op} {limit})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
