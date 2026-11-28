#!/usr/bin/env python
"""Paired comparison of two evaluation runs (``kaggle_eval.py --json``) on the same photos.

For every field: exact-match discordant counts, the exact two-sided McNemar test, and the paired
bootstrap CI of the difference in F1. A photo that was rejected counts as a miss in both runs.
    python scripts/paired_tests.py baseline.json candidate.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

FIELDS = ("national_id", "birth_date", "expiry_date")


def outcome(records: list[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    exact = np.array([bool(r[name][2]) if name in r else False for r in records])
    auto = np.array([(not r[name][1]) if name in r else False for r in records])
    return exact, auto


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def f1(exact: np.ndarray, auto: np.ndarray) -> float:
    prec = (exact & auto).sum() / auto.sum() if auto.any() else 0.0
    rec = (exact & auto).sum() / len(exact)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main() -> None:
    a = {r["id"]: r for r in json.loads(Path(sys.argv[1]).read_text())["records"]}
    b = {r["id"]: r for r in json.loads(Path(sys.argv[2]).read_text())["records"]}
    ids = sorted(set(a) & set(b))
    ra, rb = [a[i] for i in ids], [b[i] for i in ids]
    rng = np.random.default_rng(0)
    print(f"{len(ids)} paired photos")
    for name in FIELDS:
        ea, aa = outcome(ra, name)
        eb, ab = outcome(rb, name)
        only_a, only_b = int((ea & ~eb).sum()), int((~ea & eb).sum())
        diffs = []
        for _ in range(2000):
            idx = rng.integers(0, len(ids), len(ids))
            diffs.append(f1(eb[idx], ab[idx]) - f1(ea[idx], aa[idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(
            f"{name:12} exact: only-baseline {only_a}, only-candidate {only_b}, "
            f"McNemar p={mcnemar_exact(only_a, only_b):.2e}; "
            f"F1 {f1(ea, aa):.3f} -> {f1(eb, ab):.3f} (diff 95% CI [{lo:+.3f}, {hi:+.3f}])"
        )


if __name__ == "__main__":
    main()
