"""Benchmark the pipeline against a labelled (synthetic or consented real) dataset.

The headline production metric is the **silent error rate**: fields that are wrong *and* were
not flagged ``needs_review``. Wrong-but-flagged fields are caught by the human confirmation step;
silent ones are what reach downstream systems.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from idcard_ocr.errors import IdCardError
from idcard_ocr.pipeline import Pipeline
from idcard_ocr.synth.dataset import Sample, load_labels
from idcard_ocr.validation.text import ZWNJ, digits_only, normalize_persian

KINDS = {
    "national_id": "digits",
    "birth_date": "digits",
    "expiry_date": "digits",
    "serial_no": "serial",
}


def canon(field: str, value: str) -> str:
    """Comparison key: digits only for numeric fields, no ZWNJ/space for names."""
    kind = KINDS.get(field, "name")
    if kind == "digits":
        return digits_only(value)
    if kind == "serial":
        return value.strip().upper()
    return normalize_persian(value).replace(ZWNJ, "").replace(" ", "")


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def run_benchmark(
    pipeline: Pipeline,
    dataset_dir: str | Path,
    *,
    limit: int | None = None,
    use_hints: bool = False,
) -> dict[str, Any]:
    samples: list[Sample] = load_labels(dataset_dir)
    if limit:
        samples = samples[:limit]
    base = Path(dataset_dir)
    totals: Counter[str] = Counter()
    per_field: dict[str, Counter[str]] = defaultdict(Counter)
    errors: Counter[str] = Counter()
    latencies: list[float] = []

    for sample in samples:
        data = (base / sample.file).read_bytes()
        totals["images"] += 1
        started = time.perf_counter()
        try:
            result = pipeline.process(
                data,
                side=sample.side if use_hints else None,
                version=sample.version if use_hints else None,
            )
        except IdCardError as exc:
            errors[exc.code] += 1
            totals["rejected"] += 1
            continue
        latencies.append((time.perf_counter() - started) * 1000.0)
        totals["processed"] += 1
        if (result.version, result.side) == (sample.version, sample.side):
            totals["classified_ok"] += 1
        for name, truth in sample.fields.items():
            counts = per_field[name]
            counts["total"] += 1
            got = result.fields.get(name)
            if got is None:
                counts["missing"] += 1
                continue
            correct = canon(name, got.value) == canon(name, truth)
            counts["correct"] += correct
            if not correct and not got.needs_review:
                counts["silent_errors"] += 1
            if got.needs_review:
                counts["flagged"] += 1
            if not correct and got.needs_review:
                counts["caught_errors"] += 1

    field_rows = {}
    all_total = all_correct = all_silent = 0
    for name, c in sorted(per_field.items()):
        field_rows[name] = {
            "total": c["total"],
            "exact_match_pct": _pct(c["correct"], c["total"]),
            "silent_error_pct": _pct(c["silent_errors"], c["total"]),
            "flagged_for_review_pct": _pct(c["flagged"], c["total"]),
        }
        all_total += c["total"]
        all_correct += c["correct"]
        all_silent += c["silent_errors"]

    lat = sorted(latencies)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0
    return {
        "images": totals["images"],
        "processed": totals["processed"],
        "rejected": totals["rejected"],
        "rejected_by_code": dict(errors),
        "classification_accuracy_pct": _pct(totals["classified_ok"], totals["processed"]),
        "fields": field_rows,
        "overall_field_exact_match_pct": _pct(all_correct, all_total),
        "overall_silent_error_pct": _pct(all_silent, all_total),
        "latency_ms": {
            "p50": round(statistics.median(lat), 1) if lat else 0.0,
            "p95": round(p95, 1),
            "mean": round(statistics.fmean(lat), 1) if lat else 0.0,
        },
    }


def to_markdown(report: dict[str, Any], title: str = "Benchmark") -> str:
    lines = [
        f"### {title}",
        "",
        f"- images: {report['images']} (processed {report['processed']}, "
        f"rejected {report['rejected']} {report['rejected_by_code'] or ''})",
        f"- side/version classification: {report['classification_accuracy_pct']}%",
        f"- overall field exact match: {report['overall_field_exact_match_pct']}%",
        f"- overall silent error rate: {report['overall_silent_error_pct']}%",
        f"- latency p50 / p95: {report['latency_ms']['p50']} / {report['latency_ms']['p95']} ms",
        "",
        "| field | n | exact % | silent error % | flagged % |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["fields"].items():
        lines.append(
            f"| {name} | {row['total']} | {row['exact_match_pct']} | "
            f"{row['silent_error_pct']} | {row['flagged_for_review_pct']} |"
        )
    return "\n".join(lines) + "\n"


def dump_json(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
