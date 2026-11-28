#!/usr/bin/env python
"""Ablation of the line recogniser's search strategy on cached, rectified held-out cards.

Compares, with 95% bootstrap CIs over cards (resampled, 2000 draws):

* ``nominal``  one fixed window per field (no registration search);
* ``per-field`` the 63-hypothesis search run independently per field (best confident reading);
* ``joint``    the same search scored by the summed confidence of all fields (this project).

and reports precision at a confidence threshold, coverage, and the AUC of the CTC confidence as a
correctness detector. Usage::

    python scripts/crnn_ablation.py --model crnn.onnx --cache data/cache --csv labels.csv \
        --json out.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from idcard_ocr.layouts import CrnnSpec, parse_layout
from idcard_ocr.ocr.crnn import GRID, NOMINAL, CrnnEngine, CrnnReading

COLUMNS = {
    "national_id": "national_id",
    "birth_date": "date_of_birth",
    "expiry_date": "expiry_date",
}


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U); NaN if one class is empty."""
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv[np.isin(allv, pos)]):
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def single_field(spec: CrnnSpec, name: str) -> CrnnSpec:
    return CrnnSpec(spec.window, spec.anchor, {name: spec.fields[name]})


def predict(
    engine: CrnnEngine, card: np.ndarray, spec: CrnnSpec, mode: str
) -> dict[str, CrnnReading]:
    if mode == "joint":
        return engine.read(card, spec)
    if mode == "nominal":
        return engine._evaluate(card, spec, [NOMINAL])[NOMINAL]
    out: dict[str, CrnnReading] = {}
    for name in spec.fields:
        sub = single_field(spec, name)
        scored = engine._evaluate(card, sub, list(GRID))
        out[name] = max(scored.values(), key=lambda r: r[name].confidence)[name]
    return out


def metrics(
    ok: np.ndarray, conf: np.ndarray, threshold: float, draws: int, seed: int
) -> dict[str, object]:
    n = len(ok)
    rng = np.random.default_rng(seed)

    def stats(idx: np.ndarray) -> list[float]:
        o, c = ok[idx], conf[idx]
        auto = c >= threshold
        prec = (o & auto).sum() / auto.sum() if auto.any() else 0.0
        rec = (o & auto).sum() / len(idx)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return [o.mean(), prec, rec, f1, auto.mean(), auc(c, o)]

    point = stats(np.arange(n))
    boot = np.array([stats(rng.integers(0, n, n)) for _ in range(draws)])
    names = ["accuracy", "precision", "recall", "f1", "coverage", "auc"]
    return {
        k: {
            "value": point[j],
            "ci95": [
                float(np.nanpercentile(boot[:, j], 2.5)),
                float(np.nanpercentile(boot[:, j], 97.5)),
            ],
        }
        for j, k in enumerate(names)
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--layout", default="calibration/kaggle/new-front-kaggle.json")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--test-mod", type=int, default=5)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--modes", nargs="+", default=["nominal", "per-field", "joint"])
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    spec = parse_layout(json.loads(Path(args.layout).read_text())).crnn
    if spec is None:
        raise SystemExit("the layout has no `crnn` section")
    with args.csv.open(encoding="utf-8-sig") as fh:
        rows = {int(r["id"]): r for r in csv.DictReader(fh)}
    ids = [i for i in sorted(rows) if i % args.test_mod == 0 and (args.cache / f"{i}.png").exists()]
    cards = {i: cv2.imread(str(args.cache / f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in ids}
    engine = CrnnEngine.from_onnx(args.model, threads=2)
    result: dict[str, object] = {"n_cards": len(ids), "threshold": args.threshold}
    for mode in args.modes:
        preds = {i: predict(engine, cards[i], spec, mode) for i in ids}
        per: dict[str, object] = {}
        for name, col in COLUMNS.items():
            ok = np.array(
                [
                    preds[i][name].text == rows[i][col].strip() and preds[i][name].confidence > 0
                    for i in ids
                ]
            )
            conf = np.array([preds[i][name].confidence for i in ids])
            per[name] = metrics(ok, conf, args.threshold, args.draws, 0)
            m: dict[str, dict[str, float]] = per[name]  # type: ignore[assignment]
            keys = ("accuracy", "precision", "recall", "f1", "auc")
            line = " ".join(f"{k} {m[k]['value']:.3f}" for k in keys)
            print(f"{mode:10} {name:12} {line}", flush=True)
        result[mode] = per
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
