"""Calibrate and evaluate on the Kaggle "Iranian National ID Card Dataset" (new-card fronts).

    scripts/download_dataset.sh                                   # needs Kaggle credentials
    python scripts/kaggle_eval.py --images data/raw \
        --csv data/raw/iranian_national_id_card_data.csv --train-model data/digit_model_kaggle.npz

Images are split by id: ``id % --test-mod == 0`` is held out for evaluation, the rest trains the
digit model (real glyphs are added to the synthetic ones). Only aggregate numbers are printed.
Nothing here is committed: the images are not ours to redistribute.

Note: the labels in this dataset are fictional and mostly fail the national-id checksum, so the
checksum repair is switched off for the evaluation (it assumes real, valid numbers).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import idcard_ocr.fields as fields_mod
from idcard_ocr.config import Settings
from idcard_ocr.errors import IdCardError
from idcard_ocr.fields import crop_roi, preprocess_variants
from idcard_ocr.imaging.classify import classify
from idcard_ocr.imaging.decode import decode_image
from idcard_ocr.imaging.detect import detect_card, rectify
from idcard_ocr.layouts import LayoutRegistry, parse_layout
from idcard_ocr.ocr.digits import CLASSES, MODEL_PATH, DigitEngine, DigitModel, glyph_matrix
from idcard_ocr.pipeline import Pipeline
from idcard_ocr.validation.national_id import NationalIdResult

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = {
    "national_id": "national_id",
    "birth_date": "date_of_birth",
    "expiry_date": "expiry_date",
    "first_name": "first_name",
    "last_name": "last_name",
    "father_name": "father_name",
}
DIGIT_FIELDS = ("national_id", "birth_date", "expiry_date")


def load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig") as fh:
        return {int(r["id"]): r for r in csv.DictReader(fh)}


def real_glyphs(ids, images: Path, rows, registry, layout, settings):  # type: ignore[no-untyped-def]
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for i in ids:
        path = images / f"{i}.jpg"
        if not path.is_file():
            continue
        try:
            image = decode_image(path.read_bytes(), max_pixels=settings.max_pixels)
            card = rectify(
                image, detect_card(image, max_side=settings.detect_max_side).quad,
                (settings.card_width, settings.card_height),
            )  # fmt: skip
            if classify(card, registry, side="front", version="new").rotated_180:
                card = cv2.rotate(card, cv2.ROTATE_180)
        except IdCardError:
            continue
        for name in DIGIT_FIELDS:
            text = rows[i][COLUMNS[name]]
            crop = crop_roi(card, layout.fields[name].roi, layout.size)
            for variant in preprocess_variants(crop)[:2]:
                comps, matrix = glyph_matrix(variant)
                if len(comps) != len(text):  # keep only trustworthy, fully matched strings
                    continue
                for ch, row in zip(text, matrix, strict=True):
                    feats.append(row)
                    labels.append(CLASSES.index(ch))
    return np.stack(feats), np.array(labels, dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument(
        "--layout", type=Path, default=ROOT / "calibration/kaggle/new-front-kaggle.json"
    )
    ap.add_argument("--train-model", type=Path, help="write a digit model trained on real glyphs")
    ap.add_argument("--model", type=Path, help="evaluate with an existing trained digit model")
    ap.add_argument("--test-mod", type=int, default=5)
    ap.add_argument("--json", type=Path, help="write aggregate results here")
    args = ap.parse_args()

    settings = Settings.from_env({**os.environ, "IDCARD_QUALITY_MODE": "warn"})
    layout = parse_layout(json.loads(args.layout.read_text(encoding="utf-8")))
    registry = LayoutRegistry([layout])
    rows = load_rows(args.csv)
    test_ids = [i for i in sorted(rows) if i % args.test_mod == 0]
    train_ids = [i for i in sorted(rows) if i % args.test_mod != 0]

    engine = DigitEngine(DigitModel.load(args.model)) if args.model else DigitEngine()
    if args.train_model:
        x, y = real_glyphs(train_ids, args.images, rows, registry, layout, settings)
        with np.load(MODEL_PATH) as data:
            sx, sy = data["x"].astype(np.float32) / 255.0, data["y"]
        DigitModel.save(args.train_model, np.concatenate([sx, x]), np.concatenate([sy, y]))
        engine = DigitEngine(DigitModel.load(args.train_model))
        have = sum((args.images / f"{i}.jpg").is_file() for i in train_ids)
        print(f"trained on {len(y)} real glyphs from {have} training images")

    # The dataset's ids are fictional and mostly fail the checksum. Left on, the checksum would
    # prefer a wrong read that happens to pass (1 in 11) over the correct one, so for this
    # evaluation a 10-digit read counts as well-formed and the repair step is off.
    original = fields_mod.normalize_national_id

    def _lenient(text: str):  # type: ignore[no-untyped-def]
        res = original(text)
        return NationalIdResult(res.value, True, ()) if len(res.value) == 10 else res

    fields_mod.normalize_national_id = _lenient  # type: ignore[assignment]
    fields_mod.decode_national_id = lambda probs: None  # type: ignore[assignment]
    pipeline = Pipeline(settings, registry=registry, digit_engine=engine)
    stats: Counter[str] = Counter()
    started = time.perf_counter()
    for i in test_ids:
        path = args.images / f"{i}.jpg"
        if not path.is_file():
            continue
        stats["images"] += 1
        try:
            result = pipeline.process(path.read_bytes(), side="front", version="new")
        except IdCardError as exc:
            stats[f"rejected:{exc.code}"] += 1
            continue
        stats["accepted"] += 1
        for name, column in COLUMNS.items():
            f = result.fields[name]
            ok = f.value == rows[i][column]
            stats[f"{name}:exact"] += ok
            stats[f"{name}:flagged"] += f.needs_review
            stats[f"{name}:silent"] += (not ok) and not f.needs_review
    seconds = (time.perf_counter() - started) / max(stats["images"], 1)

    n = max(stats["accepted"], 1)
    print(f"held-out images: {stats['images']}, accepted: {stats['accepted']}, {seconds:.2f}s/img")
    for k, v in sorted(stats.items()):
        if k.startswith("rejected"):
            print(f"  {k}: {v}")
    print(f"{'field':12} {'exact':>7} {'flagged':>8} {'silent err':>11}   (of accepted)")
    for name in COLUMNS:
        e, f, s = (stats[f"{name}:{t}"] for t in ("exact", "flagged", "silent"))
        print(f"{name:12} {e / n:7.1%} {f / n:8.1%} {s / n:11.1%}")
    if args.json:
        args.json.write_text(json.dumps(dict(stats), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
