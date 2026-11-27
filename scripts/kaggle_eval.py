"""Calibrate and evaluate on the Kaggle "Iranian National ID Card Dataset" (new-card fronts).

    scripts/download_dataset.sh                                   # needs Kaggle credentials
    python scripts/kaggle_eval.py --images data/raw \
        --csv data/raw/iranian_national_id_card_data.csv \
        --cache data/cache --train-model data/digit_model_kaggle.npz

Stages (``--stage``, default ``all``) are resumable: ``cache`` detects, rectifies and orients every
photo once (re-run it to continue after an interruption), ``train`` fits the digit model on real
glyphs, ``eval`` measures the held-out images.

Images are split by id: ``id % --test-mod == 0`` is held out for evaluation, the rest trains the
digit model (real glyphs are added to the synthetic ones). Only aggregate numbers are printed.
Nothing here is committed: the images are not ours to redistribute.

Note: the labels in this dataset are fictional and mostly fail the national-id checksum, so the
checksum is not used for the evaluation (it assumes real, valid numbers).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import idcard_ocr.fields as fields_mod
from idcard_ocr.config import Settings
from idcard_ocr.errors import IdCardError
from idcard_ocr.fields import crop_roi, preprocess_variants
from idcard_ocr.imaging.classify import classify
from idcard_ocr.imaging.decode import decode_image
from idcard_ocr.imaging.detect import detect_card, rectify
from idcard_ocr.layouts import Box, LayoutRegistry, parse_layout
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
METRIC_HELP = """Metrics per field over ALL held-out images (a rejected photo yields no value).
  accuracy  = exact / all
  precision = exact among auto-accepted values (not flagged for review)
  recall    = exact and auto-accepted / all
  F1        = harmonic mean of precision and recall
  char acc  = position-wise character accuracy (accepted photos only)"""

# Per-process state, set by the pool initialisers.
_G: dict[str, Any] = {}


def load_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8-sig") as fh:
        return {int(r["id"]): r for r in csv.DictReader(fh)}


def _init_common(layout_path: str) -> None:
    cv2.setNumThreads(1)
    settings = Settings.from_env({**os.environ, "IDCARD_QUALITY_MODE": "warn"})
    layout = parse_layout(json.loads(Path(layout_path).read_text(encoding="utf-8")))
    _G.update(settings=settings, layout=layout, registry=LayoutRegistry([layout]))


# -- stage 1: cache oriented, rectified cards ------------------------------------------------
def _init_cache(layout_path: str) -> None:
    _init_common(layout_path)


def _cache_one(job: tuple[str, str]) -> int:
    src, dst = job
    s = _G["settings"]
    try:
        image = decode_image(Path(src).read_bytes(), max_pixels=s.max_pixels)
        card = rectify(
            image,
            detect_card(image, max_side=s.detect_max_side).quad,
            (s.card_width, s.card_height),
        )
        if classify(card, _G["registry"], side="front", version="new").rotated_180:
            card = cv2.rotate(card, cv2.ROTATE_180)
    except IdCardError:
        Path(dst + ".none").touch()  # remember failures so a resumed run skips them
        return 0
    cv2.imwrite(dst, card)
    return 1


def stage_cache(images: Path, cache: Path, ids: list[int], layout_path: Path, jobs: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    todo = []
    for i in ids:
        src, dst = images / f"{i}.jpg", cache / f"{i}.png"
        if src.is_file() and not dst.exists() and not Path(str(dst) + ".none").exists():
            todo.append((str(src), str(dst)))
    print(f"cache: {len(todo)} photos to process", flush=True)
    started = time.time()
    with Pool(jobs, _init_cache, (str(layout_path),)) as pool:
        for done, _ in enumerate(pool.imap_unordered(_cache_one, todo, chunksize=4), 1):
            if done % 100 == 0:
                print(f"  {done}/{len(todo)} ({time.time() - started:.0f}s)", flush=True)


# -- stage 2: train the digit model on real glyphs --------------------------------------------
def _init_train(layout_path: str, model_path: str | None) -> None:
    _init_common(layout_path)
    _G["engine"] = DigitEngine(DigitModel.load(model_path)) if model_path else None


def _glyphs_one(job: tuple[str, dict[str, str]]) -> tuple[list[np.ndarray], list[int]]:
    path, row = job
    layout, engine = _G["layout"], _G["engine"]
    card = cv2.imread(path)
    feats: list[np.ndarray] = []
    labels: list[int] = []
    offset = (0, 0)
    if engine is not None:  # round 2: register the text block with the round-1 model
        offset = fields_mod.locate_offset(
            engine, card, layout.fields["national_id"], layout.size
        ) or (0, 0)
    for name in DIGIT_FIELDS:
        text = row[COLUMNS[name]]
        box = layout.fields[name].roi
        crop = crop_roi(card, Box(box.x + offset[0], box.y + offset[1], box.w, box.h), layout.size)
        for variant in preprocess_variants(crop)[:2]:
            comps, matrix = glyph_matrix(variant)
            if len(comps) != len(text):  # keep only trustworthy, fully matched strings
                continue
            for ch, vec in zip(text, matrix, strict=True):
                feats.append(vec)
                labels.append(CLASSES.index(ch))
    return feats, labels


def stage_train(cache, rows, train_ids, layout_path, out, jobs, rounds) -> None:  # type: ignore[no-untyped-def]
    with np.load(MODEL_PATH) as data:
        sx, sy = data["x"].astype(np.float32) / 255.0, data["y"]
    model_path: str | None = None
    for round_no in range(1, rounds + 1):
        work = [
            (str(cache / f"{i}.png"), rows[i]) for i in train_ids if (cache / f"{i}.png").exists()
        ]
        feats: list[np.ndarray] = []
        labels: list[int] = []
        with Pool(jobs, _init_train, (str(layout_path), model_path)) as pool:
            for n, (f, lab) in enumerate(pool.imap_unordered(_glyphs_one, work, chunksize=4), 1):
                feats += f
                labels += lab
                if n % 200 == 0:
                    print(f"  round {round_no}: {n}/{len(work)}", flush=True)
        DigitModel.save(
            out,
            np.concatenate([sx, np.stack(feats)]),
            np.concatenate([sy, np.array(labels, dtype=np.uint8)]),
        )
        model_path = str(out)
        print(f"round {round_no}: {len(labels)} real glyphs from {len(work)} photos", flush=True)


# -- stage 3: evaluate ------------------------------------------------------------------------
def _init_eval(layout_path: str, model_path: str | None) -> None:
    _init_common(layout_path)
    original = fields_mod.normalize_national_id

    def _lenient(text: str):  # type: ignore[no-untyped-def]
        res = original(text)
        return NationalIdResult(res.value, True, ()) if len(res.value) == 10 else res

    # The dataset's ids are fictional and mostly fail the checksum. Left on, the checksum would
    # prefer a wrong read that happens to pass (1 in 11) over the correct one, so for this
    # evaluation a 10-digit read counts as well-formed and the repair step is off.
    fields_mod.normalize_national_id = _lenient  # type: ignore[assignment]
    fields_mod.decode_national_id = lambda probs: None  # type: ignore[assignment]
    engine = DigitEngine(DigitModel.load(model_path)) if model_path else DigitEngine()
    _G["pipeline"] = Pipeline(_G["settings"], registry=_G["registry"], digit_engine=engine)


def _eval_one(job: tuple[str, dict[str, str]]) -> dict[str, int]:
    path, row = job
    out: Counter[str] = Counter({"images": 1})
    t = time.perf_counter()
    try:
        result = _G["pipeline"].process(Path(path).read_bytes(), side="front", version="new")
    except IdCardError as exc:
        out[f"rejected:{exc.code}"] += 1
        return dict(out)
    out["seconds_x1000"] = int((time.perf_counter() - t) * 1000)
    out["accepted"] += 1
    for name, column in COLUMNS.items():
        f = result.fields[name]
        want = row[column]
        ok = f.value == want
        out[f"{name}:exact"] += ok
        out[f"{name}:auto"] += not f.needs_review
        out[f"{name}:auto_ok"] += ok and not f.needs_review
        out[f"{name}:chars"] += len(want)
        out[f"{name}:chars_ok"] += sum(
            a == b for a, b in zip(f.value.ljust(len(want)), want, strict=False)
        )
    return dict(out)


def stage_eval(images, rows, test_ids, layout_path, model, jobs, json_out) -> None:  # type: ignore[no-untyped-def]
    work = [
        (str(images / f"{i}.jpg"), rows[i]) for i in test_ids if (images / f"{i}.jpg").is_file()
    ]
    stats: Counter[str] = Counter()
    with Pool(jobs, _init_eval, (str(layout_path), str(model) if model else None)) as pool:
        for part in pool.imap_unordered(_eval_one, work, chunksize=2):
            stats.update(part)
    total = max(stats["images"], 1)
    acc = max(stats["accepted"], 1)
    secs = stats["seconds_x1000"] / 1000 / acc
    print(
        f"held-out images: {stats['images']}, accepted: {stats['accepted']}, {secs:.2f}s/img (CPU)"
    )
    for k, v in sorted(stats.items()):
        if k.startswith("rejected"):
            print(f"  {k}: {v}")
    print(METRIC_HELP)
    print(f"{'field':12} {'accuracy':>9} {'precision':>10} {'recall':>7} {'F1':>7} {'char acc':>9}")
    summary: dict[str, dict[str, float]] = {}
    for name in COLUMNS:
        ex, auto, ok = (stats[f"{name}:{t}"] for t in ("exact", "auto", "auto_ok"))
        prec = ok / auto if auto else 0.0
        rec = ok / total
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        char = stats[f"{name}:chars_ok"] / max(stats[f"{name}:chars"], 1)
        summary[name] = {
            "accuracy": ex / total,
            "accuracy_of_accepted": ex / acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "char_accuracy": char,
        }
        print(f"{name:12} {ex / total:9.1%} {prec:10.1%} {rec:7.1%} {f1:7.1%} {char:9.1%}")
    if json_out:
        json_out.write_text(
            json.dumps({"counts": dict(stats), "summary": summary}, indent=2), encoding="utf-8"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument(
        "--layout", type=Path, default=ROOT / "calibration/kaggle/new-front-kaggle.json"
    )
    ap.add_argument("--cache", type=Path, default=Path("data/cache"))
    ap.add_argument("--stage", choices=("cache", "train", "eval", "all"), default="all")
    ap.add_argument("--train-model", type=Path, help="write a digit model trained on real glyphs")
    ap.add_argument("--model", type=Path, help="evaluate with an existing trained digit model")
    ap.add_argument("--rounds", type=int, default=2, help="training rounds (2 = re-register)")
    ap.add_argument("--test-mod", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--json", type=Path, help="write aggregate results here")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    test_ids = [i for i in sorted(rows) if i % args.test_mod == 0]
    train_ids = [i for i in sorted(rows) if i % args.test_mod != 0]
    stage = args.stage
    if stage in ("cache", "all"):
        stage_cache(args.images, args.cache, sorted(rows), args.layout, args.jobs)
    if stage in ("train", "all") and args.train_model:
        stage_train(
            args.cache, rows, train_ids, args.layout, args.train_model, args.jobs, args.rounds
        )
    if stage in ("eval", "all"):
        model = args.train_model if (args.train_model and args.train_model.exists()) else args.model
        stage_eval(args.images, rows, test_ids, args.layout, model, args.jobs, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
