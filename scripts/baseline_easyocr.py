#!/usr/bin/env python
"""Generic pretrained baseline: EasyOCR (CRAFT detector + CRNN recogniser) on the same crops.

It reads each field from the nominal 360x166 window of the cached, rectified card (the same area the
proposed method starts from) with a digits-and-slash allow-list. A field is auto-accepted when the
EasyOCR confidence is at least ``--threshold`` and the string is well-formed. The output has the
same ``records`` format as ``kaggle_eval.py --json`` so ``paired_tests.py`` can compare runs;
photos without a cached card (no card found) count as rejected, as in the main evaluation.

    pip install easyocr          # models are downloaded from the EasyOCR GitHub releases
    python scripts/baseline_easyocr.py --cache data/cache --csv labels.csv --json easyocr.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np

from idcard_ocr.layouts import parse_layout
from idcard_ocr.ocr.crnn import window_box

COLUMNS = {
    "national_id": "national_id",
    "birth_date": "date_of_birth",
    "expiry_date": "expiry_date",
}
DATE = re.compile(r"^\d{4}/\d{2}/\d{2}$")


def clean(kind: str, pieces: list[tuple[str, float]]) -> tuple[str, float]:
    """Join the pieces left to right; confidence is the minimum over pieces."""
    text = "".join(t for t, _ in pieces).replace(" ", "")
    conf = min((c for _, c in pieces), default=0.0)
    ok = (len(text) == 10 and text.isdigit()) if kind == "national_id" else bool(DATE.match(text))
    return (text, conf if ok else 0.0)


PERSIAN = "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
ARABIC = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
ALLOW = "0123456789/" + PERSIAN + ARABIC
TO_LATIN = str.maketrans(PERSIAN + ARABIC, "0123456789" * 2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--layout", default="calibration/kaggle/new-front-kaggle.json")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--test-mod", type=int, default=5)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument(
        "--lang", default="fa", help="EasyOCR language (fa uses the Arabic-script model)"
    )
    args = ap.parse_args()

    spec = parse_layout(json.loads(Path(args.layout).read_text())).crnn
    if spec is None:
        raise SystemExit("the layout has no `crnn` section")
    with args.csv.open(encoding="utf-8-sig") as fh:
        rows = {int(r["id"]): r for r in csv.DictReader(fh)}
    ids = [i for i in sorted(rows) if i % args.test_mod == 0]
    reader = easyocr.Reader([args.lang], gpu=False, verbose=False)
    records: list[dict[str, object]] = []
    counts = {name: [0, 0, 0] for name in COLUMNS}  # exact, auto, auto & exact
    for n, i in enumerate(ids):
        rec: dict[str, object] = {"id": i}
        path = args.cache / f"{i}.png"
        card = cv2.imread(str(path)) if path.exists() else None
        if card is None:
            rec["rejected"] = "no_card"
            records.append(rec)
            continue
        for name, col in COLUMNS.items():
            cx, cy, _plane, kind = spec.fields[name]
            x0, y0, x1, y1 = window_box((cx, cy), spec.window, card.shape[:2])
            crop = np.ascontiguousarray(card[y0:y1, x0:x1])
            found = reader.readtext(crop, allowlist=ALLOW, detail=1, paragraph=False)
            found.sort(key=lambda r: min(p[0] for p in r[0]))
            text, conf = clean(kind, [(r[1].translate(TO_LATIN), float(r[2])) for r in found])
            want = rows[i][col].strip()
            ok, auto = text == want and conf > 0, conf >= args.threshold
            rec[name] = [text, not auto, bool(ok)]
            counts[name][0] += ok
            counts[name][1] += auto
            counts[name][2] += ok and auto
        records.append(rec)
        if n % 50 == 0:
            print(f"{n}/{len(ids)}", flush=True)
    summary = {}
    for name, (ex, auto, ok) in counts.items():
        prec = ok / auto if auto else 0.0
        rec_ = ok / len(ids)
        summary[name] = {
            "accuracy": ex / len(ids),
            "precision": prec,
            "recall": rec_,
            "f1": 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0,
        }
        print(f"{name:12} {summary[name]}")
    args.json.write_text(json.dumps({"summary": summary, "records": records}, indent=1))


if __name__ == "__main__":
    main()
