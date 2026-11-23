"""Command line interface: ``idcard-ocr <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from idcard_ocr import __version__
from idcard_ocr.config import Settings
from idcard_ocr.errors import IdCardError


def _pipeline(settings: Settings):  # type: ignore[no-untyped-def]
    from idcard_ocr.pipeline import Pipeline
    from idcard_ocr.store import EncryptedImageStore

    store = None
    if settings.store_dir and settings.store_keys:
        store = EncryptedImageStore(
            settings.store_dir, settings.store_keys, settings.retention_days
        )
    return Pipeline(settings, store=store)


def cmd_extract(args: argparse.Namespace) -> int:
    pipe = _pipeline(Settings.from_env())
    data = Path(args.image).read_bytes()
    try:
        result = pipe.process(data, side=args.side, version=args.version, store=args.store)
    except IdCardError as exc:
        print(
            json.dumps(
                {"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(result.model_dump_json(indent=2))
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from idcard_ocr.synth.dataset import generate_dataset

    samples = generate_dataset(args.out, args.count, seed=args.seed, difficulty=args.difficulty)
    print(f"wrote {len(samples)} images and labels.jsonl to {args.out}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from idcard_ocr.bench import dump_json, run_benchmark, to_markdown

    pipe = _pipeline(Settings.from_env())
    report = run_benchmark(pipe, args.dataset, limit=args.limit, use_hints=args.hints)
    print(to_markdown(report, title=f"Benchmark: {args.dataset}"))
    if args.json:
        dump_json(report, args.json)
    return 0


def cmd_overlay(args: argparse.Namespace) -> int:
    """Draw a layout's field boxes on the rectified card to calibrate layouts for real cards."""
    import cv2

    from idcard_ocr.imaging.decode import decode_image
    from idcard_ocr.imaging.detect import detect_card, rectify
    from idcard_ocr.layouts import LayoutRegistry

    settings = Settings.from_env()
    registry = LayoutRegistry.load(settings.layout_dir)
    layout = registry.get(args.layout)
    image = decode_image(Path(args.image).read_bytes(), max_pixels=settings.max_pixels)
    card = rectify(image, detect_card(image).quad, layout.size)
    palette = {"field": (0, 160, 0), "marker": (200, 0, 0), "probe": (0, 140, 255)}
    for spec in layout.fields.values():
        x, y, w, h = spec.roi.as_tuple()
        cv2.rectangle(card, (x, y), (x + w, y + h), palette["field"], 2)
        cv2.putText(
            card,
            spec.name,
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            palette["field"],
            1,
            cv2.LINE_AA,
        )
    for name, box in layout.markers.items():
        x, y, w, h = box.as_tuple()
        cv2.rectangle(card, (x, y), (x + w, y + h), palette["marker"], 2)
        cv2.putText(
            card,
            name,
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            palette["marker"],
            1,
            cv2.LINE_AA,
        )
    x, y, w, h = layout.bg_probe.as_tuple()
    cv2.rectangle(card, (x, y), (x + w, y + h), palette["probe"], 2)
    cv2.imwrite(args.out, card)
    print(f"wrote {args.out}")
    return 0


def cmd_gen_key(_: argparse.Namespace) -> int:
    from idcard_ocr.store import generate_key

    print(generate_key())
    return 0


def cmd_purge(_: argparse.Namespace) -> int:
    settings = Settings.from_env()
    pipe = _pipeline(settings)
    if pipe.store is None:
        print("no store configured (IDCARD_STORE_DIR / IDCARD_STORE_KEYS)", file=sys.stderr)
        return 2
    print(f"purged {pipe.store.purge_expired()} expired record(s)")
    return 0


def cmd_train_digits(args: argparse.Namespace) -> int:
    from idcard_ocr.ocr.digit_train import discover_fonts, train
    from idcard_ocr.ocr.digits import MODEL_PATH

    fonts = discover_fonts(args.fonts_dir, include_held_out=args.include_held_out)
    stats = train(args.out or MODEL_PATH, fonts=fonts, per_class=args.per_class, seed=args.seed)
    print(
        json.dumps(
            {"samples": stats.samples, "fonts": stats.fonts, "per_class": stats.per_class},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "idcard_ocr.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idcard-ocr", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="read one card photo and print JSON")
    p.add_argument("image")
    p.add_argument("--side", choices=["front", "back"])
    p.add_argument("--version-hint", dest="version", help="card version, e.g. old or new")
    p.add_argument("--store", action="store_true", help="retain the encrypted image")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("synth", help="generate a synthetic labelled dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("bench", help="benchmark against a labelled dataset")
    p.add_argument("dataset")
    p.add_argument("--limit", type=int)
    p.add_argument("--hints", action="store_true", help="pass true side/version as hints")
    p.add_argument("--json", help="write the full report to this file")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("overlay", help="draw layout boxes on a rectified photo (calibration)")
    p.add_argument("image")
    p.add_argument("--layout", required=True, help="layout id, e.g. new-front")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_overlay)

    p = sub.add_parser("gen-key", help="print a new AES-256 key for IDCARD_STORE_KEYS")
    p.set_defaults(func=cmd_gen_key)

    p = sub.add_parser("purge", help="delete stored images past their retention date")
    p.set_defaults(func=cmd_purge)

    p = sub.add_parser("train-digits", help="retrain the digit model from local fonts")
    p.add_argument("--fonts-dir")
    p.add_argument("--out")
    p.add_argument("--per-class", type=int, default=450)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--include-held-out",
        action="store_true",
        help="also train on the fonts the synthetic benchmark renders with",
    )
    p.set_defaults(func=cmd_train_digits)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
