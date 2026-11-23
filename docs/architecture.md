# Architecture

```
upload ──► decode ──► detect ──► rectify ──► quality gate ──► classify ──► read fields ──► (store)
           │           │          │            │                │            │
           │           │          │            │                │            └─ digits: HOG+kNN, checksum DP
           │           │          │            │                │            └─ names/serial: Tesseract attempts
           │           │          │            │                └─ side, version, rotation (0/180)
           │           │          │            └─ sharpness, exposure, resolution; glare per field ROI
           │           │          └─ perspective warp to 1011x638, long edge horizontal
           │           └─ contours over 4 foreground masks, scored by shape; flat-scan fallback
           └─ header-first checks: format allow-list, pixel cap, EXIF orientation
```

All stages live in `src/idcard_ocr`:

| Module | Responsibility |
|---|---|
| `imaging/decode.py` | Safe decode (JPEG/PNG/WEBP only, pixel cap checked before pixels are decoded) |
| `imaging/detect.py` | Card quadrilateral, perspective rectification |
| `imaging/quality.py` | Blur/exposure/resolution gates, per-field glare |
| `imaging/classify.py` | Side/version/orientation from layout-declared probes and markers |
| `layouts/*.json` | Single source of truth for ROIs, markers, palette (also drives the synthetic renderer) |
| `fields.py` | Preprocessing variants, per-kind readers, validator-guided candidate selection |
| `ocr/digits.py` | Glyph segmentation, HOG features, k-NN, checksum-constrained decoding |
| `ocr/tesseract.py` | Tesseract wrapper behind the `OcrEngine` protocol |
| `validation/` | National id, Jalali dates, Persian text normalisation |
| `store.py` | AES-256-GCM store with retention, rotation, erasure |
| `pipeline.py` | Orchestration, timings, quality enforcement, retention |
| `api/app.py` | FastAPI service: auth, bounded concurrency, limits |
| `synth/`, `bench.py` | Synthetic data, benchmark with silent-error metric |

## Key decisions

**Per-field ROI reading, not whole-card OCR.** Cards have a fixed layout, so each field has a
known region. OCR on small single-line crops is faster on CPU and far more accurate than reading
a busy card. The price is that a layout must be calibrated per card design.

**Validators choose between OCR attempts.** Fields with verifiable structure are read through
several preprocessing variants and engines; the first *valid* result wins. A valid checksum is
strong evidence (a random 10-digit string passes 1 time in 11), which is reflected in confidence.

**Checksum as a decoder.** `decode_national_id` runs a dynamic program over
`(position, partial sum mod 11)` to find the best valid id. It may change at most one glyph, only
one whose classifier probability is below 0.9, and the result is always `needs_review`. This is a
deliberate guard against silently "repairing" a wrong id into a different valid one.

**Names are never auto-trusted.** Tesseract's confidence is poorly calibrated for Persian free
text (wrong names are often read with high confidence). On the synthetic data, thresholding on
confidence could not separate right from wrong, so names always set `needs_review` and add the
`free_text_unverified` issue. `IDCARD_TRUST_NAMES=1` opts out.

**Swappable OCR engines.** `OcrEngine` is a two-method protocol. A different engine (PaddleOCR via
ONNX, or a fine-tuned vision-language model as a low-traffic fallback for flagged fields) can be
plugged in through `Pipeline(engine=...)` without touching the rest. A VLM on CPU is slow and can
hallucinate digits, so use it only behind the checksum and only for fields already flagged.

**Fail closed.** No API keys means no start. A request to retain an image when no store is
configured is an error (`409`), never silently dropped. Missing models are `503`, not partial results.

**Bounded resources.** `IDCARD_MAX_CONCURRENCY` caps parallel images (default `min(2, cpus)`),
extra requests wait `IDCARD_QUEUE_TIMEOUT` then get `429 Retry-After`, uploads are size-capped and
read in chunks, OpenCV and OpenMP are pinned to one thread per request.

## Layouts

A layout JSON describes one (version, side): card size, a palette colour sampled from `bg_probe`,
`markers` (`photo`, `barcode`, `chip`) and `fields` (each with `roi`, `kind`, optional `regex`).
Kinds: `national_id`, `name`, `date`, `serial`. Layouts are validated on load (bounds, kinds).
Extra or overriding layouts load from `IDCARD_LAYOUT_DIR`.

## Extension points

* New card design: add layout JSON files ([calibration guide](calibration.md)).
* New field kind: add a reader in `fields.py` and a validator in `validation/`.
* New OCR engine: implement `OcrEngine`.
* New storage backend: keep the `EncryptedImageStore` interface (`save/load/info/delete/purge_expired`).
