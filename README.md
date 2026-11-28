# iran-idcard-ocr

CPU-only detection, rectification and OCR for **Iranian national ID cards** (old and new design,
front and back), with **encrypted image retention** for later identity verification.

[فارسی](README.fa.md) · [Architecture](docs/architecture.md) · [API](docs/api.md) ·
[Operations](docs/operations.md) · [Adapting to real cards](docs/calibration.md) ·
[Benchmarks](docs/benchmarks.md) · [Privacy & security](docs/privacy-and-security.md)

> **Read this first: status and honest limits**
>
> * The built-in layouts describe a **synthetic specimen design**, not the real card artwork.
>   Before production you must calibrate the layouts on real, consented samples
>   ([docs/calibration.md](docs/calibration.md)).
> * First real-photo measurement (fictional cards from a public Kaggle dataset, new-card fronts,
>   400 held-out hand-held photos): **88% of photos accepted. With the optional line recogniser
>   ([docs/crnn.md](docs/crnn.md), model not distributed): national id F1 72% (precision 99.6%), dates
>   F1 about 70% (precision about 98%), names about 25% and always sent to review.** Without it: id 42%,
>   dates about 34% (see [docs/benchmarks.md](docs/benchmarks.md)). That is not production quality. Old cards and card backs have no real-photo calibration at all.
> * The bundled digit model is trained on fonts and does not read the real card numerals; a model
>   trained on real glyphs does (see [calibration/kaggle](calibration/kaggle/README.md)).
> * Free-text names are read with Tesseract's Persian model, which is weak. **Names always require
>   user confirmation** (`needs_review: true`) unless you opt out.
> * This is OCR, not identity verification. It does no liveness check, face matching, forgery
>   detection or civil-registry lookup. Treat its output as a convenience to pre-fill a form.

## What it does

* Finds the card in a phone photo, corrects perspective, handles portrait and upside-down photos.
* Tells **front vs back** and **old vs new** design apart (or accepts hints).
* Rejects unusable photos with specific reasons (`blurry`, `too_dark`, `glare`, `low_resolution`)
  so a client can ask the user to retake.
* Reads fields per region of interest instead of OCR-ing the whole card:
  national id, first/last/father name, birth date, expiry date, serial number.
* Validates what can be validated: national-id **checksum**, calendar-correct **Jalali dates**
  (with Gregorian conversion), serial patterns. Digits are normalised from Persian/Arabic-Indic.
* Flags every uncertain field with `needs_review` and an `issues` list instead of guessing.
* Keeps the original photo and the rectified card **AES-256-GCM encrypted** with retention,
  key rotation and erasure.

## Why this design (resource-limited hosts)

No GPU, no deep-learning runtime. A request takes roughly 0.3 s on one CPU core in the synthetic
benchmark, memory stays in the low hundreds of MB, and concurrency is bounded so a small VM cannot
be overloaded.

| Stage | Technique |
|---|---|
| Card detection | OpenCV contours over several foreground masks, scored by shape |
| Side/version | Colour-palette probe + marker statistics (photo, barcode, chip), no training data |
| Digits (id, dates) | HOG + k-NN glyph classifier (about 1 MB model), checksum-guided decoding |
| Names, serial | Tesseract (`fas`, `ara`, `eng`), several attempts, best confidence wins |

The checksum is used as a decoder, not just a filter: a dynamic program over the national id finds
the most probable *valid* id, and at most one low-confidence digit may be corrected (and such a
result is always flagged for review).

## Quick start

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-fas tesseract-ocr-ara tesseract-ocr-eng fonts-dejavu-core
pip install .            # Python 3.11+

idcard-ocr synth --out data/demo --count 8 --difficulty easy     # fictional, watermarked cards
idcard-ocr extract "$(ls data/demo/*.jpg | head -1)"             # JSON result
idcard-ocr bench data/demo                                       # accuracy and latency
```

### As a service

```bash
cp .env.example .env            # set IDCARD_API_KEYS and IDCARD_STORE_KEYS (idcard-ocr gen-key)
docker compose up --build
curl -H "X-API-Key: $KEY" -F file=@card.jpg -F store_image=true http://127.0.0.1:8000/v1/extract
```

The service **refuses to start without API keys** (or an explicit `IDCARD_ALLOW_ANONYMOUS=1` for
local development). See [docs/api.md](docs/api.md).

### Example response (abridged)

```json
{
  "version": "new", "side": "front", "layout_id": "new-front",
  "classification_confidence": 0.93,
  "fields": {
    "national_id": {"value": "0499370899", "confidence": 0.97, "valid": true, "needs_review": false, "issues": []},
    "birth_date": {"value": "1370/05/12", "gregorian": "1991-08-03", "valid": true, "needs_review": false, "issues": []},
    "first_name": {"value": "علی", "confidence": 0.81, "valid": true, "needs_review": true, "issues": ["free_text_unverified"]}
  },
  "quality": {"sharpness": 212.4, "brightness": 168.0, "glare_ratio": 0.0, "issues": [], "ok": true},
  "stored": {"image_id": "9f2c…", "expires_at": "2026-12-23T10:00:00+00:00"}
}
```

## Development

```bash
make install   # editable install with dev tools
make lint typecheck test
make bench     # synthetic benchmark with regression gate
```

CI (GitHub Actions) runs ruff, mypy, the test suite on Python 3.11-3.13 with real Tesseract,
a synthetic accuracy gate, and a Docker build with a smoke test. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Synthetic data

`idcard_ocr.synth` renders fictional cards (generic art, random names, checksum-valid random ids)
and degrades them like phone photos (perspective, blur, glare, noise, JPEG). Every card carries a
visible **SPECIMEN** mark and none reproduces official artwork. Do not use them as documents.

## License

MIT, see [LICENSE](LICENSE).
