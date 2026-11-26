# Changelog

## 0.1.0 - unreleased

* Real-photo work (Kaggle dataset): stronger card detector (saturation/pastel masks, edge-support and
  content scoring, outermost-candidate rule, small-card pass), chroma-only palette matching,
  anchor-based registration of the text block (`locate_offset`), shifted-box retries for digit
  fields, `calibration/kaggle` layout and `scripts/kaggle_eval.py`. Limit `opencv-python-headless` to 4.x.

* Card detection, rectification, quality gates, side/version/rotation classification.
* Per-field OCR for old/new front and back layouts; custom digit recognizer with checksum-guided decoding.
* National id checksum, Jalali date validation with Gregorian conversion.
* AES-256-GCM encrypted image store with retention, rotation, erasure.
* FastAPI service with API keys, bounded concurrency and upload limits; CLI.
* Synthetic specimen generator and benchmark with silent-error metric.
* CI, Docker image, Dependabot, weekly dependency audit.
