# Changelog

## 0.1.0 - unreleased

* Card detection, rectification, quality gates, side/version/rotation classification.
* Per-field OCR for old/new front and back layouts; custom digit recognizer with checksum-guided decoding.
* National id checksum, Jalali date validation with Gregorian conversion.
* AES-256-GCM encrypted image store with retention, rotation, erasure.
* FastAPI service with API keys, bounded concurrency and upload limits; CLI.
* Synthetic specimen generator and benchmark with silent-error metric.
* CI, Docker image, Dependabot, weekly dependency audit.
