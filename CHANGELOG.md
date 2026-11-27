# Changelog

## 0.1.0 - unreleased

* Orientation is cross-checked with the face: a card whose face is found only upside down is rotated
  by 180 degrees (warning `card_orientation_corrected`); fixed 110 of 1568 real photos. Effect on
  held-out national-id accuracy is small (34.1% -> 35.8%).
* Classification fallback: when the hints leave a single layout and the colour classifier refuses, a
  face in the photo box (upright or rotated) decides the orientation (warning `classified_by_face`).
  Held-out photos accepted rose from 325 to 354 of 400; national-id accuracy 36.2% -> 37.0%.
* Portrait crops are levelled by the eye line (YuNet landmarks, 3-35 degrees) before cropping.
* Face-photo extraction: `idcard_ocr.portrait` (OpenCV YuNet, bundled 232 KB ONNX, MIT, ~40 ms per
  card on CPU) crops the portrait inside the layout's photo box (shifted by the located text block).
  Reported as `portrait` in the result (`face_found`, `confidence`, boxes), retained encrypted as the
  `portrait` variant (`GET /v1/images/{id}?variant=portrait`); warning `portrait_face_not_found` when
  no face is detected. `IDCARD_EXTRACT_PORTRAIT=0` disables it. On 1568 real hand-held photos a face
  was found in 66% (misses are mostly frames where no card was detected).
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
