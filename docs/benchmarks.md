# Benchmarks

The first section is **synthetic** (specimen cards, simulated phone photos): it guards against
regressions and says little about real accuracy. The [second section](#real-photos-kaggle-dataset)
is the first measurement on real photos, and it is much lower.

Setup: `idcard-ocr synth --count 60 --seed 2026 --difficulty <level>`, then `idcard-ocr bench`.
Digit model trained on local fonts; the benchmark font (DejaVu Sans Bold / Condensed Bold) was held out.
Accuracy counts only accepted images; rejected ones are reported separately.

| Metric | easy | medium | hard |
|---|---|---|---|
| national_id exact | 100% | 89.7% | 64.3% |
| birth_date exact | 100% | 86.2% | 64.3% |
| side/version classification | 100% | 98.3% | 86.2% |
| silent error rate (wrong and not flagged) | ~0.5% | ~1.1% | ~3.2% |
| names exact (always flagged for review) | 38-71% across levels | | |
| serial_no exact | 67-97% across levels | | | 
| rejected as unusable | 1/60 | 3/60 | 31/60 |
| latency p50 (one CPU core) | ~0.3 s | ~0.3 s | ~0.3 s |

Rejections on `hard` are mostly `classification_failed` (21) and `poor_image_quality` (10). Rejecting
a bad photo and asking for a retake is the intended behaviour, not a failure.

## Reading the numbers

* The metric that matters for a KYC pre-fill is the **silent error rate**, not raw accuracy: a
  wrong value shown with `needs_review: true` costs the user one tap, a wrong value shown as
  confident is a data-quality incident.
* Persian free-text names are the weakest part (Tesseract `fas`); the service never marks them
  as confident unless you opt in.
* CI runs `scripts/check_benchmark.py` with thresholds for `easy` and `medium` and fails on regression.

Reproduce: `make bench`.

## Real photos (Kaggle dataset)

Data: [kibordevents/iranian-national-id-card-dataset](https://www.kaggle.com/datasets/kibordevents/iranian-national-id-card-dataset),
676 photos with labels available at the time of measurement. The cards are fictional but the photos
are realistic: hand-held, tilted, upside down, fingers over the card, mixed lighting. New-card fronts only.

Method (`scripts/kaggle_eval.py`, see [calibration/kaggle](../calibration/kaggle/README.md)):
layout calibrated on the images; digit model = synthetic glyphs + real glyphs from the 540 training
images (`id % 5 != 0`); evaluation on the 136 held-out images (`id % 5 == 0`), none of which were
used for training. Checksum not used (the dataset's labels mostly fail it).

| Metric | Result |
|---|---|
| held-out images accepted (card found, side recognised) | 115 / 136 (84.6%) |
| rejected: card not recognised | 21 |
| national_id exact | 45.2% of accepted |
| birth_date exact | 40.9% |
| expiry_date exact | 40.0% |
| first / last / father name exact | 39.1% / 33.0% / 33.0% (always flagged) |
| latency (1 CPU core) | about 1.8 s per image |

How to read it:

* Before calibration, 0 of 40 images were read at all. Calibration, orientation handling, a
  stronger card detector, a digit model trained on real glyphs and anchor-based registration of
  the text block brought it to the numbers above. It is **not production quality yet**.
* Error analysis on national_id: when the crop yields exactly ten glyphs, reading is right about
  three times in four and most remaining errors are one digit. The dominant loss is geometric:
  hand-held photos where the detected quadrilateral includes background or clips the card, so the
  text sits where the layout does not expect it. Better corner localisation is the highest-value next step.
* "Silent error" is not reported here for national_id: without the checksum nothing distinguishes a
  wrong ten-digit read from a right one. With real (checksum-valid) numbers the checksum rejects
  about 10 of every 11 wrong reads, which this dataset cannot show.
* Names stay at roughly one third: Tesseract's Persian model is weak on this typeface.
  A Persian-capable text recogniser trained on real crops would help most there.
* With 136 test images, differences of a few percentage points are noise.
