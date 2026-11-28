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
All 2,000 labelled photos were used (1,856 first, the missing 144 added later). The cards are fictional but the photos are
realistic: hand-held, tilted, upside down, fingers over the card, mixed lighting. New-card fronts only.

Method (`scripts/kaggle_eval.py`, see [calibration/kaggle](../calibration/kaggle/README.md)):
layout calibrated on the images; digit model = synthetic glyphs + 37,190 real glyphs from the
1,376 training photos in which the card was found (`id % 5 != 0`, two training rounds); evaluation on
the **400 held-out photos** (`id % 5 == 0`), none used for training. The checksum is not used (the
dataset's labels mostly fail it).

Definitions, per field, over **all** held-out photos (a rejected photo yields no value, so it is a miss):
*accuracy* = exact / all; *precision* = exact among values the service auto-accepts (not flagged for
review); *recall* = exact and auto-accepted / all; *F1* = their harmonic mean.

| Field | Accuracy | Accuracy of accepted photos | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| national_id | 41.8% | 47.2% | 53.2%* | 41.8% | 46.8% |
| birth_date | 34.2% | 38.7% | 91.3% | 34.2% | 49.8% |
| expiry_date | 33.5% | 37.9% | 94.4% | 33.5% | 49.4% |
| first_name | 26.8% | 30.2% | 0%** | 0%** | 0%** |
| last_name | 25.2% | 28.5% | 0%** | 0%** | 0%** |
| father_name | 23.2% | 26.3% | 0%** | 0%** | 0%** |

Photos accepted (card found and recognised): 354 / 400 (88.5%); 46 rejected as `classification_failed`.
CPU time about 2.7 s per accepted photo on this 2-core machine with two workers.

\* Without the checksum nothing separates a wrong ten-digit read from a right one, so national_id
precision here is essentially its raw accuracy. With real, checksum-valid numbers the checksum rejects
about 10 of every 11 wrong reads; this dataset cannot show that.
\*\* Names are always flagged for review by design, so no name is ever auto-accepted: precision and
recall are zero by construction. Their accuracy is the figure to read.

**These are not production numbers.** More than half of the ids and dates are still wrong or missing.

### What limits accuracy (error analysis)

* **Rectification, not recognition, dominates.** Looking at failing crops: in many hand-held photos
  the detected quadrilateral is not the card (fingers or background included), so the card stays
  trapezoid-shaped and the text block lands away from the layout; some boxes sit on empty background.
  Even when the ID line is located, glyphs from sheared or tilted cards differ from the training glyphs.
* More training data did not help: going from 540 to 1,266 training photos left the held-out accuracy
  unchanged (34% vs 34%) and a label-cleaned model gave the same end-to-end result.
* **Orientation is not the bottleneck.** 7% of the cached cards (110 of 1,568) were still upside down
  after the colour classifier; cross-checking with the face fixed them, and held-out national-id accuracy
  moved only from 34.1% to 35.8% (dates and names within 1 point). Corner accuracy and glyph quality
  on tilted cards remain the limit.
* **Deskewing the digit line** (rotation search, see `ocr/digits.py:deskew`) was the largest single gain
  of the last rounds: national-id accuracy 37.0% -> 41.8%. A slant (shear) search was tried and dropped:
  the date slash is itself slanted and the synthetic date test fell from 100% to 37%.
* **Tried and rejected: face-based registration.** Warping each card so its face matches the median face
  box (rotation from the eye line, scale from the face width, translation) made things worse: national-id
  accuracy 41.8% -> 36.8%, dates -1.7 to -4.8 points. The face size and position vary between people more
  than the card geometry does, and the extra resampling blurs the glyphs. Corner accuracy has to be fixed
  in the detector, not downstream of it.
* Glyph-level accuracy on held-out real glyphs is about 89% (k-NN; SVM, MLP and gradient boosting land
  within a point), and 0.89^10 is about 31% for a ten-digit id. Part of the "errors" are hard glyphs
  (2/3/4 and 5/0 confusions on tilted cards), part are noisy auto-labels.
* Names stay near a quarter: Tesseract's Persian model is weak on this typeface.

### Next steps with the best expected gain

1. A real corner detector for hand-held photos (line fitting on a Canny/LSD edge map with a
   rectangle prior, or a small learned corner model trained on annotated corners).
2. Rotation and shear augmentation of the glyph training data, and a small CNN instead of k-NN once
   geometry is fixed.
3. A Persian text recogniser trained on name crops (the dataset has the labels) instead of Tesseract.
4. Rerun on checksum-valid, real, consented cards: that is the only way to know real-world accuracy.

Earlier measurement (676 photos, 136 held-out, same method): national_id 45%, dates about 41% of
accepted. The larger, more representative test set is lower; with 136 images a few points are noise.
