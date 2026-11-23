# Benchmarks

**These are synthetic results** (fictional specimen cards, simulated phone photos). They show the
pipeline works and guard against regressions. They are *not* an estimate of real-world accuracy.

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
