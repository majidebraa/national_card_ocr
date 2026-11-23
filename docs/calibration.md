# Adapting the layouts to real cards

The bundled layouts (`src/idcard_ocr/layouts/*.json`) describe the **synthetic specimen cards**
produced by `idcard-ocr synth`. They are a faithful *mechanism*, not a measurement of the real
card artwork. Calibrate them on real, consented samples before relying on the output.

## What to collect

* 20-50 photos per design (old front, old back, new front, new back), taken the way your users
  take them, with their consent and a documented retention policy. Never commit them to the repo.
* The true values for each, entered by a human, in the same JSON-lines format as
  `labels.jsonl` produced by the synthetic generator (see `synth/dataset.py`).

## Steps

1. Copy a bundled layout into a private directory and point `IDCARD_LAYOUT_DIR` at it.
   A file with the same `id` overrides the bundled one.
2. Draw the boxes on a rectified photo and look at them:
   ```bash
   idcard-ocr overlay photo.jpg --layout new-front --out overlay.png
   ```
   Adjust each `roi` (`x, y, w, h` in layout pixels, default card size 1011x638) until the box
   hugs the printed value with a small margin. Leave 4-8 px around digits.
3. Update `bg_probe` and `palette` (the typical colour of an unprinted background region) and the
   `markers` (photo, barcode, chip positions) so side/version classification works.
4. Run the benchmark on your labelled set: `idcard-ocr bench data/real --json report.json`
   and read `silent_error_rate`, the share of fields that are wrong **and not flagged**.
   That is the number to drive down and to put in your acceptance criteria.
5. If digit accuracy is low, retrain the digit model with fonts closer to the real print:
   `idcard-ocr train-digits --fonts-dir /path/to/fonts`. Hold out at least one font to estimate
   generalisation, and never evaluate on the fonts you trained on.

## Acceptance suggestion

Do not go live until, on a few hundred real photos, the national-id and date fields have a silent
error rate you can accept, and a human confirmation step exists for names. Revisit after every
layout or model change; CI only guards the synthetic benchmark.
