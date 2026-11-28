# Line recogniser (CRNN + CTC) with joint registration

The glyph pipeline (`ocr/digits.py`: segment, HOG, k-NN) is weak on hand-held photos: per-glyph accuracy
~89% means ~31% for a ten-digit id, and every segmentation error is fatal. `ocr/crnn.py` replaces it
for the national id and both dates, and keeps the old path only as a flagged fallback.

## Method

| Idea | Source / novelty |
|---|---|
| Read the whole line with a convolutional-recurrent network trained with CTC; no segmentation | Shi, Bai, Yao, *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*, IEEE TPAMI 39(11), 2017; Graves et al., *Connectionist Temporal Classification*, ICML 2006 |
| Train from **line-level** labels only (the id/dates already in a KYC database) instead of hand-labelled glyphs | weak supervision |
| **Label-verified pseudo-alignment**: a bootstrap model finds, per training card, the window where it reads the *known* label correctly; only those windows are used to train the final model | this project (self-training with a verifiable label, cf. Lee, *Pseudo-Label*, ICML-W 2013) |
| **Joint registration by recognition**: one hypothesis (scale, dy, dx) defines the windows of *all* fields; the hypothesis with the highest summed CTC confidence wins. The dates cannot drift to the neighbouring line and each field validates the others' alignment | this project |
| Confidence = mean top-1 probability of the emitted characters. It separates right from wrong almost perfectly (AUC ~1.0 on held-out cards), so "not confident" is a reliable trigger for review | measured, see below |

Inputs are 112x240 gray windows (360x166 card pixels) plus a constant plane that says which field is
being read. The network has 1.1 M parameters (4.1 MB as ONNX). Training augmentation: rotation +-6
degrees, shear +-0.18, scale, translation, blur, noise, contrast, plus 5% "no digits here" windows.

## Results (400 held-out hand-held photos, full pipeline, checksum off)

| field | accuracy | precision | recall | F1 | glyph pipeline F1 |
|---|---:|---:|---:|---:|---:|
| national_id | 56.8% | 99.6% | 56.8% | 72.3% | 46.8% |
| birth_date | 55.0% | 97.8% | 55.0% | 70.4% | 49.8% |
| expiry_date | 53.0% | 98.6% | 52.8% | 68.7% | 49.4% |

Precision is measured over auto-accepted values only: a field the line reader is not confident about
is kept as the glyph reading but flagged `line_reader_unconfirmed` for human review. The remaining
~45% of recall is lost on photos where no hypothesis yields a confident well-formed reading: 46 of 400
are rejected before OCR (no card found) and most others are badly rectified or out of focus.

Ablations on the 325 held-out cards that reach OCR (id only unless noted):

* single fixed window: 55% exact, 96% exact when the output has ten digits;
* + window search: 66% exact, 99.5% precision at conf >= 0.95;
* joint registration (all fields): ids 65.8% at 99.5% precision; dates 67-69%, precision up from
  86-91% (independent search per field) to 99.5%.

## Cost

~0.4 s per card when the nominal window is confident, ~1.5-2 s for the full 63-hypothesis search on
one CPU thread (`IDCARD_CRNN_THREADS`). No GPU.

## Using it

```bash
pip install 'idcard-ocr[crnn]'
export IDCARD_CRNN_MODEL=/path/to/crnn.onnx    # IDCARD_CRNN_MIN_CONFIDENCE=0.9
```

A layout opts in with a `crnn` section (see `calibration/kaggle/new-front-kaggle.json`): window size,
anchor and, per field, the window centre, plane id and kind. The model is tied to that geometry.

## Reproducing

```bash
pip install torch onnx onnxruntime
python scripts/kaggle_eval.py --stage cache ...           # rectified, oriented cards
python scripts/train_crnn.py bootstrap --cache ... --csv ... --work work   # ~40 min, 2 CPUs
python scripts/train_crnn.py align     ...                                 # ~15 min
python scripts/train_crnn.py train     ...                                 # ~25 min
python scripts/train_crnn.py export    --work work --out crnn.onnx
```

The model weights are **not** distributed: they are trained on a dataset whose licence the project
cannot confirm. Train your own on consented, checksum-valid cards before any production use.

## Limits and next steps

* Only the national id and the dates; names remain Tesseract (about one in four right).
* The geometry assumes the rectified card is close to canonical (+-90 px, 0.85-1.2x). A learned
  corner detector would raise recall the most.
* Dataset ids mostly fail the checksum (synthetic cards), so the checksum could not be used as an extra
  verifier here. On real cards it is a free, strong filter on top of the CTC confidence.
* Training windows are verified only for ~65% of the training cards; a second self-training round
  (re-align with the joint model) is the obvious next experiment.
