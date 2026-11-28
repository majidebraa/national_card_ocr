Aggregate results behind `manuscript.md` (no per-photo predictions: the dataset licence is unconfirmed).
Regenerate with `scripts/kaggle_eval.py --json`, `scripts/crnn_ablation.py --json`, `scripts/paired_tests.py`.

## Figures and tables

`scripts/make_paper_figures.py --results <dir> --out docs/paper` regenerates every figure and
`table_*.tex` of `paper.tex` from *full* result files (per-photo `records` for the bootstrap intervals,
per-card `confidence`/`correct` arrays for the precision-coverage, histogram and reliability plots).
Those per-photo files are not committed (they are tied to the dataset's photographs); the committed
JSON files here hold the aggregates. `pipeline_easyocr.json` is the EasyOCR baseline
(`scripts/baseline_easyocr.py`, Persian/Arabic-script model, nominal window, no adaptation).
Build the PDF with `cd docs/paper && latexmk -pdf paper.tex`.
