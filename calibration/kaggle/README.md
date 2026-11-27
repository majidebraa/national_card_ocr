# Calibration for the Kaggle "Iranian National ID Card Dataset"

`new-front-kaggle.json` is a layout for the **new (smart) card front** as it appears in the
dataset [kibordevents/iranian-national-id-card-dataset](https://www.kaggle.com/datasets/kibordevents/iranian-national-id-card-dataset)
(fictional cards photographed in everyday conditions: hand-held, tilted, upside down, partly covered).

The images and their labels are **not** in this repository; their redistribution terms are not ours
to decide. Fetch them yourself:

```bash
scripts/download_dataset.sh                       # needs Kaggle credentials
CSV=data/raw/iranian_national_id_card_data.csv
# Resumable stages; the cache holds rectified, oriented cards (about 5 min for 1,856 photos on 2 cores)
python scripts/kaggle_eval.py --images data/raw --csv $CSV --cache data/cache --stage cache
python scripts/kaggle_eval.py --images data/raw --csv $CSV --cache data/cache --stage train \
    --train-model data/digit_model_kaggle.npz     # trains on id % 5 != 0 (about 12 min)
python scripts/kaggle_eval.py --images data/raw --csv $CSV --stage eval \
    --model data/digit_model_kaggle.npz --json data/results.json   # id % 5 == 0; accuracy, precision, recall, F1
```

Use the layout and model in the service with:

```bash
export IDCARD_LAYOUT_DIR=calibration/kaggle
```

(The bundled digit model was trained on fonts and does not read this card's numerals. A model
trained with `--train-model` does; ship it only if the dataset's license allows derived data.)

## Caveats

* The cards are fictional, and most national ids in the labels **fail the real checksum**, so the
  evaluation script does not use the checksum. With real cards the checksum is a strong filter
  that this dataset cannot demonstrate.
* Only new-card fronts are in the dataset. Backs and old cards are not covered by this calibration.
* Results on this set say nothing about forged cards or screen replays: it has no negatives.
