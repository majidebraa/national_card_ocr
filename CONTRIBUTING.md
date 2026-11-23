# Contributing

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-fas tesseract-ocr-ara tesseract-ocr-eng fonts-dejavu-core
python -m venv .venv && . .venv/bin/activate
make install
make lint typecheck test
```

* Python 3.11+. Style is enforced by `ruff` (lint + format) and `mypy`.
* Tests that need the Tesseract binary are marked `ocr` and skipped when it is missing.
* Coverage must stay at or above 80% (`make test`).
* Behaviour that affects accuracy needs a benchmark run: `make bench`. Report the before/after table.
* **Never** add real ID images, real names or real ID numbers to the repository, tests or issues.
  Use `idcard-ocr synth`.
* Keep PII out of logs and error messages.
* Use clear commits; update `CHANGELOG.md` and the docs when behaviour changes.

Security problems: see [SECURITY.md](SECURITY.md).
