.PHONY: install lint format typecheck test bench docker clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy src

test:
	pytest --cov

bench:
	idcard-ocr synth --out data/medium --count 60 --difficulty medium
	idcard-ocr bench data/medium --json data/medium.json
	python scripts/check_benchmark.py data/medium.json medium

docker:
	docker build -t iran-idcard-ocr:local .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
