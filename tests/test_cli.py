from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from idcard_ocr.cli import build_parser, main
from idcard_ocr.store import EncryptedImageStore, generate_key
from idcard_ocr.synth.dataset import Sample


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tessdata_dir: str | None) -> None:
    for name in ("IDCARD_STORE_DIR", "IDCARD_STORE_KEYS"):
        monkeypatch.delenv(name, raising=False)
    if tessdata_dir:
        monkeypatch.setenv("IDCARD_TESSDATA_DIR", tessdata_dir)


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_gen_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["gen-key"]) == 0
    assert len(capsys.readouterr().out.strip()) == 44


def test_synth_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["synth", "--out", str(tmp_path / "d"), "--count", "4", "--difficulty", "easy"]) == 0
    )
    assert (tmp_path / "d" / "labels.jsonl").is_file()
    assert "wrote 4 images" in capsys.readouterr().out


@pytest.mark.ocr
def test_extract_and_bench(
    easy_dataset: tuple[Path, list[Sample]],
    tesseract_ready,  # type: ignore[no-untyped-def]
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    directory, samples = easy_dataset
    front = next(s for s in samples if s.side == "front")
    assert main(["extract", str(directory / front.file)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["fields"]["national_id"]["value"] == front.fields["national_id"]

    report = tmp_path / "r.json"
    assert main(["bench", str(directory), "--limit", "4", "--json", str(report)]) == 0
    assert "silent error" in capsys.readouterr().out
    assert json.loads(report.read_text())["images"] == 4


@pytest.mark.ocr
def test_extract_reports_errors_as_json(
    tmp_path: Path,
    tesseract_ready,
    capsys: pytest.CaptureFixture[str],  # type: ignore[no-untyped-def]
) -> None:
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((500, 800, 3), 128, np.uint8))
    assert main(["extract", str(blank)]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "card_not_found"


def test_overlay(easy_dataset: tuple[Path, list[Sample]], tmp_path: Path) -> None:
    directory, samples = easy_dataset
    front = next(s for s in samples if s.side == "front")
    out = tmp_path / "o.png"
    assert (
        main(
            ["overlay", str(directory / front.file), "--layout", front.layout_id, "--out", str(out)]
        )
        == 0
    )
    assert cv2.imread(str(out)).shape[:2] == (638, 1011)


def test_purge_without_and_with_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["purge"]) == 2
    key = generate_key()
    monkeypatch.setenv("IDCARD_STORE_DIR", str(tmp_path / "v"))
    monkeypatch.setenv("IDCARD_STORE_KEYS", key)
    store = EncryptedImageStore(tmp_path / "v", (key,), 30)
    from datetime import UTC, datetime, timedelta

    store.save({"original": b"x" * 64}, now=datetime.now(UTC) - timedelta(days=40))
    capsys.readouterr()
    assert main(["purge"]) == 0
    assert "purged 1" in capsys.readouterr().out


def test_train_digits_writes_a_loadable_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from idcard_ocr.ocr.digits import DigitModel

    out = tmp_path / "model.npz"
    assert main(["train-digits", "--out", str(out), "--per-class", "12", "--seed", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] > 100
    DigitModel.load(out)
