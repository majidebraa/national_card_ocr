from __future__ import annotations

from pathlib import Path

import pytest

from idcard_ocr.config import Settings
from idcard_ocr.errors import (
    CardNotFoundError,
    ClassificationError,
    InvalidImageError,
    QualityError,
    StoreDisabledError,
)
from idcard_ocr.pipeline import Pipeline
from idcard_ocr.store import EncryptedImageStore, generate_key
from idcard_ocr.synth.dataset import Sample
from tests.conftest import FakeEngine


def _bytes(dataset: tuple[Path, list[Sample]], index: int) -> tuple[bytes, Sample]:
    directory, samples = dataset
    return (directory / samples[index].file).read_bytes(), samples[index]


def test_front_card_end_to_end_without_tesseract(
    fake_pipeline: Pipeline, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    fronts = [i for i, s in enumerate(easy_dataset[1]) if s.side == "front"]
    data, sample = _bytes(easy_dataset, fronts[0])
    result = fake_pipeline.process(data)
    assert (result.version, result.side) == (sample.version, sample.side)
    assert result.fields["national_id"].value == sample.fields["national_id"]
    assert result.fields["national_id"].valid and not result.fields["national_id"].needs_review
    assert result.fields["birth_date"].value == sample.fields["birth_date"]
    assert result.fields["birth_date"].gregorian is not None
    assert set(result.timings_ms) >= {"decode", "detect", "ocr", "total"}


def test_free_text_names_always_need_review(
    fake_pipeline: Pipeline, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    fronts = [i for i, s in enumerate(easy_dataset[1]) if s.side == "front"]
    result = fake_pipeline.process(_bytes(easy_dataset, fronts[0])[0])
    for name in ("first_name", "last_name", "father_name"):
        assert result.fields[name].needs_review
        assert "free_text_unverified" in result.fields[name].issues
    assert result.needs_review


def test_trusting_names_is_opt_in(
    settings: Settings, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    from dataclasses import replace

    pipe = Pipeline(replace(settings, trust_names=True), engine=FakeEngine("علی", 0.95))
    fronts = [i for i, s in enumerate(easy_dataset[1]) if s.side == "front"]
    result = pipe.process(_bytes(easy_dataset, fronts[0])[0])
    assert not result.fields["first_name"].needs_review


def test_upside_down_photo_is_corrected(
    fake_pipeline: Pipeline, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    import cv2
    import numpy as np

    fronts = [i for i, s in enumerate(easy_dataset[1]) if s.side == "front" and s.rotation == 0]
    data, sample = _bytes(easy_dataset, fronts[0])
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".jpg", cv2.rotate(image, cv2.ROTATE_180))
    assert ok
    result = fake_pipeline.process(bytes(buf.tobytes()))
    assert "card_was_upside_down" in result.warnings
    assert result.fields["national_id"].value == sample.fields["national_id"]


def test_hints_are_respected(
    fake_pipeline: Pipeline, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    data, sample = _bytes(easy_dataset, 0)
    result = fake_pipeline.process(data, side=sample.side, version=sample.version)
    assert result.layout_id == sample.layout_id


def test_garbage_and_blank_inputs(fake_pipeline: Pipeline) -> None:
    with pytest.raises(InvalidImageError):
        fake_pipeline.process(b"nope")
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".png", np.full((500, 800, 3), 128, np.uint8))
    assert ok
    with pytest.raises(CardNotFoundError):
        fake_pipeline.process(bytes(buf.tobytes()))


def test_quality_enforcement_and_warn_mode(
    settings: Settings, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    from dataclasses import replace

    data, _ = _bytes(easy_dataset, 0)
    strict = Pipeline(replace(settings, min_sharpness=1e9), engine=FakeEngine())
    with pytest.raises(QualityError) as err:
        strict.process(data)
    assert "blurry" in err.value.details["issues"]
    lenient = Pipeline(
        replace(settings, min_sharpness=1e9, quality_mode="warn"), engine=FakeEngine()
    )
    result = lenient.process(data)
    assert "quality:blurry" in result.warnings and not result.quality.ok


def test_store_requested_but_not_configured(
    fake_pipeline: Pipeline, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    with pytest.raises(StoreDisabledError):
        fake_pipeline.process(_bytes(easy_dataset, 0)[0], store=True)


def test_image_retention(
    settings: Settings, tmp_path: Path, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    store = EncryptedImageStore(tmp_path / "v", (generate_key(),), 30)
    pipe = Pipeline(settings, engine=FakeEngine(), store=store)
    data, sample = _bytes(easy_dataset, 0)
    result = pipe.process(data, store=True)
    assert result.stored is not None
    assert store.load(result.stored.image_id) == data  # original bytes, untouched
    card = store.load(result.stored.image_id, "card")
    assert card.startswith(b"\xff\xd8")
    assert store.info(result.stored.image_id).version == sample.version
    # the stored metadata holds no extracted personal data
    meta = (tmp_path / "v" / f"{result.stored.image_id}.json").read_text()
    assert sample.fields.get("national_id", "never-present") not in meta


def test_pair_processing(fake_pipeline: Pipeline, easy_dataset: tuple[Path, list[Sample]]) -> None:
    samples = easy_dataset[1]
    version = samples[0].version
    front = next(i for i, s in enumerate(samples) if s.side == "front" and s.version == version)
    back = next(i for i, s in enumerate(samples) if s.side == "back" and s.version == version)
    pair = fake_pipeline.process_pair(_bytes(easy_dataset, front)[0], _bytes(easy_dataset, back)[0])
    assert pair.consistent and pair.front.side == "front" and pair.back.side == "back"


def test_wrong_side_hint_does_not_silently_succeed(
    fake_pipeline: Pipeline, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    samples = easy_dataset[1]
    front = next(i for i, s in enumerate(samples) if s.side == "front")
    try:
        result = fake_pipeline.process(_bytes(easy_dataset, front)[0], side="back")
    except ClassificationError:
        return
    # if it does classify, it must be at low confidence and flagged
    assert result.classification_confidence < 0.9 or result.needs_review


@pytest.mark.ocr
def test_real_tesseract_reads_serial_and_flags_names(
    settings: Settings,
    tesseract_ready,
    easy_dataset: tuple[Path, list[Sample]],  # type: ignore[no-untyped-def]
) -> None:
    pipe = Pipeline(settings)
    backs = [i for i, s in enumerate(easy_dataset[1]) if s.side == "back"]
    ok = 0
    for i in backs:
        data, sample = _bytes(easy_dataset, i)
        ok += pipe.process(data).fields["serial_no"].value == sample.fields["serial_no"]
    assert ok >= max(1, len(backs) - 1)
