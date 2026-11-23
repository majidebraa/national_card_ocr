from __future__ import annotations

import cv2
import numpy as np
import pytest

from idcard_ocr.errors import ClassificationError
from idcard_ocr.imaging.classify import barcode_score, chip_score, classify, photo_score
from idcard_ocr.layouts import LayoutRegistry


@pytest.mark.parametrize("layout_id", ["old-front", "new-front", "old-back", "new-back"])
def test_flat_cards_are_classified(
    registry: LayoutRegistry, flat_cards: dict, layout_id: str
) -> None:
    res = classify(flat_cards[layout_id][0], registry)
    assert res.layout.id == layout_id and not res.rotated_180
    assert res.confidence > 0.5


@pytest.mark.parametrize("layout_id", ["old-front", "new-back"])
def test_upside_down_cards_are_detected(
    registry: LayoutRegistry, flat_cards: dict, layout_id: str
) -> None:
    flipped = cv2.rotate(flat_cards[layout_id][0], cv2.ROTATE_180)
    res = classify(flipped, registry)
    assert res.layout.id == layout_id and res.rotated_180


def test_hints_restrict_candidates(registry: LayoutRegistry, flat_cards: dict) -> None:
    res = classify(flat_cards["new-front"][0], registry, side="front", version="new")
    assert res.layout.id == "new-front"
    with pytest.raises(ClassificationError):
        classify(flat_cards["new-front"][0], registry, side="front", version="alien")


def test_blank_card_cannot_be_classified(registry: LayoutRegistry) -> None:
    with pytest.raises(ClassificationError):
        classify(np.full((638, 1011, 3), 128, np.uint8), registry)


def test_marker_scores_are_bounded() -> None:
    empty = np.zeros((0, 0, 3), np.uint8)
    assert photo_score(empty) == barcode_score(empty[:, :, 0]) == chip_score(empty) == 0.0
    noise = np.random.default_rng(1).integers(0, 255, (80, 80, 3), dtype=np.uint8)
    for score in (photo_score(noise), barcode_score(cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY))):
        assert 0.0 <= score <= 1.0
