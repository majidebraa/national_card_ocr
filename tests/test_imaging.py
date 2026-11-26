from __future__ import annotations

import io
import random

import cv2
import numpy as np
import pytest
from PIL import Image

from idcard_ocr.config import Settings
from idcard_ocr.errors import CardNotFoundError, ImageTooLargeError, InvalidImageError
from idcard_ocr.imaging.decode import decode_image, encode_jpeg
from idcard_ocr.imaging.detect import detect_card, order_quad, quad_width_px, rectify
from idcard_ocr.imaging.quality import (
    ISSUE_BLURRY,
    ISSUE_DARK,
    ISSUE_LOW_RES,
    assess_quality,
    blown_out_ratio,
    with_glare,
)
from idcard_ocr.layouts import LayoutRegistry
from idcard_ocr.synth.scene import compose_scene


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf.tobytes())


class TestDecode:
    def test_roundtrip(self) -> None:
        img = np.full((300, 400, 3), 120, np.uint8)
        out = decode_image(_png(img), max_pixels=10**7)
        assert out.shape == (300, 400, 3) and out.dtype == np.uint8

    @pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
    def test_garbage_is_rejected(self, data: bytes) -> None:
        with pytest.raises(InvalidImageError):
            decode_image(data, max_pixels=10**7)

    def test_too_many_pixels(self) -> None:
        with pytest.raises(ImageTooLargeError):
            decode_image(_png(np.zeros((400, 400, 3), np.uint8)), max_pixels=1000)

    def test_too_small(self) -> None:
        with pytest.raises(InvalidImageError):
            decode_image(_png(np.zeros((50, 50, 3), np.uint8)), max_pixels=10**7)

    def test_unsupported_format(self) -> None:
        buf = io.BytesIO()
        Image.new("RGB", (300, 300)).save(buf, format="BMP")
        with pytest.raises(InvalidImageError) as err:
            decode_image(buf.getvalue(), max_pixels=10**7)
        assert err.value.details["format"] == "BMP"

    def test_exif_orientation_is_applied(self) -> None:
        img = Image.new("RGB", (400, 300), (10, 200, 10))
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise on display
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        out = decode_image(buf.getvalue(), max_pixels=10**7)
        assert out.shape[:2] == (400, 300)

    def test_encode_jpeg(self) -> None:
        data = encode_jpeg(np.full((50, 50, 3), 90, np.uint8))
        assert data.startswith(b"\xff\xd8")


class TestDetect:
    def test_order_quad(self) -> None:
        pts = np.array([[100, 90], [10, 10], [10, 90], [100, 10]], dtype=np.float32)
        ordered = order_quad(pts)
        assert ordered.tolist() == [[10, 10], [100, 10], [100, 90], [10, 90]]

    @pytest.mark.parametrize("difficulty", ["easy", "medium"])
    def test_detects_synthetic_scenes(self, flat_cards: dict, difficulty: str) -> None:
        errors = []
        for i, (card, _) in enumerate(flat_cards.values()):
            scene = compose_scene(card, random.Random(i), difficulty)
            det = detect_card(scene.image)
            errors.append(float(np.abs(det.quad - scene.quad).max()))
            assert det.confidence > 0.35
        # corners within 4% of the frame width
        assert max(errors) < 0.04 * 1600

    def test_flat_scan_without_background(self, flat_cards: dict) -> None:
        card = next(iter(flat_cards.values()))[0]
        det = detect_card(card)
        assert det.method in {"full-frame", "border-distance", "canny", "otsu", "otsu-inv"}
        assert quad_width_px(det.quad) > 0.8 * card.shape[1]

    def test_blank_image_has_no_card(self) -> None:
        with pytest.raises(CardNotFoundError):
            detect_card(np.full((600, 900, 3), 128, np.uint8))

    def test_portrait_card_is_rotated_to_landscape(self, flat_cards: dict) -> None:
        card = next(iter(flat_cards.values()))[0]
        scene = compose_scene(cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE), random.Random(3), "easy")
        det = detect_card(scene.image)
        out = rectify(scene.image, det.quad, (1011, 638))
        assert out.shape[:2] == (638, 1011)


class TestQuality:
    def test_sharp_image_passes(self, flat_cards: dict) -> None:
        card = next(iter(flat_cards.values()))[0]
        rep = assess_quality(card, 1500, Settings())
        assert rep.ok, rep.issues

    def test_blur_dark_and_low_res_are_reported(self, flat_cards: dict) -> None:
        card = next(iter(flat_cards.values()))[0]
        bad = (cv2.GaussianBlur(card, (0, 0), 6) * 0.2).astype(np.uint8)
        rep = assess_quality(bad, 100, Settings())
        assert {ISSUE_BLURRY, ISSUE_DARK, ISSUE_LOW_RES} <= set(rep.issues)
        assert not rep.ok
        assert rep.to_dict()["ok"] is False

    def test_blown_out_regions_are_glare(self, registry: LayoutRegistry, flat_cards: dict) -> None:
        layout = registry.get("new-front")
        card = flat_cards["new-front"][0].copy()
        boxes = [f.roi for f in layout.fields.values()]
        assert blown_out_ratio(card, boxes, layout.size) == 0.0
        first = boxes[0]
        card[first.y : first.y + first.h, first.x : first.x + first.w] = 255
        ratio = blown_out_ratio(card, boxes, layout.size)
        assert ratio == pytest.approx(1 / len(boxes))
        rep = with_glare(assess_quality(card, 1500, Settings()), ratio, Settings())
        assert "glare" in rep.issues


class TestContentScore:
    QUAD = np.array([[20, 20], [380, 20], [380, 240], [20, 240]], np.float32)

    def test_flat_patch_scores_zero(self) -> None:
        from idcard_ocr.imaging.detect import _content_score

        flat = np.full((300, 400, 3), (180, 90, 160), np.uint8)  # coloured but featureless
        assert _content_score(self.QUAD, flat) == 0.0

    def test_neutral_grey_scores_zero(self) -> None:
        from idcard_ocr.imaging.detect import _content_score

        rng = np.random.default_rng(0)
        grey = rng.integers(60, 200, (300, 400, 1), dtype=np.uint8).repeat(3, axis=2)
        assert _content_score(self.QUAD, grey) == 0.0  # detailed but colourless (e.g. a portrait)

    def test_coloured_detailed_region_scores_high(self) -> None:
        from idcard_ocr.imaging.detect import _content_score

        rng = np.random.default_rng(0)
        card = np.full((300, 400, 3), (230, 215, 190), np.uint8)
        card[::6] = (60, 40, 30)  # text-like detail
        card = np.clip(card + rng.integers(0, 6, card.shape), 0, 255).astype(np.uint8)
        assert _content_score(self.QUAD, card) > 0.8
