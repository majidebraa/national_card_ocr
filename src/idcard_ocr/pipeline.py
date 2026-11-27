"""End-to-end pipeline: decode -> detect -> rectify -> quality -> classify -> OCR -> (store)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2

from idcard_ocr.config import Settings
from idcard_ocr.errors import QualityError, StoreDisabledError
from idcard_ocr.fields import SHIFTS, locate_offset, read_field
from idcard_ocr.imaging.classify import classify
from idcard_ocr.imaging.decode import decode_image, encode_jpeg
from idcard_ocr.imaging.detect import detect_card, quad_width_px, rectify
from idcard_ocr.imaging.quality import (
    QualityReport,
    assess_quality,
    blown_out_ratio,
    with_glare,
)
from idcard_ocr.layouts import LayoutRegistry
from idcard_ocr.models import (
    CardResult,
    FieldResult,
    PairResult,
    PortraitInfo,
    QualityInfo,
    StoredRef,
)
from idcard_ocr.ocr.base import OcrEngine
from idcard_ocr.ocr.digits import MODEL_PATH, DigitEngine
from idcard_ocr.ocr.tesseract import TesseractEngine
from idcard_ocr.portrait import PortraitExtractor, default_model_path

if TYPE_CHECKING:
    from idcard_ocr.store import EncryptedImageStore


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        engine: OcrEngine | None = None,
        registry: LayoutRegistry | None = None,
        store: EncryptedImageStore | None = None,
        digit_engine: DigitEngine | None = None,
    ) -> None:
        self.settings = settings
        cv2.setNumThreads(settings.cv_threads)
        self.registry = registry or LayoutRegistry.load(settings.layout_dir)
        self.engine: OcrEngine = engine or TesseractEngine(
            lang=settings.ocr_lang,
            tessdata_dir=settings.tessdata_dir,
            timeout_s=settings.ocr_timeout_s,
        )
        self.store = store
        # The digit recognizer is used for the national id and dates when its model is present;
        # otherwise Tesseract handles every field.
        self.digit_engine: DigitEngine | None = digit_engine or (
            DigitEngine() if MODEL_PATH.is_file() else None
        )

        self.portraits: PortraitExtractor | None = (
            PortraitExtractor()
            if settings.extract_portrait and default_model_path().is_file()
            else None
        )

    def check_ready(self) -> None:
        self.engine.check_ready()
        if self.digit_engine is not None:
            self.digit_engine.check_ready()

    def _gate(self, quality: QualityReport, warnings: list[str], only: str | None = None) -> None:
        issues = [i for i in quality.issues if only is None or i == only]
        if not issues:
            return
        if self.settings.quality_mode == "enforce":
            raise QualityError(
                "The image quality is too low to read reliably.", details={"issues": issues}
            )
        warnings.extend(f"quality:{issue}" for issue in issues)

    def process(
        self,
        data: bytes,
        *,
        side: str | None = None,
        version: str | None = None,
        store: bool = False,
    ) -> CardResult:
        """Process one uploaded image. Raises :class:`idcard_ocr.errors.IdCardError` subclasses."""
        if store and self.store is None:
            raise StoreDisabledError(
                "Image retention was requested but no encrypted store is configured."
            )
        s = self.settings
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - since) * 1000.0, 1)
            return now

        image = decode_image(data, max_pixels=s.max_pixels)
        t = lap("decode", t0)

        detection = detect_card(image, max_side=s.detect_max_side)
        card = rectify(image, detection.quad, (s.card_width, s.card_height))
        t = lap("detect", t)

        quality = assess_quality(card, quad_width_px(detection.quad), s)
        warnings: list[str] = []
        self._gate(quality, warnings)
        t = lap("quality", t)

        cls = classify(card, self.registry, side=side, version=version)
        if cls.rotated_180:
            card = cv2.rotate(card, cv2.ROTATE_180)
            warnings.append("card_was_upside_down")
        layout = cls.layout
        ratio = blown_out_ratio(card, (f.roi for f in layout.fields.values()), layout.size)
        quality = with_glare(quality, ratio, s)
        self._gate(quality, warnings, only="glare")
        t = lap("classify", t)

        photo_box = layout.markers.get("photo")
        if self.portraits is not None and photo_box is not None and layout.side == "front":
            card, flipped = self.portraits.orient(card, photo_box)
            if flipped:
                warnings.append("card_orientation_corrected")
            t = lap("orient", t)
        offset = (0, 0)
        max_shifts = len(SHIFTS)
        anchor = next((f for f in layout.fields.values() if f.kind == "national_id"), None)
        if anchor is not None and self.digit_engine is not None:
            found = locate_offset(self.digit_engine, card, anchor, layout.size)
            if found is None:
                # No line read as ten digits: the photo is probably unusable, so do not burn CPU
                # on retries; every digit field is flagged for review below.
                warnings.append("text_block_not_located")
                max_shifts = 1
            else:
                offset = found
                max_shifts = 3  # the anchor is accurate; only small local jitter remains
                if offset != (0, 0):
                    warnings.append("text_block_shifted")

        fields: dict[str, FieldResult] = {}
        for name, spec in layout.fields.items():
            fields[name] = read_field(
                self.engine,
                card,
                spec,
                layout.size,
                review_confidence=s.review_confidence,
                latin_lang=s.ocr_latin_lang,
                alt_lang=s.ocr_alt_lang,
                trust_names=s.trust_names,
                digit_engine=self.digit_engine,
                offset=offset,
                max_shifts=max_shifts,
            )
        t = lap("ocr", t)

        portrait_info: PortraitInfo | None = None
        portrait_jpeg: bytes | None = None
        if self.portraits is not None and photo_box is not None:
            found_portrait = self.portraits.extract(card, photo_box, offset)
            portrait_info = PortraitInfo(
                face_found=found_portrait.face_found,
                confidence=found_portrait.confidence,
                box=list(found_portrait.box),
                face_box=list(found_portrait.face_box) if found_portrait.face_box else None,
            )
            portrait_jpeg = encode_jpeg(found_portrait.image)
            if not found_portrait.face_found:
                warnings.append("portrait_face_not_found")
            t = lap("portrait", t)

        stored: StoredRef | None = None
        if store and self.store is not None:
            blobs = {"original": data, "card": encode_jpeg(card)}
            if portrait_jpeg is not None:
                blobs["portrait"] = portrait_jpeg
            saved = self.store.save(
                blobs,
                side=layout.side,
                version=layout.version,
            )
            stored = StoredRef(
                image_id=saved.image_id,
                sha256=saved.sha256,
                created_at=saved.created_at,
                expires_at=saved.expires_at,
                variants=list(saved.variants),
            )
            lap("store", t)
        timings["total"] = round((time.perf_counter() - t0) * 1000.0, 1)

        return CardResult(
            version=layout.version,
            side=layout.side,
            layout_id=layout.id,
            classification_confidence=round(cls.confidence, 4),
            detection_confidence=round(detection.confidence, 4),
            fields=fields,
            quality=QualityInfo(**quality.to_dict()),  # type: ignore[arg-type]
            warnings=warnings,
            portrait=portrait_info,
            stored=stored,
            timings_ms=timings,
        )

    def process_pair(
        self,
        front: bytes,
        back: bytes,
        *,
        version: str | None = None,
        store: bool = False,
    ) -> PairResult:
        """Process both sides of one card and cross-check that they agree."""
        front_res = self.process(front, side="front", version=version, store=store)
        # Reuse the version found on the front so a noisy back cannot flip to the other design.
        back_res = self.process(
            back, side="back", version=version or front_res.version, store=store
        )
        consistent = front_res.version == back_res.version
        warnings = [] if consistent else ["front_and_back_versions_differ"]
        return PairResult(front=front_res, back=back_res, consistent=consistent, warnings=warnings)
