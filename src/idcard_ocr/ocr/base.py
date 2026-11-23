"""The OCR engine interface.

Anything that can turn a small, single-line image crop into text can plug in here: a different
Tesseract model, PaddleOCR through ONNX, or a fine-tuned VLM used as a low-traffic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class OcrText:
    text: str
    confidence: float  # 0..1; 0 when the engine has no opinion


class OcrEngine(Protocol):
    name: str

    def recognize(
        self,
        image: np.ndarray,
        *,
        psm: int = 7,
        whitelist: str | None = None,
        lang: str | None = None,
    ) -> OcrText:
        """Recognise one line of text. ``image`` is a single-channel uint8 array.

        ``lang`` optionally overrides the engine's default language for this call.
        """
        ...

    def check_ready(self) -> None:
        """Raise :class:`idcard_ocr.errors.OcrUnavailableError` if the engine cannot run."""
        ...
