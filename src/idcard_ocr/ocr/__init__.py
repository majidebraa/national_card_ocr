"""OCR engines behind a small, swappable interface."""

from idcard_ocr.ocr.base import OcrEngine, OcrText
from idcard_ocr.ocr.tesseract import TesseractEngine

__all__ = ["OcrEngine", "OcrText", "TesseractEngine"]
