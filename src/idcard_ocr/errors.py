"""Exception hierarchy.

Every error carries a stable machine-readable ``code`` and an HTTP status so the API layer can
map it without string matching. Messages never contain extracted personal data.
"""

from __future__ import annotations

from typing import Any


class IdCardError(Exception):
    code = "idcard_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class InvalidImageError(IdCardError):
    code = "invalid_image"
    http_status = 400


class ImageTooLargeError(InvalidImageError):
    code = "image_too_large"
    http_status = 413


class CardNotFoundError(IdCardError):
    code = "card_not_found"
    http_status = 422


class ClassificationError(IdCardError):
    code = "classification_failed"
    http_status = 422


class QualityError(IdCardError):
    code = "poor_image_quality"
    http_status = 422


class OcrUnavailableError(IdCardError):
    code = "ocr_unavailable"
    http_status = 503


class OcrError(IdCardError):
    code = "ocr_failed"
    http_status = 502


class StoreDisabledError(IdCardError):
    code = "store_disabled"
    http_status = 409


class StorageError(IdCardError):
    code = "storage_error"
    http_status = 500


class ImageNotFoundError(IdCardError):
    code = "image_not_found"
    http_status = 404


class LayoutError(IdCardError):
    code = "invalid_layout"
    http_status = 500
