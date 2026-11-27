"""Result models. Field values are personal data: they are returned to the caller, never logged."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FieldResult(BaseModel):
    value: str = Field(description="Normalised value (ASCII digits, canonical date, clean name).")
    raw: str = Field(description="Text exactly as recognised, before normalisation.")
    confidence: float = Field(ge=0.0, le=1.0)
    valid: bool = Field(description="Passed the field's validation (checksum, date, pattern).")
    needs_review: bool = Field(description="Low confidence or invalid: ask the user to confirm.")
    issues: list[str] = Field(default_factory=list)
    gregorian: str | None = Field(default=None, description="ISO date for Jalali date fields.")


class QualityInfo(BaseModel):
    sharpness: float
    brightness: float
    glare_ratio: float
    source_width_px: float
    issues: list[str]
    ok: bool


class StoredRef(BaseModel):
    image_id: str
    sha256: str
    created_at: str
    expires_at: str
    variants: list[str]


class PortraitInfo(BaseModel):
    """Face photo on the card. The pixels are only available from the encrypted store."""

    face_found: bool = Field(description="A face was detected; False means the crop is a guess.")
    confidence: float = Field(ge=0.0, le=1.0)
    box: list[int] = Field(description="x, y, w, h of the portrait crop on the rectified card.")
    face_box: list[int] | None = None


class CardResult(BaseModel):
    version: str
    side: str
    layout_id: str
    classification_confidence: float = Field(ge=0.0, le=1.0)
    detection_confidence: float = Field(ge=0.0, le=1.0)
    fields: dict[str, FieldResult]
    quality: QualityInfo
    warnings: list[str] = Field(default_factory=list)
    portrait: PortraitInfo | None = None
    stored: StoredRef | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return any(f.needs_review for f in self.fields.values())


class PairResult(BaseModel):
    front: CardResult
    back: CardResult
    consistent: bool = Field(description="Both sides belong to the same card version.")
    warnings: list[str] = Field(default_factory=list)
