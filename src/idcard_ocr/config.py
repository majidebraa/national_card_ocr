"""Runtime configuration, read from environment variables (12-factor style)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_TRUE = {"1", "true", "yes", "on"}


def _str(env: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _str(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _str(env, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _str(env, name)
    return default if raw is None else raw.lower() in _TRUE


def _csv(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = _str(env, name, "") or ""
    return tuple(p.strip() for p in raw.split(",") if p.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    # --- input limits -------------------------------------------------------------------
    max_upload_bytes: int = 8 * 1024 * 1024
    max_pixels: int = 40_000_000
    # --- image processing ---------------------------------------------------------------
    detect_max_side: int = 960  # detection runs on a downscaled copy to save CPU
    card_width: int = 1011  # ID-1 card (85.6 x 54 mm) at roughly 300 dpi
    card_height: int = 638
    # --- quality gates ------------------------------------------------------------------
    min_sharpness: float = 45.0
    min_brightness: float = 50.0
    max_brightness: float = 225.0
    max_glare_ratio: float = 0.05
    min_source_width_px: int = 520
    quality_mode: str = "enforce"  # "enforce" rejects bad images, "warn" only reports
    # --- OCR ----------------------------------------------------------------------------
    ocr_lang: str = "fas"
    ocr_alt_lang: str = "ara"  # second opinion for names; empty string disables it
    ocr_latin_lang: str = "eng"  # used for Latin/digit-only fields such as the serial number
    tessdata_dir: str | None = None
    ocr_timeout_s: float = 20.0
    review_confidence: float = 0.60  # fields below this are flagged needs_review
    # Tesseract's confidence is poorly calibrated for Persian free text (wrong names are often
    # read with high confidence), so names always require user confirmation unless this is on.
    trust_names: bool = False
    extract_portrait: bool = True  # crop the face photo (YuNet, ~40 ms/card)
    # --- concurrency (resource-limited hosts: keep this small) ---------------------------
    max_concurrency: int = 2
    cv_threads: int = 1  # OpenCV threads per request; concurrency comes from parallel requests
    queue_timeout_s: float = 10.0
    request_timeout_s: float = 90.0
    # --- encrypted image store ----------------------------------------------------------
    store_dir: str | None = None
    store_keys: tuple[str, ...] = ()
    retention_days: int = 30
    # --- API auth -----------------------------------------------------------------------
    api_keys: tuple[str, ...] = ()
    allow_anonymous: bool = False
    # --- layouts ------------------------------------------------------------------------
    layout_dir: str | None = None  # extra/override layout JSON files

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e: Mapping[str, str] = os.environ if env is None else env
        d = cls()
        cpus = os.cpu_count() or 1
        return cls(
            max_upload_bytes=_int(e, "IDCARD_MAX_UPLOAD_BYTES", d.max_upload_bytes),
            max_pixels=_int(e, "IDCARD_MAX_PIXELS", d.max_pixels),
            detect_max_side=_int(e, "IDCARD_DETECT_MAX_SIDE", d.detect_max_side),
            card_width=_int(e, "IDCARD_CARD_WIDTH", d.card_width),
            card_height=_int(e, "IDCARD_CARD_HEIGHT", d.card_height),
            min_sharpness=_float(e, "IDCARD_MIN_SHARPNESS", d.min_sharpness),
            min_brightness=_float(e, "IDCARD_MIN_BRIGHTNESS", d.min_brightness),
            max_brightness=_float(e, "IDCARD_MAX_BRIGHTNESS", d.max_brightness),
            max_glare_ratio=_float(e, "IDCARD_MAX_GLARE_RATIO", d.max_glare_ratio),
            min_source_width_px=_int(e, "IDCARD_MIN_SOURCE_WIDTH", d.min_source_width_px),
            quality_mode=(_str(e, "IDCARD_QUALITY_MODE", d.quality_mode) or d.quality_mode).lower(),
            ocr_lang=_str(e, "IDCARD_OCR_LANG", d.ocr_lang) or d.ocr_lang,
            ocr_alt_lang=e.get("IDCARD_OCR_ALT_LANG", d.ocr_alt_lang).strip(),
            ocr_latin_lang=_str(e, "IDCARD_OCR_LATIN_LANG", d.ocr_latin_lang) or d.ocr_latin_lang,
            tessdata_dir=_str(e, "IDCARD_TESSDATA_DIR", d.tessdata_dir),
            ocr_timeout_s=_float(e, "IDCARD_OCR_TIMEOUT", d.ocr_timeout_s),
            trust_names=_bool(e, "IDCARD_TRUST_NAMES", d.trust_names),
            extract_portrait=_bool(e, "IDCARD_EXTRACT_PORTRAIT", d.extract_portrait),
            review_confidence=_float(e, "IDCARD_REVIEW_CONFIDENCE", d.review_confidence),
            max_concurrency=max(1, _int(e, "IDCARD_MAX_CONCURRENCY", min(2, cpus))),
            cv_threads=max(1, _int(e, "IDCARD_CV_THREADS", d.cv_threads)),
            queue_timeout_s=_float(e, "IDCARD_QUEUE_TIMEOUT", d.queue_timeout_s),
            request_timeout_s=_float(e, "IDCARD_REQUEST_TIMEOUT", d.request_timeout_s),
            store_dir=_str(e, "IDCARD_STORE_DIR", d.store_dir),
            store_keys=_csv(e, "IDCARD_STORE_KEYS"),
            retention_days=_int(e, "IDCARD_RETENTION_DAYS", d.retention_days),
            api_keys=_csv(e, "IDCARD_API_KEYS"),
            allow_anonymous=_bool(e, "IDCARD_ALLOW_ANONYMOUS", d.allow_anonymous),
            layout_dir=_str(e, "IDCARD_LAYOUT_DIR", d.layout_dir),
        )
