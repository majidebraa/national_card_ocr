"""Persian text and digit normalisation."""

from __future__ import annotations

import re
import unicodedata

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(_PERSIAN_DIGITS)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_ARABIC_DIGITS)})

_CHAR_MAP = {
    "ي": "ی",  # Arabic yeh -> Persian yeh
    "ى": "ی",
    "ك": "ک",  # Arabic kaf -> Persian keheh
    "ۀ": "ه",
    "ة": "ه",
    "ؤ": "و",
    "أ": "ا",
    "إ": "ا",
}
_WS = re.compile(r"\s+")
ZWNJ = chr(0x200C)  # zero-width non-joiner, legitimate inside Persian words


def _char_class(*spans: int | tuple[int, int], negate: bool = False) -> re.Pattern[str]:
    """Character-class regex built from code points, so no invisible characters live in source."""
    parts: list[str] = []
    for span in spans:
        lo, hi = (span, span) if isinstance(span, int) else span
        if lo == hi:
            parts.append(re.escape(chr(lo)))
        else:
            parts.append(f"{re.escape(chr(lo))}-{re.escape(chr(hi))}")
    return re.compile("[" + ("^" if negate else "") + "".join(parts) + "]")


_PERSIAN_SPANS: tuple[int | tuple[int, int], ...] = (
    (0x0621, 0x063A),
    (0x0641, 0x064A),
    0x067E,
    0x0686,
    0x0698,
    0x06A9,
    0x06AF,
    0x06CC,
)
# Marks stripped from names: Arabic diacritics, tatweel, bidi controls, zero-width joiner, BOM.
_STRIP = _char_class(
    (0x064B, 0x065F),
    0x0670,
    0x0640,
    0x200D,
    0x200E,
    0x200F,
    (0x202A, 0x202E),
    (0x2066, 0x2069),
    0xFEFF,
)
_NON_PERSIAN = _char_class(*_PERSIAN_SPANS, 0x200C, 0x20, negate=True)
_PERSIAN_LETTERS = _char_class(*_PERSIAN_SPANS)


def to_ascii_digits(text: str) -> str:
    """Map Persian and Arabic-Indic digits to ASCII digits."""
    return text.translate(_DIGIT_MAP)


def digits_only(text: str) -> str:
    return re.sub(r"\D", "", to_ascii_digits(text))


def normalize_persian(text: str) -> str:
    """Unicode-normalise, unify Arabic/Persian letter variants, drop diacritics and noise."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)
    text = _STRIP.sub("", text)
    return _WS.sub(" ", text).strip()


def clean_name(text: str) -> tuple[str, list[str]]:
    """Normalise a name field and report what had to be removed (OCR noise)."""
    issues: list[str] = []
    norm = normalize_persian(text)
    stripped = _NON_PERSIAN.sub("", norm)
    stripped = _WS.sub(" ", stripped).strip()
    if stripped != norm:
        issues.append("removed_non_persian_characters")
    if stripped and len(_PERSIAN_LETTERS.findall(stripped)) < 2:
        issues.append("too_short")
    return stripped, issues
