"""Iranian national id (کد ملی) validation.

The 10th digit is a check digit: with ``s = sum(d[i] * (10 - i) for i in 0..8)`` and
``r = s % 11``, the check digit is ``r`` when ``r < 2`` and ``11 - r`` otherwise.
Ids made of one repeated digit satisfy the formula but are never issued, so they are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from idcard_ocr.validation.text import digits_only


def check_digit(first_nine: str) -> int:
    if len(first_nine) != 9 or not first_nine.isdigit():
        raise ValueError("expected exactly nine ASCII digits")
    s = sum(int(d) * (10 - i) for i, d in enumerate(first_nine))
    r = s % 11
    return r if r < 2 else 11 - r


def is_valid_national_id(value: str) -> bool:
    if len(value) != 10 or not value.isdigit():
        return False
    if len(set(value)) == 1:
        return False
    return check_digit(value[:9]) == int(value[9])


def with_check_digit(first_nine: str) -> str:
    return first_nine + str(check_digit(first_nine))


@dataclass(frozen=True, slots=True)
class NationalIdResult:
    value: str
    valid: bool
    issues: tuple[str, ...]


def normalize_national_id(raw: str) -> NationalIdResult:
    """Normalise OCR text into a national id candidate and validate it.

    Leading zeros are frequently lost; 8-9 digit strings are zero-padded and accepted only if
    the padded value passes the checksum.
    """
    digits = digits_only(raw)
    if len(digits) == 10:
        if is_valid_national_id(digits):
            return NationalIdResult(digits, True, ())
        return NationalIdResult(digits, False, ("checksum_failed",))
    if 8 <= len(digits) < 10:
        padded = digits.zfill(10)
        if is_valid_national_id(padded):
            return NationalIdResult(padded, True, ("padded_leading_zeros",))
        return NationalIdResult(padded, False, ("wrong_length", "checksum_failed"))
    return NationalIdResult(digits, False, ("wrong_length",))
