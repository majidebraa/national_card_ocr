"""Jalali (Persian) date parsing, validation and Gregorian conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from idcard_ocr.validation.text import to_ascii_digits

MIN_JALALI_YEAR = 1200
MAX_JALALI_YEAR = 1500

_SEPARATORS = re.compile(r"[/\-.٫٬∕⁄\\|,:;]+")


def is_jalali_leap(year: int) -> bool:
    """33-year arithmetic leap rule (consistent with the official calendar for 1178-1633)."""
    return (25 * year + 11) % 33 < 8


def jalali_month_length(year: int, month: int) -> int:
    if month <= 6:
        return 31
    if month <= 11:
        return 30
    return 30 if is_jalali_leap(year) else 29


def is_valid_jalali(year: int, month: int, day: int) -> bool:
    if not MIN_JALALI_YEAR <= year <= MAX_JALALI_YEAR:
        return False
    if not 1 <= month <= 12:
        return False
    return 1 <= day <= jalali_month_length(year, month)


def jalali_to_gregorian(year: int, month: int, day: int) -> date:
    """Convert a valid Jalali date to a :class:`datetime.date`."""
    if not is_valid_jalali(year, month, day):
        raise ValueError("invalid Jalali date")
    jy = year + 1595
    days = -355668 + (365 * jy) + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + day
    days += (month - 1) * 31 if month < 7 else (month - 7) * 30 + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    month_lengths = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while gm < 12 and gd > month_lengths[gm]:
        gd -= month_lengths[gm]
        gm += 1
    return date(gy, gm + 1, gd)


@dataclass(frozen=True, slots=True)
class JalaliDateResult:
    value: str  # canonical YYYY/MM/DD (ASCII digits) when parseable, else cleaned raw text
    valid: bool
    gregorian: str | None
    issues: tuple[str, ...]


def parse_jalali(raw: str) -> JalaliDateResult:
    text = _SEPARATORS.sub("/", to_ascii_digits(raw)).strip()
    parts = [p for p in re.split(r"[/\s]+", text) if p]
    if len(parts) == 1 and re.fullmatch(r"\d{8}", parts[0]):
        parts = [parts[0][:4], parts[0][4:6], parts[0][6:]]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return JalaliDateResult(text, False, None, ("unparseable_date",))
    a, b, c = (int(p) for p in parts)
    # OCR on right-to-left text sometimes yields dd/mm/yyyy; accept it when the year is last.
    if len(parts[0]) <= 2 and len(parts[2]) == 4:
        a, c = c, a
    if not is_valid_jalali(a, b, c):
        return JalaliDateResult(f"{a:04d}/{b:02d}/{c:02d}", False, None, ("invalid_date",))
    greg = jalali_to_gregorian(a, b, c)
    return JalaliDateResult(f"{a:04d}/{b:02d}/{c:02d}", True, greg.isoformat(), ())
