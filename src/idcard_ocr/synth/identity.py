"""Random fictional identities with checksum-valid national ids and valid Jalali dates."""

from __future__ import annotations

import random
from dataclasses import dataclass

from idcard_ocr.validation.dates import jalali_month_length
from idcard_ocr.validation.national_id import with_check_digit
from idcard_ocr.validation.text import ZWNJ

FIRST_NAMES = [
    "علی",
    "محمد",
    "حسین",
    "رضا",
    "مهدی",
    "امیر",
    "حسن",
    "سعید",
    "کاوه",
    "بهرام",
    "پویا",
    "فاطمه",
    "زهرا",
    "مریم",
    "سارا",
    "نرگس",
    "پریسا",
    "الهام",
    "شیرین",
    "مینا",
    "ژاله",
    "گلناز",
]
LAST_NAMES = [
    "احمدی",
    "محمدی",
    "رضایی",
    "حسینی",
    "کریمی",
    "موسوی",
    "جعفری",
    "صادقی",
    "نوری",
    "کاظمی",
    "قاسمی",
    "رحیمی",
    "یوسفی",
    "اکبری",
    "شریفی",
    "زارعی",
    "پاکزاد",
    "چراغی",
    "گودرزی",
    f"ژیان{ZWNJ}فر",
]
FATHER_NAMES = [
    "عباس",
    "جواد",
    "اکبر",
    "اصغر",
    "ابراهیم",
    "اسماعیل",
    "یعقوب",
    "غلامرضا",
    "نصرالله",
    "فریدون",
    "منوچهر",
    "پرویز",
    "کامران",
    "بهمن",
    "داریوش",
]
SERIAL_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # no I/O: avoids look-alikes with 1/0


@dataclass(frozen=True, slots=True)
class Identity:
    national_id: str
    first_name: str
    last_name: str
    father_name: str
    birth_date: str  # YYYY/MM/DD (Jalali)
    expiry_date: str  # YYYY/MM/DD (Jalali)
    serial_no: str

    def values(self) -> dict[str, str]:
        return {
            "national_id": self.national_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "father_name": self.father_name,
            "birth_date": self.birth_date,
            "expiry_date": self.expiry_date,
            "serial_no": self.serial_no,
        }


def _date(rng: random.Random, y0: int, y1: int) -> str:
    year = rng.randint(y0, y1)
    month = rng.randint(1, 12)
    day = rng.randint(1, jalali_month_length(year, month))
    return f"{year:04d}/{month:02d}/{day:02d}"


def random_identity(rng: random.Random) -> Identity:
    while True:
        nine = "".join(str(rng.randint(0, 9)) for _ in range(9))
        nid = with_check_digit(nine)
        if len(set(nid)) > 1:
            break
    serial = (
        rng.choice(SERIAL_LETTERS)
        + rng.choice(SERIAL_LETTERS)
        + "".join(str(rng.randint(0, 9)) for _ in range(8))
    )
    return Identity(
        national_id=nid,
        first_name=rng.choice(FIRST_NAMES),
        last_name=rng.choice(LAST_NAMES),
        father_name=rng.choice(FATHER_NAMES),
        birth_date=_date(rng, 1335, 1385),
        expiry_date=_date(rng, 1405, 1415),
        serial_no=serial,
    )
