from __future__ import annotations

import pytest

from idcard_ocr.validation.dates import (
    is_jalali_leap,
    is_valid_jalali,
    jalali_month_length,
    jalali_to_gregorian,
    parse_jalali,
)
from idcard_ocr.validation.national_id import (
    check_digit,
    is_valid_national_id,
    normalize_national_id,
    with_check_digit,
)
from idcard_ocr.validation.text import (
    ZWNJ,
    clean_name,
    digits_only,
    normalize_persian,
    to_ascii_digits,
)


class TestNationalId:
    # Fictional numbers built with the public checksum rule.
    @pytest.mark.parametrize("nid", ["0499370899", "1234567891", "0012345679"])
    def test_valid(self, nid: str) -> None:
        assert is_valid_national_id(nid)

    @pytest.mark.parametrize(
        "nid", ["0499370898", "123456789", "12345678901", "abcdefghij", "", "1111111111"]
    )
    def test_invalid(self, nid: str) -> None:
        assert not is_valid_national_id(nid)

    def test_check_digit_round_trip(self) -> None:
        for nine in ("123456789", "000000001", "987654321"):
            assert is_valid_national_id(with_check_digit(nine)) or len(set(nine)) == 1

    def test_check_digit_requires_nine_digits(self) -> None:
        with pytest.raises(ValueError):
            check_digit("12345")

    def test_persian_digits_are_normalised(self) -> None:
        res = normalize_national_id("۰۴۹۹۳۷۰۸۹۹")
        assert res.valid and res.value == "0499370899" and not res.issues

    def test_arabic_indic_digits_and_noise(self) -> None:
        assert normalize_national_id("٠٤٩٩ ٣٧٠ ٨٩٩").valid

    def test_lost_leading_zero_is_padded_only_if_checksum_passes(self) -> None:
        res = normalize_national_id("499370899")
        assert res.valid and res.value == "0499370899"
        assert "padded_leading_zeros" in res.issues
        bad = normalize_national_id("499370898")
        assert not bad.valid and "checksum_failed" in bad.issues

    def test_wrong_length(self) -> None:
        res = normalize_national_id("12345")
        assert not res.valid and res.issues == ("wrong_length",)

    def test_checksum_failure_keeps_value_for_review(self) -> None:
        res = normalize_national_id("0499370898")
        assert res.value == "0499370898" and "checksum_failed" in res.issues


class TestDates:
    def test_leap_years(self) -> None:
        assert [y for y in range(1395, 1410) if is_jalali_leap(y)] == [1395, 1399, 1403, 1408]

    def test_month_lengths(self) -> None:
        assert jalali_month_length(1400, 1) == 31
        assert jalali_month_length(1400, 7) == 30
        assert jalali_month_length(1400, 12) == 29
        assert jalali_month_length(1399, 12) == 30

    @pytest.mark.parametrize(
        ("jalali", "gregorian"),
        [
            ((1403, 1, 1), "2024-03-20"),
            ((1399, 12, 30), "2021-03-20"),
            ((1370, 5, 12), "1991-08-03"),
            ((1360, 5, 10), "1981-08-01"),
            ((1404, 7, 1), "2025-09-23"),
        ],
    )
    def test_conversion(self, jalali: tuple[int, int, int], gregorian: str) -> None:
        assert jalali_to_gregorian(*jalali).isoformat() == gregorian

    @pytest.mark.parametrize(
        "ymd",
        [(1400, 13, 1), (1400, 0, 1), (1400, 1, 32), (1400, 7, 31), (1400, 12, 30), (900, 1, 1)],
    )
    def test_invalid_dates(self, ymd: tuple[int, int, int]) -> None:
        assert not is_valid_jalali(*ymd)
        with pytest.raises(ValueError):
            jalali_to_gregorian(*ymd)

    def test_parse_variants(self) -> None:
        for raw in ("۱۳۷۰/۰۵/۱۲", "1370-05-12", "1370.5.12", "13700512", "1370 05 12"):
            res = parse_jalali(raw)
            assert res.valid and res.value == "1370/05/12" and res.gregorian == "1991-08-03"

    def test_day_first_is_accepted_when_year_is_last(self) -> None:
        assert parse_jalali("12/05/1370").value == "1370/05/12"

    def test_unparseable_and_invalid(self) -> None:
        assert parse_jalali("hello").issues == ("unparseable_date",)
        res = parse_jalali("1370/13/40")
        assert not res.valid and res.issues == ("invalid_date",)


class TestText:
    def test_digit_mapping(self) -> None:
        assert to_ascii_digits("۱۲۳٤٥٦") == "123456"
        assert digits_only("کد: ۱۲ ۳") == "123"

    def test_normalize_persian_unifies_variants(self) -> None:
        assert normalize_persian("عليرضا  كريمي") == "علیرضا کریمی"
        assert normalize_persian("حَسَن") == "حسن"

    def test_clean_name_flags_noise(self) -> None:
        value, issues = clean_name("علی ۱۲ x")
        assert value == "علی" and "removed_non_persian_characters" in issues

    def test_clean_name_too_short(self) -> None:
        _, issues = clean_name("ع")
        assert "too_short" in issues

    def test_zwnj_is_preserved(self) -> None:
        value, _ = clean_name(f"ژیان{ZWNJ}فر")
        assert ZWNJ in value
