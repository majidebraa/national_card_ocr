from __future__ import annotations

import pytest

from idcard_ocr.config import Settings


def test_defaults_are_conservative() -> None:
    s = Settings.from_env({})
    assert s.max_concurrency >= 1
    assert s.quality_mode == "enforce"
    assert s.trust_names is False
    assert s.allow_anonymous is False
    assert s.api_keys == ()


def test_env_parsing() -> None:
    s = Settings.from_env(
        {
            "IDCARD_MAX_UPLOAD_BYTES": "1024",
            "IDCARD_API_KEYS": "a, b ,,c",
            "IDCARD_ALLOW_ANONYMOUS": "YES",
            "IDCARD_MIN_SHARPNESS": "12.5",
            "IDCARD_QUALITY_MODE": "WARN",
            "IDCARD_MAX_CONCURRENCY": "0",
            "IDCARD_OCR_ALT_LANG": "",
            "IDCARD_TRUST_NAMES": "1",
        }
    )
    assert s.max_upload_bytes == 1024
    assert s.api_keys == ("a", "b", "c")
    assert s.allow_anonymous and s.trust_names
    assert s.min_sharpness == 12.5
    assert s.quality_mode == "warn"
    assert s.max_concurrency == 1  # never below one worker
    assert s.ocr_alt_lang == ""


@pytest.mark.parametrize("name", ["IDCARD_MAX_PIXELS", "IDCARD_RETENTION_DAYS"])
def test_bad_integers_fail_loudly(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        Settings.from_env({name: "many"})


def test_bad_float_fails_loudly() -> None:
    with pytest.raises(ValueError, match="IDCARD_MAX_GLARE_RATIO"):
        Settings.from_env({"IDCARD_MAX_GLARE_RATIO": "lots"})
