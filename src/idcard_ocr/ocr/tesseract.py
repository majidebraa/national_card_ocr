"""Tesseract engine (CPU only, ``tessdata_fast`` models are enough)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytesseract
from pytesseract import Output, TesseractError, TesseractNotFoundError

from idcard_ocr.errors import OcrError, OcrUnavailableError
from idcard_ocr.ocr.base import OcrText


class TesseractEngine:
    name = "tesseract"

    def __init__(
        self,
        *,
        lang: str = "fas",
        tessdata_dir: str | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        self.lang = lang
        self.tessdata_dir = tessdata_dir
        self.timeout_s = timeout_s

    def _config(self, psm: int, whitelist: str | None) -> str:
        parts = [f"--oem 1 --psm {int(psm)}", "-c load_system_dawg=0 -c load_freq_dawg=0"]
        if self.tessdata_dir:
            parts.append(f'--tessdata-dir "{self.tessdata_dir}"')
        if whitelist:
            parts.append(f"-c tessedit_char_whitelist={whitelist}")
        return " ".join(parts)

    def check_ready(self) -> None:
        if shutil.which("tesseract") is None:
            raise OcrUnavailableError("The tesseract binary is not installed.")
        for code in self.lang.split("+"):
            if self.tessdata_dir:
                if not (Path(self.tessdata_dir) / f"{code}.traineddata").is_file():
                    raise OcrUnavailableError(f"Tesseract language data {code!r} is missing.")
            else:
                try:
                    langs = pytesseract.get_languages(config="")
                except (TesseractError, TesseractNotFoundError) as exc:
                    raise OcrUnavailableError("Could not list Tesseract languages.") from exc
                if code not in langs:
                    raise OcrUnavailableError(f"Tesseract language data {code!r} is missing.")

    def recognize(
        self,
        image: np.ndarray,
        *,
        psm: int = 7,
        whitelist: str | None = None,
        lang: str | None = None,
    ) -> OcrText:
        try:
            data = pytesseract.image_to_data(
                image,
                lang=lang or self.lang,
                config=self._config(psm, whitelist),
                output_type=Output.DICT,
                timeout=self.timeout_s,
            )
        except TesseractNotFoundError as exc:
            raise OcrUnavailableError("The tesseract binary is not installed.") from exc
        except TesseractError as exc:  # subclass of RuntimeError: must be caught first
            raise OcrError("Tesseract failed on this image.") from exc
        except RuntimeError as exc:  # pytesseract signals a timeout with a bare RuntimeError
            raise OcrError("OCR timed out.") from exc

        words: list[str] = []
        confs: list[float] = []
        for text, conf in zip(data["text"], data["conf"], strict=True):
            token = str(text).strip()
            if not token:
                continue
            words.append(token)
            try:
                value = float(conf)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                confs.append(value / 100.0)
        if not words:
            return OcrText("", 0.0)
        # Tesseract emits tokens in reading order; for right-to-left lines that is already
        # right-to-left logical order, so a plain join is correct.
        return OcrText(" ".join(words), float(np.mean(confs)) if confs else 0.0)
