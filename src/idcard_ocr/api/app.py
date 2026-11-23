"""FastAPI application factory.

Operational choices, all aimed at small CPU-only hosts and sensitive data:

* **Fail closed**: the app refuses to start without API keys unless anonymous access is
  explicitly enabled.
* **Bounded work**: at most ``max_concurrency`` images are processed at once (a thread pool
  of that size); extra requests wait up to ``queue_timeout_s`` and then get ``429``.
* **Bounded input**: uploads are read in chunks and rejected past ``max_upload_bytes``.
* **No personal data in logs**: only method, path, status and duration are logged.
* **No caching** of any response (``Cache-Control: no-store``).
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, TypeVar

from fastapi import Depends, FastAPI, File, Form, Header, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from idcard_ocr import __version__
from idcard_ocr.config import Settings
from idcard_ocr.errors import (
    IdCardError,
    ImageTooLargeError,
    InvalidImageError,
    OcrUnavailableError,
    StoreDisabledError,
)
from idcard_ocr.models import CardResult, PairResult
from idcard_ocr.pipeline import Pipeline
from idcard_ocr.store import VARIANTS, EncryptedImageStore

log = logging.getLogger("idcard_ocr.api")
T = TypeVar("T")
Side = Literal["front", "back"]  # module level: FastAPI resolves annotations from globals
_CHUNK = 1024 * 256


def _sniff_media_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _error(
    status: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details or {}}}
    return JSONResponse(body, status_code=status)


def create_app(settings: Settings | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    if not cfg.api_keys and not cfg.allow_anonymous:
        raise RuntimeError(
            "Refusing to start without authentication: set IDCARD_API_KEYS (comma separated) "
            "or, for local development only, IDCARD_ALLOW_ANONYMOUS=1."
        )
    store: EncryptedImageStore | None = None
    if pipeline is not None:
        store = pipeline.store
    elif cfg.store_dir:
        if not cfg.store_keys:
            raise RuntimeError("IDCARD_STORE_DIR is set but IDCARD_STORE_KEYS is empty.")
        store = EncryptedImageStore(cfg.store_dir, cfg.store_keys, cfg.retention_days)
    pipe = pipeline or Pipeline(cfg, store=store)
    pool = ThreadPoolExecutor(max_workers=cfg.max_concurrency, thread_name_prefix="idcard")
    gate = asyncio.Semaphore(cfg.max_concurrency)
    keys = [k.encode("utf-8") for k in cfg.api_keys]

    app = FastAPI(
        title="iran-idcard-ocr",
        version=__version__,
        description="CPU-only OCR and encrypted retention for Iranian national ID cards.",
    )

    @app.middleware("http")
    async def _wrap(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or secrets.token_hex(8)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id[:64]
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        log.info(
            "%s %s -> %s in %.0f ms [%s]",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id[:16],
        )
        return response

    @app.exception_handler(IdCardError)
    async def _idcard_error(_: Request, exc: IdCardError) -> JSONResponse:
        return _error(exc.http_status, exc.code, exc.message, exc.details)

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if cfg.allow_anonymous and not keys:
            return
        supplied = (x_api_key or "").encode("utf-8")
        # Compare against every key without short-circuiting to keep timing uniform.
        ok = False
        for key in keys:
            ok |= hmac.compare_digest(supplied, key)
        if not ok:
            raise _Unauthorized

    class _UnauthorizedError(Exception):
        pass

    _Unauthorized = _UnauthorizedError()

    @app.exception_handler(_UnauthorizedError)
    async def _unauthorized(_: Request, __: _UnauthorizedError) -> JSONResponse:
        resp = _error(401, "unauthorized", "A valid X-API-Key header is required.")
        resp.headers["WWW-Authenticate"] = "ApiKey"
        return resp

    auth = Depends(require_key)

    async def read_upload(upload: UploadFile) -> bytes:
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = await upload.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > cfg.max_upload_bytes:
                raise ImageTooLargeError(
                    "The upload exceeds the size limit.",
                    details={"max_upload_bytes": cfg.max_upload_bytes},
                )
            chunks.append(chunk)
        if not size:
            raise InvalidImageError("Empty upload.")
        return b"".join(chunks)

    async def run_limited(fn: Callable[[], T]) -> T:
        try:
            await asyncio.wait_for(gate.acquire(), timeout=cfg.queue_timeout_s)
        except TimeoutError as exc:
            raise _Busy from exc
        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(pool, fn), timeout=cfg.request_timeout_s
            )
        except TimeoutError as exc:
            raise _Timeout from exc
        finally:
            gate.release()

    class _BusyError(Exception):
        pass

    class _TimeoutError(Exception):
        pass

    _Busy = _BusyError()
    _Timeout = _TimeoutError()

    @app.exception_handler(_BusyError)
    async def _busy(_: Request, __: _BusyError) -> JSONResponse:
        resp = _error(429, "busy", "The service is at capacity; retry shortly.")
        resp.headers["Retry-After"] = "2"
        return resp

    @app.exception_handler(_TimeoutError)
    async def _timeout(_: Request, __: _TimeoutError) -> JSONResponse:
        return _error(504, "timeout", "Processing took too long.")

    def check_hints(side: str | None, version: str | None) -> None:
        if version is not None and version not in pipe.registry.versions:
            raise InvalidImageError(
                "Unknown card version.", details={"allowed": sorted(pipe.registry.versions)}
            )
        _ = side

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    async def readyz() -> dict[str, Any]:
        try:
            pipe.check_ready()
        except OcrUnavailableError as exc:
            return JSONResponse(  # type: ignore[return-value]
                {"status": "unavailable", "reason": exc.message}, status_code=503
            )
        return {"status": "ready", "store_enabled": store is not None}

    @app.post("/v1/extract", response_model=CardResult, tags=["extract"], dependencies=[auth])
    async def extract(
        file: Annotated[UploadFile, File(description="JPEG, PNG or WEBP photo of one card side")],
        side: Annotated[Side | None, Form()] = None,
        version: Annotated[str | None, Form()] = None,
        store_image: Annotated[bool, Form(description="Retain the encrypted image")] = False,
    ) -> CardResult:
        check_hints(side, version)
        data = await read_upload(file)
        return await run_limited(
            lambda: pipe.process(data, side=side, version=version, store=store_image)
        )

    @app.post("/v1/extract/pair", response_model=PairResult, tags=["extract"], dependencies=[auth])
    async def extract_pair(
        front: Annotated[UploadFile, File()],
        back: Annotated[UploadFile, File()],
        version: Annotated[str | None, Form()] = None,
        store_image: Annotated[bool, Form()] = False,
    ) -> PairResult:
        check_hints(None, version)
        front_data, back_data = await read_upload(front), await read_upload(back)
        return await run_limited(
            lambda: pipe.process_pair(front_data, back_data, version=version, store=store_image)
        )

    def need_store() -> EncryptedImageStore:
        if store is None:
            raise StoreDisabledError("No encrypted image store is configured.")
        return store

    @app.get("/v1/images/{image_id}", tags=["images"], dependencies=[auth])
    async def get_image(
        image_id: str,
        variant: Annotated[Literal["original", "card"], Query()] = "original",
    ) -> Response:
        st = need_store()
        data = await asyncio.get_running_loop().run_in_executor(None, st.load, image_id, variant)
        return Response(content=data, media_type=_sniff_media_type(data))

    @app.delete("/v1/images/{image_id}", status_code=204, tags=["images"], dependencies=[auth])
    async def delete_image(image_id: str) -> Response:
        from idcard_ocr.errors import ImageNotFoundError

        if not need_store().delete(image_id):
            raise ImageNotFoundError("Unknown image id.")
        return Response(status_code=204)

    app.state.pipeline = pipe
    app.state.variants = VARIANTS
    return app
