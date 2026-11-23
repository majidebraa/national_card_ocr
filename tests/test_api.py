from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from idcard_ocr.api.app import create_app
from idcard_ocr.config import Settings
from idcard_ocr.ocr.base import OcrText
from idcard_ocr.pipeline import Pipeline
from idcard_ocr.store import EncryptedImageStore, generate_key
from idcard_ocr.synth.dataset import Sample
from tests.conftest import FakeEngine

KEY = "test-key-123"


def _first(dataset: tuple[Path, list[Sample]], side: str) -> tuple[bytes, Sample]:
    directory, samples = dataset
    sample = next(s for s in samples if s.side == side)
    return (directory / sample.file).read_bytes(), sample


@pytest.fixture
def client(settings: Settings, tmp_path: Path) -> TestClient:
    cfg = replace(settings, allow_anonymous=False, api_keys=(KEY, "second-key"))
    store = EncryptedImageStore(tmp_path / "vault", (generate_key(),), 30)
    pipe = Pipeline(cfg, engine=FakeEngine(), store=store)
    return TestClient(create_app(cfg, pipe), headers={"X-API-Key": KEY})


def test_fails_closed_without_auth_configuration(settings: Settings) -> None:
    with pytest.raises(RuntimeError, match="authentication"):
        create_app(replace(settings, allow_anonymous=False, api_keys=()))


def test_store_dir_without_keys_is_rejected(settings: Settings, tmp_path: Path) -> None:
    cfg = replace(settings, store_dir=str(tmp_path / "s"), store_keys=())
    with pytest.raises(RuntimeError, match="STORE_KEYS"):
        create_app(cfg)


def test_health_endpoints_are_open(client: TestClient) -> None:
    anonymous = TestClient(client.app)
    assert anonymous.get("/healthz").json()["status"] == "ok"
    assert anonymous.get("/readyz").json() == {"status": "ready", "store_enabled": True}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_authentication_is_required(client: TestClient, headers: dict[str, str]) -> None:
    anonymous = TestClient(client.app)
    resp = anonymous.post("/v1/extract", files={"file": ("a.jpg", b"x")}, headers=headers)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
    assert resp.headers["www-authenticate"] == "ApiKey"
    assert anonymous.get("/v1/images/" + "0" * 32, headers=headers).status_code == 401


def test_any_configured_key_works(client: TestClient) -> None:
    other = TestClient(client.app, headers={"X-API-Key": "second-key"})
    assert other.get("/v1/images/" + "0" * 32).status_code == 404


def test_extract_front(client: TestClient, easy_dataset: tuple[Path, list[Sample]]) -> None:
    data, sample = _first(easy_dataset, "front")
    resp = client.post("/v1/extract", files={"file": ("card.jpg", data, "image/jpeg")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["fields"]["national_id"]["value"] == sample.fields["national_id"]
    assert body["side"] == "front" and body["stored"] is None
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-request-id"]


def test_retention_roundtrip_and_erasure(
    client: TestClient, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    data, _ = _first(easy_dataset, "front")
    body = client.post(
        "/v1/extract", files={"file": ("c.jpg", data, "image/jpeg")}, data={"store_image": "true"}
    ).json()
    image_id = body["stored"]["image_id"]
    original = client.get(f"/v1/images/{image_id}")
    assert original.status_code == 200 and original.content == data
    assert original.headers["content-type"] == "image/jpeg"
    card = client.get(f"/v1/images/{image_id}", params={"variant": "card"})
    assert card.status_code == 200 and card.content.startswith(b"\xff\xd8")
    assert client.delete(f"/v1/images/{image_id}").status_code == 204
    assert client.get(f"/v1/images/{image_id}").status_code == 404
    assert client.delete(f"/v1/images/{image_id}").status_code == 404


def test_pair_endpoint(client: TestClient, easy_dataset: tuple[Path, list[Sample]]) -> None:
    directory, samples = easy_dataset
    version = samples[0].version
    front = next(s for s in samples if s.side == "front" and s.version == version)
    back = next(s for s in samples if s.side == "back" and s.version == version)
    resp = client.post(
        "/v1/extract/pair",
        files={
            "front": ("f.jpg", (directory / front.file).read_bytes(), "image/jpeg"),
            "back": ("b.jpg", (directory / back.file).read_bytes(), "image/jpeg"),
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["consistent"] is True


def test_error_responses_have_a_stable_shape(client: TestClient) -> None:
    resp = client.post("/v1/extract", files={"file": ("x.jpg", b"not an image")})
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "invalid_image" and set(err) == {"code", "message", "details"}


def test_validation_of_hints(client: TestClient, easy_dataset: tuple[Path, list[Sample]]) -> None:
    data, _ = _first(easy_dataset, "front")
    files = {"file": ("c.jpg", data, "image/jpeg")}
    assert client.post("/v1/extract", files=files, data={"side": "left"}).status_code == 422
    bad = client.post("/v1/extract", files=files, data={"version": "alien"})
    assert bad.status_code == 400 and "allowed" in bad.json()["error"]["details"]


def test_oversized_upload_is_rejected(settings: Settings) -> None:
    cfg = replace(settings, max_upload_bytes=1000)
    client = TestClient(create_app(cfg, Pipeline(cfg, engine=FakeEngine())))
    resp = client.post("/v1/extract", files={"file": ("big.jpg", b"0" * 5000)})
    assert resp.status_code == 413 and resp.json()["error"]["code"] == "image_too_large"


def test_empty_upload_is_rejected(client: TestClient) -> None:
    resp = client.post("/v1/extract", files={"file": ("e.jpg", b"")})
    assert resp.status_code in (400, 422)


def test_store_endpoints_without_store(settings: Settings) -> None:
    client = TestClient(create_app(settings, Pipeline(settings, engine=FakeEngine())))
    assert client.get("/v1/images/" + "0" * 32).status_code == 409
    assert client.delete("/v1/images/" + "0" * 32).status_code == 409


def test_unreadable_ids_are_404(client: TestClient) -> None:
    assert client.get("/v1/images/not-an-id").status_code == 404


def test_readyz_reports_missing_ocr(settings: Settings) -> None:
    from idcard_ocr.errors import OcrUnavailableError

    class Broken(FakeEngine):
        def check_ready(self) -> None:
            raise OcrUnavailableError("tesseract missing")

    client = TestClient(create_app(settings, Pipeline(settings, engine=Broken())))
    resp = client.get("/readyz")
    assert resp.status_code == 503 and resp.json()["status"] == "unavailable"


class SlowEngine(FakeEngine):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    def recognize(self, image, **kwargs) -> OcrText:  # type: ignore[no-untyped-def, override]
        time.sleep(self.delay)
        return super().recognize(image, **kwargs)


def test_overload_returns_429_with_retry_after(
    settings: Settings, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    import asyncio

    import httpx

    cfg = replace(settings, max_concurrency=1, queue_timeout_s=0.05)
    app = create_app(cfg, Pipeline(cfg, engine=SlowEngine(0.4)))
    data, _ = _first(easy_dataset, "back")  # back cards use the slow engine for the serial

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            calls = [
                client.post("/v1/extract", files={"file": ("c.jpg", data, "image/jpeg")})
                for _ in range(3)
            ]
            return list(await asyncio.gather(*calls))

    responses = asyncio.run(run())
    codes = sorted(r.status_code for r in responses)
    assert 429 in codes and 200 in codes, codes
    busy = next(r for r in responses if r.status_code == 429)
    assert busy.headers["retry-after"] == "2" and busy.json()["error"]["code"] == "busy"


def test_request_timeout(settings: Settings, easy_dataset: tuple[Path, list[Sample]]) -> None:
    cfg = replace(settings, request_timeout_s=0.05)
    client = TestClient(create_app(cfg, Pipeline(cfg, engine=SlowEngine(0.5))))
    data, _ = _first(easy_dataset, "back")
    resp = client.post("/v1/extract", files={"file": ("c.jpg", data, "image/jpeg")})
    assert resp.status_code == 504 and resp.json()["error"]["code"] == "timeout"


def test_openapi_schema_lists_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/v1/extract", "/v1/extract/pair", "/v1/images/{image_id}", "/healthz"} <= set(paths)
