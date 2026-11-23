# HTTP API

Start with `idcard-ocr serve` or Docker. Interactive docs are served at `/docs`.
Every endpoint except `/healthz` and `/readyz` requires the header `X-API-Key`.
Responses carry `Cache-Control: no-store` and an `X-Request-ID`.

## `POST /v1/extract`

`multipart/form-data`:

| Field | Type | Notes |
|---|---|---|
| `file` | file | JPEG, PNG or WEBP, one card side |
| `side` | `front` \| `back` | optional hint |
| `version` | string | optional hint (`old`, `new`, or any layout version you added) |
| `store_image` | bool | retain the encrypted original and rectified card |

Returns a `CardResult` (see the schema at `/openapi.json`). Important members:

* `fields.<name>.value` normalised value; `raw` text as recognised; `confidence` 0-1;
  `valid` passed validation; `needs_review` ask the user to confirm; `issues` stable codes.
* `quality` measured sharpness/brightness/glare and `issues`.
* `warnings`, e.g. `card_was_upside_down`.
* `stored` image reference when `store_image=true`.

## `POST /v1/extract/pair`

Fields `front`, `back` (files), optional `version`, `store_image`. The version found on the front
is applied to the back. Returns `{front, back, consistent, warnings}`.

## `GET /v1/images/{id}?variant=original|card`, `DELETE /v1/images/{id}`

Decrypts and returns a retained image (`original` is byte-identical to the upload), or erases it.
Ids are random 128-bit hex strings. Expired records answer `404` and are removed by
`idcard-ocr purge`.

## Errors

```json
{"error": {"code": "poor_image_quality", "message": "…", "details": {"issues": ["blurry"]}}}
```

| HTTP | `code` | Meaning |
|---|---|---|
| 400 | `invalid_image` | not a decodable JPEG/PNG/WEBP, too small, unknown `version` |
| 401 | `unauthorized` | missing or wrong `X-API-Key` |
| 404 | `image_not_found` | unknown, malformed or expired image id |
| 409 | `store_disabled` | retention requested but no store configured |
| 413 | `image_too_large` | upload or pixel count over the limit |
| 422 | `card_not_found`, `classification_failed`, `poor_image_quality` | ask the user to retake the photo |
| 429 | `busy` | at capacity; honour `Retry-After` |
| 502/503 | `ocr_failed`, `ocr_unavailable` | OCR engine problem |
| 504 | `timeout` | processing exceeded `IDCARD_REQUEST_TIMEOUT` |

For `422` responses a client can show targeted guidance from `details.issues`
(`blurry`, `too_dark`, `too_bright`, `glare`, `low_resolution`).

## Health

`GET /healthz` is liveness. `GET /readyz` returns `503` when Tesseract or its language data is
missing; use it for orchestrator readiness.
