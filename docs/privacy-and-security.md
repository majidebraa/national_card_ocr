# Privacy and security

National ID cards are highly sensitive personal data. This document describes what the software
does; it is not legal advice. Check the data-protection law that applies to you, obtain explicit
consent, and keep the retention period as short as your verification process allows.

## Data handling

| Data | Where | Protection |
|---|---|---|
| Uploaded photo | memory; optionally the store | AES-256-GCM at rest, id bound as AAD |
| Rectified card | memory; optionally the store | same |
| Extracted fields | response only | never written to disk or logs |
| Store metadata | `<id>.json` | sha256 of the original, side/version, timestamps; no personal fields |

* Retention is opt-in per request (`store_image=true`) and needs `IDCARD_STORE_DIR` and keys.
* Every record expires (`IDCARD_RETENTION_DAYS`); run `idcard-ocr purge` regularly.
* `DELETE /v1/images/{id}` erases a record (erasure requests).
* Key rotation: put the new key first in `IDCARD_STORE_KEYS`, keep old keys until old data expires.
* Files are mode 0600 in a 0700 directory; writes are atomic and fsynced.
* Integrity: the original is checked against the stored sha256 on every load.

## Service hardening

* Fail closed: no API keys, no start (unless `IDCARD_ALLOW_ANONYMOUS=1`).
* Keys compared in constant time; 401 for missing/invalid.
* Image ids are random 128-bit hex and strictly validated, so they cannot form paths.
* Upload size and decoded-pixel caps are enforced before pixel decoding (decompression bombs).
* Only JPEG/PNG/WEBP are decoded; the Docker image runs as a non-root user, read-only root
  filesystem, all capabilities dropped (see `docker-compose.yml`).
* Responses are `no-store`; logs hold method, path, status, duration and request id only.
* The CI `security` workflow runs `pip-audit` weekly; Dependabot covers pip, Docker and Actions.

## Not covered

This is OCR. It does not detect forged or tampered cards, replayed photos, screen captures,
or liveness, and it does not match the face to the person or consult any civil registry. For
identity *verification* combine it with those controls; the retained image exists so that such
checks can be done later.

## Test data

Never commit real ID photos. The synthetic generator produces fictional, watermarked specimens
that do not imitate official artwork.
