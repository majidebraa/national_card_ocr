"""Encrypted, retention-aware storage for card images (needed later for identity verification).

Design:

* AES-256-GCM per blob. The image id is bound as associated data, so a ciphertext copied
  under another id fails authentication.
* Key ring: the first key encrypts, every key can decrypt, so keys can be rotated without
  re-encrypting. A 4-byte key fingerprint is stored in the blob header.
* Ids are random 128-bit hex strings (unguessable) and strictly validated, so user input can
  never become a path.
* Atomic writes (temp file + ``os.replace``), mode 0600 files in a 0700 directory.
* Every record has ``expires_at``; ``purge_expired`` implements retention and ``delete``
  implements erasure requests.
* Only non-personal metadata is stored next to the blob (sha256 of the plaintext, card
  side/version, timestamps). Extracted field values are never written here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from idcard_ocr.errors import ImageNotFoundError, StorageError

MAGIC = b"IDOC1"
_ID_RE = re.compile(r"^[0-9a-f]{32}$")
VARIANTS = ("original", "card")


def generate_key() -> str:
    """A fresh urlsafe-base64 256-bit key, suitable for ``IDCARD_STORE_KEYS``."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")


def _decode_key(text: str) -> bytes:
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, TypeError) as exc:
        raise StorageError("A store key is not valid base64.") from exc
    if len(raw) != 32:
        raise StorageError("Store keys must decode to exactly 32 bytes (AES-256).")
    return raw


def _fingerprint(key: bytes) -> bytes:
    return hashlib.sha256(b"idcard-ocr-key-id" + key).digest()[:4]


@dataclass(frozen=True, slots=True)
class StoredImage:
    image_id: str
    sha256: str
    created_at: str
    expires_at: str
    side: str | None
    version: str | None
    variants: tuple[str, ...]


class EncryptedImageStore:
    def __init__(self, directory: str | Path, keys: tuple[str, ...], retention_days: int) -> None:
        if not keys:
            raise StorageError("At least one encryption key is required.")
        if retention_days < 1:
            raise StorageError("retention_days must be at least 1.")
        self._dir = Path(directory)
        self._keys = [_decode_key(k) for k in keys]
        self._by_fp = {_fingerprint(k): k for k in self._keys}
        self._retention = timedelta(days=retention_days)
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self._dir, 0o700)
        except OSError as exc:  # pragma: no cover - platform dependent
            raise StorageError("Could not restrict permissions on the store directory.") from exc

    # -- helpers ---------------------------------------------------------------------------
    @staticmethod
    def _check_id(image_id: str) -> str:
        if not _ID_RE.fullmatch(image_id):
            raise ImageNotFoundError("Unknown image id.")
        return image_id

    def _blob_path(self, image_id: str, variant: str) -> Path:
        return self._dir / f"{image_id}.{variant}.enc"

    def _meta_path(self, image_id: str) -> Path:
        return self._dir / f"{image_id}.json"

    def _write_atomic(self, path: Path, data: bytes) -> None:
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                os.fchmod(fh.fileno(), 0o600)
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            Path(tmp).unlink(missing_ok=True)
            raise StorageError("Could not write to the image store.") from exc

    def _encrypt(self, image_id: str, plaintext: bytes) -> bytes:
        key = self._keys[0]
        nonce = secrets.token_bytes(12)
        ct = AESGCM(key).encrypt(nonce, plaintext, image_id.encode("ascii"))
        return MAGIC + _fingerprint(key) + nonce + ct

    def _decrypt(self, image_id: str, blob: bytes) -> bytes:
        if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 4 + 12 + 16:
            raise StorageError("Stored blob is corrupt.")
        off = len(MAGIC)
        fp, nonce, ct = blob[off : off + 4], blob[off + 4 : off + 16], blob[off + 16 :]
        key = self._by_fp.get(fp)
        if key is None:
            raise StorageError("No configured key can decrypt this blob.")
        try:
            return AESGCM(key).decrypt(nonce, ct, image_id.encode("ascii"))
        except InvalidTag as exc:
            raise StorageError("Stored blob failed authentication.") from exc

    # -- public API ------------------------------------------------------------------------
    def save(
        self,
        blobs: dict[str, bytes],
        *,
        side: str | None = None,
        version: str | None = None,
        now: datetime | None = None,
    ) -> StoredImage:
        if "original" not in blobs:
            raise StorageError("The original image must be stored.")
        unknown = set(blobs) - set(VARIANTS)
        if unknown:
            raise StorageError(f"Unknown variants: {sorted(unknown)}")
        stamp = now or datetime.now(UTC)
        image_id = secrets.token_hex(16)
        for variant, data in blobs.items():
            self._write_atomic(self._blob_path(image_id, variant), self._encrypt(image_id, data))
        sha256 = hashlib.sha256(blobs["original"]).hexdigest()
        created_at = stamp.isoformat()
        expires_at = (stamp + self._retention).isoformat()
        meta = {
            "image_id": image_id,
            "sha256": sha256,
            "created_at": created_at,
            "expires_at": expires_at,
            "side": side,
            "version": version,
            "variants": sorted(blobs),
        }
        self._write_atomic(self._meta_path(image_id), json.dumps(meta).encode("utf-8"))
        return StoredImage(
            image_id,
            sha256,
            created_at,
            expires_at,
            side,
            version,
            tuple(sorted(blobs)),
        )

    def info(self, image_id: str) -> StoredImage:
        self._check_id(image_id)
        path = self._meta_path(image_id)
        if not path.is_file():
            raise ImageNotFoundError("Unknown image id.")
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StorageError("Stored metadata is corrupt.") from exc
        return StoredImage(
            m["image_id"],
            m["sha256"],
            m["created_at"],
            m["expires_at"],
            m.get("side"),
            m.get("version"),
            tuple(m["variants"]),
        )

    def load(
        self, image_id: str, variant: str = "original", *, now: datetime | None = None
    ) -> bytes:
        record = self.info(image_id)
        if variant not in record.variants:
            raise ImageNotFoundError("Unknown image variant.")
        if datetime.fromisoformat(record.expires_at) <= (now or datetime.now(UTC)):
            raise ImageNotFoundError("This image has expired and awaits purging.")
        path = self._blob_path(image_id, variant)
        try:
            blob = path.read_bytes()
        except OSError as exc:
            raise ImageNotFoundError("Unknown image id.") from exc
        plain = self._decrypt(image_id, blob)
        if variant == "original" and hashlib.sha256(plain).hexdigest() != record.sha256:
            raise StorageError("Integrity check failed for the stored image.")
        return plain

    def delete(self, image_id: str) -> bool:
        self._check_id(image_id)
        existed = self._meta_path(image_id).is_file()
        for variant in VARIANTS:
            self._blob_path(image_id, variant).unlink(missing_ok=True)
        self._meta_path(image_id).unlink(missing_ok=True)
        return existed

    def purge_expired(self, *, now: datetime | None = None) -> int:
        """Delete every record past its retention date; returns how many were removed."""
        cutoff = now or datetime.now(UTC)
        removed = 0
        for meta in self._dir.glob("*.json"):
            image_id = meta.stem
            if not _ID_RE.fullmatch(image_id):
                continue
            try:
                record = self.info(image_id)
            except (StorageError, ImageNotFoundError):
                continue
            if datetime.fromisoformat(record.expires_at) <= cutoff and self.delete(image_id):
                removed += 1
        return removed
