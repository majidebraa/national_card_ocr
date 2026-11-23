from __future__ import annotations

import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from idcard_ocr.errors import ImageNotFoundError, StorageError
from idcard_ocr.store import EncryptedImageStore, generate_key


@pytest.fixture
def key() -> str:
    return generate_key()


@pytest.fixture
def store(tmp_path: Path, key: str) -> EncryptedImageStore:
    return EncryptedImageStore(tmp_path / "vault", (key,), retention_days=30)


PAYLOAD = b"\xff\xd8\xff fake jpeg payload " * 50


def test_roundtrip_and_metadata(store: EncryptedImageStore) -> None:
    saved = store.save({"original": PAYLOAD, "card": b"card-bytes"}, side="front", version="new")
    assert len(saved.image_id) == 32 and saved.variants == ("card", "original")
    assert store.load(saved.image_id) == PAYLOAD
    assert store.load(saved.image_id, "card") == b"card-bytes"
    info = store.info(saved.image_id)
    assert (info.side, info.version) == ("front", "new")
    assert datetime.fromisoformat(info.expires_at) > datetime.now(UTC) + timedelta(days=29)


def test_nothing_is_stored_in_plaintext(store: EncryptedImageStore, tmp_path: Path) -> None:
    saved = store.save({"original": PAYLOAD})
    blob = (tmp_path / "vault" / f"{saved.image_id}.original.enc").read_bytes()
    assert b"fake jpeg payload" not in blob
    meta = (tmp_path / "vault" / f"{saved.image_id}.json").read_text()
    assert "fake jpeg" not in meta


def test_permissions_are_private(store: EncryptedImageStore, tmp_path: Path) -> None:
    saved = store.save({"original": PAYLOAD})
    vault = tmp_path / "vault"
    assert stat.S_IMODE(os.stat(vault).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(vault / f"{saved.image_id}.original.enc").st_mode) == 0o600


def test_tampering_is_detected(store: EncryptedImageStore, tmp_path: Path) -> None:
    saved = store.save({"original": PAYLOAD})
    path = tmp_path / "vault" / f"{saved.image_id}.original.enc"
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0x01
    path.write_bytes(bytes(blob))
    with pytest.raises(StorageError, match="authentication"):
        store.load(saved.image_id)


def test_blob_cannot_be_moved_to_another_id(store: EncryptedImageStore, tmp_path: Path) -> None:
    a = store.save({"original": PAYLOAD})
    b = store.save({"original": b"other" * 40})
    vault = tmp_path / "vault"
    (vault / f"{b.image_id}.original.enc").write_bytes(
        (vault / f"{a.image_id}.original.enc").read_bytes()
    )
    with pytest.raises(StorageError):
        store.load(b.image_id)


@pytest.mark.parametrize(
    "bad", ["../etc/passwd", "a" * 31, "A" * 32, "g" * 32, "", "../../" + "a" * 32, "a" * 32 + "/x"]
)
def test_ids_never_become_paths(store: EncryptedImageStore, bad: str) -> None:
    with pytest.raises(ImageNotFoundError):
        store.load(bad)
    with pytest.raises(ImageNotFoundError):
        store.delete(bad)


def test_unknown_id_and_variant(store: EncryptedImageStore) -> None:
    with pytest.raises(ImageNotFoundError):
        store.load("0" * 32)
    saved = store.save({"original": PAYLOAD})
    with pytest.raises(ImageNotFoundError):
        store.load(saved.image_id, "card")


def test_key_rotation(tmp_path: Path) -> None:
    old, new = generate_key(), generate_key()
    first = EncryptedImageStore(tmp_path, (old,), 30)
    saved = first.save({"original": PAYLOAD})
    rotated = EncryptedImageStore(tmp_path, (new, old), 30)
    assert rotated.load(saved.image_id) == PAYLOAD  # old key still decrypts
    fresh = rotated.save({"original": PAYLOAD})
    only_new = EncryptedImageStore(tmp_path, (new,), 30)
    assert only_new.load(fresh.image_id) == PAYLOAD  # new data uses the new key
    with pytest.raises(StorageError, match="No configured key"):
        only_new.load(saved.image_id)


def test_wrong_key_fails(tmp_path: Path) -> None:
    saved = EncryptedImageStore(tmp_path, (generate_key(),), 30).save({"original": PAYLOAD})
    with pytest.raises(StorageError):
        EncryptedImageStore(tmp_path, (generate_key(),), 30).load(saved.image_id)


def test_retention_and_purge(store: EncryptedImageStore) -> None:
    past = datetime.now(UTC) - timedelta(days=31)
    old = store.save({"original": PAYLOAD}, now=past)
    fresh = store.save({"original": PAYLOAD})
    with pytest.raises(ImageNotFoundError, match="expired"):
        store.load(old.image_id)
    assert store.purge_expired() == 1
    assert store.load(fresh.image_id) == PAYLOAD
    with pytest.raises(ImageNotFoundError):
        store.info(old.image_id)
    assert store.purge_expired() == 0


def test_delete(store: EncryptedImageStore, tmp_path: Path) -> None:
    saved = store.save({"original": PAYLOAD, "card": b"x"})
    assert store.delete(saved.image_id) is True
    assert store.delete(saved.image_id) is False
    assert list((tmp_path / "vault").iterdir()) == []


def test_invalid_configuration(tmp_path: Path) -> None:
    with pytest.raises(StorageError):
        EncryptedImageStore(tmp_path, (), 30)
    with pytest.raises(StorageError):
        EncryptedImageStore(tmp_path, ("not base64 !!",), 30)
    with pytest.raises(StorageError, match="32 bytes"):
        EncryptedImageStore(tmp_path, ("c2hvcnQ",), 30)
    with pytest.raises(StorageError):
        EncryptedImageStore(tmp_path, (generate_key(),), 0)


def test_save_requires_original_and_known_variants(store: EncryptedImageStore) -> None:
    with pytest.raises(StorageError):
        store.save({"card": b"x"})
    with pytest.raises(StorageError):
        store.save({"original": b"x", "thumbnail": b"y"})


def test_corrupt_blob_header(store: EncryptedImageStore, tmp_path: Path) -> None:
    saved = store.save({"original": PAYLOAD})
    (tmp_path / "vault" / f"{saved.image_id}.original.enc").write_bytes(b"junk")
    with pytest.raises(StorageError, match="corrupt"):
        store.load(saved.image_id)


def test_keys_are_unique() -> None:
    assert len({generate_key() for _ in range(20)}) == 20
