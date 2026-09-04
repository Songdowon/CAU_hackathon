"""Durable, immutable-by-submission-ID checkpoint snapshots.

The metadata file is the commit marker.  An artifact without metadata is an
incomplete snapshot left by an interrupted writer and may be replaced on the
next retry; metadata without its matching artifact is corruption and is never
silently repaired from the student's current model.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Iterator


CHUNK_SIZE = 8 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CREATE_LOCK = threading.Lock()


class SnapshotError(Exception):
    """Base error raised while creating or reading a snapshot."""


class InvalidSubmissionId(SnapshotError):
    pass


class SnapshotNotFound(SnapshotError):
    pass


class SnapshotConflict(SnapshotError):
    pass


class SourceChanged(SnapshotError):
    pass


class UnsafeSource(SnapshotError):
    pass


def normalize_submission_id(submission_id: str) -> str:
    """Return the canonical spelling of a UUID and reject arbitrary paths."""
    try:
        return str(uuid.UUID(submission_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise InvalidSubmissionId("submission_id must be a UUID") from exc


def snapshot_paths(submission_id: str, submissions_dir: Path) -> tuple[Path, Path]:
    """Return flat artifact and commit-marker paths for ``submission_id``."""
    canonical_id = normalize_submission_id(submission_id)
    root = Path(submissions_dir)
    return (
        root / f"{canonical_id}.pt",
        root / f"{canonical_id}.snapshot.json",
    )


def _stat_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _copy_and_hash(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    max_bytes: int | None = None,
) -> tuple[str, int]:
    """Stream bytes to ``destination`` while calculating their SHA-256."""
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(CHUNK_SIZE):
        next_size = size + len(chunk)
        if max_bytes is not None and next_size > max_bytes:
            raise UnsafeSource(
                f"checkpoint grew beyond the {max_bytes}-byte size limit"
            )
        destination.write(chunk)
        digest.update(chunk)
        size = next_size
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _lstat_optional(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _read_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise UnsafeSource("this platform cannot safely open the checkpoint")
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def open_checkpoint_safely(source_path: Path) -> tuple[BinaryIO, os.stat_result]:
    """Open a regular checkpoint without following its final path component."""
    source_path = Path(source_path)
    try:
        path_stat = os.lstat(source_path)
    except FileNotFoundError as exc:
        raise SnapshotNotFound(f"checkpoint does not exist: {source_path}") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise UnsafeSource("checkpoint must be a regular file, not a symlink")

    try:
        source_fd = os.open(source_path, _read_flags())
    except FileNotFoundError as exc:
        raise SnapshotNotFound(f"checkpoint does not exist: {source_path}") from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EINVAL):
            raise UnsafeSource("checkpoint must be a regular file, not a symlink") from exc
        raise

    try:
        opened_stat = os.fstat(source_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise UnsafeSource("checkpoint must be a regular file")
        if _stat_signature(opened_stat) != _stat_signature(path_stat):
            raise SourceChanged("checkpoint changed before copying started")
        return os.fdopen(source_fd, "rb", closefd=True), opened_stat
    except Exception:
        os.close(source_fd)
        raise


def _open_committed_file(path: Path, description: str) -> BinaryIO:
    """Open a committed snapshot component without following symlinks."""
    try:
        file_fd = os.open(path, _read_flags())
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise SnapshotNotFound("submission snapshot does not exist") from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EINVAL):
            raise SnapshotConflict(f"existing snapshot {description} is unsafe") from exc
        raise SnapshotConflict(f"existing snapshot {description} cannot be opened") from exc

    try:
        opened_stat = os.fstat(file_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise SnapshotConflict(
                f"existing snapshot {description} is not a regular file"
            )
        return os.fdopen(file_fd, "rb", closefd=True)
    except Exception:
        os.close(file_fd)
        raise


def _hash_open_file(file_obj: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while chunk := file_obj.read(CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _load_metadata(metadata_path: Path) -> dict:
    try:
        with _open_committed_file(metadata_path, "metadata") as metadata_file:
            raw = metadata_file.read()
        value = json.loads(raw.decode("utf-8"))
    except SnapshotError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotConflict("existing snapshot metadata is invalid") from exc
    if not isinstance(value, dict):
        raise SnapshotConflict("existing snapshot metadata is invalid")
    return value


def _validate_metadata(metadata: dict, canonical_id: str) -> None:
    required = {"submission_id", "sha256", "size_bytes", "created_at"}
    if set(metadata) != required:
        raise SnapshotConflict("existing snapshot metadata is invalid")
    if metadata["submission_id"] != canonical_id:
        raise SnapshotConflict("existing snapshot metadata has the wrong ID")
    if not isinstance(metadata["sha256"], str) or not _SHA256_RE.fullmatch(
        metadata["sha256"]
    ):
        raise SnapshotConflict("existing snapshot metadata has an invalid SHA-256")
    if (
        isinstance(metadata["size_bytes"], bool)
        or not isinstance(metadata["size_bytes"], int)
        or metadata["size_bytes"] < 0
    ):
        raise SnapshotConflict("existing snapshot metadata has an invalid size")
    if not isinstance(metadata["created_at"], str) or not metadata["created_at"]:
        raise SnapshotConflict("existing snapshot metadata has an invalid timestamp")
    try:
        created_at = datetime.fromisoformat(
            metadata["created_at"].replace("Z", "+00:00")
        )
        if created_at.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
    except ValueError as exc:
        raise SnapshotConflict(
            "existing snapshot metadata has an invalid timestamp"
        ) from exc


def get_snapshot(
    submission_id: str,
    submissions_dir: Path,
    *,
    verify_hash: bool = True,
) -> dict:
    """Load a committed snapshot, optionally re-hashing its artifact."""
    canonical_id = normalize_submission_id(submission_id)
    artifact_path, metadata_path = snapshot_paths(canonical_id, submissions_dir)

    # Metadata is deliberately checked first: it is the commit marker.  A lone
    # artifact is an uncommitted crash remnant, not a snapshot clients may use.
    if _lstat_optional(metadata_path) is None:
        raise SnapshotNotFound("submission snapshot does not exist")
    if _lstat_optional(artifact_path) is None:
        raise SnapshotConflict("snapshot metadata exists without its artifact")

    metadata = _load_metadata(metadata_path)
    _validate_metadata(metadata, canonical_id)

    with _open_committed_file(artifact_path, "artifact") as artifact_file:
        opened_stat = os.fstat(artifact_file.fileno())
        if opened_stat.st_size != metadata["size_bytes"]:
            raise SnapshotConflict("existing snapshot size does not match metadata")
        if verify_hash:
            actual_hash, actual_size = _hash_open_file(artifact_file)
            if (
                actual_size != metadata["size_bytes"]
                or actual_hash != metadata["sha256"]
            ):
                raise SnapshotConflict(
                    "existing snapshot hash does not match metadata"
                )
    return metadata


def _ensure_snapshot_directory(submissions_dir: Path) -> None:
    submissions_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    directory_stat = os.lstat(submissions_dir)
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise SnapshotConflict("submissions path must be a real directory")


@contextmanager
def _process_lock(canonical_id: str, submissions_dir: Path) -> Iterator[None]:
    """Serialize the same ID across API worker processes on this volume."""
    lock_dir = submissions_dir / ".locks"
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    if not stat.S_ISDIR(os.lstat(lock_dir).st_mode):
        raise SnapshotConflict("snapshot lock path must be a real directory")

    lock_path = lock_dir / f"{canonical_id}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise UnsafeSource("this platform cannot safely create snapshots")
    flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise SnapshotConflict("snapshot lock file is unsafe") from exc
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise SnapshotConflict("snapshot lock is not a regular file")
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _remove_stale_temporary_files(
    canonical_id: str, submissions_dir: Path
) -> None:
    for stale_path in submissions_dir.glob(f".{canonical_id}.*.tmp"):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass
        except (IsADirectoryError, PermissionError, OSError) as exc:
            raise SnapshotConflict("stale snapshot temporary file is unsafe") from exc


def _remove_uncommitted_files(canonical_id: str, submissions_dir: Path) -> None:
    artifact_path, metadata_path = snapshot_paths(canonical_id, submissions_dir)
    if _lstat_optional(metadata_path) is not None:
        return

    # With no commit marker, a final-looking artifact is an interrupted write.
    # The directory is root-owned in production, so removing the name cannot
    # delete participant-owned data outside this managed namespace; unlink also
    # never follows a malicious symlink.
    if _lstat_optional(artifact_path) is not None:
        try:
            artifact_path.unlink()
        except OSError as exc:
            raise SnapshotConflict(
                "incomplete snapshot artifact cannot be removed"
            ) from exc

    _remove_stale_temporary_files(canonical_id, submissions_dir)


def _new_temp_file(
    submissions_dir: Path, canonical_id: str, suffix: str
) -> tuple[int, Path]:
    file_fd, raw_path = tempfile.mkstemp(
        prefix=f".{canonical_id}.", suffix=suffix, dir=submissions_dir
    )
    return file_fd, Path(raw_path)


def _publish_no_replace(temp_path: Path, final_path: Path) -> None:
    """Atomically publish a fully synced file without overwriting a winner."""
    try:
        os.link(temp_path, final_path, follow_symlinks=False)
    except FileExistsError as exc:
        raise SnapshotConflict(f"snapshot path already exists: {final_path.name}") from exc
    temp_path.unlink()


def create_snapshot(
    submission_id: str,
    source_path: Path,
    submissions_dir: Path,
    *,
    max_bytes: int | None = None,
) -> dict:
    """Durably freeze ``source_path`` under a canonical submission UUID."""
    canonical_id = normalize_submission_id(submission_id)
    submissions_dir = Path(submissions_dir)

    # Uvicorn dispatches sync endpoints through threads; flock supplies the
    # corresponding same-ID exclusion between worker processes.
    with _CREATE_LOCK:
        _ensure_snapshot_directory(submissions_dir)
        with _process_lock(canonical_id, submissions_dir):
            return _create_snapshot(
                canonical_id,
                Path(source_path),
                submissions_dir,
                max_bytes=max_bytes,
            )


def _create_snapshot(
    canonical_id: str,
    source_path: Path,
    submissions_dir: Path,
    *,
    max_bytes: int | None,
) -> dict:
    artifact_path, metadata_path = snapshot_paths(canonical_id, submissions_dir)

    if _lstat_optional(metadata_path) is not None:
        metadata = get_snapshot(canonical_id, submissions_dir, verify_hash=True)
        _remove_stale_temporary_files(canonical_id, submissions_dir)
        return metadata
    _remove_uncommitted_files(canonical_id, submissions_dir)

    artifact_fd, temp_artifact = _new_temp_file(
        submissions_dir, canonical_id, ".pt.tmp"
    )
    destination = os.fdopen(artifact_fd, "wb", closefd=True)
    temp_metadata: Path | None = None
    artifact_published = False
    try:
        source, opened_before = open_checkpoint_safely(source_path)
        with source, destination:
            if max_bytes is not None and opened_before.st_size > max_bytes:
                raise UnsafeSource(
                    "checkpoint is too large "
                    f"({opened_before.st_size} bytes; limit {max_bytes})"
                )
            sha256, size_bytes = _copy_and_hash(
                source, destination, max_bytes=max_bytes
            )
            destination.flush()
            os.fchmod(destination.fileno(), 0o444)
            os.fsync(destination.fileno())
            opened_after = os.fstat(source.fileno())

        try:
            path_after = os.lstat(source_path)
        except FileNotFoundError as exc:
            raise SourceChanged(
                "checkpoint disappeared while it was being copied"
            ) from exc
        expected = _stat_signature(opened_before)
        if (
            not stat.S_ISREG(path_after.st_mode)
            or _stat_signature(opened_after) != expected
            or _stat_signature(path_after) != expected
            or size_bytes != opened_before.st_size
        ):
            raise SourceChanged("checkpoint changed while it was being copied")

        metadata = {
            "submission_id": canonical_id,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        metadata_fd, temp_metadata = _new_temp_file(
            submissions_dir, canonical_id, ".snapshot.json.tmp"
        )
        with os.fdopen(metadata_fd, "w", encoding="utf-8", closefd=True) as metadata_file:
            json.dump(metadata, metadata_file, ensure_ascii=False, sort_keys=True)
            metadata_file.write("\n")
            metadata_file.flush()
            os.fchmod(metadata_file.fileno(), 0o444)
            os.fsync(metadata_file.fileno())

        # Publish and persist the artifact before publishing metadata.  Readers
        # regard the metadata link as the single commit point for the pair.
        _publish_no_replace(temp_artifact, artifact_path)
        artifact_published = True
        _fsync_directory(submissions_dir)
        _publish_no_replace(temp_metadata, metadata_path)
        temp_metadata = None
        _fsync_directory(submissions_dir)
        return metadata
    except Exception:
        # A normal exception is cleaned eagerly.  A process crash can leave the
        # same artifact-only state, which the next lock holder removes above.
        if artifact_published and _lstat_optional(metadata_path) is None:
            try:
                artifact_path.unlink()
                _fsync_directory(submissions_dir)
            except FileNotFoundError:
                pass
        raise
    finally:
        if not destination.closed:
            destination.close()
        for temp_path in (temp_artifact, temp_metadata):
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass
