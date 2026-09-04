"""Privilege-separated conversion of participant ``.pt`` checkpoints.

The trusted FastAPI process never deserializes pickle-based participant data.
It opens a stable source descriptor, launches the converter as the participant
UID with a scrubbed environment, and publishes only the resulting safetensors
blob into the root-owned submission store.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from app.core import config
from app.services.submissions import SourceChanged, UnsafeSource, open_checkpoint_safely


class ConversionFailed(RuntimeError):
    pass


def safe_checkpoint_path(submission_id: str) -> Path:
    return config.SUBMISSIONS_DIR / f"{submission_id}.safetensors"


def _signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _terminate_process_group(process: subprocess.Popen) -> None:
    for signal_value in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signal_value)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=5)
            if signal_value == signal.SIGTERM:
                # Clear normal descendants that outlived the group leader.
                continue
            break
        except subprocess.TimeoutExpired:
            continue


def _converter_environment(runtime_root: Path) -> dict[str, str]:
    allowed = {"LD_LIBRARY_PATH", "PATH"}
    environment = {
        name: value for name, value in os.environ.items() if name in allowed
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HOME": str(runtime_root / "home"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(config.TRUSTED_SCORER_ROOT),
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(runtime_root / "tmp"),
        }
    )
    return environment


def _prepare_runtime_root(runtime_root: Path) -> None:
    uid = config.PARTICIPANT_UID if os.geteuid() == 0 else os.geteuid()
    gid = config.PARTICIPANT_GID if os.geteuid() == 0 else os.getegid()
    for directory in (runtime_root, runtime_root / "home", runtime_root / "tmp"):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        if os.geteuid() == 0:
            os.chown(directory, uid, gid)


def _drop_privilege_arguments() -> dict:
    if os.geteuid() != 0:
        return {}
    if config.PARTICIPANT_UID == 0 or config.PARTICIPANT_GID == 0:
        raise ConversionFailed("checkpoint converter must not run as root")
    return {
        "user": config.PARTICIPANT_UID,
        "group": config.PARTICIPANT_GID,
        "extra_groups": (),
        "umask": 0o077,
    }


def _validate_existing_destination(path: Path, max_bytes: int) -> None:
    try:
        file_stat = os.lstat(path)
    except FileNotFoundError:
        raise
    if not stat.S_ISREG(file_stat.st_mode):
        raise UnsafeSource("converted checkpoint must be a regular file")
    if file_stat.st_mode & 0o022:
        raise UnsafeSource("converted checkpoint must not be group/other writable")
    if os.geteuid() == 0 and file_stat.st_uid != 0:
        raise UnsafeSource("converted checkpoint must be owned by root")
    if not 0 < file_stat.st_size <= max_bytes:
        raise UnsafeSource("converted checkpoint has an invalid size")


def convert_checkpoint(
    source_path: Path,
    destination_path: Path,
    *,
    max_bytes: int,
) -> Path:
    """Convert and atomically publish ``destination_path`` without trusted pickle."""
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    if max_bytes <= 0:
        raise ConversionFailed("checkpoint conversion size limit must be positive")
    destination_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        _validate_existing_destination(destination_path, max_bytes)
        return destination_path

    output_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    temporary_path = Path(temporary_name)
    source_file = None
    try:
        source_file, source_before = open_checkpoint_safely(source_path)
        if not config.TRUSTED_CONVERTER_PATH.is_file():
            raise ConversionFailed(
                "trusted checkpoint converter is missing: "
                f"{config.TRUSTED_CONVERTER_PATH}"
            )
        with tempfile.TemporaryDirectory(prefix="hackathon-converter-") as raw_runtime:
            runtime_root = Path(raw_runtime)
            _prepare_runtime_root(runtime_root)
            command = [
                sys.executable,
                str(config.TRUSTED_CONVERTER_PATH),
                "--input",
                f"/proc/self/fd/{source_file.fileno()}",
                "--output-fd",
                str(output_fd),
                "--max-bytes",
                str(max_bytes),
                "--cpu-seconds",
                str(max(1, int(config.CONVERSION_TIMEOUT_SECONDS))),
            ]
            process = subprocess.Popen(
                command,
                cwd=config.TRUSTED_SCORER_ROOT,
                env=_converter_environment(runtime_root),
                pass_fds=(source_file.fileno(), output_fd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                **_drop_privilege_arguments(),
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=config.CONVERSION_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(process)
                raise ConversionFailed(
                    "checkpoint conversion timed out after "
                    f"{config.CONVERSION_TIMEOUT_SECONDS:g}s"
                ) from exc

        if process.returncode:
            detail = (stderr or stdout or "invalid or unsafe checkpoint")[-2000:].strip()
            raise ConversionFailed(f"checkpoint conversion failed: {detail}")

        source_after = os.fstat(source_file.fileno())
        try:
            path_after = os.lstat(source_path)
        except FileNotFoundError as exc:
            raise SourceChanged("checkpoint disappeared during conversion") from exc
        if (
            _signature(source_before) != _signature(source_after)
            or _signature(source_before) != _signature(path_after)
            or not stat.S_ISREG(path_after.st_mode)
        ):
            raise SourceChanged("checkpoint changed during conversion")

        os.fsync(output_fd)
        output_before = os.fstat(output_fd)
        if not 0 < output_before.st_size <= max_bytes:
            raise ConversionFailed("converter produced an invalid output size")

        # Never publish the inode whose writable descriptor crossed the
        # privilege boundary. A compromised converter could keep that FD in a
        # daemonized child. Copy with pread into a fresh root-owned inode and
        # reject any concurrent mutation of the converter output.
        publish_fd, publish_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.",
            suffix=".publish.tmp",
            dir=destination_path.parent,
        )
        publish_path = Path(publish_name)
        try:
            position = 0
            while position < output_before.st_size:
                chunk = os.pread(
                    output_fd,
                    min(8 * 1024 * 1024, output_before.st_size - position),
                    position,
                )
                if not chunk:
                    raise ConversionFailed("converter output changed while restaging")
                view = memoryview(chunk)
                while view:
                    written = os.write(publish_fd, view)
                    if written <= 0:
                        raise ConversionFailed("could not restage converter output")
                    view = view[written:]
                position += len(chunk)
            output_after = os.fstat(output_fd)
            if _signature(output_before) != _signature(output_after):
                raise ConversionFailed("converter output changed while restaging")
            os.fchmod(publish_fd, 0o444)
            os.fsync(publish_fd)
            try:
                os.link(publish_path, destination_path, follow_symlinks=False)
            except FileExistsError as exc:
                raise ConversionFailed(
                    "converted checkpoint path already exists"
                ) from exc
            publish_path.unlink()
        finally:
            os.close(publish_fd)
            publish_path.unlink(missing_ok=True)
        directory_fd = os.open(
            destination_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return destination_path
    finally:
        if source_file is not None:
            source_file.close()
        os.close(output_fd)
        temporary_path.unlink(missing_ok=True)
