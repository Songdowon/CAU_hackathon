"""Runtime configuration for the trusted participant-side API."""

from __future__ import annotations

import math
import os
from pathlib import Path


WORKSPACE_ROOT = Path(os.environ.get("STUDENT_WORKSPACE_ROOT", "/workspace"))

MODELS_DIR = WORKSPACE_ROOT / "models"
SUBMISSIONS_DIR = WORKSPACE_ROOT / "submissions"
NUM_CLASSES = 100
MAX_ARTIFACT_BYTES = int(
    os.environ.get("MAX_ARTIFACT_BYTES", str(512 * 1024 * 1024))
)

if MAX_ARTIFACT_BYTES <= 0:
    raise RuntimeError("MAX_ARTIFACT_BYTES must be positive")

TRUSTED_SCORER_ROOT = Path(
    os.environ.get("TRUSTED_SCORER_ROOT", "/opt/hackathon/scorer")
)
TRUSTED_CONVERTER_PATH = Path(
    os.environ.get(
        "TRUSTED_CONVERTER_PATH",
        str(TRUSTED_SCORER_ROOT / "convert_checkpoint.py"),
    )
)
PARTICIPANT_UID = int(os.environ.get("PARTICIPANT_UID", "10001"))
PARTICIPANT_GID = int(os.environ.get("PARTICIPANT_GID", "10001"))
CONVERSION_TIMEOUT_SECONDS = float(
    os.environ.get("CONVERSION_TIMEOUT_SECONDS", "300")
)
if PARTICIPANT_UID <= 0 or PARTICIPANT_GID <= 0:
    raise RuntimeError("PARTICIPANT_UID/PARTICIPANT_GID must be positive and non-root")
if (
    not math.isfinite(CONVERSION_TIMEOUT_SECONDS)
    or CONVERSION_TIMEOUT_SECONDS <= 0
):
    raise RuntimeError("CONVERSION_TIMEOUT_SECONDS must be positive")
TEAM_ID = os.environ.get("TEAM_ID", "").strip()
GRADING_SERVER_URL = os.environ.get("GRADING_SERVER_URL", "").strip().rstrip("/")
GRADING_API_TOKEN_FILE = Path(
    os.environ.get(
        "GRADING_API_TOKEN_FILE", "/run/secrets/grading_api_token"
    )
)
GRADING_SERVER_ALLOW_HTTP = os.environ.get("GRADING_SERVER_ALLOW_HTTP", "0") == "1"
GRADING_CONNECT_TIMEOUT_SECONDS = float(
    os.environ.get("GRADING_CONNECT_TIMEOUT_SECONDS", "20")
)
GRADING_UPLOAD_TIMEOUT_SECONDS = float(
    os.environ.get("GRADING_UPLOAD_TIMEOUT_SECONDS", "1200")
)
GRADING_STATUS_TIMEOUT_SECONDS = float(
    os.environ.get("GRADING_STATUS_TIMEOUT_SECONDS", "20")
)

for timeout_name, timeout_value in (
    ("GRADING_CONNECT_TIMEOUT_SECONDS", GRADING_CONNECT_TIMEOUT_SECONDS),
    ("GRADING_UPLOAD_TIMEOUT_SECONDS", GRADING_UPLOAD_TIMEOUT_SECONDS),
    ("GRADING_STATUS_TIMEOUT_SECONDS", GRADING_STATUS_TIMEOUT_SECONDS),
):
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise RuntimeError(f"{timeout_name} must be positive")

# Participant-facing authentication is intentionally separate from the hidden
# grading upload token. Values are supported only for local/backward-
# compatible development; deployments should use root-readable secret files.
STUDENT_API_KEY_FILE = Path(
    os.environ.get("STUDENT_API_KEY_FILE", "/run/secrets/student_api_key")
)
SUBMIT_PASSWORD_FILE = Path(
    os.environ.get("SUBMIT_PASSWORD_FILE", "/run/secrets/submit_password")
)
API_KEY_HEADER_NAME = os.environ.get("API_KEY_HEADER_NAME", "X-API-Key")


def read_secret(path: Path, fallback_environment: str, description: str) -> str:
    """Read a secret file, with an env fallback for local development only."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        value = os.environ.get(fallback_environment, "").strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read {description} file {path}: {exc}") from exc
    if not value:
        raise RuntimeError(
            f"{description} is not configured; mount {path} or set "
            f"{fallback_environment} for local development"
        )
    return value


def student_api_key() -> str:
    return read_secret(STUDENT_API_KEY_FILE, "HACKATHON_API_KEY", "student API key")


def submit_password() -> str:
    return read_secret(SUBMIT_PASSWORD_FILE, "TEAM_SUBMIT_PASSWORD", "submit password")


def grading_api_token() -> str:
    return read_secret(
        GRADING_API_TOKEN_FILE,
        "GRADING_API_TOKEN",
        "grading API token",
    )
