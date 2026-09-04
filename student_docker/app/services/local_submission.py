"""Freeze a selected experiment checkpoint and upload it for private testing."""

from __future__ import annotations

import re
import secrets
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.core import config
from app.services.submissions import create_snapshot, snapshot_paths


_SUBMISSION_LOCK = threading.Lock()
_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}\.pt\Z")


class SubmissionBusy(RuntimeError):
    pass


class DeliveryPending(RuntimeError):
    pass


class DeliveryRejected(RuntimeError):
    pass


def selected_model_path(filename: str) -> Path:
    if not isinstance(filename, str) or not _MODEL_NAME_RE.fullmatch(filename):
        raise ValueError(
            "model_filename must be a plain .pt filename using letters, digits, '.', '_', or '-'"
        )
    path = config.MODELS_DIR / filename
    if path.parent != config.MODELS_DIR:
        raise ValueError("model_filename must stay inside /workspace/models")
    return path


def _grading_endpoint(path: str) -> str:
    base = config.GRADING_SERVER_URL
    if not base:
        raise DeliveryPending("grading server URL is not configured")
    parsed = urlparse(base)
    allowed = {"https"} | ({"http"} if config.GRADING_SERVER_ALLOW_HTTP else set())
    if parsed.scheme not in allowed or not parsed.netloc:
        raise DeliveryPending("grading server URL must use HTTPS")
    if (
        parsed.username
        or parsed.password
        or parsed.path.rstrip("/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise DeliveryPending("grading server URL contains forbidden URL components")
    return f"{base}{path}"


def _auth_headers() -> dict[str, str]:
    if not config.TEAM_ID:
        raise DeliveryRejected("TEAM_ID is not configured")
    return {
        "Authorization": f"Bearer {config.grading_api_token()}",
        "X-Team-ID": config.TEAM_ID,
    }


def _upload(submission_id: str, filename: str, snapshot: dict) -> dict:
    artifact, _ = snapshot_paths(submission_id, config.SUBMISSIONS_DIR)
    endpoint = _grading_endpoint("/api/v1/submissions")
    headers = {
        **_auth_headers(),
        "Content-Type": "application/octet-stream",
        "Content-Length": str(snapshot["size_bytes"]),
        "X-Submission-ID": submission_id,
        "X-Model-Filename": filename,
        "X-Artifact-SHA256": snapshot["sha256"],
    }
    try:
        with artifact.open("rb") as source:
            response = requests.post(
                endpoint,
                headers=headers,
                data=source,
                timeout=(
                    config.GRADING_CONNECT_TIMEOUT_SECONDS,
                    config.GRADING_UPLOAD_TIMEOUT_SECONDS,
                ),
                allow_redirects=False,
            )
    except (OSError, requests.RequestException) as exc:
        raise DeliveryPending(f"model upload failed: {exc}") from exc
    try:
        if response.status_code not in (200, 201, 202):
            detail = response.text[:1000]
            if response.status_code >= 500 or response.status_code in (408, 425, 429):
                raise DeliveryPending(
                    f"grading server temporarily unavailable: HTTP {response.status_code} {detail}"
                )
            raise DeliveryRejected(
                f"grading server rejected the model: HTTP {response.status_code} {detail}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise DeliveryPending("grading server returned invalid JSON") from exc
        expected = {
            "submission_id": submission_id,
            "team_name": config.TEAM_ID,
            "artifact_sha256": snapshot["sha256"],
            "artifact_size_bytes": snapshot["size_bytes"],
        }
        if not isinstance(body, dict) or any(
            body.get(key) != value for key, value in expected.items()
        ):
            raise DeliveryRejected("grading server acknowledgement mismatch")
        return body
    finally:
        response.close()


def submit_selected_checkpoint(filename: str) -> dict:
    if not _SUBMISSION_LOCK.acquire(blocking=False):
        raise SubmissionBusy("another model upload is already active")
    try:
        source = selected_model_path(filename)
        submission_id = str(uuid.uuid4())
        snapshot = create_snapshot(
            submission_id,
            source,
            config.SUBMISSIONS_DIR,
            max_bytes=config.MAX_ARTIFACT_BYTES,
        )
        artifact, _ = snapshot_paths(submission_id, config.SUBMISSIONS_DIR)
        try:
            response = _upload(submission_id, filename, snapshot)
        except DeliveryPending as exc:
            raise DeliveryPending(
                f"{exc}; 결과가 불확실하면 submission_id={submission_id} 상태를 조회하세요"
            ) from exc
        return {
            **response,
            "model_filename": filename,
            "snapshot_path": str(artifact),
        }
    finally:
        _SUBMISSION_LOCK.release()


def get_remote_submission(submission_id: str) -> dict:
    canonical = str(uuid.UUID(submission_id))
    if not secrets.compare_digest(canonical, submission_id.lower()):
        raise ValueError("submission_id must be a canonical UUID")
    try:
        response = requests.get(
            _grading_endpoint(f"/api/v1/submissions/{canonical}"),
            headers=_auth_headers(),
            timeout=config.GRADING_STATUS_TIMEOUT_SECONDS,
            allow_redirects=False,
        )
    except (requests.RequestException, RuntimeError) as exc:
        raise DeliveryPending(f"grading status request failed: {exc}") from exc
    try:
        if response.status_code == 404:
            raise DeliveryRejected("submission is not known to the grading server")
        if response.status_code != 200:
            raise DeliveryPending(
                f"grading status unavailable: HTTP {response.status_code} {response.text[:500]}"
            )
        body = response.json()
        if not isinstance(body, dict) or body.get("submission_id") != canonical:
            raise DeliveryRejected("grading status response mismatch")
        return body
    finally:
        response.close()
