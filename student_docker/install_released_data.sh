#!/bin/bash
# Install or verify the complete participant-side v2 validation bundle.
# Installation is root-only and authenticated. --verify-only rehashes every
# image; --verify-ready checks the marker and critical assets in each team's
# private workspace copy without rescanning every JPEG at container startup.
set -euo pipefail

readonly RAW_DATA_ROOT="${DATASET_ROOT-${STUDENT_WORKSPACE_ROOT-/workspace}}"
if [ -z "${RAW_DATA_ROOT}" ]; then
    printf '[student-data-init] DATASET_ROOT must not be empty\n' >&2
    exit 1
fi
case "${RAW_DATA_ROOT}" in
    /*) ;;
    *)
        printf '[student-data-init] DATASET_ROOT must be absolute: %s\n' "${RAW_DATA_ROOT}" >&2
        exit 1
        ;;
esac
if [ -L "${RAW_DATA_ROOT}" ]; then
    printf '[student-data-init] DATASET_ROOT must not be a symlink: %s\n' "${RAW_DATA_ROOT}" >&2
    exit 1
fi
normalized_data_root="$(realpath -ms -- "${RAW_DATA_ROOT}")"
resolved_data_root="$(realpath -m -- "${RAW_DATA_ROOT}")"
if [ "${normalized_data_root}" != "${resolved_data_root}" ]; then
    printf '[student-data-init] DATASET_ROOT must not traverse symlinked parents: %s\n' "${RAW_DATA_ROOT}" >&2
    exit 1
fi
readonly DATA_ROOT="${resolved_data_root}"
unset normalized_data_root resolved_data_root
case "${DATA_ROOT}" in
    /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var)
        printf '[student-data-init] refusing broad DATASET_ROOT target: %s\n' "${DATA_ROOT}" >&2
        exit 1
        ;;
esac
readonly LEGACY_DATA_ROOT="${STUDENT_DATA_LINK_ROOT:-/data/hai_ssh/datasets/imagenet}"

# Organizer-published immutable pins. Environment overrides remain available
# for a future dataset version, and the same strict validation applies to them.
readonly REPO_ID="${HF_STUDENT_REPO_ID:-cau-ai-hackathon/imagenet-released}"
readonly REPO_REVISION="${HF_STUDENT_REPO_REVISION:-dbd6208837a986ba32139a6275b30efb5ed071f5}"
readonly ARCHIVE_NAME="${HF_STUDENT_ARCHIVE:-student_docker.tar}"
readonly TOKEN_PATH="${HF_TOKEN_PATH:-/run/secrets/hf_student_token}"
readonly EXPECTED_ARCHIVE_BYTES="${HF_STUDENT_EXPECTED_ARCHIVE_BYTES:-15576883200}"
readonly EXPECTED_ARCHIVE_SHA256="${HF_STUDENT_EXPECTED_ARCHIVE_SHA256:-749bfe3f2313987f35b2155a7b17a3c0ad312a91733458e0047aa4a2262c29ec}"
readonly EXPECTED_DATASET_REVISION="${HF_STUDENT_EXPECTED_DATASET_REVISION:-3490c3295fcf00c192011e3d570db670275d1859a7a8ed91ae52eebfd6eb88b9}"

readonly EXPECTED_RELEASED_COUNT="${HF_STUDENT_EXPECTED_RELEASED_IMAGES:-113566}"
readonly EXPECTED_VALIDATION_COUNT="${HF_STUDENT_EXPECTED_VALIDATION_IMAGES:-15000}"
readonly EXPECTED_MODEL_BYTES="${HF_STUDENT_EXPECTED_MODEL_BYTES:-343562706}"

readonly BOOTSTRAP_ROOT="${DATA_ROOT}/.bootstrap"
readonly DOWNLOAD_ROOT="${BOOTSTRAP_ROOT}/student-validation-v2-download"
readonly STAGING_ROOT="${BOOTSTRAP_ROOT}/student-validation-v2-staging"
readonly MARKER_PATH="${BOOTSTRAP_ROOT}/student-validation-v2.ready"

readonly MANIFEST_PATH="${DATA_ROOT}/dataset_manifest.json"
readonly IMAGE_ROOT="${DATA_ROOT}/imagenet_released"
readonly MODEL_PATH="${DATA_ROOT}/m_o/M_o.pt"
readonly SPLIT_PATH="${DATA_ROOT}/splits/student_split.pt"
readonly CACHE_PATH="${DATA_ROOT}/validation_cache/M_o__validation.npz"
readonly REFS_PATH="${DATA_ROOT}/validation_cache/refs.pt"

VERIFY_ONLY=0
VERIFY_READY=0
case "${1:-}" in
    "") ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --verify-ready) VERIFY_READY=1 ;;
    -h|--help)
        printf 'usage: %s [--verify-only|--verify-ready]\n' "$0"
        exit 0
        ;;
    *)
        printf '[student-data-init] unsupported argument: %s\n' "$1" >&2
        exit 2
        ;;
esac
[ "$#" -le 1 ] || {
    printf '[student-data-init] too many arguments\n' >&2
    exit 2
}

log() {
    printf '[student-data-init] %s\n' "$*"
}

fail() {
    log "$*" >&2
    exit 1
}

is_placeholder() {
    case "$1" in
        REPLACE_WITH_*) return 0 ;;
        *) return 1 ;;
    esac
}

require_positive_integer() {
    case "$2" in
        ''|*[!0-9]*) fail "$1 must be a positive integer" ;;
    esac
    [ "$2" -gt 0 ] || fail "$1 must be a positive integer"
}

require_sha256() {
    case "$2" in
        *[!0-9a-f]*|'') fail "$1 must be 64 lowercase hexadecimal characters" ;;
    esac
    [ "${#2}" -eq 64 ] || fail "$1 must be 64 lowercase hexadecimal characters"
}

require_install_pins() {
    is_placeholder "${REPO_ID}" && fail "HF_STUDENT_REPO_ID still has its v2 placeholder value"
    is_placeholder "${REPO_REVISION}" && fail "HF_STUDENT_REPO_REVISION still has its v2 placeholder value"
    is_placeholder "${EXPECTED_ARCHIVE_BYTES}" && fail "HF_STUDENT_EXPECTED_ARCHIVE_BYTES still has its v2 placeholder value"
    is_placeholder "${EXPECTED_ARCHIVE_SHA256}" && fail "HF_STUDENT_EXPECTED_ARCHIVE_SHA256 still has its v2 placeholder value"
    is_placeholder "${EXPECTED_DATASET_REVISION}" && fail "HF_STUDENT_EXPECTED_DATASET_REVISION still has its v2 placeholder value"
    [[ "${REPO_ID}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || \
        fail "HF_STUDENT_REPO_ID must be an owner/repository identifier"
    [[ "${REPO_REVISION}" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]] || \
        fail "HF_STUDENT_REPO_REVISION must be an immutable commit hash"
    [[ "${ARCHIVE_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || \
        fail "HF_STUDENT_ARCHIVE must be a plain file name"
    require_positive_integer HF_STUDENT_EXPECTED_ARCHIVE_BYTES "${EXPECTED_ARCHIVE_BYTES}"
    require_sha256 HF_STUDENT_EXPECTED_ARCHIVE_SHA256 "${EXPECTED_ARCHIVE_SHA256}"
    require_sha256 HF_STUDENT_EXPECTED_DATASET_REVISION "${EXPECTED_DATASET_REVISION}"
}

for count_pair in \
    "released:${EXPECTED_RELEASED_COUNT}" \
    "validation:${EXPECTED_VALIDATION_COUNT}" \
    "M_o.pt:${EXPECTED_MODEL_BYTES}"; do
    require_positive_integer "expected ${count_pair%%:*}" "${count_pair#*:}"
done

marker_value() {
    local key="$1"
    sed -n "s/^${key}=//p" "${MARKER_PATH}" | head -n 1
}

marker_has_valid_format() {
    [ -f "${MARKER_PATH}" ] && [ ! -L "${MARKER_PATH}" ] || return 1
    [ "$(marker_value format)" = "2" ] || return 1
    [ -n "$(marker_value repo)" ] || return 1
    [ -n "$(marker_value revision)" ] || return 1
    [ -n "$(marker_value archive)" ] || return 1

    local marker_archive_bytes marker_archive_sha marker_dataset_revision
    marker_archive_bytes="$(marker_value archive_bytes)"
    marker_archive_sha="$(marker_value archive_sha256)"
    marker_dataset_revision="$(marker_value dataset_revision)"
    case "${marker_archive_bytes}" in ''|*[!0-9]*) return 1 ;; esac
    [ "${#marker_archive_sha}" -eq 64 ] || return 1
    [ "${#marker_dataset_revision}" -eq 64 ] || return 1
    case "${marker_archive_sha}${marker_dataset_revision}" in *[!0-9a-f]*) return 1 ;; esac

    for key in manifest_sha256 split_sha256 refs_sha256 cache_sha256 model_sha256; do
        local digest
        digest="$(marker_value "${key}")"
        [ "${#digest}" -eq 64 ] || return 1
        case "${digest}" in *[!0-9a-f]*) return 1 ;; esac
    done
}

marker_matches_requested_pins() {
    marker_has_valid_format || return 1
    if ! is_placeholder "${REPO_ID}"; then
        [ "$(marker_value repo)" = "${REPO_ID}" ] || return 1
    fi
    if ! is_placeholder "${REPO_REVISION}"; then
        [ "$(marker_value revision)" = "${REPO_REVISION}" ] || return 1
    fi
    [ "$(marker_value archive)" = "${ARCHIVE_NAME}" ] || return 1
    if ! is_placeholder "${EXPECTED_ARCHIVE_BYTES}"; then
        [ "$(marker_value archive_bytes)" = "${EXPECTED_ARCHIVE_BYTES}" ] || return 1
    fi
    if ! is_placeholder "${EXPECTED_ARCHIVE_SHA256}"; then
        [ "$(marker_value archive_sha256)" = "${EXPECTED_ARCHIVE_SHA256}" ] || return 1
    fi
    if ! is_placeholder "${EXPECTED_DATASET_REVISION}"; then
        [ "$(marker_value dataset_revision)" = "${EXPECTED_DATASET_REVISION}" ] || return 1
    fi
    [ "$(marker_value released_count)" = "${EXPECTED_RELEASED_COUNT}" ] || return 1
    [ "$(marker_value validation_count)" = "${EXPECTED_VALIDATION_COUNT}" ] || return 1
    [ "$(marker_value model_bytes)" = "${EXPECTED_MODEL_BYTES}" ] || return 1
}

marker_digests_match_files() {
    local key path
    while read -r key path; do
        [ -f "${path}" ] && [ ! -L "${path}" ] || return 1
        [ "$(sha256sum "${path}" | awk '{print $1}')" = "$(marker_value "${key}")" ] || return 1
    done <<EOF
manifest_sha256 ${MANIFEST_PATH}
split_sha256 ${SPLIT_PATH}
refs_sha256 ${REFS_PATH}
cache_sha256 ${CACHE_PATH}
model_sha256 ${MODEL_PATH}
EOF
}

validate_dataset() {
    local validation_root="$1"
    VALIDATION_ROOT="${validation_root}" \
    VALIDATE_RELEASED_COUNT="${EXPECTED_RELEASED_COUNT}" \
    VALIDATE_VALIDATION_COUNT="${EXPECTED_VALIDATION_COUNT}" \
    VALIDATE_MODEL_BYTES="${EXPECTED_MODEL_BYTES}" \
    VALIDATE_DATASET_REVISION="${EXPECTED_DATASET_REVISION}" \
    python - <<'PY'
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import torch


root = Path(os.environ["VALIDATION_ROOT"])
expected_counts = {
    "released": int(os.environ["VALIDATE_RELEASED_COUNT"]),
    "validation": int(os.environ["VALIDATE_VALIDATION_COUNT"]),
}
expected_model_bytes = int(os.environ["VALIDATE_MODEL_BYTES"])
expected_revision = os.environ["VALIDATE_DATASET_REVISION"]


def abort(message: str) -> None:
    print(f"dataset format-v2 validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_content(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        abort(f"cannot open dataset image {path}: {exc}")
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            abort(f"dataset image is not regular: {path}")
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(source.fileno())
    signature = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if signature(before) != signature(after):
        abort(f"dataset image changed while hashing: {path}")
    return after.st_size, digest.hexdigest()


def require_regular(relative: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        abort(f"required regular file is missing: {relative}")
    return path


if root.is_symlink() or not root.is_dir():
    abort("dataset root is missing or symlinked")
for relative in ("imagenet_released", "m_o", "splits", "validation_cache"):
    directory = root / relative
    if directory.is_symlink() or not directory.is_dir():
        abort(f"required directory is missing or symlinked: {relative}")

manifest_path = require_regular("dataset_manifest.json")
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
    abort(f"dataset_manifest.json is invalid: {exc}")

required_manifest = {
    "schema_version": 2,
    "phase": "validation",
    "score_version": "unlearning-v2",
    "runtime_root": "/workspace/datasets/imagenet_released",
}
if not isinstance(manifest, dict) or set(manifest) != {
    *required_manifest, "dataset_revision", "splits", "assets"
}:
    abort("dataset manifest keys do not match the v2 schema")
for key, expected in required_manifest.items():
    if manifest.get(key) != expected:
        abort(f"dataset manifest {key} mismatch")
revision = manifest.get("dataset_revision")
if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{64}", revision) is None:
    abort("dataset manifest dataset_revision is not a SHA-256")
if not expected_revision.startswith("REPLACE_WITH_") and revision != expected_revision:
    abort("dataset manifest dataset_revision does not match the deployment pin")

for forbidden in ("test", "holdout_A", "holdout_B", "val"):
    if (root / "imagenet_released" / forbidden).exists():
        abort(f"private/legacy split must not be present: {forbidden}")
image_children = {path.name for path in (root / "imagenet_released").iterdir()}
if image_children != {"train", "validation"}:
    abort("image root must contain exactly train and validation")
if {path.name for path in (root / "m_o").iterdir()} != {"M_o.pt"}:
    abort("m_o directory contains missing or unexpected assets")
if {path.name for path in (root / "splits").iterdir()} != {"student_split.pt"}:
    abort("splits directory contains missing or unexpected assets")
if {path.name for path in (root / "validation_cache").iterdir()} != {
    "M_o__validation.npz", "refs.pt"
}:
    abort("validation_cache contains missing or unexpected assets")

split_path = require_regular("splits/student_split.pt")
refs_path = require_regular("validation_cache/refs.pt")
cache_path = require_regular("validation_cache/M_o__validation.npz")
model_path = require_regular("m_o/M_o.pt")
if model_path.stat().st_size != expected_model_bytes:
    abort("M_o.pt size mismatch")

try:
    split = torch.load(split_path, map_location="cpu", weights_only=True)
    refs = torch.load(refs_path, map_location="cpu", weights_only=True)
except Exception as exc:
    abort(f"cannot load split/reference metadata: {exc}")
if not isinstance(split, dict) or set(split) != {"meta", "splits"}:
    abort("student_split.pt must contain exactly meta and splits")
if not isinstance(split["meta"], dict) or not isinstance(split["splits"], dict):
    abort("student split meta/splits must be mappings")
if set(split["splits"]) != set(expected_counts):
    abort("student split must contain released and validation only")

meta = split["meta"]
if set(meta) != {
    "schema_version", "phase", "score_version", "root", "dataset_revision",
    "wnids", "counts", "sha256", "records_sha256", "content_sha256",
}:
    abort("student split meta keys do not match the v2 schema")
for key, expected in (
    ("schema_version", 2),
    ("phase", "validation"),
    ("score_version", "unlearning-v2"),
    ("root", "/workspace/datasets/imagenet_released"),
    ("dataset_revision", revision),
):
    if meta.get(key) != expected:
        abort(f"student split meta {key} mismatch")
wnids = meta.get("wnids")
if (
    not isinstance(wnids, list)
    or not wnids
    or len(wnids) != len(set(wnids))
    or not all(isinstance(value, str) and value for value in wnids)
):
    abort("student split wnids are invalid")

directory_for = {
    "released": "train",
    "validation": "validation",
}
computed_info = {}
records_hashes = {}
validation_labels = None
for name, expected_count in expected_counts.items():
    raw_records = split["splits"][name]
    if not isinstance(raw_records, list) or len(raw_records) != expected_count:
        abort(f"{name} count mismatch: expected {expected_count}, got {len(raw_records) if isinstance(raw_records, list) else 'invalid'}")
    records = []
    seen = set()
    byte_total = 0
    content_entries = []
    expected_prefix = directory_for[name]
    for raw in raw_records:
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            abort(f"{name} contains a malformed record")
        relative, label = raw
        if not isinstance(relative, str):
            abort(f"{name} contains a non-string path")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or relative != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or len(pure.parts) < 3
            or pure.parts[0] != expected_prefix
            or pure.suffix != ".JPEG"
        ):
            abort(f"{name} contains an unsafe/wrong-prefix path: {relative!r}")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(wnids):
            abort(f"{name} contains an invalid label")
        if relative in seen:
            abort(f"{name} contains a duplicate path")
        seen.add(relative)
        image = root / "imagenet_released" / relative
        if image.is_symlink() or not image.is_file():
            abort(f"{name} image is missing or not regular: {relative}")
        image_bytes, image_sha256 = image_content(image)
        byte_total += image_bytes
        content_entries.append((relative, image_bytes, image_sha256))
        records.append((relative, label))

    split_dir = root / "imagenet_released" / expected_prefix
    actual = set()
    if split_dir.is_symlink() or not split_dir.is_dir():
        abort(f"{name} image directory is missing or symlinked")
    for candidate in split_dir.rglob("*"):
        if candidate.is_symlink():
            abort(f"{name} image tree contains a symlink")
        if candidate.is_file():
            if candidate.suffix != ".JPEG":
                abort(f"{name} image tree contains a non-JPEG file")
            actual.add(candidate.relative_to(root / "imagenet_released").as_posix())
    if actual != seen:
        abort(f"{name} image files do not exactly match the split manifest")

    path_hash = hashlib.sha256("".join(sorted(seen)).encode()).hexdigest()
    records_hash = hashlib.sha256(
        "".join(f"{path}\t{label}\n" for path, label in sorted(records)).encode()
    ).hexdigest()
    content_hash = hashlib.sha256(
        "".join(
            f"{path}\0{size}\0{digest}\n"
            for path, size, digest in sorted(content_entries)
        ).encode()
    ).hexdigest()
    computed_info[name] = {
        "count": expected_count,
        "bytes": byte_total,
        "path_sha256": path_hash,
        "records_sha256": records_hash,
        "content_sha256": content_hash,
    }
    records_hashes[name] = records_hash
    if name == "validation":
        validation_labels = np.asarray([label for _, label in records])

if manifest.get("splits") != computed_info:
    abort("dataset manifest split metadata does not match files/records")
if meta.get("counts") != expected_counts:
    abort("student split meta counts mismatch")
if meta.get("sha256") != {name: info["path_sha256"] for name, info in computed_info.items()}:
    abort("student split meta path hashes mismatch")
if meta.get("records_sha256") != records_hashes:
    abort("student split meta record hashes mismatch")
if meta.get("content_sha256") != {
    name: info["content_sha256"] for name, info in computed_info.items()
}:
    abort("student split meta content hashes mismatch")
calculated_revision = hashlib.sha256(
    "".join(
        f"{name}\t{records_hashes[name]}\n"
        for name in ("validation",)
    ).encode()
).hexdigest()
if calculated_revision != revision:
    abort("dataset_revision does not match validation records")

assets = manifest.get("assets")
if not isinstance(assets, dict) or set(assets) != {"m_o", "representation_cache", "refs"}:
    abort("dataset manifest assets do not match the v2 schema")
asset_specs = {
    "m_o": ("m_o/M_o.pt", model_path),
    "representation_cache": ("validation_cache/M_o__validation.npz", cache_path),
}
for name, (relative, path) in asset_specs.items():
    spec = assets.get(name)
    expected = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **({"status": "ready"} if name == "representation_cache" else {}),
    }
    if spec != expected:
        abort(f"dataset manifest {name} metadata mismatch")
if assets.get("refs") != {"path": "validation_cache/refs.pt", "schema_version": 2}:
    abort("dataset manifest refs metadata mismatch")

if not isinstance(refs, dict) or set(refs) != {
    "schema_version", "phase", "score_version", "dataset_revision",
    "accuracy_split", "representation_split", "reference_accuracy",
    "score_depth", "depths", "forget_labels", "forget_wnids",
}:
    abort("refs.pt must contain a mapping")
for key, expected in (
    ("schema_version", 2),
    ("phase", "validation"),
    ("score_version", "unlearning-v2"),
    ("dataset_revision", revision),
    ("accuracy_split", "validation"),
    ("representation_split", "validation"),
):
    if refs.get(key) != expected:
        abort(f"refs.pt {key} mismatch")
if refs.get("depths") != ["b4", "b8", "b12", "pre"] or refs.get("score_depth") != "pre":
    abort("refs.pt feature depth configuration mismatch")
forget_labels = refs.get("forget_labels")
forget_wnids = refs.get("forget_wnids")
if (
    not isinstance(forget_labels, list)
    or not forget_labels
    or len(forget_labels) != len(set(forget_labels))
    or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(wnids) for value in forget_labels)
    or forget_wnids != [wnids[value] for value in forget_labels]
):
    abort("refs.pt forget label/wnid mapping is invalid")
reference_accuracy = refs.get("reference_accuracy")
if not isinstance(reference_accuracy, dict) or set(reference_accuracy) != {"acc_f", "acc_r"}:
    abort("refs.pt reference_accuracy is invalid")
for name, value in reference_accuracy.items():
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or not 0 <= value <= 100
    ):
        abort(f"refs.pt reference_accuracy {name} is invalid")

try:
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {
            "schema_version", "phase", "dataset_revision", "split_name",
            "correct", "total", "labels", "f_b4", "f_b8", "f_b12", "f_pre",
        }
        if len(cache.files) != len(set(cache.files)) or set(cache.files) != required:
            abort("reference cache arrays do not match the v2 schema")
        for key, expected in (
            ("schema_version", 2),
            ("phase", "validation"),
            ("dataset_revision", revision),
            ("split_name", "validation"),
        ):
            if cache[key].ndim != 0 or cache[key].item() != expected:
                abort(f"reference cache {key} mismatch")
        if cache["labels"].dtype != np.dtype(np.int64) or cache["labels"].ndim != 1:
            abort("reference cache labels must be a 1-D int64 array")
        if not np.array_equal(cache["labels"], validation_labels):
            abort("reference cache labels/order do not match validation")
        for name in ("correct", "total"):
            values = cache[name]
            if (
                values.dtype != np.dtype(np.float64)
                or values.shape != (len(wnids),)
                or not np.isfinite(values).all()
                or (values < 0).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                abort(f"reference cache {name} is invalid")
        if (cache["correct"] > cache["total"]).any():
            abort("reference cache correct exceeds total")
        expected_total = np.bincount(validation_labels, minlength=len(wnids)).astype(np.float64)
        if not np.array_equal(cache["total"], expected_total):
            abort("reference cache totals do not match validation labels")
        for depth in ("b4", "b8", "b12", "pre"):
            features = cache[f"f_{depth}"]
            if (
                features.dtype != np.dtype(np.float32)
                or features.shape != (expected_counts["validation"], 768)
                or not np.isfinite(features).all()
            ):
                abort(f"reference cache f_{depth} shape/dtype mismatch")
except (OSError, ValueError, KeyError) as exc:
    abort(f"reference cache is invalid: {exc}")
PY
}

write_marker() {
    local marker_tmp="${MARKER_PATH}.tmp.$$"
    local dataset_revision
    dataset_revision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_revision"])' "${MANIFEST_PATH}")"
    printf '%s\n' \
        'format=2' \
        "repo=${REPO_ID}" \
        "revision=${REPO_REVISION}" \
        "archive=${ARCHIVE_NAME}" \
        "archive_bytes=${EXPECTED_ARCHIVE_BYTES}" \
        "archive_sha256=${EXPECTED_ARCHIVE_SHA256}" \
        "dataset_revision=${dataset_revision}" \
        "released_count=${EXPECTED_RELEASED_COUNT}" \
        "validation_count=${EXPECTED_VALIDATION_COUNT}" \
        "model_bytes=${EXPECTED_MODEL_BYTES}" \
        "manifest_sha256=$(sha256sum "${MANIFEST_PATH}" | awk '{print $1}')" \
        "split_sha256=$(sha256sum "${SPLIT_PATH}" | awk '{print $1}')" \
        "refs_sha256=$(sha256sum "${REFS_PATH}" | awk '{print $1}')" \
        "cache_sha256=$(sha256sum "${CACHE_PATH}" | awk '{print $1}')" \
        "model_sha256=$(sha256sum "${MODEL_PATH}" | awk '{print $1}')" \
        > "${marker_tmp}"
    chmod 0444 "${marker_tmp}"
    mv -f "${marker_tmp}" "${MARKER_PATH}"
    sync -f "${MARKER_PATH}"
}

cleanup_transient_files() {
    rm -rf -- "${DOWNLOAD_ROOT}" "${STAGING_ROOT}"
    find "${BOOTSTRAP_ROOT}" -maxdepth 1 -type f -name 'student-validation-v2.ready.tmp.*' -delete
}

verify_installed() {
    marker_matches_requested_pins || fail "format-v2 completion marker is missing or does not match requested pins"
    validate_dataset "${DATA_ROOT}"
    marker_digests_match_files || fail "installed dataset files do not match the completion marker"
    [ "$(marker_value dataset_revision)" = "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_revision"])' "${MANIFEST_PATH}")" ] || \
        fail "completion marker dataset_revision mismatch"
}

verify_ready_marker() {
    # The marker is authoritative only under the deployment contract: it was
    # written after dataset-init performed validate_dataset, all managed files
    # are root-owned/non-writable, and this container mounts /datasets read-only.
    marker_matches_requested_pins || fail "format-v2 completion marker is missing or does not match requested pins"
    marker_digests_match_files || fail "critical dataset assets do not match the completion marker"
    for required_directory in \
        "${IMAGE_ROOT}" \
        "${IMAGE_ROOT}/train" \
        "${IMAGE_ROOT}/validation" \
        "${DATA_ROOT}/m_o" \
        "${DATA_ROOT}/splits" \
        "${DATA_ROOT}/validation_cache"; do
        [ -d "${required_directory}" ] && [ ! -L "${required_directory}" ] || \
            fail "required dataset directory is missing or symlinked: ${required_directory}"
    done
    [ "$(marker_value dataset_revision)" = "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_revision"])' "${MANIFEST_PATH}")" ] || \
        fail "completion marker dataset_revision mismatch"
}

if [ "${VERIFY_ONLY}" -eq 1 ]; then
    require_install_pins
    [ -d "${DATA_ROOT}" ] || fail "dataset root is missing: ${DATA_ROOT}"
    verify_installed
    log "format-v2 validation data fully verified (read-only mode)"
    exit 0
fi

if [ "${VERIFY_READY}" -eq 1 ]; then
    require_install_pins
    [ -d "${DATA_ROOT}" ] || fail "dataset root is missing: ${DATA_ROOT}"
    verify_ready_marker
    log "format-v2 trusted init marker and critical assets verified (read-only mode)"
    exit 0
fi

require_install_pins
if [ "$(id -u)" != "0" ] && [ "${HACKATHON_INSTALLER_TEST_MODE:-0}" != "1" ]; then
    fail "dataset installation must run as root"
fi
if [ -L "${DATA_ROOT}" ]; then
    fail "dataset root must not be a symlink: ${DATA_ROOT}"
fi
if [ -L "${BOOTSTRAP_ROOT}" ] || { [ -e "${BOOTSTRAP_ROOT}" ] && [ ! -d "${BOOTSTRAP_ROOT}" ]; }; then
    fail "dataset bootstrap root must be a real directory: ${BOOTSTRAP_ROOT}"
fi
mkdir -p "${BOOTSTRAP_ROOT}"
if [ "${HACKATHON_INSTALLER_TEST_MODE:-0}" != "1" ]; then
    chown root:root "${BOOTSTRAP_ROOT}"
fi
chmod 0700 "${BOOTSTRAP_ROOT}"
for transient_root in "${DOWNLOAD_ROOT}" "${STAGING_ROOT}"; do
    [ ! -L "${transient_root}" ] || fail "transient dataset path must not be a symlink: ${transient_root}"
done

if marker_matches_requested_pins; then
    if validate_dataset "${DATA_ROOT}" && marker_digests_match_files; then
        cleanup_transient_files
        log "format-v2 validation data already installed; skipping download"
        exit 0
    fi
    fail "completion marker exists but installed format-v2 data failed verification"
elif [ -f "${MARKER_PATH}" ]; then
    log "installed data marker does not match requested v2 pins; reinstalling"
fi

if [ -n "${HF_STUDENT_TOKEN:-}" ]; then
    hf_token="${HF_STUDENT_TOKEN}"
    unset HF_STUDENT_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN
elif [ -f "${TOKEN_PATH}" ] && [ ! -L "${TOKEN_PATH}" ] && [ -r "${TOKEN_PATH}" ]; then
    hf_token="$(tr -d '\r\n' < "${TOKEN_PATH}")"
else
    fail "HF token is unavailable; set HF_STUDENT_TOKEN or mount ${TOKEN_PATH}"
fi
[ -n "${hf_token}" ] || fail "HF token is empty"

mkdir -p "${DOWNLOAD_ROOT}"
log "downloading ${REPO_ID}@${REPO_REVISION}/${ARCHIVE_NAME}"
xet_version="$(python -c 'from importlib.metadata import version; print(version("hf-xet"))' 2>/dev/null)" || \
    fail "hf_xet is not installed; refusing to use an unaccelerated fallback"
log "starting hf_xet download (version=${xet_version}, high_performance=${HF_XET_HIGH_PERFORMANCE:-0})"
HF_TOKEN="${hf_token}" \
HF_HOME="${DOWNLOAD_ROOT}/.hf-home" \
HF_HUB_CACHE="${DOWNLOAD_ROOT}/.hf-home/hub" \
HF_XET_CACHE="${DOWNLOAD_ROOT}/.hf-home/xet" \
hf download "${REPO_ID}" "${ARCHIVE_NAME}" \
    --repo-type dataset \
    --revision "${REPO_REVISION}" \
    --local-dir "${DOWNLOAD_ROOT}"
hf_token=''
unset HF_TOKEN

archive_path="${DOWNLOAD_ROOT}/${ARCHIVE_NAME}"
[ -f "${archive_path}" ] || fail "download completed without expected archive ${archive_path}"
archive_bytes="$(stat -c '%s' "${archive_path}" 2>/dev/null || true)"
[ "${archive_bytes}" = "${EXPECTED_ARCHIVE_BYTES}" ] || \
    fail "archive size mismatch: expected ${EXPECTED_ARCHIVE_BYTES}, got ${archive_bytes}"
log "verifying archive SHA-256"
archive_sha256="$(sha256sum "${archive_path}" | awk '{print $1}')"
[ "${archive_sha256}" = "${EXPECTED_ARCHIVE_SHA256}" ] || fail "archive SHA-256 mismatch"

# Reject links, devices, traversal, duplicate names, and unexpected top-level
# content even though the archive is pinned. This keeps extraction fail-closed.
python - "${archive_path}" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

archive_path = sys.argv[1]
allowed = {
    "dataset_manifest.json",
    "imagenet_released",
    "m_o",
    "splits",
    "validation_cache",
}
seen = set()
with tarfile.open(archive_path, "r:*") as archive:
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or member.name != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
            or not path.parts
            or path.parts[0] != "student_docker"
            or (len(path.parts) > 1 and path.parts[1] not in allowed)
            or not (member.isdir() or member.isfile())
            or member.name in seen
        ):
            raise SystemExit(f"unsafe or unexpected archive member: {member.name!r}")
        seen.add(member.name)
PY

rm -rf -- "${STAGING_ROOT}"
mkdir -p "${STAGING_ROOT}"
log "extracting format-v2 student validation bundle"
tar --extract --file "${archive_path}" \
    --directory "${STAGING_ROOT}" \
    --strip-components=1 \
    --no-same-owner \
    student_docker/dataset_manifest.json \
    student_docker/imagenet_released \
    student_docker/m_o \
    student_docker/splits \
    student_docker/validation_cache

validate_dataset "${STAGING_ROOT}"

# Publish all managed paths before writing the marker. A crash during these
# moves leaves no valid marker, so the next init safely downloads/rebuilds.
rm -rf -- \
    "${MANIFEST_PATH}" \
    "${IMAGE_ROOT}" \
    "${DATA_ROOT}/m_o" \
    "${DATA_ROOT}/splits" \
    "${DATA_ROOT}/validation_cache"
mv "${STAGING_ROOT}/dataset_manifest.json" "${MANIFEST_PATH}"
mv "${STAGING_ROOT}/imagenet_released" "${IMAGE_ROOT}"
mv "${STAGING_ROOT}/m_o" "${DATA_ROOT}/m_o"
mv "${STAGING_ROOT}/splits" "${DATA_ROOT}/splits"
mv "${STAGING_ROOT}/validation_cache" "${DATA_ROOT}/validation_cache"
rmdir "${STAGING_ROOT}"

if [ "${HACKATHON_INSTALLER_TEST_MODE:-0}" != "1" ]; then
    chown -R root:root \
        "${MANIFEST_PATH}" "${IMAGE_ROOT}" "${DATA_ROOT}/m_o" \
        "${DATA_ROOT}/splits" "${DATA_ROOT}/validation_cache"
    find "${IMAGE_ROOT}" "${DATA_ROOT}/m_o" "${DATA_ROOT}/splits" \
        "${DATA_ROOT}/validation_cache" -type d -exec chmod 0555 {} +
    find "${MANIFEST_PATH}" "${IMAGE_ROOT}" "${DATA_ROOT}/m_o" \
        "${DATA_ROOT}/splits" "${DATA_ROOT}/validation_cache" -type f -exec chmod 0444 {} +
fi

write_marker

mkdir -p "${LEGACY_DATA_ROOT}"
if [ -e "${LEGACY_DATA_ROOT}/train" ] && [ ! -L "${LEGACY_DATA_ROOT}/train" ]; then
    fail "refusing to replace non-symlink path ${LEGACY_DATA_ROOT}/train"
fi
ln -sfn "${IMAGE_ROOT}/train" "${LEGACY_DATA_ROOT}/train"

cleanup_transient_files
log "format-v2 validation data installation complete"
