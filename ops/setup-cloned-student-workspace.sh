#!/bin/bash
# Install the known student archive directly into a cloned student workspace.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly WORKSPACE="${REPOSITORY_ROOT}/student_docker"
readonly EXPECTED_BYTES="15576883200"
readonly EXPECTED_SHA256="749bfe3f2313987f35b2155a7b17a3c0ad312a91733458e0047aa4a2262c29ec"

fail() {
    printf '[workspace-setup] %s\n' "$*" >&2
    exit 1
}

case "${1:-}" in
    -h|--help)
        printf 'usage: %s [ARCHIVE]\n' "$0"
        printf 'default ARCHIVE: ~/dataset/student_docker.tar\n'
        exit 0
        ;;
esac
[ "$#" -le 1 ] || fail "too many arguments"

archive_input="${1:-${HOME}/dataset/student_docker.tar}"
case "${archive_input}" in
    /*) ;;
    *) archive_input="${PWD}/${archive_input}" ;;
esac
[ -f "${archive_input}" ] && [ ! -L "${archive_input}" ] || \
    fail "archive is missing or is not a regular file: ${archive_input}"
readonly ARCHIVE="$(realpath -e -- "${archive_input}")"
[ -d "${WORKSPACE}" ] && [ ! -L "${WORKSPACE}" ] || \
    fail "student_docker workspace is missing or unsafe"

actual_bytes="$(stat -c '%s' "${ARCHIVE}")"
[ "${actual_bytes}" = "${EXPECTED_BYTES}" ] || \
    fail "archive size mismatch: expected ${EXPECTED_BYTES}, got ${actual_bytes}"
printf '[workspace-setup] checking archive SHA-256...\n'
actual_sha256="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
[ "${actual_sha256}" = "${EXPECTED_SHA256}" ] || fail "archive SHA-256 mismatch"

for target in dataset_manifest.json m_o splits validation_cache; do
    [ ! -e "${WORKSPACE}/${target}" ] && [ ! -L "${WORKSPACE}/${target}" ] || \
        fail "target already exists; refusing to overwrite: ${WORKSPACE}/${target}"
done
if [ -d "${WORKSPACE}/imagenet_released" ]; then
    unexpected="$(find "${WORKSPACE}/imagenet_released" -mindepth 1 \
        ! -path "${WORKSPACE}/imagenet_released/PLACEHOLDER.md" -print -quit)"
    [ -z "${unexpected}" ] || fail "imagenet_released already contains data"
else
    mkdir -p -- "${WORKSPACE}/imagenet_released"
fi

stage="$(mktemp -d "${WORKSPACE}/.dataset-install.XXXXXX")"
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT

printf '[workspace-setup] extracting into the cloned workspace...\n'
tar --extract --file "${ARCHIVE}" \
    --directory "${stage}" \
    --strip-components=1 \
    --no-same-owner \
    student_docker/dataset_manifest.json \
    student_docker/imagenet_released \
    student_docker/m_o \
    student_docker/splits \
    student_docker/validation_cache

for required in \
    dataset_manifest.json \
    imagenet_released/train \
    imagenet_released/validation \
    m_o/M_o.pt \
    splits/student_split.pt \
    validation_cache/M_o__validation.npz \
    validation_cache/refs.pt; do
    [ -e "${stage}/${required}" ] || fail "archive is missing ${required}"
done

released_count="$(find "${stage}/imagenet_released/train" -type f -name '*.JPEG' | wc -l)"
validation_count="$(find "${stage}/imagenet_released/validation" -type f -name '*.JPEG' | wc -l)"
[ "${released_count}" = "113566" ] || fail "released image count mismatch"
[ "${validation_count}" = "15000" ] || fail "validation image count mismatch"

mv -- "${stage}/dataset_manifest.json" "${WORKSPACE}/dataset_manifest.json"
mv -- "${stage}/m_o" "${WORKSPACE}/m_o"
mv -- "${stage}/splits" "${WORKSPACE}/splits"
mv -- "${stage}/validation_cache" "${WORKSPACE}/validation_cache"
mv -- "${stage}/imagenet_released/train" "${WORKSPACE}/imagenet_released/train"
mv -- "${stage}/imagenet_released/validation" "${WORKSPACE}/imagenet_released/validation"
mkdir -p -- "${WORKSPACE}/models" "${WORKSPACE}/results"

printf '[workspace-setup] ready: released=%s, validation=%s\n' \
    "${released_count}" "${validation_count}"
printf '[workspace-setup] next: source %q\n' \
    "${WORKSPACE}/activate_local.sh"
