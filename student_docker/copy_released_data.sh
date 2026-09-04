#!/bin/bash
# Copy the one downloaded student package into one team's private workspace.
set -euo pipefail

readonly SOURCE_ROOT="${SOURCE_DATA_ROOT:-/dataset-source}"
readonly WORKSPACE_ROOT="${STUDENT_WORKSPACE_ROOT:-/workspace}"
readonly TARGET_ROOT="${TARGET_DATA_ROOT:-${WORKSPACE_ROOT}/datasets}"
readonly READY_RELATIVE=".bootstrap/student-validation-v2.ready"

fail() {
    printf '[student-data-copy] %s\n' "$*" >&2
    exit 1
}

[ "$(id -u)" = "0" ] || fail "data copy must run as root"
for path in "${SOURCE_ROOT}" "${WORKSPACE_ROOT}" "${TARGET_ROOT}"; do
    case "${path}" in /*) ;; *) fail "paths must be absolute: ${path}" ;; esac
done
[ -d "${SOURCE_ROOT}" ] && [ ! -L "${SOURCE_ROOT}" ] || \
    fail "downloaded source dataset is missing"
[ -f "${SOURCE_ROOT}/${READY_RELATIVE}" ] && \
    [ ! -L "${SOURCE_ROOT}/${READY_RELATIVE}" ] || \
    fail "downloaded source dataset has no trusted ready marker"

mkdir -p "${WORKSPACE_ROOT}"
[ ! -L "${WORKSPACE_ROOT}" ] || fail "workspace must not be a symlink"
if [ -e "${TARGET_ROOT}" ] || [ -L "${TARGET_ROOT}" ]; then
    [ -d "${TARGET_ROOT}" ] && [ ! -L "${TARGET_ROOT}" ] || \
        fail "target dataset path is unsafe"
    cmp -s "${SOURCE_ROOT}/${READY_RELATIVE}" "${TARGET_ROOT}/${READY_RELATIVE}" || \
        fail "target dataset exists but does not match the downloaded source"
    printf '[student-data-copy] private dataset copy already exists; skipping\n'
    exit 0
fi

stage="${WORKSPACE_ROOT}/.datasets-copy-${RANDOM}-$$"
[ ! -e "${stage}" ] || fail "copy staging path already exists"
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
mkdir -m 0700 "${stage}"
printf '[student-data-copy] copying dataset into %s\n' "${TARGET_ROOT}"
# --reflink=never guarantees that later in-place augmentation cannot mutate a
# shared copy-on-write extent through filesystem-specific surprises.
cp -a --reflink=never -- "${SOURCE_ROOT}/." "${stage}/"
chown -R participant:participant "${stage}"
find "${stage}" -type d -exec chmod u+rwx {} +
find "${stage}" -type f -exec chmod u+rw {} +
mv -- "${stage}" "${TARGET_ROOT}"
trap - EXIT
sync -f "${WORKSPACE_ROOT}"
printf '[student-data-copy] independent writable dataset is ready\n'
