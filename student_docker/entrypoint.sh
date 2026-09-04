#!/bin/bash
# Prepare the participant workspace without ever placing trusted API/scorer
# code in it. The API runs as root; code-server is the only participant-user
# process and therefore cannot replace immutable submission snapshots.
set -euo pipefail

readonly WORKSPACE_ROOT="${STUDENT_WORKSPACE_ROOT:-/workspace}"
readonly DATA_ROOT="${DATASET_ROOT:-${WORKSPACE_ROOT}}"
readonly DATA_MODE="${STUDENT_DATA_MODE:-auto}"
readonly SEED_ROOT="${STUDENT_WORKSPACE_SEED_ROOT:-/opt/workspace-seed}"
readonly INSTALLER="${STUDENT_DATA_INSTALLER:-/usr/local/sbin/install-released-data}"
readonly SUPERVISOR="${STUDENT_SUPERVISORD_BIN:-/usr/bin/supervisord}"
readonly SUPERVISOR_CONFIG="${STUDENT_SUPERVISORD_CONFIG:-/etc/supervisor/conf.d/supervisord.conf}"
readonly LEGACY_DATA_ROOT="${STUDENT_DATA_LINK_ROOT:-/data/hai_ssh/datasets/imagenet}"
readonly SOURCE_SECRET_ROOT="${HACKATHON_SECRET_SOURCE_ROOT:-/run/secrets}"
readonly STAGED_SECRET_ROOT="${HACKATHON_STAGED_SECRET_ROOT:-/run/hackathon-secrets}"
readonly PUBLIC_TEMPLATE_ROOT="${HACKATHON_PUBLIC_TEMPLATE_ROOT:-/opt/hackathon/public}"
readonly PUBLIC_RUNTIME_ROOT="${HACKATHON_PUBLIC_RUNTIME_ROOT:-/run/hackathon-public}"
readonly CODE_PUBLIC_PORT="${CODE_SERVER_PUBLIC_PORT:-8080}"

log() {
    printf '[student-entrypoint] %s\n' "$*"
}

fail() {
    log "$*" >&2
    exit 1
}

# One-shot init containers supply the installer as their command. They must
# not seed or mutate a participant workspace before downloading shared data.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

[ "$(id -u)" = "0" ] || fail "the runtime entrypoint must run as root"
participant_uid="$(id -u participant 2>/dev/null)" || fail "participant user is missing"
case "${participant_uid}" in ''|*[!0-9]*) fail "participant UID is invalid" ;; esac
[ "${participant_uid}" -ge 10000 ] || fail "participant UID must be isolated from common host UIDs"
unset participant_uid
[ -d "${SEED_ROOT}" ] || fail "workspace seed is missing: ${SEED_ROOT}"
[ -x "${INSTALLER}" ] || fail "dataset installer is missing: ${INSTALLER}"
case "${CODE_PUBLIC_PORT}" in
    ''|*[!0-9]*) fail "CODE_SERVER_PUBLIC_PORT must be numeric" ;;
esac
[ "${CODE_PUBLIC_PORT}" -ge 1 ] && [ "${CODE_PUBLIC_PORT}" -le 65535 ] || \
    fail "CODE_SERVER_PUBLIC_PORT must be in 1..65535"

validate_scoped_root() {
    local description="$1"
    local candidate="$2"
    local lexical resolved
    [ -n "${candidate}" ] || fail "${description} must not be empty"
    case "${candidate}" in
        /*) ;;
        *) fail "${description} must be absolute: ${candidate}" ;;
    esac
    lexical="$(realpath -ms -- "${candidate}")"
    resolved="$(realpath -m -- "${candidate}")"
    [ "${lexical}" = "${resolved}" ] || \
        fail "${description} must not traverse symlinked parents: ${candidate}"
    case "${resolved}" in
        /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var)
            fail "refusing broad ${description} target: ${resolved}"
            ;;
    esac
}

validate_scoped_root STUDENT_WORKSPACE_ROOT "${WORKSPACE_ROOT}"
validate_scoped_root DATASET_ROOT "${DATA_ROOT}"
validate_scoped_root STUDENT_DATA_LINK_ROOT "${LEGACY_DATA_ROOT}"
validate_scoped_root HACKATHON_SECRET_SOURCE_ROOT "${SOURCE_SECRET_ROOT}"
validate_scoped_root HACKATHON_STAGED_SECRET_ROOT "${STAGED_SECRET_ROOT}"
validate_scoped_root HACKATHON_PUBLIC_RUNTIME_ROOT "${PUBLIC_RUNTIME_ROOT}"

mkdir -p "${WORKSPACE_ROOT}"
if [ -L "${WORKSPACE_ROOT}" ]; then
    fail "workspace root must not be a symlink: ${WORKSPACE_ROOT}"
fi

if [ -L "${WORKSPACE_ROOT}/.initialized" ]; then
    fail "workspace initialization marker must not be a symlink"
fi
if [ ! -f "${WORKSPACE_ROOT}/.initialized" ]; then
    cp -a "${SEED_ROOT}/." "${WORKSPACE_ROOT}/"

    # Only the copied seed is participant-editable. Keeping the workspace
    # itself root:root 0755 prevents renaming root-owned submission/data paths.
    find "${WORKSPACE_ROOT}" -mindepth 1 -maxdepth 1 \
        ! -name '.initialized' ! -name 'submissions' \
        -exec chown -R participant:participant {} +
    chown root:root "${WORKSPACE_ROOT}"
    chmod 0755 "${WORKSPACE_ROOT}"
    touch "${WORKSPACE_ROOT}/.initialized"
fi

# Migrate older workspaces that used a participant-writable/sticky root.
chown root:root "${WORKSPACE_ROOT}" "${WORKSPACE_ROOT}/.initialized"
chmod 0755 "${WORKSPACE_ROOT}"

# Add newly released baseline examples to old persistent workspaces, without
# replacing anything the participant has already created.
if [ -d "${SEED_ROOT}/baselines" ] && [ ! -e "${WORKSPACE_ROOT}/baselines" ]; then
    cp -a "${SEED_ROOT}/baselines" "${WORKSPACE_ROOT}/baselines"
    chown -R participant:participant "${WORKSPACE_ROOT}/baselines"
fi

for editable_directory in models results; do
    [ ! -L "${WORKSPACE_ROOT}/${editable_directory}" ] || \
        fail "${editable_directory} directory must not be a symlink"
done
mkdir -p "${WORKSPACE_ROOT}/models" "${WORKSPACE_ROOT}/results"
chown -R participant:participant "${WORKSPACE_ROOT}/models" "${WORKSPACE_ROOT}/results"

# Only the root-run trusted API may create or replace snapshots. Existing
# committed artifacts remain readable to participants for final collection.
if [ -L "${WORKSPACE_ROOT}/submissions" ]; then
    fail "submission directory must not be a symlink"
fi
mkdir -p "${WORKSPACE_ROOT}/submissions"
chown -R --no-dereference root:root "${WORKSPACE_ROOT}/submissions"
chmod 0755 "${WORKSPACE_ROOT}/submissions"
find "${WORKSPACE_ROOT}/submissions" -mindepth 1 -type d -exec chmod 0755 {} +
find "${WORKSPACE_ROOT}/submissions" -type f -exec chmod go-w {} +

safe_link() {
    local source_path="$1"
    local destination_path="$2"

    [ -e "${source_path}" ] || fail "dataset asset is missing: ${source_path}"
    if [ -L "${destination_path}" ]; then
        [ "$(readlink "${destination_path}")" = "${source_path}" ] || \
            fail "refusing to replace mismatched link ${destination_path}"
    elif [ -e "${destination_path}" ]; then
        fail "refusing to replace existing path ${destination_path}"
    else
        ln -s "${source_path}" "${destination_path}"
    fi
    chown -h root:root "${destination_path}"
}

case "${DATA_MODE}" in
    auto)
        if [ "${DATA_ROOT}" = "${WORKSPACE_ROOT}" ]; then
            effective_data_mode=install
        else
            effective_data_mode=verify-ready
        fi
        ;;
    install|verify-only|verify-ready) effective_data_mode="${DATA_MODE}" ;;
    *) fail "STUDENT_DATA_MODE must be auto, install, verify-only, or verify-ready" ;;
esac

if [ "${effective_data_mode}" = "install" ]; then
    # A standalone container owns its dataset volume and performs the
    # authenticated, pinned installation itself.
    DATASET_ROOT="${DATA_ROOT}" "${INSTALLER}"
elif [ "${effective_data_mode}" = "verify-only" ]; then
    # Full operator diagnostic: rehash every JPEG and validate every record.
    DATASET_ROOT="${DATA_ROOT}" "${INSTALLER}" --verify-only
else
    # Each team receives an already-populated private workspace copy.
    # dataset-init did the full scan before publishing its pinned marker, so
    # startup checks the copied marker and critical assets without rescanning
    # every JPEG. Participant image edits made later are intentionally allowed.
    DATASET_ROOT="${DATA_ROOT}" "${INSTALLER}" --verify-ready
fi

if [ "${DATA_ROOT}" != "${WORKSPACE_ROOT}" ]; then
    # Verification/installation touches only DATA_ROOT. These root-owned links
    # expose the canonical /datasets layout from participant tools.
    safe_link "${DATA_ROOT}/imagenet_released" "${WORKSPACE_ROOT}/imagenet_released"
    safe_link "${DATA_ROOT}/m_o" "${WORKSPACE_ROOT}/m_o"
    safe_link "${DATA_ROOT}/splits" "${WORKSPACE_ROOT}/splits"
    safe_link "${DATA_ROOT}/validation_cache" "${WORKSPACE_ROOT}/validation_cache"
    safe_link "${DATA_ROOT}/dataset_manifest.json" "${WORKSPACE_ROOT}/dataset_manifest.json"

    mkdir -p "${LEGACY_DATA_ROOT}"
    safe_link "${DATA_ROOT}/imagenet_released/train" "${LEGACY_DATA_ROOT}/train"
fi

stage_runtime_secret() {
    local staged_name="$1"
    local file_variable="$2"
    local fallback_variable="$3"
    local value="${!fallback_variable:-}"
    local configured_path="${!file_variable:-}"
    local secret_path="${configured_path:-${SOURCE_SECRET_ROOT}/${staged_name}}"
    local staged_path="${STAGED_SECRET_ROOT}/${staged_name}"
    local temporary_path

    if [ -n "${configured_path}" ] || [ -e "${secret_path}" ] || [ -L "${secret_path}" ]; then
        [ -f "${secret_path}" ] && [ ! -L "${secret_path}" ] || \
            fail "${file_variable} must reference a regular, non-symlink file: ${secret_path}"
        [ -r "${secret_path}" ] || fail "${file_variable} is not readable: ${secret_path}"
        local resolved_source resolved_root
        resolved_source="$(realpath -e -- "${secret_path}")"
        resolved_root="$(realpath -e -- "${SOURCE_SECRET_ROOT}")"
        [ "$(dirname -- "${resolved_source}")" = "${resolved_root}" ] || \
            fail "${file_variable} must be located directly under ${SOURCE_SECRET_ROOT}"
        value="$(tr -d '\r\n' < "${secret_path}")"
    fi
    [ -n "${value}" ] || fail "${fallback_variable} must be set or supplied via ${file_variable}"

    temporary_path="$(mktemp "${STAGED_SECRET_ROOT}/.${staged_name}.XXXXXX")"
    printf '%s' "${value}" > "${temporary_path}"
    chown root:root "${temporary_path}"
    chmod 0400 "${temporary_path}"
    mv -f "${temporary_path}" "${staged_path}"
    printf -v "${file_variable}" '%s' "${staged_path}"
    export "${file_variable}"
    unset "${fallback_variable}"
}

# The app natively reads these paths on each request. Accept conventional
# legacy aliases as well, without copying their values into the environment.
if [ -n "${HACKATHON_API_KEY_FILE:-}" ] && [ -z "${STUDENT_API_KEY_FILE:-}" ]; then
    export STUDENT_API_KEY_FILE="${HACKATHON_API_KEY_FILE}"
fi
if [ -n "${TEAM_SUBMIT_PASSWORD_FILE:-}" ] && [ -z "${SUBMIT_PASSWORD_FILE:-}" ]; then
    export SUBMIT_PASSWORD_FILE="${TEAM_SUBMIT_PASSWORD_FILE}"
fi

if [ -L "${SOURCE_SECRET_ROOT}" ]; then
    fail "secret source root must not be a symlink: ${SOURCE_SECRET_ROOT}"
fi
mkdir -p "${SOURCE_SECRET_ROOT}"
[ -d "${SOURCE_SECRET_ROOT}" ] || fail "secret source root is not a directory"
if [ -L "${STAGED_SECRET_ROOT}" ]; then
    fail "staged secret root must not be a symlink: ${STAGED_SECRET_ROOT}"
fi
mkdir -p "${STAGED_SECRET_ROOT}"
chown root:root "${STAGED_SECRET_ROOT}"
chmod 0700 "${STAGED_SECRET_ROOT}"

stage_runtime_secret code_server_password CODE_SERVER_PASSWORD_FILE CODE_SERVER_PASSWORD
stage_runtime_secret student_api_key STUDENT_API_KEY_FILE HACKATHON_API_KEY
stage_runtime_secret submit_password SUBMIT_PASSWORD_FILE TEAM_SUBMIT_PASSWORD
stage_runtime_secret grading_api_token GRADING_API_TOKEN_FILE GRADING_API_TOKEN

# Docker Compose file-backed secrets may appear as 0444 bind mounts. Hide the
# source directory after staging so the participant process cannot read them.
chown root:root "${SOURCE_SECRET_ROOT}"
chmod 0700 "${SOURCE_SECRET_ROOT}"

# Compose's local file-backed secret mode may be ignored because it is a bind
# mount. Verify the effective kernel permission boundary, not just mode text.
for protected_secret_root in "${SOURCE_SECRET_ROOT}" "${STAGED_SECRET_ROOT}"; do
    if /usr/sbin/runuser -u participant -- test -r "${protected_secret_root}"; then
        fail "participant can read protected secret root: ${protected_secret_root}"
    fi
done
for protected_secret in "${STAGED_SECRET_ROOT}"/*; do
    [ -f "${protected_secret}" ] || continue
    if /usr/sbin/runuser -u participant -- test -r "${protected_secret}"; then
        fail "participant can read staged secret: ${protected_secret}"
    fi
done

CODE_SERVER_PASSWORD="$(tr -d '\r\n' < "${CODE_SERVER_PASSWORD_FILE}")"
export CODE_SERVER_PASSWORD

# Generate the tiny deployment-specific public page in a root-owned runtime
# directory. This keeps custom team port mappings out of participant control.
[ -f "${PUBLIC_TEMPLATE_ROOT}/index.html" ] || fail "public index template is missing"
grep -q '__CODE_SERVER_PUBLIC_PORT__' "${PUBLIC_TEMPLATE_ROOT}/index.html" || \
    fail "public index template has no code-server port placeholder"
mkdir -p "${PUBLIC_RUNTIME_ROOT}"
chown root:root "${PUBLIC_RUNTIME_ROOT}"
chmod 0755 "${PUBLIC_RUNTIME_ROOT}"
sed "s/__CODE_SERVER_PUBLIC_PORT__/${CODE_PUBLIC_PORT}/g" \
    "${PUBLIC_TEMPLATE_ROOT}/index.html" > "${PUBLIC_RUNTIME_ROOT}/index.html"
cp "${PUBLIC_TEMPLATE_ROOT}/favicon.png" "${PUBLIC_RUNTIME_ROOT}/favicon.png"
chown root:root "${PUBLIC_RUNTIME_ROOT}/index.html" "${PUBLIC_RUNTIME_ROOT}/favicon.png"
chmod 0644 "${PUBLIC_RUNTIME_ROOT}/index.html" "${PUBLIC_RUNTIME_ROOT}/favicon.png"

# The download credential must never reach FastAPI or code-server. Other
# credential environment fallbacks are cleared specifically for code-server
# by supervisord; production should use the root-readable *_FILE variants.
unset HACKATHON_API_KEY TEAM_SUBMIT_PASSWORD GRADING_API_TOKEN
unset HF_STUDENT_TOKEN HF_TOKEN HUGGING_FACE_HUB_TOKEN

exec "${SUPERVISOR}" -c "${SUPERVISOR_CONFIG}"
