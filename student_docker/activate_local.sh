#!/bin/bash
# Source this file before running the cloned workspace directly on a server.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    printf 'Run this with: source %s\n' "$0" >&2
    exit 1
fi

student_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
student_repository="$(cd -- "${student_workspace}/.." && pwd -P)"
export STUDENT_WORKSPACE_ROOT="${student_workspace}"
export DATASET_ROOT="${student_workspace}"
export TRUSTED_SCORER_ROOT="${student_repository}/grading_docker"
cd -- "${student_workspace}"
printf '[student-workspace] activated %s\n' "${student_workspace}"
unset student_workspace student_repository
