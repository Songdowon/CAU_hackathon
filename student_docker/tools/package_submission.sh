#!/bin/bash
# 제출 버퍼 폴더 만들기 — 파이썬 파일 + 가중치, 딱 2개.
#
#   bash tools/package_submission.sh          # EXPERIMENTS.md에서 최고점 모델 자동 선택
#   bash tools/package_submission.sh r007     # 특정 실험 지정
set -euo pipefail
cd "$(dirname "$0")/.."

name="${1:-$(awk -F'|' '/^\| r[0-9]/ {gsub(/[ *]/,"",$10); gsub(/ /,"",$2);
              if ($10+0 > s) {s=$10+0; n=$2}} END{print n}' EXPERIMENTS.md)}"
ckpt="models/${name}.pt"
[ -f "$ckpt" ] || { echo "체크포인트 없음: $ckpt" >&2; exit 1; }

dest="submission_ready/${name}"
rm -rf "$dest"; mkdir -p "$dest"

python validate_submission.py --ckpt "$ckpt"   # 구조 검증 실패 시 여기서 중단
cp -- "$ckpt" "$dest/model.pt"

# 코드 1개 파일로. 이 실험의 config를 맨 위에 주석으로 박아 단독 재현 가능하게 둔다.
{
    echo "# 제출: ${name}  (git $(git rev-parse --short HEAD))"
    echo "# 로컬 점수(public validation):"
    python tools/fasteval.py "$ckpt" 2>/dev/null | tail -3 | sed 's/^/#   /'
    echo "# 사용한 config (configs/${name}.yaml):"
    sed 's/^/#   /' "configs/${name}.yaml"
    echo
    cat unlearn_remap.py
} > "$dest/unlearn_remap.py"

ls -la "$dest"

# --submit 을 주면 ~/submissions 바로 아래에 평평하게 놓는다.
# (하위 폴더로 두면 주최측 시스템이 인식하지 못함)
if [ "${2:-}" = "--submit" ] || [ "${1:-}" = "--submit" ]; then
    cp -f "$dest/model.pt" "$dest/unlearn_remap.py" ~/submissions/
    echo; echo "제출 완료 — ~/submissions 내용:"; ls -la ~/submissions/
else
    echo "제출하려면:  bash tools/package_submission.sh ${name} --submit"
fi
