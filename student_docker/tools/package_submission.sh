#!/bin/bash
# 제출 패키징: 가중치 + 코드 스냅샷 + 점수 기록을 ~/submissions 아래에 묶는다.
#
#   bash tools/package_submission.sh models/r003.pt
#
# 주최측이 이 폴더를 수거해 대시보드에 반영한다. 매시간 채점이므로 HH:50까지
# 배치 완료를 원칙으로 한다. (폴더/파일 네이밍 규칙은 주최측 확인 필요)
set -euo pipefail

ckpt="${1:?사용법: bash tools/package_submission.sh models/r00X.pt}"
[ -f "$ckpt" ] || { echo "체크포인트 없음: $ckpt" >&2; exit 1; }

name="$(basename "$ckpt" .pt)"
stamp="$(date +%Y%m%d_%H%M)"
dest="$HOME/submissions/${stamp}_${name}"
mkdir -p "$dest"

# 1) 구조 검증 — 여기서 실패하면 제출해도 채점 불가
python validate_submission.py --ckpt "$ckpt"

# 2) 가중치
cp -- "$ckpt" "$dest/model.pt"

# 3) 코드 스냅샷 (데이터/체크포인트 제외, git 추적 파일만)
git -C "$(git rev-parse --show-toplevel)" archive --format=tar.gz \
    -o "$dest/code.tar.gz" HEAD student_docker

# 4) 이 제출이 무엇인지
cfg="configs/${name}.yaml"
{
    echo "# 제출 ${stamp} — ${name}"
    echo
    [ -f "$cfg" ] && { echo '## config'; echo '```yaml'; cat "$cfg"; echo '```'; }
    echo "## 로컬 점수 (public validation)"
    echo '```'
    python tools/fasteval.py "$ckpt" 2>/dev/null | tail -4
    echo '```'
    echo "git commit: $(git rev-parse HEAD)"
} > "$dest/README.md"

echo
echo "패키징 완료: $dest"
ls -la "$dest"
