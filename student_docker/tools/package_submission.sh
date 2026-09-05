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
    # 보간 모델(r017_a0.95 등)은 자체 config가 없으므로 베이스 실험 것을 쓴다.
    base="${name%%_a[0-9]*}"
    [ "$base" != "$name" ] && echo "# 위 가중치는 tools/interpolate.py로 M_o와 ${base}를 섞은 것: ${name#${base}_a} 비율"
    # _hf는 학습 config가 아니라 후처리 단계다. 떼어내야 베이스 config를 찾는다.
    # (안 떼면 config를 못 찾아 아래 soup 분기로 빠져서, 단일 모델을 "가중치 평균"이라고
    #  잘못 설명하는 제출물이 나갔다.)
    if [ "${name%_hf}" != "$name" ]; then
        base="${base%_hf}"
        cat <<'NOTE'
#
# 이 가중치는 2단계로 만들어졌다.
#   1) unlearning 학습 (아래 config + unlearn_remap.py) — M_o에서 출발, 재초기화 없음
#   2) head 보정 (tools/headfit.py, 파일 하단에 첨부) — 1)의 결과에서 classifier
#      head만 released retain split으로 이어서 학습. backbone은 건드리지 않는다.
#
# 2)에 대한 설명: 채점되는 pre-logits feature는 classifier 이전에서 나오므로 이
# 단계는 CKA_f / CKA_r을 비트 단위로 바꾸지 않는다(실측 확인). 즉 망각의 실체는
# 1)에서 이미 끝나 있고, 위에 적힌 CKA_f 값이 그것을 보인다 — 표현이 지워지지
# 않은 채 head만 조정했다면 CKA_f는 1에 가깝고 RUS_o = 0이 되어 최종 점수가
# 0이었을 것이다. head는 재초기화하지 않고 기존 가중치에서 이어 학습하며,
# 출력 차원은 100개 그대로다. 학습에는 released split만 쓰고 validation은
# 로컬 점수 확인에만 쓴다.
NOTE
    fi
    if [ -f "configs/${base}.yaml" ]; then
        echo "# 사용한 config (configs/${base}.yaml):"
        sed 's/^/#   /' "configs/${base}.yaml"
    else
        # soup 모델은 자체 config가 없다. 재료 실험들의 config를 모두 싣는다.
        echo "# 이 가중치는 tools/soup.py로 아래 실험들의 가중치를 평균한 것이다:"
        for ing in ${base#soup_}; do :; done
        for ing in $(echo "${base#soup_}" | tr '_' ' '); do
            [ -f "configs/${ing}.yaml" ] || continue
            echo "# --- configs/${ing}.yaml ---"
            sed 's/^/#   /' "configs/${ing}.yaml"
        done
    fi
    echo
    cat unlearn_remap.py
    # _hf 모델은 head 보정 단계를 거쳤으므로 그 코드도 같이 실어야 재현된다.
    [[ "$name" == *_hf ]] && { echo; echo "# ===== tools/headfit.py (head 보정 단계) ====="; cat tools/headfit.py; }
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
