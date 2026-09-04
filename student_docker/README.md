# 학생 환경 사용법

## 데이터

`/workspace`에서 다음 경로를 사용합니다.

| 경로 | 용도 |
|---|---|
| `imagenet_released/train` | released 학습 이미지 113,566장 |
| `imagenet_released/validation` | 공개 validation 15,000장 |
| `splits/student_split.pt` | released/validation record와 label |
| `m_o/M_o.pt` | 출발 모델 |
| `validation_cache/M_o__validation.npz` | 로컬 RUS_o 기준 feature |

각 팀의 실제 데이터는 `/workspace/datasets`에 복사되며 위 경로들은 그 복사본을 가리킵니다. 다른 학생과 공유되지 않고 writable입니다.

## 실험과 로컬 점수

실험마다 다른 이름을 사용하세요.

```bash
python unlearn.py --config configs/unlearn.yaml
python validate_submission.py --ckpt models/experiment-001.pt
python score_model.py models/experiment-001.pt
```

`configs/unlearn.yaml`의 `output.save_path`를 다음 실험에서는
`models/experiment-002.pt`처럼 바꿉니다. `score_model.py`는 공개 validation에서 AUS, RUS_o, final score를 출력할 뿐 중앙으로 전송하지 않습니다.

## 공식 제출

학생 웹 페이지에서 `experiment-001.pt`처럼 `models` 바로 아래의 파일명을 입력합니다. API를 직접 쓰려면:

```bash
curl -X POST http://localhost/api/submit \
  -H 'Content-Type: application/json' \
  -d '{"submit_password":"발급값","model_filename":"experiment-001.pt"}'
```

업로드가 중앙에서 완전히 검증되면 최대 10회 중 1회를 사용합니다. 받은 submission UUID로 학생 웹이 자동 polling하며, 직접 조회할 수도 있습니다.

```bash
curl http://localhost/api/submissions/<submission-uuid>
```

중앙 서버는 공개 validation 점수가 아니라 비공개 test의 AUS/RUS_o와 조화평균을 반환합니다.
