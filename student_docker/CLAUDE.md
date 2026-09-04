# CLAUDE.md — 2026 오픈소스 SW·AI 딥러닝 해커톤 (Machine Unlearning)

> 이 파일은 `참가자매뉴얼.pptx` 내용을 정리한 프로젝트 컨텍스트입니다.
> Claude Code는 세션 시작 시 이 파일을 자동으로 읽습니다. 매뉴얼을 다시 첨부/설명할 필요 없이,
> 여기 적힌 규칙과 지표를 기준으로 코드를 작성/검토하세요.

## 대회 개요

- 대회명: 2026 오픈소스 SW·AI 딥러닝 해커톤 대회 (중앙대)
- 대회 기간: **09.04(금) 18:30 ~ 09.05(토) 18:00** (약 24시간)
- 예선 결과 발표: 09.07(월), 상위 6팀 선정
- 본선 결과 발표: 09.10(목)
- 시상: 대상 1팀(100만원) / 최우수상 2팀(80만원) / 우수상 3팀(50만원)

## Task: Machine Unlearning

- 배경: 이미 학습을 마친 모델에서, 학습에 동의하지 않은(또는 저작권 문제가 있는) 특정 데이터의 영향력을
  선택적으로 제거해 모델이 그 내용을 "잊게" 만드는 기술.
- 목표: **100개 이미지 클래스 중 10개 클래스(Forget Class)를 모델에서 명시적으로 제거**하고,
  나머지 90개 클래스(Retain Class)에 대한 성능은 유지.
- 모델 구조(슬라이드 13 기준): `Input → ViT Model → Final Output Z → Classifier Head → Class 예측`
  - `Final Output Z` (ViT 임베딩)에서 **CKA** 측정
  - `Classifier Head` 출력에서 **Accuracy** 측정

## 실행 환경 & 레포 구조 (서버에서 직접 확인함)

- 컨테이너: 팀별 독립 컨테이너로 보임 (hostname이 docker container ID 형식), GPU RTX 5090(32GB) 단독 할당, CPU 12코어
- 코드 레포: `~/cau-ai-hackathon-26` (origin: `github.com/leemarshal/cau-ai-hackathon-26`)
- **실제 작업 디렉토리는 레포 루트가 아니라 `~/cau-ai-hackathon-26/student_docker/` 입니다.** 모든 스크립트(`unlearn.py`, `configs/`, `utils/`, `imagenet_vit.py`, `validate_submission.py`, `score_model.py`, `models/`)가 여기 있음. (레포 루트에는 `grading_docker/`, `ops/`, `student_docker/`, README만 있고 실행 파일은 없음)
- 데이터: `student_docker/imagenet_released/`, `student_docker/m_o/`, `student_docker/splits/`, `student_docker/validation_cache/` — 팀 전용 복사본이라 **writable**, 다른 학생과 공유 안 됨 (`student_docker/README.md`에 명시)
- 제출 폴더: `~/submissions` → `/mnt/Team8_db504` (팀8 전용 공유 스토리지) — 단, 공식 제출은 아래 "제출 프로세스" 참고, 이 폴더는 별개일 수 있음(확인 필요)
- 채점 관련 디렉토리 둘 존재: `student_docker/`(학생이 로컬로 돌려보는 `score_model.py`) vs `grading_docker/`(organizer 측 실제 채점 로직으로 보임 — `score_unlearning.py`, `convert_checkpoint.py`, `imagenet_vit.py`). **`grading_docker/score_unlearning.py`를 읽으면 Metric 1(AUS) 최종 결합 수식을 정확히 확인할 수 있을 것으로 보임.**

### 베이스라인 코드 (`student_docker/baselines/ga_example.py`)
- **NegGrad(gradient ascent) 방식**: forget set에 대해 `loss = -cross_entropy`로 **손실을 최대화**시켜 강제로 잊게 만드는 가장 단순한 방법. retain 데이터는 아예 안 보고 모든 파라미터를 업데이트하기 때문에, forget뿐 아니라 다른 클래스 성능도 같이 깎아먹는 **의도적으로 약한 예제**임 (organizer 코멘트: "Do not build your submission by copying this and tuning lr/epochs" — 그대로 튜닝만 해서 제출하지 말 것)
- 목적은 **체크포인트 shape(= `{"model": state_dict}` 형태)를 보여주는 것**뿐, 경쟁력 있는 시작점이 아님. 실제 organizer baseline은 학생 환경에 안 들어있음.
- 실행(반드시 `student_docker/`에서): `python baselines/ga_example.py --config configs/unlearn.yaml`
- `M_o.pt` 로드 → `ViTWrapper(...)` 사용 → `utils.data.get_loaders(...)`가 `forget`/`retain` 로더 제공

### 실험 workflow (`student_docker/README.md` 기준)
```bash
# student_docker/ 안에서
python unlearn.py --config configs/unlearn.yaml
python validate_submission.py --ckpt models/experiment-001.pt   # 구조 체크
python score_model.py models/experiment-001.pt                   # 로컬 채점 (public validation 기준 AUS, RUS_o, final score 출력)
```
- 실험마다 `configs/unlearn.yaml`의 `output.save_path`를 `models/experiment-002.pt`처럼 바꿔서 다른 이름으로 저장할 것
- `score_model.py`는 로컬(public validation)에서만 채점하고 중앙에 전송하지 않음 — **공식 점수가 아니라 참고용**

### `configs/unlearn.yaml` (실험 config — 재현성 담당)
```yaml
seed: 0
model:
  arch: vit_base_patch16_224.mae   # 고정, 변경 금지 (채점 서버와 동일해야 함)
  num_classes: 100                 # 고정, 변경 금지
  mo_ckpt: m_o/M_o.pt
data:
  split: splits/student_split.pt
  forget: es/es_imagenet_mo/forget10.json   # 잊어야 할 10개 클래스 목록 (경로 확인됨)
  batch_size: 128
  workers: 8
train:                              # 자유롭게 튜닝 가능, 필요하면 항목 추가 가능
  epochs: 2
  lr: 1.0e-5
  optimizer: AdamW
  weight_decay: 0.0
output:
  save_path: models/experiment-001.pt   # 실험마다 고유한 이름으로 바꿀 것
```
- **`model.arch`, `model.num_classes`는 절대 변경 금지** — 채점 서버가 같은 아키텍처를 기대함.
- 튜닝은 `train:` 항목 위주로 (epochs, lr, optimizer, weight_decay), 필요시 항목 추가 가능.
- 하이퍼파라미터를 `unlearn.py`에 하드코딩하지 말고 반드시 이 yaml에서 관리 — 재현성 때문에 organizer가 명시적으로 요구.
- forget class 목록은 **`es/es_imagenet_mo/forget10.json`**에 있음 (앞서 `refs.pt`의 `forget_labels`와 같은 10개 클래스를 가리킬 것으로 보이나 형식은 다름 — 둘 다 확인해서 일치하는지 검증할 가치 있음).

### 데이터셋 구조 (`dataset_manifest.json` 기준)
- `schema_version: 2`, `phase: validation`, `score_version: unlearning-v2`
- 이미지: `imagenet_released/train`(released split, 113,566장) + `imagenet_released/validation`(15,000장) — ImageNet-100 계열로 추정
- `m_o/M_o.pt`: **원본 모델(Original AI Model) 체크포인트** — unlearning의 시작점, 여기서부터 작업해야 함 (재학습 금지 규칙과 일치)
- `validation_cache/refs.pt`: **forget class 지정 정보(`forget_labels`, `forget_wnids`)가 여기 들어있음** — forget 10개 클래스는 팀이 정하는 게 아니라 organizer가 고정 지정
- `refs.pt`의 `reference_accuracy: {acc_f, acc_r}` — 원본 모델의 forget/retain accuracy 기준값
- `refs.pt`의 `depths: ["b4","b8","b12","pre"]`, `score_depth: "pre"` — 대회 명칭 기준 CKA(=RUS_o)는 `pre`(분류기 직전 임베딩, 슬라이드의 "Final Output Z")로 최종 채점
- 데이터 설치 스크립트(`install_released_data.sh`)는 이미 설치·검증된 상태로 보이므로 **직접 재실행할 필요 없음**

## 평가 방식 (✅ `grading_docker/score_unlearning.py`에서 실제 코드 확인 완료)

> 대회 매뉴얼(pptx)의 "Metric 1 / Metric 2"는 코드상 각각 **AUS**(Accuracy 기반)와
> **RUS_o**(CKA 기반)로 불림. 아래는 `compute_score()` 함수를 그대로 옮긴 것 — 더 이상 추측 아님.

```python
retain_drop = max(reference_acc_r - acc_r, 0.0) / 100   # acc_r이 기준보다 떨어진 만큼만 (0 하한)
forget_gap  = abs(acc_f - reference_acc_f) / 100          # acc_f가 기준과 얼마나 다른지 (절댓값!)
AUS   = (1 - retain_drop) / (1 + forget_gap)
RUS_o = harmonic(1 - CKA_f_o, CKA_r_o)                     # = 이전에 유도한 조화평균 수식과 동일
final_score = harmonic(AUS, RUS_o)
# harmonic(a, b) = 0 if a<=0 or b<=0 else 2ab/(a+b)
```

- `reference_acc_f`, `reference_acc_r`는 **`refs.pt`의 `reference_accuracy`에서 옴 — 실측 확인 완료: `{'acc_f': 0.0, 'acc_r': 95.89876543209876}`**
  - `reference_acc_f = 0.0`이므로 `forget_gap = |acc_f − 0| / 100 = acc_f / 100` — **결국 Acc_f는 낮을수록(0%에 가까울수록) 좋음**, 처음 직관이 맞았음. 아래 남겨둔 "무조건 0%로 밀어붙이지 말 것" 경고는 해제.
  - `reference_acc_r ≈ 95.9%`가 retain accuracy의 목표 기준선 — Acc_r을 이 값 밑으로 떨어뜨리지 않는 게 목표 (그 이상 올라가도 추가 보너스는 없음, `retain_drop`이 0 하한이라).
- `retain_drop`은 **0 하한** — retain accuracy가 기준보다 좋아져도 AUS에 보너스 없음, 나빠진 만큼만 페널티.
- `linear_cka`는 slide의 CKA 정의와 수학적으로 동일(centered linear CKA), `[0,1]`로 클램프됨.

### 최종 점수
- **최종 평가 = harmonic(AUS, RUS_o)**
- ⚠️ 로컬 `score_model.py`는 **public validation** 기준, 중앙 서버는 **private test set** 기준 — 다를 수 있음.

## 대회 규칙 (반드시 준수)

- 🚫 **Retain class에 대해서만 모델을 처음부터 재학습(Retrained AI Model)하는 방식은 금지.**
  반드시 "이미 학습된 원본 모델"에서 unlearning을 수행해야 함 — from-scratch 재학습 금지.
- 제출물: **① Unlearning된 모델 가중치 ② 코드**

## 🚨 공식 제출 프로세스 — 횟수 제한 있음!

- 공식 제출은 **최대 10회**까지만 가능하고, **중앙 서버가 업로드를 완전히 검증했을 때만 1회 차감**됨
  (즉 형식 오류로 실패하면 안 깎이는 것으로 보이지만, 확실친 않으니 함부로 낭비하지 말 것)
- 방법 1 — 학생 웹 페이지에서 `models/` 바로 아래 파일명(예: `experiment-001.pt`)을 입력
- 방법 2 — API 직접 호출:
  ```bash
  curl -X POST http://localhost/api/submit \
    -H 'Content-Type: application/json' \
    -d '{"submit_password":"발급값","model_filename":"experiment-001.pt"}'
  ```
  결과는 `submission UUID`로 받고 `curl http://localhost/api/submissions/<uuid>`로 상태 조회 (학생 웹이 자동 polling도 함)
- **중앙 서버는 public validation이 아니라 private test set 기준 AUS/RUS_o 조화평균을 반환** — 최종 순위는 이 값 기준.
- ⚠️ **`submit_password`는 이 채팅에 붙여넣지 마세요** — 저는 그걸로 대신 제출할 수도 없고, 채팅에 남기지 않는 게 안전합니다.
- **전략: 로컬 `score_model.py`로 public validation 점수를 충분히 확인하고 확신이 설 때만 공식 제출을 아껴서 쓸 것.** 24시간 안에 10번이면 실험 사이클 대비 넉넉하지 않음.

## 아직 확인 안 된 것

- [x] ~~데이터셋 상세 스펙~~ → `dataset_manifest.json` / `refs.pt`로 확인됨
- [x] ~~베이스라인 코드 내용~~ → `ga_example.py` = 의도적으로 약한 NegGrad baseline
- [x] ~~레포/작업 디렉토리 구조~~ → `student_docker/`가 실제 작업 디렉토리로 확인됨
- [x] ~~제출 프로세스~~ → 학생 웹페이지 또는 API, 최대 10회 (위 섹션 참고)
- [x] ~~`configs/unlearn.yaml` 내용~~ → 확인됨 (위 섹션 참고)
- [x] ~~`grading_docker/score_unlearning.py` 내용~~ → **AUS/RUS_o/final_score 정확한 수식 확인 완료** (위 평가방식 섹션 참고)
- [x] ~~`refs.pt`의 `reference_accuracy` 실제 숫자값~~ → `{'acc_f': 0.0, 'acc_r': 95.89876543209876}` 확인됨
- [ ] `refs.pt`의 `forget_labels`와 `es/es_imagenet_mo/forget10.json`이 같은 10개 클래스를 가리키는지 교차 검증
- [ ] `~/submissions`(`/mnt/Team8_db504`) 폴더가 공식 제출 프로세스와 별개인지, 코드 백업용인지
- [ ] 외부 데이터/사전학습 가중치 사용 가능 여부

## 작업 원칙

- Retrain-from-scratch 금지 규칙을 어기는 코드(즉, retain class만으로 모델을 새로 초기화해 학습하는 코드)는
  작성하지 말 것 — 대회 실격 사유.
- `ga_example.py`(NegGrad, `student_docker/baselines/`에서)는 **체크포인트 shape 확인용 샘플일 뿐 시작점으로 삼지 말 것** — organizer가 명시적으로 "이거 복사해서 lr/epoch만 튜닝해서 내지 말라"고 안내함. retain 데이터를 아예 안 쓰는 방식이라 Acc_r이 크게 깎여서 좋은 점수가 나오기 어려움. retain set을 같이 활용하는 방법(예: forget에서 ascent + retain에서 동시에 원래 성능 유지하도록 loss 추가, 또는 influence function/scrubbing 계열)을 고려할 것.
- **Acc_f는 0%에 가까울수록 좋음이 확정됨** (`reference_acc_f = 0.0`이라 gap = acc_f/100). **Acc_r은 95.9% 밑으로 떨어뜨리지 않는 게 목표** — 이 두 값의 트레이드오프가 AUS의 핵심.
- 실험할 때마다 Acc_f, Acc_r, CKA_f, CKA_r, 최종 점수를 `EXPERIMENTS.md`에 기록.
- 시간이 24시간으로 매우 짧으므로, 완벽한 구현보다 **제출 가능한 baseline을 먼저 끝까지 돌리는 것**을 최우선으로.