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

## 공식 제출 프로세스 (실측 확인 완료 — 대회 중 규칙이 변경됨)

- **제출 방법: `~/submissions/`(= `/mnt/Team8_db504`) 바로 아래에 가중치와 코드를 평평하게 둔다.**
  주최측이 수거해 대시보드에 반영한다. 하위 폴더로 두면 인식되지 않는다.
  API(`/api/submit`)와 학생 웹페이지는 이 컨테이너에 설정돼 있지 않으므로 쓰지 않는다.
- **제출 한도는 30회**(당초 안내된 10회에서 변경). 채점은 **실시간**이며, 당초의
  "10:00부터 매시간 1회"는 폐기됐다. 배치 후 2~5분이면 리더보드에 반영된다.
- **리더보드는 팀별 최고점만 기록한다.** 낮은 점수를 제출해도 기존 기록이 깎이지
  않으므로, 제출의 유일한 비용은 슬롯이다.
- 리더보드 조회: `https://api.minds.ai.kr/scores` (전체), `https://api.minds.ai.kr/team/<팀명>` (우리 이력).
  웹 UI는 `https://hackathon2026.minds.ai.kr/`.
- **제출은 매번 사용자 승인을 받고 실행한다.** 검증·패키징까지만 하고 실행 여부를 묻는다.

## 실험 도구 (이번 대회에서 만든 것들)

| 도구 | 용도 |
|---|---|
| `tools/fasteval.py` | validation 15,000장을 fp16으로 캐시해 **6초** 채점 (원본 `score_model.py`는 1.5~4분). 실제 grader와 final 기준 2.4e-4 이내 일치 |
| `tools/run_exp.py` | flock으로 GPU 직렬화, 남의 작업 중이면 대기, 결과를 `EXPERIMENTS.md`에 자동 기록, 실행별 로그를 `logs/<이름>.train.log`에 보존 |
| `tools/mkcfg.py` | 점 표기 오버라이드로 sweep config 생성 |
| `tools/greedy_soup.py` | 재료를 하나씩 추가하며 개선될 때만 채택하는 앙상블 탐색 (학습 불필요) |
| `tools/soup.py` / `soup_search.py` | 균등 / 가중치 탐색 앙상블 |
| `tools/robust_select.py` | validation A/B 절반 분할로 후보의 표본 민감도 측정 |
| `tools/sensitivity.py` | **무작위 층화 분할 5회**로 표본 민감도 측정. 고정 A/B 분할 하나는 그 분할에 과적합해 잘못된 결론을 준다 |
| `tools/headfit.py` | **unlearning 끝난 모델의 head만 retain으로 재보정.** CKA는 head 이전에서 측정되므로 비용 0으로 AUS만 회수 (+0.0012~0.0016) |
| `tools/normfit.py` | 채점 지점 앞 LayerNorm affine 1536개 사후 최적화 — **여유 없음이 확인된 음성 결과** |
| `tools/interpolate.py` | M_o와의 가중치 보간 스캔 |
| `tools/package_submission.sh` | 구조 검증 → `~/submissions`에 평평하게 배치 (`--submit`) |

**실험 실행은 반드시 `tools/run_exp.py`를 통한다.** 직접 실행하면 GPU가 충돌하고
결과가 기록되지 않는다. 그리고 여러 큐가 같은 로그 파일에 append하면 줄이 섞여
결과를 잘못 읽는 사고가 두 번 났으므로, 큐마다 로그 파일을 분리한다.

## 작업 원칙

- Retrain-from-scratch 금지 규칙을 어기는 코드(즉, retain class만으로 모델을 새로 초기화해 학습하는 코드)는
  작성하지 말 것 — 대회 실격 사유.
- `ga_example.py`(NegGrad, `student_docker/baselines/`에서)는 **체크포인트 shape 확인용 샘플일 뿐 시작점으로 삼지 말 것** — organizer가 명시적으로 "이거 복사해서 lr/epoch만 튜닝해서 내지 말라"고 안내함. retain 데이터를 아예 안 쓰는 방식이라 Acc_r이 크게 깎여서 좋은 점수가 나오기 어려움. retain set을 같이 활용하는 방법(예: forget에서 ascent + retain에서 동시에 원래 성능 유지하도록 loss 추가, 또는 influence function/scrubbing 계열)을 고려할 것.
- **후보 선별은 raw final이 아니라 `(1−CKA_f) + CKA_r` 합으로 한다.** headfit이 Acc_f를
  사후에 0으로 만들므로 raw final의 AUS 항은 정보가 아니다. ckar4는 raw 0.9886으로
  relfe_seed0(0.9917)보다 낮아 보였지만 합이 1.9876 vs 1.9758이라 실제로는 훨씬 낫고,
  headfit 후 0.99623으로 우리 최고가 됐다. **raw final로 골랐으면 기각했을 실험이다.**
- **제출 후보는 조합 탐색을 하지 마라.** 갭은 모델이 아니라 선택 과정의 성질이다 —
  단일 모델 −0.0006, 게이트만 −0.0002, 로컬 점수로 조합 탐색 −0.0033.
- **Acc_f는 0%에 가까울수록 좋음이 확정됨** (`reference_acc_f = 0.0`이라 gap = acc_f/100). **Acc_r은 95.9% 밑으로 떨어뜨리지 않는 게 목표** — 이 두 값의 트레이드오프가 AUS의 핵심.
- 실험할 때마다 Acc_f, Acc_r, CKA_f, CKA_r, 최종 점수를 `EXPERIMENTS.md`에 기록.
- 시간이 24시간으로 매우 짧으므로, 완벽한 구현보다 **제출 가능한 baseline을 먼저 끝까지 돌리는 것**을 최우선으로.
## 이번 대회에서 실측으로 확인한 것 (다음 세션에서 재유도하지 말 것)

**방법론**
- **NegGrad류(forget만 ascent)는 retain이 붕괴해 0.118에 그친다.** 핵심은 "잊게 하는 것"이
  아니라 "잊게 하면서 나머지를 안 건드리는 것"이다.
- 우리 방법은 **forget 이미지를 매 스텝 무작위 retain 파트너의 teacher feature/label로
  재매핑**한다. CKA가 scale·rotation 불변이라 feature를 줄이거나 회전시켜선 안 떨어지고,
  샘플 간 2차 구조 자체를 바꿔야 하기 때문이다.
- **효과가 확인된 축**: 스텝 수(4800이 최적, 7200은 손해), lr 3e-5, 미니배치 CKA 직접
  최소화, 뒤쪽 6블록만 학습, CKA floor(포화 후 forget 압력 차단), final norm 동결,
  관계 기반 retain 앵커(rel), forget 압력 강화(relf), EMA 0.99, 가중치 앙상블.
- **효과가 없거나 역효과인 축**: 전체 블록 학습, 학습 범위 K=2, retain 앵커에 eval
  transform, 증강 강화(일반화 갭 3.6배 악화), gradient 투영(S04), Fisher 앵커(S08,
  −0.015), retain 복구 학습(Acc_f가 68%까지 되살아남), EMA 0.999(0.99보다 나쁨).

**측정과 선택**
- **학습은 완전히 결정적이다.** 같은 config·seed면 152개 텐서가 비트 단위로 일치한다.
  따라서 같은 seed로 짝지은 단일 변수 비교가 올바른 설계다.
- **seed 간 편차는 크다** (같은 설정에서 final 0.908~0.985). 3~5회에 1번꼴로 retain
  표현이 붕괴하는 실패 모드가 있다. 재료 게이트: `CKA_f < 0.03` **그리고** `CKA_r > 0.96`.
- **로컬 점수는 private를 거의 예측하지 못한다.** 제출 11건에서 로컬 final과 private의
  상관은 +0.375(n=11, 유의하지 않음)였고, 로컬 최고가 private 3위인 일이 반복됐다.
  이전율은 대략 10~20% 수준이다.
- **forget 삭제는 일반화되지만 retain 보존은 그렇지 않다.** 학습에 쓴 이미지와 처음 보는
  이미지 사이에서 CKA_f는 +0.0003, CKA_r은 −0.004~−0.0066 차이가 난다.
- **곡선의 정체**: forget 활성의 90~94%가 retain 부분공간 안에 있다(층별 SVD로 측정).
  두 목적이 같은 공간을 공유하므로 손실 설계로는 우회할 수 없다. 이 측정 덕분에
  GPM(직교 투영)을 3시간 들이기 전에 기각할 수 있었다.

**앙상블**
- **서로 다른 지점의 모델을 섞는 것이 단일 최고 모델보다 항상 낫다.** 최고 조합은
  forget 극단(r019) + 중간(relf) + retain 극단(s06) + 궤적 평균(S03)이었다.
- 개별 성능이 좋아도 `CKA_f`가 높은 재료를 넣으면 soup 전체가 무너진다.
- 가중치 최적화는 균등 평균 대비 이득이 없다(+0.00004, A/B에서 순위 뒤집힘).

## 09-05 오전에 추가로 확정된 것

**앙상블은 포화됐다.** 6재료 크로스 config 소프에 7번째 재료를 넣으면 최선이
+0.00003이다. 같은 config의 시드를 더 뽑는 것은 로컬 점수를 못 움직인다. 시드
평균의 이득은 config마다 다르다 — s06 +0.0033, relf +0.0016, **rel −0.0062**.
rel 계열은 시드끼리 섞으면 무너지므로 모든 소프에서 제외한다.

**M_o 방향 사후 보간은 전 방향 손해다.** 전체·블록별·α 전 범위를 훑었고 모든
지점이 기준보다 낮다. CKA_r은 오르지만 CKA_f가 더 빨리 오르고 Acc_f가 되살아나
AUS까지 무너진다. 조화평균의 두 편미분이 0.496/0.504로 같아서 **CKA_f를 팔아
CKA_r을 사는 트레이드는 원리적으로 이득이 없다** — 합 자체를 올려야 한다.

**탐색을 줄이면 이전이 좋아진다(방향은 확인, 크기는 예측 불가).** 게이트만 걸고
조합 탐색 없이 균등 평균한 `uniform16`은 로컬이 0.0022 낮은데 private이 +0.0005
높았다(갭 −0.0005 vs greedy −0.0032). 다만 탐색량 중간인 `gate987`의 갭이 가장
컸으므로(−0.0041) 단조 규칙은 아니다.

**후보 선택 지표를 새로 만들 때는 반드시 무작위 분할 여러 개로 재라.** 고정
A/B 분할의 `|A−B|`로 갭이 완벽히 설명되는 것처럼 보여 `private ≈ 로컬 − |A−B|
− 0.0005`를 법칙으로 썼는데, 무작위 층화 분할 5회로 재니 세 모델의 민감도가
사실상 같고 순서까지 뒤집혔다. 점 3개에 파라미터 1개를 맞춘 것이었다.
`tools/sensitivity.py`를 쓸 것.

**[폐기됨 — 아래 09-05 오후 참조] private 천장이 ≈0.991이다.** 제출 4건의 private이 0.98986/0.99034/0.99085/
0.98986로 한 점에 모여 있고, 같은 기간 로컬은 0.9913→0.9940으로 움직였다.
1등(0.99670) 수준은 `CKA_r ≥ 0.995`와 `CKA_f ≤ 0.005`를 동시에 요구하는데 우리
최고 CKA_r은 0.9883이다. **선택이 아니라 방법의 한계다.**

**L2-SP** (`train.lambda_l2sp`) — 가중치를 학습 시작점(M_o) 쪽으로 당기는
decoupled 정규화. clip 뒤에 gradient로 더한다. retain의 정답은 모든 이미지에서
M_o이므로 가중치가 M_o 근처에 묶일수록 CKA_r이 유지된다. 세기 보정: 학습 후
`‖p−a‖ = 19.7`, clip된 grad norm = 1.0이라 평형점은 `‖p−a‖ ≈ 1/α`. 유효 범위는
**α = 0.1~1.0**이고 1e-3 같은 값은 아무 효과가 없다.


## 09-05 오후 — 천장은 없었다 (private 0.99085 → 0.99565, 9위 → 3위)

**"private 천장 ≈0.991"은 틀렸다.** 천장처럼 보인 것은 방법의 한계가 아니라
**AUS에 0.0045가 놀고 있었던 것**이다. 지표 구조에서 도출한 결론("AUS는 1을
넘을 수 없으니 차이는 전부 RUS_o에서 난다")을 실측값(AUS = 0.9954)과 대조하지
않고 굴린 것이 원인이었다. **"상한이 있다"와 "상한에 도달했다"는 다른 명제다.**

**headfit — 측정 지점의 분리.** 채점 feature는 classifier **이전**('pre'), Accuracy는
classifier **이후**에서 잰다. 따라서 head를 바꿔도 CKA는 비트 단위로 안 변한다
(3개 모델 실측). unlearning 끝난 모델의 head만 released retain split으로 이어서
학습하면 RUS_o 비용 0으로 AUS만 오른다. `Acc_f → 0.000`은 하이퍼파라미터 18개
조합 전부에서 나왔다 — forget 라벨이 타깃에 없어 그 행 logit이 구조적으로 눌린다.
STRATEGY.md §2.2의 "head만 조작하면 0점"은 head 조작이 방법 *전체*일 때 얘기다.

**headfit이 연 설계 공간이 진짜 이득이다.** Acc_f가 공짜로 0이 되므로 학습 목표에서
`lambda_ce_f`를 지울 수 있고, 그 용량을 CKA_r로 옮길 수 있다.

| | CKA_f | CKA_r | 합 | raw final | headfit 후 |
|---|---|---|---|---|---|
| relfe_seed0 | 0.0085 | 0.9843 | 1.9758 | 0.9917 | 0.99330 |
| noce (`lambda_ce_f` 0) | 0.0236 | 0.9893 | 1.9657 | 0.9840 | — |
| **ckar4** (+ `lambda_cka_r` 4.0) | **0.0046** | **0.9922** | **1.9876** | 0.9886 | **0.99623** |

**두 CKA가 동시에 좋아진 첫 사례** — 프론티어 위 이동이 아니라 프론티어를 민 것이다.

**L2-SP는 종결됐다.** λ=2에서 CKA_r +0.0034를 사고 CKA_f +0.0268을 냈다(교환비
8:1, 합 −0.023). 학습 중 M_o로 당기는 것은 §29의 사후 보간과 같은 손해 트레이드를
만난다. λ→∞면 모델이 M_o가 되어 final=0이므로 곡선 방향이 정해져 있다.
구현 주의: `p.grad`에 더하면 AdamW가 좌표별로 정규화해 폭발한다 —
`opt.step()` 뒤 `p -= lr·λ·(p−a)`.

**남은 격차는 일반화이고 대부분이 forget 쪽이다.** 같은 n=1500에서 학습 forget의
CKA_f는 0.0047, 검증 forget은 0.0117이다(표본 크기 편향 아님을 n=500~4000으로
확인). retain 격차(−0.0035)의 2배다. **지금까지 모든 축이 retain 일반화만
겨냥했다.** → `faug`(forget 브랜치만 증강 강화). §27이 기각한 증강은 retain 앵커
쪽이고 논리가 반대다 — retain은 원본을 붙잡아야 하니 변형이 해롭고, forget은
넓게 지워야 하니 도움이 된다.

**GPU는 이미 포화(99%, 530W, bf16)라 처리량으로 짜낼 여지가 없다.** 실제 손실은
세션 여러 개가 같은 GPU에 각자 큐를 넣고 서로 취소하는 데서 났다(하루에 7개
실험 분량). **큐는 하나로 합쳐 순서를 확정할 것.** `run_exp.py`는 학습이 끝난
뒤에야 로그 파일을 쓰므로(`capture_output`), **진행 중에는 로그 부재가 정상이다 —
프로세스와 GPU로 확인하라.**


## 09-05 오후 2부 — 궤적 채굴로 2위 (private 0.99634)

**최적점은 최종 스텝이 아니었다.** `train.ckpt_every`/`ckpt_from`으로 학습 중
궤적을 저장하니, 한 번의 학습이 후보 1개가 아니라 13개를 낸다. lr이 상수라
**4600스텝에서 끊어도 그 구간 스냅샷은 7200스텝 실행과 동일**하므로 관심 구간만
돌리면 회당 19분에 13뽑기다.

| | CKA_f | CKA_r | 합 |
|---|---|---|---|
| ckar4 최종 | 0.0046 | 0.99224 | 1.9876 |
| ckar8 최종(7200) | 0.0231 | 0.99487 | 1.9718 |
| **같은 궤적 3000~3800** | **0.0032~0.0035** | **0.9952~0.9957** | **1.9917~1.9922** |

"ckar4급 forget + ckar8급 retain"이 한 궤적 안에 있었다. **궤적 내에서 CKA_r은
0.9951~0.9958로 안정적이고 CKA_f는 5배 요동친다.**

### 가장 비싼 교훈 — 고르지 말고 평균하라

같은 스냅샷 13개로 두 후보를 만들어 둘 다 제출한 대조 실험:

| 방식 | 로컬 | private | 갭 |
|---|---|---|---|
| 로컬 최고 1개 선택 | 0.99729 | 0.99494 | −0.00235 |
| **13개 전부 균등 평균** | 0.99723 | **0.99634** | **−0.00089** |

로컬 차이 0.00006에 private 차이 0.0014. CKA_f는 forget 1500장으로 재는
추정량이라 **최대값 선택은 표본 노이즈를 고르는 일**이 된다.

**"평균 금지"는 계열이 다를 때만 유효하다.** ckar4+r019는 CKA_f가 0.0046 →
0.1021로 폭발하지만, 같은 궤적의 인접 스냅샷은 삭제 방향이 정렬돼 있어 평균이
두 지표를 동시에 개선한다(mall_hf의 CKA_r 0.99640은 개별 13개 중 최고보다 높다).

### 갭 표 — 5점으로 확정

| 후보 | 선택 방식 | 로컬 | private | 갭 |
|---|---|---|---|---|
| uniform16_hf | 게이트만 (16재료) | 0.99302 | 0.99282 | −0.0002 |
| ckar4_hf | 단일 모델, 탐색 0 | 0.99623 | 0.99565 | −0.0006 |
| **mall_hf** | **스냅샷 13개 전부 평균** | 0.99723 | **0.99634** | −0.0009 |
| m3800_hf | 스냅샷 13개 중 1개 선택 | 0.99729 | 0.99494 | −0.0024 |
| gate987_hf | 게이트+임계값 (9재료) | 0.99516 | 0.99182 | −0.0033 |

**갭은 로컬 점수가 아니라 선택량이 만든다. 로컬 최고를 제출하지 말고, 규칙으로
집계한 것을 제출한다.**

### 상대 점수 역산에서 얻는 것은 "필요조건"뿐이다

1위(0.9971869)에 대해 AUS=1이어도 RUS ≥ 0.99439가 필요하고 CKA_r ≤ 1이므로
**CKA_f ≤ 0.0112가 수학적 필요조건**이다 — 이건 유효한 정보다. 반면 "우리 두
모델의 지표를 합치면 1위 점수가 재현된다"는 추론은 **레벨셋 위 무한히 많은 조합
중 하나를 우연히 짚은 것**이라 상대의 방법에 대한 증거가 되지 못한다.

### 채점식이 이미 0으로 만든 항을 다시 제어하려 하지 말 것

`lambda_ce_r`은 원래 0이고 `lambda_ce_f`는 headfit 도입 후 0으로 뺐다(그 용량을
`lambda_cka_r`에 넣은 것이 ckar 계열의 출발점이다). 따라서 "정확도가 목표에
도달하면 classification 압력을 끈다"는 종류의 제어기는 **우리 config에서 no-op**다.
그리고 우리 Acc_r(95.73)은 기준선(95.899)보다 낮아 plateau 위가 아니라 벌점
구간 안에 있다.

## 최종 결과 (09-05 17:45 마감 직전)

**private 0.99639 — 전체 5위** (1위 FOV 0.99877, 2위 qwer 0.99793, 3위 Yang LeKang
0.99712, 4위 잘하고싶다 0.99643 / 6위 AGENTS.md 0.99634). 상위 6팀 예선 통과선 안.

오전 10:49 시점 10위 0.99085에서 출발해 **+0.0055** 올렸다.

### 최종 제출물

`~/submissions/final/Team_노광탈_final.tar.gz` — 모델은 **`d25_hf`**.
λ=8 seed0 궤적을 100스텝 간격으로 저장해 step 2200~4600의 스냅샷 25개를 균등
평균하고 headfit(ep8/lr3e-3)을 건 것. 실채점기 0.99721 → private 0.99639.

### 마감 직전 재추첨의 결과

같은 0-임계값 레시피를 스냅샷 간격·구간만 바꿔 다시 뽑은 것들:

| 후보 | 구성 | 로컬 | private |
|---|---|---|---|
| mall_hf | 2200~4600 @200, 13개 | 0.99723 | 0.99634 |
| **d25_hf** | **2200~4600 @100, 25개** | 0.99721 | **0.99639** |
| d36_hf / meta3_hf | 구간 확대 / 세 후보 재평균 | 0.99698 / 0.99723 | 0.99636 |

**로컬이 0.0002 안에서 같은 후보들의 private이 0.99634~0.99639로 흩어진다.**
0-임계값 레시피의 갭 산포가 ±0.0003 정도라는 뜻이고, 마감 직전 남는 슬롯으로
같은 레시피를 재추첨하는 것은 실제로 +0.00005를 벌어줬다(5위와 6위를 가른 차이).

### 실패로 확정된 마지막 시도들

- **λ=4 궤적 평균(c4all)**: 0.99431. 개별 스냅샷은 CKA_f 0.0042~0.0068인데 평균이
  0.0130으로 **나빠졌다**. λ=8과 반대로 λ=4는 궤적 내 삭제 방향이 덜 정렬돼 있다.
  **"같은 궤적이면 평균이 안전하다"는 규칙조차 조건부다.**
- **seed 1 궤적 평균(s1all)**: CKA_f 0.019, 합 1.977. seed 0만 낮은 CKA_f 영역을
  지났다(seed 1·2 모두 꽝). 궤적 자체가 시드 복권이고 당첨률은 1/3 정도다.
