# 2026 오픈소스 SW·AI 해커톤 — Machine Unlearning 최종 제출

팀: **노광탈**

제출 모델 `model.pt`의 로컬 점수 (public validation 15,000장, 채점 지표는 주최측과 동일):

{{SCORES}}

---

## 1. 개요 — 3단계 파이프라인

```
M_o (m_o/M_o.pt)
  │
  │  ① unlearning 학습 (unlearn_remap.py, seed 0, 4600 step)
  │     학습 도중 100 step마다 가중치 스냅샷 저장 (step 1000부터, 36개)
  ▼
models/{{STEM}}_s2000.pt ... models/{{STEM}}_s4400.pt   (그중 25개 사용)
  │
  │  ② step 2000~4400 구간의 스냅샷 25개를 균등 평균 (tools/average_snapshots.py)
  ▼
models/averaged.pt
  │
  │  ③ classifier head만 보정 (tools/headfit.py, backbone freeze)
  ▼
model.pt   ← 최종 제출물
```

**재학습(from scratch)은 없습니다.** 모든 단계가 제공된 원본 모델 `m_o/M_o.pt`에서
출발하며, ③에서도 head를 재초기화하지 않고 기존 가중치에서 이어 학습합니다.
모델 아키텍처(`vit_base_patch16_224.mae`)와 출력 차원(100)은 변경하지 않았습니다.

---

## 2. 방법

### 2.1 ① unlearning 학습 — forget feature를 retain 분포로 재매핑

채점식이 `RUS_o = harmonic(1 - CKA_f_o, CKA_r_o)`이고 CKA는 등방 스케일과 직교
변환에 불변이므로, forget feature를 단순히 줄이거나 회전시키면 `CKA_f_o`가 1
근처로 남아 최종 점수가 0이 됩니다. **forget 샘플들 사이의 2차 구조 자체를
바꿔야 합니다.**

그래서 매 스텝 forget 이미지마다 retain 배치에서 무작위 파트너를 뽑아 그
파트너의 **teacher(M_o) feature**를 목표로 삼습니다. 파트너가 매 스텝 새로
뽑히므로 구조가 섞이고, feature가 분포 안에 머물러 발산하지 않습니다.

동시에 retain 배치에서는 teacher와의 feature 코사인 + logit KD로 표현을 붙잡고,
미니배치 CKA를 직접 최대화해 `CKA_r_o`를 지킵니다.

손실 항 (config의 `lambda_*`가 각 가중치):

| 항 | 대상 | 내용 |
|---|---|---|
| `lambda_feat_r` 2.0 | retain | teacher feature와의 코사인 유지 |
| `lambda_kd_r` 2.0 | retain | teacher logit KD (T=2.0) |
| `lambda_cka_r` {{LAMBDA}} | retain | 미니배치 linear CKA 최대화 |
| `lambda_remap_f` 1.0 | forget | 무작위 retain 파트너의 teacher feature로 끌어당김 |
| `lambda_cka_f` 3.0 | forget | 미니배치 linear CKA 최소화 |
| `lambda_ce_r` 0.0 / `lambda_ce_f` 0.0 | — | 분류 손실은 사용하지 않음 (③에서 처리) |

추가 설정:
- `cka_floor: 0.05` — `CKA_f`가 충분히 낮아지면 forget 압력을 끕니다(포화 후
  retain을 불필요하게 훼손하지 않기 위해).
- `trainable_blocks: 6` — 뒤쪽 6개 블록 + head만 학습. 손상이 후반부에
  집중되므로 앞단을 얼리면 retain을 싸게 지킬 수 있습니다.
- `freeze_norm: true` — 채점되는 pre-logits feature는 `backbone.norm` 직후
  값입니다. 그 affine 파라미터는 feature를 차원별로 스케일하는데 CKA는 차원별
  스케일에 불변이 아니므로, 전역 왜곡의 통로가 됩니다. 얼려서 막습니다.
- `ema_decay: 0.99` — 저장되는 가중치는 학습 궤적의 EMA입니다.
- `clip_grad: 1.0`, `optimizer: AdamW`, `weight_decay: 0.0`, `lr: 3.0e-05`

### 2.2 ② 같은 궤적 스냅샷의 균등 평균

학습 중 `CKA_r`은 궤적 전체에서 0.9951~0.9958로 안정적인 반면 **`CKA_f`는
0.0032~0.0156으로 5배 요동칩니다.** 마지막 스텝 하나만 쓰면 그 요동의 한 점을
그대로 받게 됩니다.

`CKA_f`는 forget 이미지 1,500장으로 재는 추정량이므로 요동의 상당 부분이 표본
노이즈입니다. 따라서 **로컬 점수가 가장 높은 스냅샷을 고르지 않고 구간 안의 25개를
전부 균등 평균**했습니다(제출 실측: 최고 1개 선택 시 로컬 대비 −0.0024, 전부 평균 시
−0.0009).

**같은 궤적 안에서만 평균합니다.** seed가 다른 궤적을 섞으면 forget 삭제 방향이
서로 달라 상쇄되어 `CKA_f`가 0.0044에서 0.2205로 붕괴하는 것을 실측했습니다.

### 2.3 ③ classifier head 보정

채점되는 feature는 classifier **이전**(pre-logits)이고 Accuracy는 classifier
**이후**에서 측정됩니다. 따라서 **head를 바꿔도 `CKA_f` / `CKA_r`은 변하지
않습니다**(세 모델에서 소수점 5자리까지 동일함을 확인).

②의 결과에서 backbone을 고정하고 head만 released retain split으로 이어서
학습하면 `RUS_o` 비용 없이 AUS만 회복됩니다. forget 클래스 라벨이 타깃에 한 번도
등장하지 않으므로 해당 행의 logit이 구조적으로 눌리고 `Acc_f`가 0으로 갑니다.

**보정은 released split으로만 수행합니다** (validation은 로컬 점수 확인 전용).

---

## 3. 데이터 전처리 및 증강

모든 데이터는 `splits/student_split.pt`의 **`released`** split만 사용합니다.
`validation` split은 학습·선택 어디에도 사용하지 않았고 로컬 점수 확인에만
썼습니다.

- forget 클래스 10개: `es/es_imagenet_mo/forget10.json`의 wnid 목록
- retain: 나머지 90개 클래스

**학습 transform** (retain / forget 브랜치 공통) — `imagenet_vit.build_train_transform`,
timm `create_transform(is_training=True)`:

| 항목 | 값 |
|---|---|
| 입력 크기 | 224 |
| RandAugment | `rand-m9-mstd0.5-inc1` |
| Random Erasing | p=0.25, mode=pixel, count=1 |
| interpolation | bicubic |
| color jitter | 없음 (RandAugment가 대체) |
| 정규화 | 모델의 `data_config`의 mean/std |

**중요**: 이 transform이 내부에서 정규화를 수행하므로 모델은 반드시
`ViTWrapper(..., in_model_norm=False)`로 생성해야 합니다. 그렇지 않으면 이중
정규화됩니다.

**평가 transform** (③의 feature 추출 및 채점) — `train_ft.eval_transform`,
`is_training=False`, `crop_pct=0.9`, bicubic.

---

## 4. 실행 방법 (재현)

`student_docker` 디렉토리에서 실행합니다. 데이터셋(`imagenet_released/`),
원본 모델(`m_o/M_o.pt`)은 제출물에서 제외했으므로 원래 위치에 있어야 합니다.

```bash
# ① unlearning 학습 (약 19분, RTX 5090 기준)
#    스텝 1000부터 100스텝마다 models/{{STEM}}_s1000.pt ... _s4600.pt 36개 저장
python unlearn_remap.py --config configs/final.yaml

# ② 그중 step 2000~4400 구간의 25개만 균등 평균
#    (구간은 결과를 보고 고른 것이 아니라 앞서 확정한 채굴 구간을 그대로 쓴 것)
python tools/average_snapshots.py \
    $(for s in $(seq 2000 100 4400); do echo -n "models/{{STEM}}_s$s.pt "; done) \
    --out models/averaged.pt

# ③ head만 보정 (backbone freeze)
python tools/headfit.py models/averaged.pt --out model.pt --epochs 8 --lr 3e-3

# 확인
python validate_submission.py --ckpt model.pt
python score_model.py model.pt
```

### ③의 인자에 대하여

`tools/headfit.py`의 기본값은 `--epochs 3 --lr 1e-3`이지만 최종 모델은
`--epochs 8 --lr 3e-3`을 사용했습니다. ②의 평균 모델은 보정 전 `Acc_f`가 약 5%로
높아 head를 **수렴까지** 학습해야 `Acc_f`가 0에 도달하기 때문입니다. 기본값으로는
`Acc_f`가 0.6% 수준에서 멈춥니다. backbone을 건드리지 않으므로 이 선택은
`CKA_f`/`CKA_r`에 아무 영향이 없습니다.

---

## 5. Random seed 및 재현성

- `seed: 0` (config). `unlearn_remap.set_seed()`가 `random`, `numpy`,
  `torch`, `torch.cuda`의 seed를 모두 설정하고, DataLoader에도 같은 seed의
  generator를 전달합니다.
- **학습은 완전히 결정적입니다.** 같은 config·seed로 두 번 실행했을 때 152개
  텐서가 비트 단위로 일치하는 것을 확인했습니다.
- `③ headfit`도 `--seed 0`(기본값)으로 배치 순서를 고정합니다.
- 학습률 스케줄이 없으므로(상수 lr) 4600 스텝에서 끊은 궤적의 스냅샷은
  더 긴 실행의 같은 스텝 스냅샷과 동일합니다.

## 6. 파일 구성

```
README.md                      이 문서
model.pt                       최종 제출 모델 (③의 결과)
configs/final.yaml             ①에 사용한 config (원본: configs/{{CFG}})

unlearn_remap.py               ① unlearning 학습
tools/average_snapshots.py     ② 궤적 스냅샷 균등 평균
tools/headfit.py               ③ head 보정
tools/fasteval.py              빠른 로컬 채점기 (참고용, 채점 지표는 주최측과 동일)

imagenet_vit.py                모델 정의 / transform (주최측 제공)
train_ft.py                    Dataset / eval transform (주최측 제공)
utils/data.py                  retain·forget 로더 (주최측 제공)
splits/student_split.pt        split 정의 (주최측 제공)
es/es_imagenet_mo/forget10.json  forget 클래스 목록 (주최측 제공)
```

제출물에서 제외한 것: 데이터셋(`imagenet_released/`), 원본 모델(`m_o/`),
`validation_cache/`, 중간 체크포인트와 캐시(`models/`, `cache/`, `logs/`,
`results/`), 최종 모델 생성에 사용되지 않은 다른 실험 스크립트.
민감 정보(API key/token 등)는 포함되어 있지 않습니다.
