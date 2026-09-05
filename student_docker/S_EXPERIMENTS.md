# S 실험 기록

## 이름 규칙 (2026-09-04 요청)

- S01.py에서 시작한 실험 라인은 대문자 S와 두 자리 번호를 사용한다: S01, S02, S03, ...
- 기존 팀원 실험 r001, r002, ...는 이름과 번호를 유지한다.
- 코드: S02.py / 설정: configs/S02.yaml / 모델: models/S02.pt.
- 설정 첫 줄에는 실험 ID와 변경 목적을 적고, script: S02.py와 output.save_path: models/S02.pt를 지정한다.
- 평가 결과와 로그에도 S02를 포함하고, 재실행은 시간이나 실행 번호를 덧붙여 과거 결과를 보존한다.
- S02는 레이어 선택, S03은 EMA/체크포인트 평균 실험이다. 다음 독립 실험 번호는 S04이다. 다른 가설/주요 설정 변경은 새 번호를 사용한다.
- unlearn.py는 복구된 대회 원본 템플릿으로 보존한다. S 실험은 별도 파일에서 진행한다.
- GPU를 공유하므로 실행할 때는 기존 tools/run_exp.py의 GPU 잠금을 사용한다. 실행 예: python tools/run_exp.py configs/S02.yaml.
- 최초 이름 정리 단계에서는 학습을 실행하지 않았다. 이후 실행과 결과는 각 실험 절에 기록한다.

## S01 — Retain CE + NegGrad

- 코드/설정: S01.py, configs/S01.yaml
- 목적: 유지 클래스의 CE를 줄이면서 삭제 클래스의 CE를 증가시킨다.
- 현재 설정: seed 0, 1 epoch, learning rate 1e-6, AdamW, weight decay 0.05, forget weight 0.1, batch size 128.
- 목적함수: CE_retain - 0.1 * CE_forget. 두 번의 backward 후 한 번 optimizer.step을 수행한다.
- 다음 실행의 저장 경로: models/S01.pt. 기존 체크포인트와 결과파일은 변경하지 않는다.
- 이름 정리: 남아 있던 Experiment_S01 경로를 S01으로 통일하고 script: S01.py를 지정했다. 학습 계산은 변경하지 않았다.

### 기존 S01 평가 기록

출처: results/S01_retain_ce_neggrad-validation-20260904T123015Z.json (tag: S01_retain_ce_neggrad.pt).
기존 파일에 기록된 public validation 결과이며, 여기서 재학습하거나 재채점한 결과가 아니다.

| 지표 | 값 |
|---|---:|
| final_score | 0.009667616 |
| Acc_f | 97.3333% |
| Acc_r | 95.6667% |
| CKA_f_o | 0.997553727 |
| CKA_r_o | 0.988644043 |

해석: 유지 클래스 정확도는 보존됐지만, 삭제 클래스 정확도와 표현 유사도가 높아 이 평가에서는 충분히 잊지 못했다.

## 새 실험 기록 항목

실험 ID, 가설, 직전 실험과의 차이, 코드/설정/모델/로그 경로, 실행 상태, 평가파일, Acc_f/Acc_r/CKA_f_o/CKA_r_o/final_score, 결론을 기록한다.
기존 EXPERIMENTS.md에는 공용 실행기가 결과를 계속 추가하며 S 실험의 해석과 계획은 이 문서에 함께 남긴다.


## S02 — Layer-selective Unlearning (2026-09-04)

### 비교 목적과 기준

레이어 선택만 바꾸고 r014의 seed=0, AdamW, lr=3e-5, 2400 steps, batch size=128 및 손실 가중치를 유지한다. 매 후보는 동일한 M_o에서 새로 시작한다.
기존 r005(전체 모델, 0.6379)와 같은 조건의 r001(뒤 6블록, 0.6241)만으로 전체 모델 학습이 더 나쁘다고 단정할 수 없다. S02-7에서 같은 조건으로 다시 비교한다.
코드는 S02.py이며, 변경 없는 학습 루프를 S02_reference.py에 스냅샷으로 보존했다. 팀원의 원본 unlearn_remap.py와 복구한 unlearn.py는 수정하지 않는다.

| 실험 ID | 학습 대상 | 목적 |
|---|---|---|
| S02 | CE gradient 비율 상위 6블록 + norm/head | 데이터 기반 자동 선택 |
| S02-1 | 마지막 2블록 + norm/head | 얕은 변경 범위 |
| S02-2 | 마지막 4블록 + norm/head | 변경 범위 확대 |
| S02-3 | 마지막 6블록 + norm/head | r014와 동일한 레이어 대조군 |
| S02-4 | 마지막 8블록 + norm/head | 변경 범위 확대 |
| S02-5 | 마지막 2블록 + norm, head 동결 | classifier 영향 분리 |
| S02-6 | classifier만, norm과 backbone 동결 | 표현이 고정되는 음성 대조군 |
| S02-7 | 전체 파라미터 | 전체 학습을 같은 조건에서 비교 |

원안의 마지막 2블록과 classifier+마지막 2블록은 친구 코드의 기본 조건에서는 중복된다. 기본 스윕의 norm/head 조건을 통일하고 S02-5를 head 동결 대조군으로 구분했다.
각 설정은 configs/<ID>.yaml, 모델은 models/<ID>.pt, 레이어 기록은 results/<ID>.selection.json에 저장한다. 기존 산출물이 있으면 새 실행 ID를 요구한다.

### Gradient 측정

- released 학습 데이터에서 forget/retain 각각 8배치 x 최대 32개를 측정한다. validation은 gradient 측정과 학습에 사용하지 않는다.
- 원본 M_o에서 정답 CE(mean)의 gradient를 FP32로 각각 계산한다. CE에 음수를 붙여도 norm은 같다.
- R_l = mean_batch(||g_forget,l||_2) / (mean_batch(||g_retain,l||_2) + 1e-8).
- 동일한 teacher와 student의 유지용 KD/특징 손실은 처음에 거의 0이므로 선택 기준의 분모로 사용하지 않는다.
- 모델 eval 모드에서 측정하고 가중치/학습 여부/모듈 모드/난수 상태를 보존한다. 별도 seed=1702와 독립적인 로더를 사용한다.
- norm/head도 진단에 기록하지만 순위 선정은 transformer block만 대상으로 한다. 평균 norm이 1e-8 이하인 저신호 블록은 선택에서 제외한다.
- 평균, 표준편차, 배치별 norm, 표본 수, 선택된 블록, 학습 파라미터 수, 설정/소스/원본 모델/분할 파일 해시를 저장한다.
- R은 초기 CE 민감도의 탐색 지표다. 큰 R이 실제 선택적 삭제나 더 높은 점수를 보장하지 않는다. 최종 판단은 Acc_f/Acc_r/CKA_f_o/CKA_r_o와 final_score로 한다.
- head-only는 pre-logits 표현을 직접 바꿀 수 없어 CKA 기반 점수의 음성 대조군으로 해석한다.

### 검증 및 실행

CPU 단위 테스트 8개와 실제 M_o를 로드한 ViT 마스크 검증이 통과했다. 비연속 블록의 gradient 전달, 동결 파라미터 보존, gradient 비율 계산, RNG 복원, 중복 기록 보호, r014 마스크 일치를 확인했다.
S02-3과 S02의 학습·평가 결과는 아래 초기 결과 재검토에 기록했다. 다른 대조군의 상태는 실행 로그에서 확인한다.

독립 측정: python S02.py --config configs/S02.yaml --probe-only
학습: python tools/run_exp.py configs/S02-3.yaml configs/S02.yaml
전체 스윕 순서: S02-3 → S02 → S02-1 → S02-2 → S02-4 → S02-5 → S02-6 → S02-7.
모든 GPU 실행은 /tmp/hackathon_gpu.lock을 통해 팀원 실행과 직렬화한다. 로컬 public validation 결과를 기준으로 비교한다.

### 실행 등록 — S02-suite-20260904T132409Z

- 등록 시각: 2026-09-04 22:24 KST. 컨트롤러 PID: 145198.
- 상태 확인 파일: `logs/S02-suite-20260904T132409Z.state.json`
- 전체 로그: `logs/S02-suite-20260904T132409Z.log`
- 실행 스크립트: `logs/S02-suite-20260904T132409Z.py`
- 등록 확인 시 S02 probe가 공유 GPU 잠금을 대기 중이었다. 대기 여부와 완료 목록은 위 상태 파일이 갱신한다.
- 측정 후 S02-3 → S02 → S02-1 → S02-2 → S02-4 → S02-5 → S02-6 → S02-7 순서로 기존 tools.run_exp.run을 호출한다. 실패하면 나머지 실행을 중단하고 원인을 상태 파일과 로그에 남긴다.
- 각 실험의 원시 로컬 평가 결과는 `results/<ID>.validation.json`, 요약은 `EXPERIMENTS.md`에 기록한다. 선택/학습 기록과 평가 완료 여부는 별도 파일로 구분한다.
- 최초 실행 래퍼는 줄바꿈 인코딩 오류로 GPU 작업 전에 종료됐다. 문법 검사 후 수정본을 실행했으며 최초 로그는 보존했다.

### S02 초기 결과 재검토 — 2026-09-05

동일한 tools/fasteval.py의 public validation 수치로 비교한다. 정식 로컬 score_model.py 결과와는 별도로 표기한다.

| 실험 | Acc_f (%) ↓ | Acc_r (%) ↑ | CKA_f_o ↓ | CKA_r_o ↑ | RUS_o ↑ | Final ↑ |
|---|---:|---:|---:|---:|---:|---:|
| r014 / S02-3 | 0.13 | 95.64 | 0.1043 | 0.9679 | 0.9304 | 0.9621 |
| r015 | 0.33 | 95.64 | 0.0239 | 0.9254 | 0.9501 | 0.9716 |
| S02 자동 선택 | 0.07 | 95.54 | 0.1272 | 0.9592 | 0.9139 | 0.9531 |
| r017 | 0.13 | 95.61 | 0.0224 | 0.9675 | 0.9725 | 0.9840 |

- S02-3은 r014와 seed/model/data/train 및 마지막 6블록+norm/head 조건이 같다. 표의 모든 지표가 표시 정밀도에서 일치하므로 기준 실험 재현으로 해석한다.
- r015 역시 마지막 6블록+norm/head를 학습한다. r014 대비 차이는 lambda_feat_r와 lambda_kd_r가 각각 1에서 2로 증가한 것이다. 따라서 “r015 loss + S02-3 layer”는 이미 r015이며 새 결합 실험이 아니다.
- RUS_o는 H(1-CKA_f_o, CKA_r_o)이므로 retain 보존만 나타내지 않는다. S02-3은 r015보다 retain CKA가 높지만 forget CKA도 높다. Acc_f가 더 낮다는 사실만으로 표현 수준의 삭제까지 더 좋다고 결론 내릴 수 없다.
- 실제 선택 효과는 같은 loss의 S02 대 S02-3으로 비교한다. S02는 [4,5,7,8,9,12]번 블록을 선택했다(1-based). Final 차이는 -0.0090338441이며 forget CKA 악화(0.1043→0.1272)와 retain CKA 악화(0.9679→0.9592)가 함께 발생했다.
- 확인된 결론은 seed=0, r014 손실, 초기 CE gradient 비율 상위 6블록 선택이 마지막 6블록보다 열세라는 것이다. 아직 layer-selective 접근 전체를 실패로 판정할 근거는 부족하다.
- r017은 r015 설정에서 steps만 2400→4800으로 늘린다. 낮은 forget CKA를 유지하면서 retain CKA를 회복했다. 다음 실험의 비교 기준은 이미 개선된 r017 계열로 갱신한다.

별도 정식 로컬 score_model.py 기록: r017=0.9839958567, r017_a0.95=0.9856048186. 후자는 0.95*r017 + 0.05*M_o 가중치 보간 모델이다. 이 수치는 public validation이며 공식 private leaderboard 점수가 아니다.

근거: configs/r014.yaml, configs/r015.yaml, configs/r017.yaml, configs/S02-3.yaml, results/S02-3.selection.json, results/S02-3.validation.json, results/S02.selection.json, results/S02.validation.json, EXPERIMENTS.md, results/r017-validation-20260904T135220Z.json, results/r017_a0.95-validation-20260904T142731Z.json.

검토 당시 S02-3과 S02의 학습·평가가 완료되었고, 기존 큐는 S02-1의 GPU 잠금을 기다리고 있었다. 이번 검토에서는 새 중복 실험을 추가하지 않았다. 실시간 상태는 기존 suite state/log를 확인한다.

## S03 — EMA / 학습 경로의 체크포인트 평균 (2026-09-05)

사용자가 지정한 세 번째 S 실험이다. 가설은 학습 후반의 가중치를 평균하면 마지막 한 시점보다 검증 성능이 안정될 수 있다는 것이다. private 성능 개선은 별도로 확인해야 하며 로컬 향상만으로 입증하지 않는다.

### 한 번의 학습에서 비교하는 네 후보

기준은 r017과 동일하다: M_o에서 시작, seed=0, 마지막 6블록+norm/head, AdamW, lr=3e-5, 4800 steps, batch=128, retain feature/KD 가중치 2/2, forget remap/CE/CKA 가중치 1/0.5/2. 기존 모델을 이어 학습하거나 원본 가중치와 보간하지 않는다.

| 후보 | 정의 | 저장 경로 |
|---|---|---|
| S03-last | 4800번째 optimizer 업데이트 직후 가중치 | models/S03-last.pt |
| S03-ema099 | step 2400에서 초기화, 이후 매 step beta=0.99 EMA | models/S03-ema099.pt |
| S03-ema0999 | step 2400에서 초기화, 이후 매 step beta=0.999 EMA | models/S03-ema0999.pt |
| S03 | step 3600/4000/4400/4800 가중치의 동일 가중 평균 | models/S03.pt |

EMA 초기값은 step 2400의 학습된 가중치이며 M_o를 평균에 별도로 넣지 않는다. 관측 횟수는 EMA 각 2401회, 체크포인트 평균 4회다. 평균과 EMA는 학습에 다시 주입하지 않는다. 평균 구간/감쇠율은 평가 전에 고정했으며 r017의 학습률 스케줄을 유지한다. ViT는 LayerNorm을 사용하고 BatchNorm 통계 재추정은 필요하지 않다.

### 구현·검증

- S03.py가 EMA/평균과 저장을 담당한다. S03_reference.py는 기존 검증된 학습 코드의 고정 사본에 설정 전달·업데이트 후 관찰·모델 반환만 추가한다.
- configs/S03.yaml에 모든 평균 설정과 출력 경로를 둔다. 과거 결과가 있으면 실행을 거부한다.
- 원시 체크포인트 4개는 models/S03_checkpoints/step-003600.pt 등의 경로에 보존한다.
- tests/test_s03.py의 7개 CPU 테스트: 손계산 평균/EMA, 초기·워밍업 제외, 원본 모델/RNG/gradient 보존, 스텝 누락·중복 거부, 기존 파일 보호, BatchNorm 거부, 실제 기준 loss loop의 관찰 전후 학습 결과 동일성.
- 실제 M_o ViT에서도 네 출력의 strict load, 가중치 보존, r017 설정 동일성을 확인했다. 전체 학습 종료 시 실제 EMA/평균 횟수와 저장한 last의 모든 텐서가 반환된 최종 모델과 같은지 다시 검사한다.
- PyTorch 2.8 AveragedModel을 사용한다: https://docs.pytorch.org/docs/2.8/generated/torch.optim.swa_utils.AveragedModel.html

### 실행·결과 위치

실행: python tools/S03_run.py --config configs/S03.yaml
기존 tools.run_exp.run을 통해 GPU 잠금을 잡고 한 번 학습한다. 네 후보를 fasteval과 score_model.py로 각각 로컬 public validation 평가한다. 평가도 같은 GPU 잠금을 사용하며 별도 프로세스 종료로 CUDA 메모리를 반환한다.

- 전체 상태: logs/S03.state.json
- 실행 로그: logs/S03.runner.log
- 학습 step/loss 로그: logs/S03.training.log
- 평균 구간/관측 횟수/소스와 데이터 해시: results/S03.trajectory.json
- 네 후보 전체 결과와 같은 학습의 last 대비 차이: results/S03.comparison.json 및 results/S03.comparison.md
- 정식 로컬 채점 원본: results/S03*-validation-<timestamp>.json

공식 제출은 이 실행에 포함되지 않는다. 현재 단계는 구현·검증 완료이며 실행 등록 상태는 아래에 기록한다. 원본 보간은 별도 가설로 남기고 이번 S03에 섞지 않는다.

### S03 실행 등록

- 등록: 2026-09-05 00:40 KST, PID 234291.
- 확인 시 공유 GPU 잠금을 대기 중. logs/S03.runner.log에서 실제 학습 시작을 확인할 수 있다.
- 학습과 네 후보의 로컬 평가가 연결되어 실행된다. 이번 등록 시점에는 S03 학습 점수가 없다.

## S04 — Gradient Surgery (2026-09-05)

- 목적: forget/retain gradient 충돌 제거가 r017의 retain/forget 표현 균형을 개선하는지 검증한다. CKA_r 저하의 원인이 충돌이라는 설명은 아직 가설이며, private 성능 향상도 미검증이다.
- 기준 설정: r017, seed 0, M_o에서 새로 시작, 마지막 6 blocks + norm/head, AdamW lr 3e-5, weight decay 0, 4800 steps, batch 128, clip 1.0.
- retain objective: 2 * feature cosine + 2 * KD (CE weight 0).
- forget objective: 1 * remap cosine + 0.5 * remapped CE + 2 * minibatch CKA.
- 방법: 모든 학습 파라미터를 하나의 벡터로 보아 d = dot(g_r, g_f)를 계산한다. d < 0이고 ||g_r|| >= 1e-8이면 g_f' = g_f - d / ||g_r||^2 * g_r. 최종 gradient는 g_r + g_f'. 이외에는 원래 합을 사용한다. retain gradient 자체는 유지한다.
- S04-control: 동일한 두-gradient 계산 경로에서 투영만 끈 대조군. S04: 투영 활성화. EMA와는 결합하지 않은 별도 실험이다.
- 원 논문의 대칭 PCGrad와 다른 retain 우선 단방향 변형이다. raw gradient 직교성은 AdamW 실제 업데이트의 retain 손실 감소나 평가 CKA_r 상승을 보장하지 않는다.
- 진단: 매 step weighted gradient norm/cosine/dot, 충돌 여부, 투영 크기, loss를 CSV 기록. step 1과 매 100 step은 실제 AdamW 파라미터 변화와 g_r의 내적도 측정한다. 양수이면 해당 retain surrogate의 1차 변화는 증가 방향이다.
- 검증: analytical/zero/tiny/None/nonfinite/SGD/control 동등성 및 원본 loss-loop 재현 검증 통과. 실행기 모의 검증은 smoke → control → surgery → 두 local full 평가의 순서와 기존 결과 보호를 확인한다. 검증 로그: logs/S04.verification.txt.
- 실행: tools/S04_run.py. 먼저 기존 GPU lock 아래에서 CUDA 음수-dot 합성 사례와 실제 ViT 배치 128, 4 steps smoke를 실행한다. 통과 시 tools.run_exp.run으로 control과 surgery를 순차 학습/fast 평가하고, 각각 score_model.py의 전체 local public validation을 실행한다.
- 소스/설정: S04.py, S04_reference.py, configs/S04.yaml, configs/S04-control.yaml. 원본 unlearn.py와 팀원의 r* 파일은 변경하지 않는다.
- 체크포인트: models/S04-control.pt, models/S04.pt. smoke 결과는 models/S04-smoke.pt에 별도 저장한다.
- 상태: logs/S04.state.json, logs/S04.runner.log. 학습 로그와 gradient CSV는 logs/S04*.training.log, logs/S04*.gradients.csv. 완료 시 results/S04.comparison.json 및 results/S04.comparison.md에 같은 대조군 대비 Acc_f, Acc_r, CKA_f, CKA_r, AUS, RUS_o, Final 차이를 기록한다.
- 판단: CKA_r만이 아니라 forget CKA/accuracy와 Final의 동반 변화를 본다. single seed의 local 결과이며, private 개선 여부는 별도로 확인해야 한다.
- 다음 독립 실험 ID: S05.

실행 접수: 2026-09-05 01:23 KST, PID 264374. 확인 시 상태: smoke_or_waiting_for_gpu. 실제 GPU smoke와 학습/평가 완료 여부는 logs/S04.state.json 및 각 run report를 확인한다.

## S05 — Weighted Soup Search (2026-09-05)

사용자가 S05를 weighted soup 탐색으로 지정했다. 이전에 제안된 CKA 분산 축소는 S05로 실행하지 않는다. 새 학습 없이 r016, r015, r012의 가중치만 convex combination으로 섞는다.

- 가중치 순서: **r016 / r015 / r012**, 각 계수 >= 0, 합 = 1.
- 최종 목적: **원본 score_model.py AUS >= 0.995를 만족하는 확인 완료 후보 중 RUS_o 최대화**. 같은 RUS_o이면 Final로 동점을 정리한다. 전체 simplex의 전역 최적해를 보장하는 검색은 아니다.
- 기존 확인된 soup 보고서는 r017+r016+r015+r012 (Final 0.9884468063), S03+r016+r015+r012 (Final 0.9886242889)의 네 모델 조합이다. 요청한 세 모델의 균등 평균은 이번 S05에서 새로 평가한다. 네 모델 결과는 별도 역사적 비교로만 표시한다.

| 초기 후보 | r016 | r015 | r012 |
|---|---:|---:|---:|
| C000 (균등 기준) | 1/3 | 1/3 | 1/3 |
| C001 | 0.50 | 0.30 | 0.20 |
| C002 | 0.50 | 0.20 | 0.30 |
| C003 | 0.40 | 0.40 | 0.20 |
| C004 | 0.40 | 0.20 | 0.40 |
| C005 | 0.60 | 0.20 | 0.20 |
| C006 (r016 단독) | 1 | 0 | 0 |
| C007 (r015 단독) | 0 | 1 | 0 |
| C008 (r012 단독) | 0 | 0 | 1 |

그다음 빠른 평가의 상위 2점에서 계수 0.05를 한 모델에서 다른 모델로 옮기는 6방향을 탐색하고, 갱신된 최고점에서 0.025로 6방향을 탐색한다. 중복/음수 계수는 제거하며 최대 27개다. 모든 fast 평가는 전체 public validation을 사용한다.

빠른 채점의 fp16 픽셀 캐시와 원본 이미지 디코딩 결과는 다를 수 있다. 탐색 단계는 AUS >= 0.994의 여유 범위에서 RUS_o 순위를 사용한다. 아무도 이 범위에 들지 않으면 AUS 순위로 다음 탐색 중심을 고른다. 0.001 여유는 휴리스틱이며 채점 오차의 보장된 상한이 아니다.

원본 채점 대상은 fast AUS >= 0.995 상위 3개, 여유 범위 상위 3개, AUS 최고, 균등 기준의 합집합(최대 8개)이다. 최종 선택에서는 여유를 적용하지 않고 원본 AUS >= 0.995를 정확히 검사한다. 조건을 만족하는 확인 완료 후보가 없으면 no feasible candidate로 끝내고 models/S05.pt를 생성하지 않는다. 만족하면 선택된 후보 파일을 그대로 복사하고 SHA-256 일치를 검사한다.

- 구현: S05.py, tools/S05_run.py, configs/S05.yaml.
- 검증: tests/test_s05.py + tests/test_s05_runner.py의 11개 테스트 통과. 계수/형태/dtype/finite 검사, 입력 불변, refinement 중복 방지, 제약 우선 선택, 원본 채점 후 후보 변경, 조건 미달 시 미선택, parent의 torch 미로드를 확인했다.
- 실제 checkpoint CPU 검증: 152개 텐서 strict ViT load, 세 endpoint 모두 원본과 모든 텐서 일치, 공통 76개 텐서 정확 보존, 저장/재로드 일치, 입력 checkpoint SHA-256 보존. logs/S05.cpu-validation.json, logs/S05.verification.txt.
- 모든 평균은 CPU float64 누적 후 원래 dtype으로 저장한다. 일치하는 텐서는 그대로 복제하고 비실수/정수 버퍼는 세 원본의 값이 같아야 한다.
- 실행: python tools/S05_run.py --config configs/S05.yaml. 기존 tools.run_exp.LOCK 및 wait_for_gpu를 사용하고, 후보/원본 평가는 각각 별도 자식 프로세스로 실행해 GPU를 반환한다.
- 상태: logs/S05.state.json, logs/S05.runner.log. 개별 평가 로그: logs/S05/.
- 출처: results/S05.manifest.json (원본 체크포인트/소스/참조 데이터 해시 및 픽셀 캐시 fingerprint).
- 후보: models/S05/S05-Cxxx.pt, results/S05/S05-Cxxx.fast.json. 최종 조건 충족 시 models/S05.pt.
- 비교표/전체 점수: results/S05.comparison.md 및 results/S05.comparison.json. 모든 값은 local public validation이며 private 개선은 미검증이다.
- 다음 독립 실험 ID: S06.

S05 실행 등록: 2026-09-04T17:18:43.594944+00:00 (UTC), PID 305434. 접수 확인 상태: fast_or_waiting_for_gpu. 실시간 상태는 logs/S05.state.json.

### S03 실행 완료

평가 완료: results/S03.comparison.md. 같은 학습의 last 대비 차이를 모든 후보에 기록했다. private 개선 여부는 미검증이다.

## S06 - CKA Target-Floor (2026-09-05)

- Baseline: r017, with the same seed, optimizer, 4,800 steps, trainable last 6 blocks, and all retain/forget loss weights.
- Single experimental change: `train.cka_floor: 0.02`.
- Effective forget CKA loss: `max(0, minibatch_CKA_f - 0.02)`.
- Goal: stop the saturated CKA_f term from consuming retain representation after it reaches the target region.
- Interpretation limit: the gate uses the 128-sample training minibatch statistic; it is not identical to the 1,500-sample validation CKA_f.
- Artifacts: `S06.py`, `configs/S06.yaml`, `tests/test_s06.py`.
- Verification: 6 focused tests passed before launch.
- Status: queued through `tools/run_exp.py` under the shared GPU lock.

## S08 — Fisher-weighted retain anchoring

- Baseline: matched s06_seed0 objective and seed; same 4,800 steps, last 6 blocks, CKA floor 0.05, norm freeze, and all retain/forget loss weights.
- Single change: estimate a diagonal empirical Fisher from 4,096 retain samples at M_o, normalize it per transformer block, then add 0.001 * sum(F_i * (theta_i - theta_i0)^2) on the last 6 blocks.
- Difference from S07/rel_seed: no direct batch CKA_r loss; the anchor is a fixed parameter-space constraint, so S07's batch-level retain/forget gradient competition is absent.
- Primary check: compare seed 0 against s06_seed0 and rel_seed0; only continue to seeds 2/4 if AUS and forget geometry stay within guardrails.
- Guardrails: Acc_f <= 0.5%, AUS >= matched S06 - 0.001; rank by worst-seed RUS_o/final.
- Artifacts: S08.py, configs/S08.yaml, tests/test_s08.py.
- Verification: 6 focused S08 tests and 6 S06 regression tests passed before launch.
- Status: ready; queue after the active rel_seed4 run.


## S09 — Mixed-loss Cosine Tail (2026-09-05)

- 목적: r019가 7200 steps 동안 고정 lr로 이동하면서 후반의 좋은 지점을 지나치는지 검증한다. retain-only recovery와 달리 마지막까지 동일한 retain+forget 혼합 loss를 유지하고 lr만 줄인다.
- 기준: r019와 동일한 M_o, seed=0, 마지막 6 blocks+norm/head, AdamW, 7200 steps, base lr=3e-5, weight decay=0, clip=1.0, loss 가중치 전부 동일.
- S09-1: updates 0~4800은 3e-5, 이후 cosine decay하여 마지막 update에서 3e-6.
- S09-2: updates 0~4800은 3e-5, 이후 cosine decay하여 마지막 update에서 0.
- hard guardrail: CKA_f_o <= 0.03을 통과한 후보만 유효.
- primary: final_score. secondary: CKA_r_o와 RUS_o.
- stability: 동일 seed의 r019보다 개선되는지 비교한다. seed=0 후보에서 guardrail과 r019 개선을 확인하기 전에는 추가 seed로 확장하지 않는다.
- 코드/설정/출력: S09.py, configs/S09-1.yaml, configs/S09-2.yaml, models/S09-1.pt, models/S09-2.pt.
- 구현: zero-based update 0~4800까지 base lr를 유지하고 4801부터 cosine tail을 적용하며 update 7199에서 지정한 final lr에 정확히 도달한다. step 로그에 실제 lr를 함께 남긴다.
- 검증: tests/test_s09.py unittest 4개 통과, S09.py py_compile 통과. 두 config가 script/schedule/output을 제외하면 r019와 동일함을 확인했다.
- 실행 등록: tools/run_exp.py의 기존 /tmp/hackathon_gpu.lock 큐에 S09-1 -> S09-2 순서로 등록. PID 737371, 로그 logs/S09.runner.log. 등록 확인 상태는 GPU lock 대기 중이다.
- 완료 결과는 EXPERIMENTS.md의 완료 표와 각 validation 결과에만 추가한다.


## S10 — Proximal L2-SP (2026-09-05)

- 진단: 이전 구현은 clip 뒤 gradient에 lambda*(theta-anchor)를 직접 더해 lambda 스케일이 task gradient와 섞였다. 현재 구현은 optimizer step 뒤 theta <- theta - lr*lambda*(theta-anchor)를 적용한다.
- Fisher S08과 달리 데이터로 추정한 중요도 가중치를 사용하지 않으며, 모든 학습 파라미터를 M_o의 대응 파라미터로 직접 수축한다.
- 기준 설정: relfe_seed0 계열(seed=0, 7200 steps, cka_f=3, cka_r=2, floor=0.05, norm 동결, EMA=0.99). proximal lambda만 변경한다.
- S10-1: lambda=2. lr*lambda*7200=0.432.
- S10-2: lambda=5. lr*lambda*7200=1.08.
- S10-3: lambda=15. lr*lambda*7200=3.24.
- 목표: CKA_r을 현재 최고 0.9883보다 높이면서 CKA_f를 낮게 유지해 기존 trade-off 곡선을 벗어나는지 확인한다. 최종 목표 구간은 CKA_r >= 0.995, CKA_f <= 0.005다.
- 코드/설정/출력: unlearn_remap.py, configs/S10-1.yaml, configs/S10-2.yaml, configs/S10-3.yaml, models/S10-1.pt, models/S10-2.pt, models/S10-3.pt.
- 검증: proximal helper 단위 테스트 4개 통과. 부호, lr 스케일, anchor overshoot 방지, params/anchors 길이 검증을 확인했고 unlearn_remap.py py_compile을 통과했다.
- 구버전 queue PID 729219는 l2p020 실행 중 중단했고 l2p080/l2p040/l2p005/l2p300은 실행하지 않았다. 비교 불가능한 구버전 결과를 완료 표에 기록하지 않는다.
- 새 실행 등록: S10-1 -> S10-2 -> S10-3 순서, PID 749378, 로그 logs/S10.runner.log. 등록 확인 상태는 기존 GPU lock 대기 중이다.

- 우선순위 변경(2026-09-05): CKA_r을 직접 겨냥하는 proximal L2-SP를 먼저 검증하기 위해 S09 대기 PID 737371을 GPU 실행 전에 취소했다. S09 코드/설정/테스트는 재현 기록으로 보존하며 결과가 있는 완료 실험으로 취급하지 않는다.


## S11 — State-triggered Cosine Tail (2026-09-05, queue 대기)

- 냉정 평가: 기대값은 **중상**. forget term이 이미 자주 OFF인데 후반 CKA_r 변동이 큰 로그와, 추가 backward/loss 없이 step size만 줄인다는 비용 구조가 근거다. 단, minibatch CKA_f는 validation CKA_f의 noisy proxy라 조기 trigger 위험이 있어 guard를 넣었다.
- matched control: configs/noce.yaml (seed 0, 7200 steps, lr 3e-5, lambda_ce_f=0.0). loss와 optimizer state는 그대로 두고 schedule만 바꾼다.
- trigger: step >= 3600 이후 최근 200 step의 forget OFF 비율 >= 0.40 **또는** raw CKA_f EMA(beta=0.98) <= cka_floor(0.05). 한 번만 trigger하며 마지막 update에서는 trigger하지 않는다.
- tail: trigger 시점의 lr 3e-5에서 step 7199의 3e-6까지 cosine decay. trigger가 없으면 3e-5를 유지한다. Adam moments는 유지한다.
- 평가 우선순위: hard guardrail CKA_f <= 0.03, primary final, secondary CKA_r/RUS, stability는 동일 seed noce 대비. 최고점 목표선은 CKA_r >= 0.995와 CKA_f <= 0.005 동시 달성 여부도 본다.
- 코드/설정/출력: S11.py, configs/S11.yaml, models/S11.pt.
- 검증: controller 단위 테스트 5개 통과(최소 step, OFF ratio trigger, tail 종점 lr, 미trigger 상수 lr, 마지막-step 0분모 방지), py_compile S11.py 통과. noce 대비 차이는 script/schedule/output과 설명 주석뿐이다.
- queue: PID 771558, logs/S11.runner.log, 현재 GPU lock 대기 중. 기존 팀 queue(PID 733772)와 S10 proximal queue(PID 749378)는 유지했다.
- 중복성: ckar4는 retain batch CKA loss, S10은 weight-space proximal update, S11은 optimizer step-size schedule이므로 정확히 겹치지 않는다.


## S13 - Anchor-Preserving Consensus Merge (2026-09-05)

- Purpose: create a submission checkpoint using CPU-only weight-space merging while preventing the cross-seed forget-direction cancellation observed in naive averaging.
- Base/anchor/helpers: m_o/M_o.pt / models/mall_hf.pt / models/s1all.pt + models/s2all.pt.
- Rule: retain the anchor delta sign; use median magnitude only from helper deltas with matching signs; restore each tensor's anchor delta norm. Opposing helper deltas never cancel the anchor. The classifier head is copied exactly from mall_hf.
- Numerical guard: max-absolute anchor delta <= 1e-5 is treated as EMA rounding drift and restored exactly to M_o. Evidence showed a clean gap from 6.68e-6 to 1.18e-3; 78 tensors were restored.
- Artifacts: S13.py, configs/S13.yaml, models/S13.pt, results/S13.merge-audit.json, tests/test_s13.py.
- CPU verification: 152 tensors; 2 head tensors exactly preserved; 78 tensors exactly restored to M_o; 72 consensus tensors; all values finite and on CPU; saved delta-norm maximum relative error 5.23e-6; exact reload and SHA-256 check passed.
- Status: checkpoint generated. No GPU evaluation was run, and S13 was not registered in the shared GPU queue. Local/private score is unverified; require CKA_f_o <= 0.03 before submission.
