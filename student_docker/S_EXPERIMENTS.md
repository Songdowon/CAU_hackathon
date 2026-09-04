# S 실험 기록

## 이름 규칙 (2026-09-04 요청)

- S01.py에서 시작한 실험 라인은 대문자 S와 두 자리 번호를 사용한다: S01, S02, S03, ...
- 기존 팀원 실험 r001, r002, ...는 이름과 번호를 유지한다.
- 코드: S02.py / 설정: configs/S02.yaml / 모델: models/S02.pt.
- 설정 첫 줄에는 실험 ID와 변경 목적을 적고, script: S02.py와 output.save_path: models/S02.pt를 지정한다.
- 평가 결과와 로그에도 S02를 포함하고, 재실행은 시간이나 실행 번호를 덧붙여 과거 결과를 보존한다.
- S02를 현재 레이어 선택 실험으로 사용하며, 다음 독립 실험 번호는 S03이다. 다른 가설/주요 설정 변경은 새 번호를 사용한다.
- unlearn.py는 복구된 대회 원본 템플릿으로 보존한다. S 실험은 별도 파일에서 진행한다.
- GPU를 공유하므로 실행할 때는 기존 tools/run_exp.py의 GPU 잠금을 사용한다. 실행 예: python tools/run_exp.py configs/S02.yaml.
- 이 문서의 명령은 다음 실행을 위한 안내이며, 이번 정리에서는 학습을 실행하지 않았다.

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
학습 점수는 아직 없으며 측정/학습 완료 후 기록한다.

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
