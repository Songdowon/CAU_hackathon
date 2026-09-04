# 전략 기록 — Machine Unlearning (Team 8)

숫자 로그는 `EXPERIMENTS.md`, **왜 그렇게 했는지**는 이 문서에 남긴다.
본선 발표 자료의 근거가 되도록 의사결정 순서대로 계속 갱신한다.

---

## 1. 문제와 채점식

100개 클래스로 학습된 원본 모델 `M_o`(ViT-B/16, MAE 사전학습)에서 지정된 10개
클래스(snowplow, seat belt, sea slug, cicada, robin, tiger, marmot, lawn mower,
swimming trunks, baseball)의 영향력만 제거하고 나머지 90개 클래스 성능은
유지해야 한다. **처음부터 재학습하는 것은 금지**이며 반드시 `M_o`에서 출발한다.

`grading_docker/score_unlearning.py`를 읽어 채점식을 정확히 확인했다.

```
retain_drop = max(95.89876543209876 - Acc_r, 0)/100
forget_gap  = |Acc_f - 0|/100
AUS   = (1 - retain_drop) / (1 + forget_gap)
RUS_o = harmonic(1 - CKA_f_o, CKA_r_o)
final = harmonic(AUS, RUS_o),   harmonic(a,b) = 0 if a<=0 or b<=0 else 2ab/(a+b)
```

- `CKA_f_o` = unlearning된 모델과 `M_o`의 **forget 클래스 표현** 유사도 → 낮을수록 좋음
- `CKA_r_o` = 같은 것의 **retain 클래스 표현** 유사도 → 높을수록 좋음
- 표현은 `pre` depth, 즉 classifier 직전 pre-logits CLS 토큰 768차원에서만 측정
  (b4/b8/b12도 리포트에 찍히지만 **점수에는 안 들어간다**)
- 채점 대상은 validation 15,000장(클래스당 150장), grader와 동일한 eval transform

## 2. 채점식에서 곧바로 나오는 네 가지 결론

1. **AUS는 1을 넘을 수 없다.** `retain_drop`에 0 하한이 있어 retain 정확도를
   기준선 위로 올려도 보너스가 없다. 즉 **팀 간 차이는 전부 RUS_o에서 난다.**
2. **head만 조작하는 방법은 0점이다.** forget 클래스의 logit을 눌러 Acc_f를 0으로
   만들어도 내부 표현이 그대로면 `CKA_f_o ≈ 1` → `1 - CKA_f_o ≈ 0` →
   조화평균 정의상 `RUS_o = 0` → **final = 0**.
3. **CKA는 isotropic scaling과 orthogonal 변환에 불변이다.** 따라서 forget
   feature를 단순히 크기만 줄이거나 회전시키는 접근은 CKA가 전혀 안 떨어진다.
   forget 샘플들 *사이의* 2차 구조(공분산 구조) 자체를 바꿔야 한다.
   이것이 우리 방법 설계의 출발점이다.
4. **retain 붕괴가 실패의 주 원인이다.** 조화평균이라 어느 한쪽이 낮으면
   전체가 끌려 내려간다. 제공된 NegGrad 예제가 이걸 정확히 보여준다.

## 3. 베이스라인 분석 — NegGrad는 왜 실패하는가

`baselines/ga_example.py`는 forget set에 대해 `loss = -CE`로 gradient ascent만
수행한다. retain 데이터를 아예 보지 않으므로 공유 backbone이 함께 망가진다.

| | Acc_f | Acc_r | CKA_f_o | CKA_r_o | AUS | RUS_o | final |
|---|---|---|---|---|---|---|---|
| NegGrad | 0.13 | **3.97** | 0.206 | **0.127** | 0.081 | 0.219 | **0.118** |

forget은 확실히 지웠지만(Acc_f 0.13) retain 정확도가 95.9% → 4.0%로 무너졌고,
retain 표현도 함께 파괴되어(CKA_r_o 0.127) RUS_o가 0.219에 묶였다.
**"잊게 만드는 것"이 아니라 "잊게 만들면서 나머지를 건드리지 않는 것"이 문제의
본질**이라는 걸 확인한 지점.

## 4. 우리 방법 (r 라인) — forget→retain feature 재매핑

`unlearn_remap.py`. `M_o`를 얼린 teacher로 두고, `M_o`에서 복제한 student를 학습한다.
매 스텝 retain 배치와 forget 배치를 함께 본다.

**retain 쪽 — 표현을 그 자리에 못 박는다**
- `L_feat_r = 1 - cos(z_student, z_teacher)` — `z`는 채점되는 pre-logits 그 자체
- `L_kd_r = KL(student logits ‖ teacher logits)` (T=2)

**forget 쪽 — 표현을 retain 분포로 옮긴다**
- 각 forget 이미지를 같은 스텝의 retain 배치에서 **무작위로 뽑은 파트너**에 짝지어,
  그 파트너의 teacher feature로 끌어당기고(`L_remap_f`), 그 파트너의 label로
  CE를 건다(`L_ce_f`).

**왜 이 형태인가.** 결론 3 때문이다. forget feature를 하나의 고정 anchor로
축소시키면 잔차가 원래 구조를 그대로 유지해 CKA가 안 떨어진다. 반면 매 스텝
서로 다른 retain 샘플로 재매핑하면 forget 샘플 간 유사도 구조가 retain 쪽 구조로
치환되어 원본과 실제로 decorrelate된다. 부수적으로 (a) feature가 학습 분포 안에
머물러 발산하지 않고, (b) 예측이 retain 클래스로 가므로 Acc_f가 자연히 0으로 간다.

**학습 범위.** NegGrad 결과에서 b4/b8의 CKA는 0.93/0.89로 멀쩡했고 `pre`만 0.21로
무너졌다. 손상이 후반부에 집중된다는 뜻이므로, 뒤쪽 K개 블록 + final norm + head만
학습해 앞단을 보호하는 것을 기본값(K=6)으로 두고 K를 주요 탐색 축으로 삼았다.

**선택적 항.** 채점식과 동일한 centered linear CKA를 미니배치에서 계산해 직접
최소화하는 항(`lambda_cka_f`)도 넣어두었다. 지표 정합성은 최고지만 n=128
추정 잡음이 있어 별도 실험으로 검증 중이다.

## 5. 인프라 판단 — 실험 속도가 곧 탐색량

제공된 `score_model.py`는 매번 JPEG 15,000장을 디코딩하느라 1.5~4분이 걸린다.
24시간짜리 대회에서 이건 탐색량을 직접 깎아먹는다. validation 이미지를 grader와
**동일한 transform**으로 한 번만 디코딩해 fp16 캐시(4.5GB)로 저장하고, 이후에는
캐시에서 forward만 돌리는 `tools/fasteval.py`를 만들었다. **6초**로 줄었고,
`ga_example.pt`에서 원본 채점기와 final 기준 2.4e-4 이내 일치를 확인했다
(차이는 fp16 반올림으로 경계선 샘플 몇 장의 예측이 흔들린 것).

GPU 1장을 팀원과 공유하므로 `tools/run_exp.py`가 flock으로 실행을 직렬화하고
결과를 `EXPERIMENTS.md`에 자동 기록한다.

## 6. 진행 로그

- **20:16 — r001 (기준 실험).** final **0.6241**. Acc_r 95.68 / CKA_r 0.973으로
  **retain 보존은 사실상 해결**. 남은 병목은 forget 쪽(Acc_f 17.27, CKA_f 0.670).
  → 다음 축: forget 압력을 스텝 수·손실 가중치·lr·학습 블록 범위·CKA 직접
  최소화의 5가지로 나눠 탐색 (r002~r006).

- **20:19~ — 1차 스윕에서 얻은 핵심 발견: 손실 가중치보다 학습 스텝 수가 훨씬
  강한 레버다.**
  - r003(가중치 강화, 400스텝) = 0.7095
  - r002(가중치 그대로, **1200스텝**) = **0.8074**

  해석: forget 표현을 retain 분포로 옮기는 것은 "세게 미는" 문제가 아니라
  "충분히 오래 미는" 문제에 가깝다. 가중치를 키우면 한 스텝의 이동량은 커지지만
  retain 앵커와의 균형이 깨져 CKA_r_o가 0.975 → 0.931로 내려앉고 Acc_r도 같이
  떨어진다(r003). 반면 스텝을 늘리면 retain 앵커가 매 스텝 함께 작동하므로
  retain을 지킨 채(Acc_r 95.76, CKA_r 0.975) forget만 계속 밀려난다
  (CKA_f 0.670 → 0.474, Acc_f 17.27 → 1.13).
  → 스텝 축을 2400 / 3600까지 확장해 한계를 확인한다 (r007, r008).

  이 시점의 남은 병목은 **CKA_f_o 단 하나**다. AUS는 이미 0.9875로 상한(1.0)에
  근접했고, RUS_o의 두 항 중 CKA_r_o도 0.975로 충분하다.

## 7. 대회 운영 전략

상위 6팀만 본선에 오르므로 **7등과 꼴등의 가치가 같다.** 따라서 기대점수가 아니라
**"6위 이내에 들 확률"을 최대화**하는 쪽으로 의사결정한다. 공식 채점이 9/5 10:00부터
매시간 1회이고 공개 리더보드가 있으므로, 매시간 우리 위치를 보고 리스크 수위를
조절한다: 6위 안에서 여유가 있으면 검증된 모델로 수렴하고, 컷 밖이거나 아슬아슬하면
분산이 큰 공격적 변형을 투입한다.
