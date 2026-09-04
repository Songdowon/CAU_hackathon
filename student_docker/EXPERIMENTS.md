# 실험 기록

`r###` = 이 라인의 실험, `s###` = 팀원 라인. 각 config 첫 줄 주석에 그 실험이 무엇인지 적는다.
점수는 `tools/fasteval.py`(public validation, grader와 동일 지표) 기준. 제출 후보는 반드시 `score_model.py`로 재확인.

기준선: M_o 무처리 = final 0 (CKA_f_o=1) / NegGrad 베이스라인 `ga_example` = **0.1179**

| 실험 | 설명 | Acc_f | Acc_r | CKA_f | CKA_r | AUS | RUS_o | **final** | 학습(s) | 시각 |
|---|---|---|---|---|---|---|---|---|---|---|
| r001 | forget→retain feature 재매핑 + retain KD (기준 실험) | 17.27 | 95.68 | 0.6701 | 0.9726 | 0.8509 | 0.4927 | **0.6241** | 87 | 09-04 20:16 |
