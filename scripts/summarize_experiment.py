"""Create an aggregate research report and plots from the frozen experiment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/local_parallel_v1_run2'
r=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
pred=pd.read_json(OUT/'test_predictions.json')
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for name,col in [('parallel_nn','probability'),('logistic_regression','logistic_regression'),('hist_gradient_boosting','hist_gradient_boosting')]:
    m=r['models'][name]['test_2024']
    fpr,tpr,_=roc_curve(pred.target,pred[col],sample_weight=pred.weight)
    precision,recall,_=precision_recall_curve(pred.target,pred[col],sample_weight=pred.weight)
    axs[0].plot(fpr,tpr,label=f'{name} ({m["roc_auc"]:.3f})')
    axs[1].plot(recall,precision,label=f'{name} ({m["pr_auc_ap"]:.3f})')
    rel=m['reliability']
    axs[2].plot([b['predicted'] for b in rel],[b['observed'] for b in rel],marker='o',label=name)
axs[0].plot([0,1],[0,1],'--',color='gray')
axs[1].axhline(r['models']['parallel_nn']['test_2024']['prevalence'],linestyle='--',color='gray')
axs[2].plot([0,1],[0,1],'--',color='gray')
for ax,title,xlabel,ylabel in zip(axs,['2024 ROC (weighted)','2024 Precision-Recall (weighted)','2024 Calibration (weighted)'],['False positive rate','Recall','Mean predicted probability'],['True positive rate','Precision','Observed fraction']):
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=7,loc='best');ax.grid(alpha=.2)
fig.savefig(OUT/'evaluation.png',dpi=170)

lines=['# 청년층 미인지 대사이상 특화 로컬 신경망: 1차 실험',
       '', '## 구현과 증거 범위', '',
       'Jev의 병렬·스키마 제한 출력 아이디어를 적용한 독립적인 tabular MLP다. Jev 원본 모델이나 RLCD 재현물이 아니다.',
       '입력 전처리 → 48/24 은닉층 → 16개 대사이상 조합 logits를 한 번에 계산한다. 결합분포에서 4개 성분 확률, 1개 이상 이상일 확률, 예상 이상 항목 수를 유도한다.',
       '학습은 조사 가중치를 적용한 교차엔트로피이며, 별도 2023년 표본으로 temperature scaling을 적합했다. 별도의 강화학습은 사용하지 않는다.',
       '', '## 대상자와 분할', '',
       '19–39세, 세 질환 의사진단 없음, 관련 약물/인슐린 미사용, 임신개월수 비기록, 12시간 이상 공복, 목표 검사값 완비, 유효 조사설계 정보를 요구했다. 원자료는 보존했다.',
       '미복약만으로 미인지라고 간주하지 않는다. 기진단 미복약자를 포함하는 untreated 대안과 8시간 공복 조건도 CLI로 지원하나 이번 주 실험에는 적용하지 않았다.',
       '진단 경험 없음은 위험요인 수치 자체를 모른다는 직접 측정이 아니다. 본 목표는 대사증후군 진단이나 미래 발병 예측이 아니라 횡단면상 하나 이상의 이상 소견 선별이다.',
       '', '| 구분 | 표본 수 | 양성 수 | 용도 |', '|---|---:|---:|---|']
purpose={'train':'2021–2022 학습','validation':'2021–2022 조기종료','calibration':'2023 확률 보정','threshold':'2023 민감도 90% 기준 컷오프','test_2024':'2024 최종 분리 평가'}
for k,v in r['partitions'].items():lines.append(f'| {k} | {v["n"]} | {v["positive"]} | {purpose[k]} |')
lines+=['', '연도+PSU 단위 그룹 중복을 차단했다. 결측치 대치와 스케일링은 학습 표본에만 적합했다. 2024년으로 구조·학습 종료·보정·컷오프를 선택하지 않았다.',
        '', '## 2024년 평가', '', '| 모델 | AUROC | AP(PR-AUC) | Brier | 민감도 | 특이도 | PPV |', '|---|---:|---:|---:|---:|---:|---:|']
for name,model in r['models'].items():
    m=model['test_2024']
    lines.append(f'| {name} | {m["roc_auc"]:.4f} | {m["pr_auc_ap"]:.4f} | {m["brier"]:.4f} | {m["sensitivity"]:.1%} | {m["specificity"]:.1%} | {m["ppv"]:.1%} |')
m=r['models']['parallel_nn']['test_2024']
lines += ['', f'신경망 컷오프: {m["threshold"]:.6f}. ECE(10 bins): {m["ece_10_bins"]:.4f}. AP의 무정보 기준은 가중 양성률 {m["prevalence"]:.1%}다.',
    '민감도는 높지만 특이도가 낮아 위양성 부담이 크다. 부스팅의 특이도가 더 높고 AUROC 차이도 작다. 신경망의 우월성이나 임상 유용성이 입증됐다고 해석하지 않는다.',
    '', '### 신경망 탐색적 95% 구간', '']
for k,v in r['models']['parallel_nn']['test_ci95_exploratory'].items():lines.append(f'- {k}: {v[0]:.4f}–{v[1]:.4f}')
lines+=['', '표본 내 층별 PSU 재표집 200회 구간이다. 제외 대상자를 포함한 정식 KNHANES domain 분산 추정 구간은 아니다.',
        '', '## 실행 시간 및 출력', '',
        f'CPU warm inference 100회: p50 {r["latency_ms"]["p50"]:.2f}ms, p95 {r["latency_ms"]["p95"]:.2f}ms. 전처리·단일 forward·구조화 출력을 포함하며 모델 로딩, HTTP 전송, 외부 API 시간은 제외한다.',
        'Choice는 선별 양성/음성, Noul은 1개 이상 이상일 확률, Score는 0–4개 예상 이상 항목 수다. Score를 0–100으로 변환해도 질환 발생 확률이 되는 것은 아니다.',
        'Choice confidence는 선택된 클래스에 부여한 확률이다. 인식론적 신뢰도나 판정 정확도 보증이 아니다. 성분별 확률과 count 분포의 별도 보정은 보장하지 않는다.',
        '', '## 남은 연구 한계', '',
        '- KNHANES 검진자가 측정한 신체계측으로 학습했으므로 자가 측정 입력에 대한 검증은 별도로 필요하다.',
        '- 완전 검사 사례·공복 조건으로 선택편향 가능성이 있다. 결측 검사값을 정상으로 대체하지 않았다.',
        '- 특이도 개선은 향후 개발 데이터에서 연구해야 한다. 이미 확인한 2024년 성능을 기준으로 반복 조정하면 독립 평가의 의미가 줄어든다.',
        '- 단일 seed 탐색 실험이다. 임상 승인, 실제 서비스 배포, 진단 성능 인증을 의미하지 않는다.',
        '', '## 근거와 산출물', '',
        '- 공식 원시자료: https://knhanes.kdca.go.kr/knhanes/rawDataDwnld/rawDataDwnld.do',
        '- 이용지침서: data/reference/knhanes/guide8.pdf, guide9.pdf. 진단·복약 코딩, 검진기본조사, 이상지질혈증 지표 및 통합 가중치 절을 확인했다.',
        '- TypeSafe 공개 설명: https://typesafe.ai/blog/introducing-system-one-models-and-jev',
        '- report.json: 분할·성능·학습기록·입력 및 소스 SHA-256',
        '- network.pt, preprocessor.joblib, config.json: 저장 모델',
        '- example_request.json, example_response.json: 실행 예시',
        '- test_predictions.json: 로컬 개별 평가값. 외부 공유하지 않음.', '']
(OUT/'REPORT_KO.md').write_text('\n'.join(lines),encoding='utf-8')
print(OUT/'REPORT_KO.md')
