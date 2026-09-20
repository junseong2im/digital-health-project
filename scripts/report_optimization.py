"""Consolidate the optimization evidence without selecting on the reused holdout."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metabolic.comparison import dump,ROOT

OUT=ROOT/'artifacts/optimized_v2'
r=json.loads((OUT/'report.json').read_text());b=json.loads((OUT/'latency.json').read_text());aux=json.loads((OUT/'pattern_report.json').read_text())
old=json.loads((ROOT/'artifacts/local_parallel_v1_run2/report.json').read_text())
labels={'frozen_v1':'기존 신경망','refined_nn':'새 신경망 후보','tuned_lr':'동일 조건 튜닝 LR','matched_basis_lr':'동일 기저 LR(C=1)'}
L=['# 병목 개선 및 신경망 재실험 결과','',
'## 결론','',
'추론 병목을 제거해 기존 모델은 판정 변화 없이 크게 빨라졌다. 새 신경망 후보는 기존보다 AUC가 소폭 높았으나 AP는 소폭 낮았고, 같은 조건으로 전처리·튜닝·고속화한 LR을 AUC와 속도 모두에서 넘지는 못했다. 목표 달성을 주장하지 않는다.',
'기존 기본 모델의 가중치와 판정 정책은 유지하고 `/v1/decide` 및 기존 CLI에 고속 런타임을 적용했다. `/v2/decide`는 신규 연구 후보를 명시적으로 사용하는 별도 경로다.',
'','## 1. 병목 측정','', '| 단계 | 중앙값(ms) | p95(ms) |', '|---|---:|---:|']
for name,item in b['legacy_stage_profile'].items():L.append(f'| {name} | {item["p50_ms"]:.4f} | {item["p95_ms"]:.4f} |')
L+=['','매 요청마다 DataFrame을 만들고 파생변수 계산·범주 처리·sklearn ColumnTransformer를 실행하는 비용이 모델 계산보다 컸다.',
'학습된 대치값·평균·표준편차·원핫 범주를 배열 연산으로 옮겼다. 배치 1개는 배열 할당과 범주 검색을 최소화한 경로를 사용한다. 기존 MLP의 LayerNorm·SiLU·선형 연산도 NumPy로 동일하게 계산한다.',
f'원래 적격자 {b["v1_equivalence"]["n"]:,}명에서 확률 최대 차이 {b["v1_equivalence"]["max_probability_difference"]:.3g}, 저장 컷오프 기준 판정 변화 {b["v1_equivalence"]["binary_changes_at_saved_threshold"]}건. 부동소수점 허용오차 내의 동등성으로, 임의의 모든 입력에서 비트 단위 일치를 의미하지 않는다.',
'','## 2. 공정한 처리시간 비교','',
'CPU 1스레드, 준비 실행 후 모델 순서를 무작위로 바꿔 1,000회 측정했다. 동일한 가상 입력, 입력 검증·전처리·확률 보정·이진 JSON 출력을 포함한다. 디스크 로딩·서버 시작·HTTP·GPU 실행은 제외한다.',
'LR에도 동일한 배열 기반 최적화를 적용했다. 튜닝 LR의 계수는 메모리에 미리 올려 추론 중 파일 읽기가 발생하지 않도록 했다.',
'','| 구현 | 중앙값(ms) | p95(ms) |', '|---|---:|---:|']
latnames={'v1_legacy_nn':'기존 신경망/기존 경로','v1_fast_nn':'기존 신경망/고속 경로','v2_refined_nn':'새 신경망/고속 경로',
'lr_legacy':'기존 LR/기존 경로','lr_fast':'기존 LR/고속 경로','lr_tuned_fast':'튜닝 LR/고속 경로'}
for name,item in b['common'].items():L.append(f'| {latnames[name]} | {item["p50_ms"]:.4f} | {item["p95_ms"]:.4f} |')
speed=b['common']['v1_legacy_nn']['p50_ms']/b['common']['v1_fast_nn']['p50_ms']
L += ['',f'동일한 기존 신경망 자체의 고속화는 약 {speed:.1f}배다. 새 신경망은 더 빠르지만, 고속화한 LR보다 빠르지는 않았다. 실행 순서·운영체제 부하에 따라 작은 차이는 변동할 수 있다.',
'', '| 전체 Choice/Score/Noul 출력 | 중앙값(ms) | p95(ms) |', '|---|---:|---:|']
for name,item in b['full_schema'].items():L.append(f'| {latnames[name]} | {item["p50_ms"]:.4f} | {item["p95_ms"]:.4f} |')
L+=['','## 3. 데이터 점검과 전처리','',
'원자료는 변경하지 않았다. 기존 4,068명 코호트를 유지했다. 중복 ID/연도 없음, 임신 코드 0 또는 남성 비해당 8만 존재함, 요청 범주에 허용되지 않은 값 없음, 수치 예측변수는 입력 범위 안에 있음을 확인했다. 공복 12시간·진단 및 약물 제외·목표 검사 완비 조건도 유지했다.',
'목표 검사 결측을 정상으로 간주하거나 채혈 결과를 입력으로 사용하지 않는다. 혈당·TG 등 실제 높은 검사값을 임의로 제거하지 않는다.',
'새 전처리는 학습 자료의 중앙값 대치, 학습 자료의 평균/표준편차, 모든 수치 변수의 결측 지시자, 범주형 결측 전용 수준을 사용한다. hinge 후보는 학습 자료의 25/50/75분위 매듭과 사전 정의한 성별×수치 기울기를 포함한다. 매듭·대치·스케일은 각 fold 학습 부분에만 적합했다.',
'','## 4. 후보 선택과 학습','',
'2021–2022의 2,052명만 사용해 3-fold StratifiedGroupKFold를 구성했다. 연도+PSU가 학습/검증에 겹치지 않으며, NN의 조기종료는 각 fold 내부에서 별도로 수행했다. 선택한 epoch로 outer train 전체에 재적합한 뒤 outer validation을 평가했다.',
'NN: standard/hinge 기저 × 은닉 8/16/32의 6개 후보. LR: 두 기저 × C=0.1/1/10의 6개 후보. 평균 fold 가중 AUC로 선택하고 Brier를 동률 기준으로 사용했다. 기존 NN만 1,680명으로 학습했고 새 NN과 비교 LR은 같은 2,052명으로 재학습했으므로 기존 대비 향상에는 학습 표본 증가도 영향을 줄 수 있다.',
'','| 후보 종류 | 선택 설정 | 개발 CV 평균 AUC |', '|---|---|---:|']
for name,item in r['selection'].items():L.append(f'| {name} | {json.dumps(item["config"],ensure_ascii=False)} | {item["mean_auc"]:.4f} |')
L += ['', 'NN은 선형 skip + tanh 잔차 구조로, primary risk 로짓 하나와 양성 조건부 15개 조합을 한 번에 출력한다. P(정상)=1−r, P(양성 조합 j)=r×q_j이므로 Choice·Noul·Score와 성분 확률이 같은 결합분포에서 나온다.',
f'최종 primary risk 학습 epoch는 내부 조기종료 epoch의 중앙값인 {r["nn_config"]["epochs"]}였다. 조기종료가 매우 빨라 개발 데이터에서 추가 비선형 학습의 이득이 크지 않았음을 함께 보고한다.',
'primary risk에 대한 조기종료 때문에 조건부 head가 거의 학습되지 않는 문제를 방지하기 위해, primary 분기와 hidden 표현을 고정하고 양성 사례에서 조건부 multinomial head를 별도로 적합했다. 이 단계는 binary risk·그 보정값·컷오프를 변경하지 않았다.',
'확률보정은 기존 2023년 calibration 집합, 임계치는 별도의 2023년 threshold 집합에서 결정했다. 2024년으로 구조·시드·컷오프를 선택하지 않았다.',
'','## 5. 재사용한 2024년 평가: 1,006명','',
'2024년 결과는 앞선 실험에서 이미 보았다. 새 외부검증이나 완전히 미열람인 시험집합이라고 부르지 않는다. 주 성능은 조사 가중치를 적용한다.',
'','| 모델 | AUC | AP | Brier | Youden 민감도 | 특이도 |', '|---|---:|---:|---:|---:|---:|']
for name in ['frozen_v1','refined_nn','tuned_lr','matched_basis_lr']:
    m=r['models'][name]['policies']['youden']['weighted'];L.append(f'| {labels[name]} | {m["roc_auc"]:.4f} | {m["pr_auc_ap"]:.4f} | {m["brier"]:.4f} | {m["sensitivity"]:.1%} | {m["specificity"]:.1%} |')
L += ['', '| 차이: 새 NN − LR | ΔAUC의 탐색적 95% 구간 |', '|---|---|']
for name,v in r['paired_nn_minus_lr'].items():
    lo,hi=v['delta_auc']['ci95'];L.append(f'| {labels[name]} | {lo:+.5f}–{hi:+.5f} |')
L+=['', '구간은 동일 PSU를 층 안에서 300회 재표집한 탐색적 구간이며 정식 domain 분산추정은 아니다. 두 비교의 구간 모두 0을 포함한다. NN의 우월성도 LR의 확정적인 우월성도 이 구간만으로 주장하지 않는다.',
'','## 6. 부가 출력 및 가구유형','',
f'조건부 head를 보완한 새 모델의 이상 항목 수 가중 MAE는 {aux["count_weighted_mae"]:.4f}다. 기존 모델은 {old["models"]["parallel_nn"]["count_test_mae"]:.4f}였다. 개선 여부는 아래 성분별 지표와 함께 판단해야 한다.',
'', '| 성분 | AUC | Brier |', '|---|---:|---:|']
for name,m in aux['components'].items():L.append(f'| {name} | {m["roc_auc"]:.4f} | {m["brier"]:.4f} |')
L += ['', '이번 후보는 원래 10개 비침습적 변수로 개발했다. 가구유형은 입력에 추가하지 않고 2024년 하위집단 성능으로 확인했다. 기존 가구정보 추가 모델과 `/v1/decide-household`는 보존했다.',
'','| 가구집단 | n | 새 모델 AUC | Youden 민감도 | 특이도 |', '|---|---:|---:|---:|---:|']
for name in ['living_1','living_0']:
    s=r['models']['refined_nn']['subgroups'][name];m=s['metrics']['youden'];L.append(f'| {name} | {s["n"]} | {m["roc_auc"]:.4f} | {m["sensitivity"]:.1%} | {m["specificity"]:.1%} |')
L+=['','## 7. 사용 및 검증','',
'- 기존 `/v1/decide`와 `python -m metabolic.predict`는 동일 모델의 고속 경로를 사용한다.',
'- 기존 `metabolic.predict.Predictor` 클래스는 수치 비교를 위한 원본 실행 경로로 보존했다. Python 호출자는 `metabolic.fast.FastPredictor`를 사용한다.',
'- 새 후보: `metabolic.refined.RefinedPredictor("artifacts/optimized_v2/model_complete")` 또는 `/v2/decide`.',
'- `/v2/decide` 요청은 기존 state와 선택적 `policy: "youden"` 또는 `"sensitivity90"`를 받는다.',
'- API TestClient, 결측 입력, scalar/batch 변환 일치, 내보낸 신경망 수치 일치, 목표 입력 거부 등을 포함해 21개 테스트 통과.',
'- `PLAN.json`, `cv_results.json`, `SELECTION.json`, `data_quality.json`, `report.json`, `pattern_report.json`, `latency.json`에 근거를 저장했다.',
'- 테스트 성능을 보고 추가 후보를 선택하지 않았으며 외부 배포하지 않았다.',
'', '```powershell', '.\\.venv\\Scripts\\python.exe -m pytest tests -q',
'.\\.venv\\Scripts\\python.exe -m metabolic.predict --artifact artifacts/local_parallel_v1_run2 --input artifacts/local_parallel_v1_run2/example_request.json',
'```','', '속도 개선은 구현상 성과다. 이 데이터에서 신경망이 로지스틱 회귀보다 반드시 잘 맞거나 더 빨라진다는 보장은 없으며, 이번 실험에서는 두 조건을 함께 달성하지 못했다.']
(OUT/'OPTIMIZATION_REPORT_KO.md').write_text('\n'.join(L),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
keys=['v1_legacy_nn','v1_fast_nn','v2_refined_nn','lr_tuned_fast'];labelsplot=['Legacy NN','Same NN optimized','Refined NN','Tuned LR optimized']
vals=[b['common'][k]['p50_ms'] for k in keys]
axes[0].barh(labelsplot[::-1],vals[::-1],color=['#748293','#159b88','#3266bc','#a4adbb']);axes[0].set_xscale('log');axes[0].set(xlabel='Median milliseconds (log scale)',title='Warm CPU: identical binary output')
for i,v in enumerate(vals[::-1]):axes[0].text(v*1.1,i,f'{v:.3f}',va='center')
auckeys=['frozen_v1','refined_nn','tuned_lr'];auc=[r['models'][k]['policies']['youden']['weighted']['roc_auc'] for k in auckeys]
axes[1].bar(['Frozen NN','Refined NN','Tuned LR'],auc,color=['#3266bc','#159b88','#748293']);axes[1].set(ylim=(0,1),ylabel='Weighted AUROC',title='Reused 2024 holdout (exploratory)')
for i,v in enumerate(auc):axes[1].text(i,v+.015,f'{v:.4f}',ha='center')
fig.savefig(OUT/'optimization.png',dpi=170)
dump(OUT/'verification.json',{'tests':'21 passed','runtime_equivalence':b['v1_equivalence'],
    'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'metabolic').glob('*.py'))+list((ROOT/'tests').glob('*.py'))},
    'model_hashes':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'model_complete').glob('*')}})
print(OUT/'OPTIMIZATION_REPORT_KO.md')
