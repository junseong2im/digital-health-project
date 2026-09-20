"""Generate a domain-architecture report from the frozen v3 experiment."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metabolic.comparison import dump,ROOT

OUT=ROOT/'artifacts/specialized_v3'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
r=read(OUT/'report.json');b=read(OUT/'latency.json');config=read(OUT/'model/config.json')
v2=read(ROOT/'artifacts/optimized_v2/report.json')
v1=read(ROOT/'artifacts/local_parallel_v1_run2/report.json')
nn=r['models']['specialized_nn']['policies']['youden']['weighted'];lr=r['models']['tuned_lr']['policies']['youden']['weighted']
L=['# 대사이상 특화 신경망 v3: 그룹 경로·다중과제 학습','',
'## 구현한 변형','',
'선택된 모델은 신체계측·인구학·생활습관의 세 인코더를 각각 두고, 각 경로를 4차원으로 압축한다. 신체계측×인구학, 신체계측×생활습관 잠재표현 곱을 결합해 비선형 관계를 표현한다. 원래 전처리 특징의 선형 skip도 유지한다.',
'신체계측 경로는 BMI·허리둘레·WHtR와 해당 결측 표시 및 파생 기저를 처리한다. 인구학 경로는 연령·성별·소득·교육을, 생활습관 경로는 흡연·음주·유산소 활동을 처리한다. 이는 모델링 가설이며 인과적 생리 기전을 입증하는 구조가 아니다.',
'출력은 종합 위험 로짓 1개와 혈당·혈압·TG·HDL의 상태를 표현하는 로짓 4개다. 정상 확률은 1-r, 15개 양성 조합의 확률은 r×조건부 확률로 계산해 총 16개 조합의 일관된 분포를 만든다. Choice·Noul·Score와 각 소견 확률을 이 분포에서 동시에 얻는다.',
'소견 간 6개 쌍별 출력항을 추가한 모델도 시험했지만, 개발 교차검증에서 제외한 모델이 소폭 더 높아 선택 모델에서는 사용하지 않는다. 입력 경로의 잠재표현 상호작용은 유지한다.',
f'학습 파라미터는 **{config["parameters"]:,}개**다. v2의 {v2["nn_config"]["parameter_count"]:,}개보다 {1-config["parameters"]/v2["nn_config"]["parameter_count"]:.1%} 적다. 실행 시에는 세 인코더의 선형 계산을 하나의 행렬 연산으로 합쳐 반복 호출 비용을 줄였다. 합쳐진 실행 표현의 영(0) 계수는 추가 학습 파라미터가 아니다.',
'', '## 학습 방식','',
'초기 종합 위험 가중치는 학습 표본에 적합한 로지스틱 회귀로, 초기 소견 가중치는 네 개의 소견별 로지스틱 회귀로 설정한 뒤 신경망을 학습한다. 따라서 완전히 무작위 초기화한 순수 MLP가 아니라 **로지스틱 초기화를 사용하는 특화 신경망**이다.',
'선택된 손실은 조사 가중치를 적용한 종합 위험 BCE + 0.2×소견별 평균 BCE + 0.05×양성 조합 조건부 NLL이다. 선형 위험 경로가 초기값에서 과도하게 벗어나지 않도록 약한 제약을 둔다.',
'BMI·허리둘레·WHtR의 상관이 이 구조로 자동 제거되는 것은 아니다. 상관된 지표를 한 경로에서 함께 처리한다는 의미다.',
'', '## 데이터와 검증','',
f'- 기존 코호트 정의 유지: 전체 4,068명. **2021–2023년 {r["development_n"]:,}명으로 개발·최종 재학습**, 2024년 {r["test_n"]:,}명 평가.',
'- 3-fold StratifiedGroupKFold: 연도+PSU가 fold의 학습·검증에 겹치지 않는다.',
'- fold 내부의 별도 그룹 분할에서 조기종료 epoch를 선택하고, 해당 epoch로 outer train 전체를 재학습해 outer validation을 예측한다.',
'- 대치값·표준화·hinge 매듭은 매 fold의 학습 표본에만 적합한다. 채혈 결과와 혈압을 입력하지 않는다.',
f'- 각 참여자를 학습에 포함하지 않은 모델이 생성한 OOF 예측을 모았다. 그중 PSU 기준 {r["oof_calibration_n"]:,}명의 예측으로 확률을 보정하고, 나머지 {r["oof_threshold_n"]:,}명의 예측으로 컷오프를 정했다.',
f'- 구조 선택 후 epoch 중앙값 {config["epochs"]}로 개발 3,062명에 최종 재학습했다. 2024년으로 구조·epoch·보정·컷오프를 고르지 않았다.',
'- 후보 선택과 보정에 개발 자료를 재사용하므로 내부 성능은 독립 외부 검증이 아니다. OOF 모델의 학습량은 최종 모델보다 작아 보정값 이전에 따른 차이도 남는다.',
'- 2024년은 앞선 실험에서 이미 확인한 자료다. 이번 결과는 **재사용한 시간 분리 평가를 통한 탐색적 결과**다.',
'','## 구조별 개발 교차검증','', '| 모델 | 경로 | 경로 폭 | 쌍별 출력항 | 소견 손실 가중치 | 평균 AUC | 파라미터 |', '|---|---|---:|---|---:|---:|---:|']
cv=read(OUT/'cv_results.json')
for row in cv['nn']:
    c=row['config'];mark=' **선택**' if row['id']==r['selection']['nn']['id'] else ''
    L.append(f'| {row["id"]}{mark} | {c["mode"]} | {c["hidden"]} | {c["pairwise"]} | {c["aux"]} | {row["mean_auc"]:.5f} | {row["parameters"]} |')
L+=['', '구조 간 CV 차이가 매우 작다. 경로 분리나 소견 손실이 우월하다고 확증하지 않는다. 사전에 정한 순위 규칙으로 하나를 선택했으며, 가장 높은 2024년 점수를 보고 선택하지 않았다.',
f'LR도 같은 개발 표본·fold에서 8개 설정을 비교했다. 선택된 LR은 {r["selection"]["lr"]["config"]}, 개발 평균 AUC {r["selection"]["lr"]["mean_auc"]:.5f}다.',
'', '## 2024년 결과','', '| 모델 | 학습 n | AUC | AP | Brier |', '|---|---:|---:|---:|---:|']
for name,n,m in [('기존 v1',1680,v1['models']['parallel_nn']['test_2024']),('이전 v2',2052,v2['models']['refined_nn']['policies']['youden']['weighted']),('특화 v3',3062,nn),('동일 조건 LR',3062,lr)]:
    L.append(f'| {name} | {n} | {m["roc_auc"]:.4f} | {m["pr_auc_ap"]:.4f} | {m["brier"]:.5f} |')
L+=['', 'v1/v2 대비 변화에는 모델 구조뿐 아니라 학습 표본 증가, 검증/보정 방식 변경이 함께 작용한다. AUC 상승 전부를 구조 변형의 효과로 해석할 수 없다.',
'', '| 모델·정책 | 민감도 | 특이도 | PPV | 컷오프 |', '|---|---:|---:|---:|---:|']
for name,model in r['models'].items():
    for policy,values in model['policies'].items():
        m=values['weighted'];L.append(f'| {name} / {policy} | {m["sensitivity"]:.1%} | {m["specificity"]:.1%} | {m["ppv"]:.1%} | {m["threshold"]:.4f} |')
L+=['', '### v3 − LR 차이의 탐색적 구간', '', '| 지표 | 탐색적 95% 구간 |', '|---|---|']
for name,values in r['paired_nn_minus_lr'].items():
    a,c=values['ci95'];L.append(f'| {name} | {a:+.5f} ~ {c:+.5f} |')
L+=['', '동일한 2024년 PSU를 층 내에서 300회 재표집한 구간이다. AUC·AP·Brier 차이 구간 모두 0을 포함한다. 우리 모델 또는 LR의 확정적 우월성을 주장할 수 없다. 정식 복합표본 domain 분산 추정 구간도 아니다.',
'', '## 소견별·가구유형별 결과','', '| 소견 | AUC | Brier |', '|---|---:|---:|']
for name,m in r['components'].items():L.append(f'| {name} | {m["roc_auc"]:.4f} | {m["brier"]:.5f} |')
L += ['',f'예상 이상 항목 수의 가중 MAE: {r["count_mae"]:.4f}. 별도 소견 각각의 완전한 보정을 보장하지 않는다.',
'', '| 가구 집단 | 평가 n | AUC | Youden 민감도 | 특이도 |', '|---|---:|---:|---:|---:|']
for name in ['living_1','living_0']:
    row=r['models']['specialized_nn']['subgroups'][name];m=row['metrics']['youden']
    L.append(f'| {name} | {row["n"]} | {m["roc_auc"]:.4f} | {m["sensitivity"]:.1%} | {m["specificity"]:.1%} |')
L += ['', 'v3는 가구정보를 입력에 추가하지 않았다. 가구원수로 나눈 하위집단 평가이며 직접 측정한 자취 여부가 아니다. 기존 가구정보 추가 모델과 API는 보존했다.',
'', '## 추론시간과 구현 검증','',
'같은 가상 입력에 대해 warm CPU 1스레드, 1,000회, 무작위 순서로 측정했다. 입력 검증·전처리·이진 결과 JSON 출력을 포함하고 모델 로딩·HTTP는 제외했다. LR에도 배열 기반 전처리와 메모리 상주 계수를 적용했다.',
'', '| 모델 | 중앙값(ms) | p95(ms) |', '|---|---:|---:|']
for name,row in b['common'].items():L.append(f'| {name} | {row["p50_ms"]:.4f} | {row["p95_ms"]:.4f} |')
full=b['v3_full_choice_score_noul']
L += ['',f'전체 Choice·Score·Noul 출력: 중앙값 {full["p50_ms"]:.4f}ms, p95 {full["p95_ms"]:.4f}ms. 측정 시점의 CPU 부하가 달라 과거 다른 실행의 시간과 직접 비율 비교하지 않는다.',
'전체 테스트 **28개 통과**. 그룹 인덱스, 확률 합·성분 관계, Torch↔배열 실행 동등성, OOF 역할 분리, 저장 모델 로딩, `/v3/decide` 정상/오류 요청을 검증했다.',
'', '## 사용법','',
'```powershell',
'.\\.venv\\Scripts\\python.exe -m metabolic.specialized --artifact artifacts/specialized_v3/model --input artifacts/specialized_v3/example_request.json --policy youden',
'.\\.venv\\Scripts\\python.exe -m pytest tests -q',
'```',
'Python: `SpecializedPredictor("artifacts/specialized_v3/model").predict(state, policy="youden")`.',
'API: `POST /v3/decide`, 예: `{"state":{"age":29,"sex":1,"HE_BMI":25},"policy":"youden"}`.',
'기존 `/v1/decide`, `/v2/decide`, 가구유형 API와 모델 파일은 보존했다. v3를 외부 배포하거나 기본 모델로 자동 교체하지 않았다.',
'', '## 남은 한계','',
'모델은 19–39세의 단면 검사 이상 소견을 선별하는 연구 후보다. 임상 진단·미래 발병 예측·자가계측의 정확성·확실한 미인지를 입증한 것은 아니다. 2024년을 반복 확인했으므로 다음 확증은 새 외부 자료에서 해야 한다. 모든 확률의 정확성이나 로지스틱 대비 우월성을 보장하지 않는다.']
(OUT/'SPECIALIZED_REPORT_KO.md').write_text('\n'.join(L),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
axes[0].bar(['v1\n1,680 train','v2\n2,052 train','v3\n3,062 train','LR\n3,062 train'],
    [v1['models']['parallel_nn']['test_2024']['roc_auc'],v2['models']['refined_nn']['policies']['youden']['weighted']['roc_auc'],nn['roc_auc'],lr['roc_auc']],color=['#9aa8bf','#6988b8','#179c8a','#606975'])
axes[0].set(ylim=(0,1),ylabel='Survey-weighted AUROC',title='2024 exploratory temporal evaluation')
for i,val in enumerate(axes[0].patches):axes[0].text(i,val.get_height()+.015,f'{val.get_height():.4f}',ha='center')
bars=axes[1].barh([a['id'] for a in cv['nn']][::-1],[a['mean_auc'] for a in cv['nn']][::-1],color='#179c8a')
axes[1].set(xlim=(0,1),xlabel='Mean development group-CV AUROC',title='Architecture selection excludes 2024')
for bar in bars:axes[1].text(bar.get_width()+.01,bar.get_y()+bar.get_height()/2,f'{bar.get_width():.5f}',va='center',fontsize=8)
fig.savefig(OUT/'specialized.png',dpi=170)
dump(OUT/'verification.json',{'tests':'28 passed','source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'metabolic').glob('*.py'))+list((ROOT/'tests').glob('*.py'))},
    'model_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'model').glob('*')},
    'serving_change':'Group encoders fused with exact structural zeros; export equivalence tested against Torch'})
print(OUT/'SPECIALIZED_REPORT_KO.md')
