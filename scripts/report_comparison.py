"""Write the comparison report and aggregate figures from completed artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/comparison_household_v1'
read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
r=read('report.json');audit=read('peer_audit.json');ha=read('household_audit.json')
ci=read('subgroup_uncertainty.json');imp=read('validation_importance.json');inter=read('interaction_multiplicity.json')
names={'base_seed42':'기존 병렬 신경망','logistic_regression':'Logistic Regression','random_forest':'Random Forest',
       'xgboost':'XGBoost','lightgbm':'LightGBM','household_seed42':'1인 가구 변수 추가 신경망'}
fmt=lambda x:f'{x:.4f}' if x is not None else '산출불가'
pct=lambda x:f'{x:.1%}' if x is not None else '산출불가'
interval=lambda v:f'{v[0]:.4f}–{v[1]:.4f}'
L=['# 비교연구 및 1인 가구 하위집단 분석', '',
'## 핵심 결론', '',
'동일한 원자료·코호트·시간 분할·가중치 조건에서 기존 로컬 병렬 신경망과 동료 연구의 4개 알고리즘을 비교했다. 신경망과 로지스틱 회귀는 비슷했고, 이번 고정 설정에서는 신경망이 세 트리 모델보다 높은 AUC를 보였다. 이는 모든 하이퍼파라미터 조건에서의 우월성 검증은 아니다.',
'가구원수로 1인/다인 가구를 나누고 가구유형을 입력으로 받는 선별 기능을 구현했다. 1인 가구 전용 신경망도 평가했지만 표본이 적어 통합 모델보다 좋지 않았다. 가구정보 추가의 이득은 시드에 따라 달랐다.',
'**자취 여부를 직접 분류한 연구가 아니다.** 확인한 변수는 가구원수 `cfam`과 세대구성 `genertn`이다. 혼자 사는 1인 가구는 식별할 수 있지만, 부모와 따로 사는지·룸메이트와 자취하는지·학생인지·기숙사 생활인지까지 확정할 수 없다.',
'**2024년은 이전 실험에서 이미 확인한 평가 연도다.** 이번 결과는 그 연도를 재사용한 탐색적 확장이다. 독립 외부 검증이나 임상적 유용성 확증으로 표현하지 않는다.',
'', '## 1. 동료 문서·공개 노트북과의 정합성', '',
f'확인한 공개 저장소: https://github.com/kdw123654/digital-project/tree/{audit["commit"]}',
'사용자가 준 문서는 원본 그대로 보관했다. 공개 노트북은 다운로드하여 읽기만 했고 실행하지 않았다. 아래 재구성은 로컬 원자료로 해당 필터 및 목표 생성 논리를 별도로 계산한 결과다.',
'', '| 항목 | 공유 문서 | 공개 노트북/현재 데이터 확인 |', '|---|---|---|',
f'| 19–39세 | 5,640명 | 노트북 출력과 현재 자료 모두 {audit["current_data_reconstruction"]["young"]:,}명 |',
f'| 치료 제외 후 | 5,204명 | 현재 같은 *_pt 필터 {audit["current_data_reconstruction"]["after_treatment_exclusion"]:,}명 |',
f'| 설명변수 완전 사례 | 4,678명 | 노트북 train 3,276 + test 1,404 = 4,680명; 현재 {audit["current_data_reconstruction"]["feature_complete"]:,}명 |',
'| 전체 양성 수 | 1,693명 | 현재 노트북 논리 재구성 1,729명 |',
'| Youden 컷오프 | 0.347 | 공개 노트북 저장 출력 0.349 |',
'| 소득 incm | 소득 5분위 | 공식 코드: 개인소득 4분위 |',
'', '수치 차이는 먼저 정합성 확인이 필요한 부분이다. 동료의 SAS 원본 해시와 패키지 버전까지 제공받지 않았으므로 모든 차이의 원인을 특정하지 않았다.',
'', '### 비교 설계에서 바로잡은 사항', '',
'- 공개 코드의 cell 9는 `y_test`로 Youden 임계치를 정하고 같은 시험 데이터에서 민감도·특이도를 계산한다. 우리는 2023년 별도 임계치 집합에서만 결정했다.',
'- cell 12의 연령별 모델은 같은 데이터에 학습·평가하여 훈련 AUC를 출력한다. 우리는 연령 전용 모델도 2024년에서 평가했다.',
f'- 목표 생성 전에 검사 결측을 확인하지 않아, 설명변수 완전 사례 중 {audit["current_data_reconstruction"]["missing_any_target_measurement"]}명은 적어도 하나의 목표 검사값이 없다. 그중 {audit["current_data_reconstruction"]["missing_target_coded_negative"]}명이 정상(0)으로 처리된다. 실제로 모두 정상인지 알 수 없으므로 오류율로 단정하지 않지만, 확정 음성으로 간주할 근거도 부족하다.',
'- `DI1_pt`, `DI2_pt`, `DE1_pt`는 치료 여부다. 전용 복약 변수와 같지 않으며, 치료자 제외만으로 모든 기진단자를 제외한 것은 아니다.',
'- 변수 중요도는 원인 기전의 입증이 아니며, AUC 0.80은 정답률 80%가 아니다. 회귀계수의 비율을 흡연 위해성의 배수로 해석하지 않는다.',
'', '## 2. 공정 비교 프로토콜', '',
'주 분석 대상: 19–39세, 고혈압·당뇨·이상지질혈증 진단 경험 없음, 관련 복약·인슐린 미사용, 임신개월수 비기록, 12시간 이상 공복, 목표 검사 완비. 미인지라는 표현은 진단 경험 없음에 기반한 대리 정의다.',
'기존 실험의 4,068명 코호트 및 개인별 분할 해시를 그대로 복원했다. 동료 문서의 AUC 0.804와 우리 0.832를 직접 승패 비교하지 않는다.',
'', '| 역할 | n | 1인 가구 | 다인 가구 |', '|---|---:|---:|---:|']
for k,v in r['partitions'].items():L.append(f'| {k} | {v["n"]} | {v["single"]} | {v["multi"]} |')
L+=['', '학습·조기종료: 2021–2022 / 확률보정·임계치 선정: 2023 / 평가: 2024. 연도+PSU 중복을 차단했다. 전처리는 학습 데이터에만 적합했다. 모든 모델에 같은 가중치와 보정·임계치 집합을 적용했다.',
'LR·RF·XGBoost·LightGBM은 사전에 고정한 설정으로 학습했다. 신경망은 기존 48/24 은닉층과 16조합 출력 구조를 유지했다. 세 시드 42/43/44를 모두 보고하며 가장 좋은 시드만 고르지 않는다.',
'학습 전 저장한 PLAN.json에 분석 범위·시드·평가 방식을 기록했다. 이 파일은 사전등록 플랫폼을 통한 정식 preregistration은 아니다.',
'', '## 3. 동일 조건의 모델 비교: 2024년 1,006명', '',
'아래 지표는 wt_itvex 가중치를 적용했다. AP는 average precision 방식의 PR-AUC다.',
'', '| 모델 | AUROC | AP | Brier | Youden 민감도 | Youden 특이도 | 컷오프 |', '|---|---:|---:|---:|---:|---:|---:|']
for name,label in names.items():
    m=r['models'][name]['policies']['youden']['weighted']
    L.append(f'| {label} | {m["roc_auc"]:.4f} | {m["pr_auc_ap"]:.4f} | {m["brier"]:.4f} | {pct(m["sensitivity"])} | {pct(m["specificity"])} | {m["threshold"]:.4f} |')
L+=['', '| 모델 | 민감도 90% 기준: 실제 민감도 | 실제 특이도 | PPV |', '|---|---:|---:|---:|']
for name,label in names.items():
    m=r['models'][name]['policies']['sensitivity90']['weighted'];L.append(f'| {label} | {pct(m["sensitivity"])} | {pct(m["specificity"])} | {pct(m["ppv"])} |')
L+=['', '민감도 90%는 2023년 임계치 집합에서의 선정 조건으로, 2024년에서 보장되는 값이 아니다. 같은 모델도 컷오프 정책에 따라 민감도와 특이도가 달라진다.',
'', '### 기존 신경망 − 비교모델: 동일 PSU 재표집에 의한 AUC 차이', '',
'| 비교 | 실제 ΔAUC | 탐색적 95% 구간 |', '|---|---:|---|']
base=r['models']['base_seed42']['policies']['youden']['weighted']['roc_auc']
for name in ['logistic_regression','random_forest','xgboost','lightgbm']:
    delta=base-r['models'][name]['policies']['youden']['weighted']['roc_auc'];v=r['paired_differences'][f'base_seed42_minus_{name}']['delta_auc']['ci95']
    L.append(f'| 신경망 − {names[name]} | {delta:+.4f} | {interval(v)} |')
L+=['', '300회 층별 PSU bootstrap 구간이며 정식 복합표본 domain 분산 추정은 아니다. 다중 비교 보정을 적용한 확증 검정도 아니다. 특히 LR 대비 구간은 0을 포함한다.',
'', '## 4. 가구유형 정의·자료 품질', '',
'`cfam=1`을 1인 가구, `cfam=2–6`을 다인 가구로 정의했다. 6은 6명 이상인 top coding이다. 9·결측은 미상으로 처리하고 다인 가구로 강제 배정하지 않는다. `genertn=1`과 교차 점검했다.',
'', '| 연도 | 19–39세 전체 | 1인 가구 | 가구정보 미상 | 두 코드 불일치 |', '|---|---:|---:|---:|---:|']
for year,v in ha['by_year'].items():L.append(f'| {year} | {v["young_n"]} | {v["single"]} | {v["unknown"]} | {v["cfam_genertn_conflict"]} |')
L+=['', '원자료 청년 5,341명에는 미상 1명이 있으나 주 연구 코호트에는 미상이 없다. 따라서 가구정보 추가 모델은 추론 시 미상 입력을 거부하고 원래 모델 사용을 안내한다.',
'', '## 5. 가구유형별 이상 소견 비율', '', '| 집단 | 4개년 n | 가중 비율 | 탐색적 95% 구간 | 2024년 n | 2024년 가중 비율 |', '|---|---:|---:|---|---:|---:|']
for group,label in [('living_1','1인 가구'),('living_0','다인 가구')]:
    v=r['prevalence_all_years'][group];t=r['prevalence_2024'][group];q=v['ci95_exploratory']
    L.append(f'| {label} | {v["n"]} | {pct(v["weighted"])} | {pct(q[0])}–{pct(q[1])} | {t["n"]} | {pct(t["weighted"])} |')
L+=['', '이 연구 코호트에서 1인 가구의 단순 가중 비율이 더 높지는 않았다. 연령·성별·소득·신체계측 구성과 선택조건이 다르므로 이를 자취의 보호효과나 유해효과로 해석할 수 없다.',
'', '## 6. 1인 가구 여부에 따른 선별 성능', '',
'컷오프는 집단별로 시험 결과를 보고 조정하지 않았다. 아래 민감도·특이도는 전체 2023년에서 정한 각 모델의 Youden 컷오프를 그대로 사용했다.',
'', '| 집단 | n/양성 | 모델 | AUROC (탐색적 95% 구간) | 민감도 | 특이도 |', '|---|---:|---|---|---:|---:|']
for group,label in [('living_1','1인 가구'),('living_0','다인 가구')]:
    for name in ['base_seed42','household_seed42']:
        s=r['models'][name]['subgroups'][group];m=s['metrics']['youden'];bounds=ci[name][group]['auc']['ci95']
        L.append(f'| {label} | {s["n"]}/{s["positive"]} | {names[name]} | {m["roc_auc"]:.4f} ({interval(bounds)}) | {pct(m["sensitivity"])} | {pct(m["specificity"])} |')
L+=['', '### 가구유형·연령별 전용 신경망', '', '| 전용 모델 | 학습 n | 평가 n | 전용 모델 AUC | 같은 평가자에 대한 통합 모델 AUC |', '|---|---:|---:|---:|---:|']
for group,item in r['specialists'].items():
    if item['status']!='trained_exploratory':L.append(f'| {group} | 표본 부족 | - | - | - |');continue
    a=item['specialist']['policies']['youden']['weighted']['roc_auc'];b=item['pooled_on_same_rows']['policies']['youden']['weighted']['roc_auc']
    L.append(f'| {group} | {item["counts"]["train"]["n"]} | {item["counts"]["test_2024"]["n"]} | {a:.4f} | {b:.4f} |')
L+=['', '별도 모델은 각 역할 집합에서 양성·음성 10명 이상 및 학습 100명 이상일 때만 수행했다. 표본 기준 충족은 임상 충분성을 뜻하지 않는다. 1인 가구 전용 모델은 통합 모델보다 낮아 기본 경로로 채택하지 않았다.',
'', '## 7. 가구정보 추가 효과: 3개 시드', '', '| 시드 | 기본 AUC | 가구 추가 AUC | 차이 | 차이의 탐색적 95% 구간 |', '|---|---:|---:|---:|---|']
deltas=[]
for seed in [42,43,44]:
    a=r['models'][f'base_seed{seed}']['policies']['youden']['weighted']['roc_auc'];b=r['models'][f'household_seed{seed}']['policies']['youden']['weighted']['roc_auc'];deltas.append(b-a)
    bounds=r['paired_differences'][f'household_minus_base_seed{seed}']['delta_auc']['ci95']
    L.append(f'| {seed} | {a:.4f} | {b:.4f} | {b-a:+.4f} | {interval(bounds)} |')
L += ['',f'세 시드의 단순 평균 AUC 변화는 {np.mean(deltas):+.4f}다. 시드 42·43의 구간은 0을 포함하고 시드 44에서는 양수였다. 개선이 일관되게 확증되었다고 말할 수 없다. 확률 보정·AP 결과도 report.json에 모두 포함했다.',
'', '## 8. 성별·연령 및 가구유형 교차 분석', '',
'전체 코호트의 가중 이상 소견 비율과 2024년 기존 신경망 성능이다. 19세는 19–29세 집단에 포함하며 단순히 20대라고 표기하지 않는다.',
'', '| 집단 | 4개년 n | 가중 비율 | 2024년 n/양성 | AUC | Youden 민감도 | 특이도 |', '|---|---:|---:|---:|---:|---:|---:|']
for group in [k for k in r['prevalence_all_years'] if k.startswith('sex_') or ('_sex_' in k)]:
    v=r['prevalence_all_years'][group];s=r['models']['base_seed42']['subgroups'][group];m=s['metrics']['youden'];flag='*' if s['low_information'] else ''
    L.append(f'| {group}{flag} | {v["n"]} | {pct(v["weighted"])} | {s["n"]}/{s["positive"]} | {fmt(m["roc_auc"])} | {pct(m["sensitivity"])} | {pct(m["specificity"])} |')
L+=['', '`sex_1`=남성, `sex_2`=여성, `living_1`=1인 가구, `living_0`=다인 가구. *는 평가집단의 양성 또는 음성이 20명 미만인 불안정한 추정이다. 각 구간은 subgroup_uncertainty.json에서 확인할 수 있다.',
'', '## 9. 탐색적 관련성: 오즈비와 상호작용', '',
f'분석 n={r["associations"]["n"]:,}. 별도의 비벌점 로지스틱 최대우도 추정에 연도+PSU 군집 표준오차를 적용했다. 이 분석은 **비가중 완전사례 분석**이며 국가 인구의 설계기반 인과효과 추정이 아니다. 교육·소득은 범주형 처리하고 연도를 보정했다.',
'연령·성별·소득·교육·흡연·음주·신체활동·허리둘레·가구유형·연도를 포함했다. BMI·WC·WHtR 세 가지를 동시에 넣어 기전을 해석하는 대신 관련성 모형에는 WC를 대표 계측으로 사용했다.',
'', '| 변수 | aOR | 95% 구간 | p (보정 전) |', '|---|---:|---|---:|']
for k in ['living_alone','sm_presnt','age','HE_wc']:
    t=r['associations']['models']['adjusted']['terms'][k];L.append(f'| {k} | {t["aOR"]:.3f} | {interval(t["ci95"])} | {t["p"]:.4g} |')
L+=['', 'aOR는 오즈의 비이며 위험률의 배수가 아니다. 1인 가구 aOR가 1보다 작더라도 잔여 교란·선택편향과 단면 자료의 한계로 자취의 보호효과로 해석할 수 없다.',
'', '| 상호작용 | 오즈비의 비 | 95% 구간 | p | 두 상호작용 BH q |', '|---|---:|---|---:|---:|']
for k,t in inter.items():L.append(f'| {k} | {t["aOR"]:.3f} | {interval(t["ci95"])} | {t["p"]:.4f} | {t["q_bh_two_prespecified_interactions"]:.4f} |')
L+=['', '연령×흡연은 실제 interaction term을 사용했다. 두 상호작용에 대한 다중검정 보정 후 기준과 단면 설계를 함께 고려해야 한다. 회귀계수 두 개의 비율만으로 흡연 피해가 몇 배 폭증했다고 주장하지 않는다.',
'', '## 10. 신경망 변수 중요도', '',
'경사 학습에서 제외하고 조기종료에는 사용한 2021–2022 validation 372명에서 변수별 20회 순열 중요도를 계산했다. 이번 중요도 결과를 구조 변경에 사용하지 않았다. 독립 시험집합의 중요도 추정은 아니다. 단위는 가중 AUC 하락이며 트리의 split count와 직접 비교할 수 없다.',
'', '| 변수/블록 | 평균 AUC 하락 | 반복 표준편차 |', '|---|---:|---:|']
for name,v in sorted(imp['importance'].items(),key=lambda z:z[1]['mean_auc_drop'],reverse=True):L.append(f'| {name} | {v["mean_auc_drop"]:.4f} | {v["sd"]:.4f} |')
L+=['', 'anthropometry_joint는 BMI·WC·WHtR를 같은 순열로 함께 섞은 결과다. 상관된 변수의 개별 중요도는 작게 나타날 수 있고, 순열 자체가 실제로 드문 조합을 만들 수도 있다. 원인 기전을 입증하지 않는다.',
'', '## 11. 실행·산출물', '',
'```powershell',
'.\\.venv\\Scripts\\python.exe -m metabolic.household --artifact artifacts/comparison_household_v1/models/household_seed42 --input artifacts/comparison_household_v1/example_request.json --policy youden',
'.\\.venv\\Scripts\\python.exe -m pytest tests -q',
'```',
'로컬 API `/v1/decide-household`에 기존 비침습적 입력과 `cfam`을 전달하면 가구유형 표기 및 Choice/Score/Noul 결과를 함께 반환한다. 기존 `/v1/decide`와 기본 모델은 그대로 보존했다. 실제 서버를 외부 배포하지 않았다.',
'', '- `report.json`: 전체 성능, 모든 시드, 분할, 하위집단, 전용모델, 오즈비, 해시',
'- `peer_audit.json`: 동료 문서와 공개 코드 확인 근거',
'- `household_audit.json`: 4개년 변수 가용성 및 일치성',
'- `models/`: 학습 모델·전처리기·보정값·컷오프',
'- `subgroup_uncertainty.json`: 하위집단 탐색적 구간',
'- `validation_importance.json`: 검증집합 순열 중요도',
'- `comparison.png`: 비교 성능 및 가구분석 그림',
'- `test_predictions.json`: 로컬 평가용 개별 예측값, 외부 공유 대상 아님',
'', '## 해석의 한계와 다음 검증 조건', '',
'관찰된 결과는 특정 코호트와 고정 설정의 탐색 실험이다. 새로운 외부 자료·아직 보지 않은 조사연도에서 재검증하고, 자취 여부를 직접 묻는 별도 설문이 있어야 자취와 대사이상의 관련성을 연구할 수 있다. 검진 계측을 자가 계측으로 바꾸었을 때의 측정오차도 별도 검증이 필요하다.',
'공식 자료: https://knhanes.kdca.go.kr/knhanes/dataAnlsGd/utztnGd.do . 제8·9기 지침서의 가구공통설문에서 cfam/genertn을 확인했다. 문서는 data/reference/knhanes에 보존했다.',
'통계 구현 참고: https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Logit.html', '']
(OUT/'COMPARISON_REPORT_KO.md').write_text('\n'.join(L),encoding='utf-8')

fig,axs=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
model_keys=['base_seed42','logistic_regression','random_forest','xgboost','lightgbm','household_seed42']
labels=['Our NN','LR','RF','XGB','LGBM','NN + household']
auc=[r['models'][k]['policies']['youden']['weighted']['roc_auc'] for k in model_keys]
axs[0].barh(labels[::-1],auc[::-1],color=['#159b88']+['#687ca0']*4+['#3266bc'])
axs[0].set(xlim=(0,1),xlabel='Survey-weighted AUROC',title='Same 2024 rows (n=1,006)')
for i,v in enumerate(auc[::-1]):axs[0].text(v+.01,i,f'{v:.3f}',va='center',fontsize=9)
for model,offset,color,label in [('base_seed42',-.08,'#3266bc','Our NN'),('household_seed42',.08,'#159b88','NN + household')]:
    for i,g in enumerate(['living_1','living_0']):
        val=r['models'][model]['subgroups'][g]['metrics']['youden']['roc_auc'];lo,hi=ci[model][g]['auc']['ci95']
        axs[1].errorbar(val,i+offset,xerr=[[val-lo],[hi-val]],fmt='o',color=color,capsize=4,label=label if i==0 else None)
axs[1].set(yticks=[0,1],yticklabels=['Single-person (n=189)','Multi-person (n=817)'],xlim=(.65,.95),xlabel='AUROC, exploratory 95% CI',title='Household subgroup discrimination')
axs[1].legend(fontsize=8)
for i,seed in enumerate([42,43,44]):
    a=r['models'][f'base_seed{seed}']['policies']['youden']['weighted']['roc_auc'];b=r['models'][f'household_seed{seed}']['policies']['youden']['weighted']['roc_auc'];delta=b-a
    lo,hi=r['paired_differences'][f'household_minus_base_seed{seed}']['delta_auc']['ci95']
    axs[2].errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color='#159b88',capsize=4)
axs[2].axvline(0,linestyle='--',color='gray');axs[2].set(yticks=[0,1,2],yticklabels=['Seed 42','Seed 43','Seed 44'],xlabel='AUROC difference (household - base)',title='Household feature ablation')
for ax in axs:ax.grid(axis='x',alpha=.2)
fig.savefig(OUT/'comparison.png',dpi=180)
print(OUT/'COMPARISON_REPORT_KO.md')
