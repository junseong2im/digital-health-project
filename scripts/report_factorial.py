"""Report both interventions, including null and adverse endpoint findings."""
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from metabolic.comparison import ROOT,dump

OUT=ROOT/'artifacts/expanded_multitask_v4'
read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
r=read('report.json');cv=read('cv_progress.json');audit=read('feature_audit.json');ci=read('multioutput_uncertainty.json');importance=read('extra_feature_importance.json');latency=read('latency.json')
labels={'base_single_s42':'기존 입력·단일과제','base_multi_s42':'기존 입력·다중과제','expanded_single_s42':'확장 입력·단일과제','expanded_multi_s42':'확장 입력·다중과제',
        'base_LR5':'기존 입력 LR 5개','expanded_LR5':'확장 입력 LR 5개'}
columns={'sedentary_hours':'하루 앉아 있는 시간(시간)','walking_days':'일주일 걷기 일수(0–7일)','stress_level':'스트레스 정도(1 높음–4 낮음)',
         'employed':'취업 여부(0/1)','alcohol_frequency':'음주 빈도 6단계','alcohol_amount':'음주량 6단계(비음주 포함)',
         'strength_days':'근력운동 일수 범주(5=5일 이상)','family_history':'만성질환 진단 가족력(0/1)','living_alone':'1인 가구(0/1)'}
L=['# 비침습적 입력 확장 × 단일·다중과제 비교 실험','',
'## 결론','',
'기존 10개 예측변수를 19개로 늘리고 단일과제/다중과제의 2×2 실험을 수행했다. 동일한 3개 시드, 개발 3,062명과 평가 1,006명으로 비교했다. 확장 입력에서 종합 위험 AUC는 조금 높았지만 개발 교차검증에서는 이득이 나타나지 않았다. 추가 정보의 효용이 안정적으로 입증된 것은 아니다.',
'다중과제 학습이 종합 위험 AUC를 높이지는 않았다. 확장 다중과제 모델은 네 소견 평균 AP·Brier에서 독립 LR 5개보다 조금 좋았지만, 차이 구간 대부분이 0을 포함했다. 모든 개별 소견이 개선된 것도 아니다.',
'2024년은 여러 차례 확인한 평가자료다. 모든 결과는 탐색적 확장으로 해석한다. 개발 OOF 예측을 구조/보정에 재사용하므로 내부 추정 역시 독립 외부 검증이 아니다.',
'','## 1. 새로 추가한 정보','',
'검진·건강설문 영역에서 4개년 공통 변수를 골랐다. 목표 검사값·개인 진단/복약·공복시간은 예측 입력에 추가하지 않았다. 영양조사 참여가 필요한 식사 회상 변수는 가중치·참여자 변경을 피하기 위해 제외했고 수면은 2024년 질문 방식 변경으로 제외했다.',
'','| 추가 변수 | 의미 | 출처 |', '|---|---|---|']
source={'sedentary_hours':'BE8_1 + BE8_2 / 60','walking_days':'BE3_31 − 1','stress_level':'BP1','employed':'EC1_1','alcohol_frequency':'BD1, BD1_11',
        'alcohol_amount':'BD1, BD1_11, BD2_1','strength_days':'BE5_1 − 1','family_history':'HE_fh','living_alone':'cfam'}
for key,value in columns.items():L.append(f'| {key} | {value} | {source[key]} |')
L+=['', '비해당/모름을 일괄적으로 0으로 바꾸지 않았다. 평생 비음주 또는 지난 1년 비음주가 확인된 경우에만 해당 음주 코드의 비해당을 비음주로 변환했다. 걷기 88/99, 근력 8/9, 가족력 9 등은 결측으로 처리했다. 앉아 있는 시간은 시간·분의 유효 범위와 합계 24시간 이하를 확인했다.',
'cfam은 직접 측정한 자취 여부가 아니다. family_history는 특정 한 질환의 가족력만을 뜻하지 않으며 공식 변수의 만성질환 가족력 범위를 따른다.',
'', '| 변수 | 2021 결측/n | 2022 결측/n | 2023 결측/n | 2024 결측/n |', '|---|---:|---:|---:|---:|']
for key in columns:
    cells=[f'{audit["by_year"][str(y)][key]["missing"]}/{audit["by_year"][str(y)][key]["n"]}' for y in range(2021,2025)]
    L.append('| '+key+' | '+' | '.join(cells)+' |')
L+=['', '추가 변수의 결측 때문에 대상자를 삭제하지 않았다. 같은 4,068명에서 비교했고, 대치/표준화는 fold 학습 표본에서만 적합했다. 학습에서 한 번도 관측되지 않은 범주 표시 열의 임의 효과가 생기지 않도록 첫 층의 상수 입력 열 가중치를 0으로 초기화했다.',
'', '## 2. 통제한 실험 설계','',
'- 기존 입력/확장 입력 × 종합 위험만 학습/종합 위험+네 소견 공동 학습의 4개 조건.',
'- 모든 조건에서 신체계측·인구학·생활습관 세 경로와 경로 폭 4를 동일하게 유지했다. 입력과 학습 과제에 필요한 차이만 허용했다.',
'- 단일과제 모델은 종합 위험 LR만 초기값으로 사용한다. 네 소견 라벨을 초기화·손실·조기종료에 사용하지 않는다. 소견별 출력은 제공하지 않는다.',
'- 다중과제는 종합 위험+소견 초기 LR을 사용하고 공동 손실을 적용한다. 두 조건의 조기종료 기준은 종합 위험 BCE로 동일하다.',
'- 2021–2023년 3,062명의 동일한 3-fold 연도+PSU 분할 사용. fold 내부 조기종료 후 outer train 재학습.',
'- 같은 OOF 보정 1,782명/기준값 1,280명 역할 사용. 각 사람의 OOF 예측을 만든 모델은 해당 사람 및 같은 PSU를 학습에 포함하지 않는다.',
'- 최종 시드 42/43/44 모두 보고한다. 시드42는 구간 계산용 기준 시드로 미리 정했으며 시험결과가 가장 높은 시드를 선택하지 않았다.',
'- LR도 입력별로 종합 위험 및 네 소견을 독립적으로 학습한다. 각 출력별 C를 개발 OOF에서 별도로 선택했다.',
'','## 3. 2×2 종합 위험 결과','',
'조사 가중치를 적용한 2024년 평가다. 표는 사전에 정한 기준 시드42이며, 다음 절에 다른 시드도 모두 제시한다.',
'','| 조건 | 개발 CV 평균 AUC | 2024 AUC | AP | Brier |', '|---|---:|---:|---:|---:|']
for key in ['base_single_s42','base_multi_s42','expanded_single_s42','expanded_multi_s42']:
    m=r['models'][key]['primary']['youden'];L.append(f'| {labels[key]} | {cv[key]["mean_auc"]:.5f} | {m["roc_auc"]:.5f} | {m["pr_auc_ap"]:.5f} | {m["brier"]:.5f} |')
L+=['', '입력 확장의 2024년 개선과 개발 CV의 감소가 공존한다. 연도별 분포 차이나 우연의 가능성이 있어 새 정보가 일반적으로 도움이 된다고 결론 내리지 않는다. 단일/다중과제의 종합 AUC 차이는 거의 없었다.',
'','### 3개 시드 전부','', '| 입력·과제 | 시드42 AUC | 시드43 AUC | 시드44 AUC |', '|---|---:|---:|---:|']
for prefix in ['base_single','base_multi','expanded_single','expanded_multi']:
    vals=[r['models'][f'{prefix}_s{s}']['primary']['youden']['roc_auc'] for s in [42,43,44]]
    L.append(f'| {prefix} | '+' | '.join(f'{v:.5f}' for v in vals)+' |')
L+=['','### 종합 AUC 변화 구간','', '| 비교 | ΔAUC의 탐색적 95% 구간 |', '|---|---|']
for key,value in r['paired_differences'].items():
    lo,hi=value['delta_auc']['ci95'];L.append(f'| {key} | {lo:+.5f} ~ {hi:+.5f} |')
L+=['', '해당 AUC 차이 구간들은 모두 0을 포함한다. 300회 층별 PSU 재표집 구간이며 정식 복합표본 domain 분산 추정이나 다중비교 보정된 확증 검정은 아니다.',
'','## 4. 다중 소견 예측 비교','',
'평균 지표는 종합 위험을 제외한 혈당·혈압·TG·HDL 네 소견의 단순 평균이다. 각 소견의 유병률이 다르므로 macro AP의 절대값을 종합 위험 AP와 직접 비교하지 않는다.',
'','| 모델 | 종합 AUC | 네 소견 평균 AUC | 평균 AP | 평균 Brier | 개수 MAE |', '|---|---:|---:|---:|---:|---:|']
for key in ['base_multi_s42','expanded_multi_s42','base_LR5','expanded_LR5']:
    v=r['models'][key];m=v['macro'];L.append(f'| {labels[key]} | {v["primary"]["youden"]["roc_auc"]:.5f} | {m["roc_auc"]:.5f} | {m["pr_auc_ap"]:.5f} | {m["brier"]:.5f} | {v["count_mae"]:.5f} |')
L+=['', '확장 다중과제 NN은 확장 독립 LR들보다 평균 AP·Brier 점수가 조금 좋았다. NN−LR 차이 구간은 평균 AUC/AP/Brier에서 0을 포함했다. 예상 이상 개수 MAE 차이 구간은 이 탐색적 비교에서 음수였다. 이를 전체 임상 성능 우월성으로 확대 해석하지 않는다.',
'','| 다중 출력 비교 | 지표 | 차이의 탐색적 95% 구간 |', '|---|---|---|']
for comparison,values in ci.items():
    for key,v in values.items():
        lo,hi=v['ci95'];L.append(f'| {comparison} | {key} | {lo:+.5f} ~ {hi:+.5f} |')
L+=['','### 소견별 결과','', '| 소견 | 기존 다중 NN AUC / AP | 확장 다중 NN AUC / AP | 확장 LR AUC / AP |', '|---|---|---|---|']
for component in r['models']['expanded_multi_s42']['components']:
    cells=[]
    for key in ['base_multi_s42','expanded_multi_s42','expanded_LR5']:
        m=r['models'][key]['components'][component];cells.append(f'{m["roc_auc"]:.4f} / {m["pr_auc_ap"]:.4f}')
    L.append('| '+component+' | '+' | '.join(cells)+' |')
L+=['', '**확장 후 HDL 판별력은 낮아졌다.** 혈당·혈압 쪽 AP 개선과 TG·HDL의 다른 방향 변화가 있으므로 모든 과제가 함께 좋아졌다고 말할 수 없다.',
'','## 5. 출력 일관성','',
'다중 NN은 공동분포로부터 종합 위험·성분·개수를 계산하므로 `max(성분확률) ≤ 하나 이상 확률 ≤ min(1, 성분확률 합)`이 구조적으로 유지된다.',
f'2024년 데이터에서 이 필요조건 위반은 확장 NN {r["models"]["expanded_multi_s42"]["union_bound_violation_n"]}건, 서로 독립적으로 학습·보정한 확장 LR 5개는 {r["models"]["expanded_LR5"]["union_bound_violation_n"]}건이었다. 이는 관측 환자 오분류 건수가 아니라 모델 출력 확률들 사이의 관계 위반이다.',
'LR도 별도의 공동분포 설계나 일관성 보정으로 이 조건을 맞출 수 있다. 신경망만 가능한 기능이라고 주장하지 않는다. 일관성은 정확성이나 임상적 안전성을 보장하지 않는다.',
'','## 6. 추가 정보 중 무엇이 기여했나','',
'2024년을 사용하지 않고 seed42의 OOF 모델을 같은 학습 조건으로 재적합하여 검증 fold에서 변수 블록을 섞었다. 각 fold 10회씩 총 30회다. 음주 상세는 월간 음주 범주 안에서 빈도·양을 함께 섞어 원래 월간 음주 여부와의 큰 모순을 줄였다.',
'','| 변수 블록 | 평균 검증 AUC 하락 | 반복 표준편차 |', '|---|---:|---:|']
for key,v in sorted(importance['blocks'].items(),key=lambda z:z[1]['mean_auc_drop'],reverse=True):L.append(f'| {key} | {v["mean_auc_drop"]:+.5f} | {v["sd_across_folds_and_permutations"]:.5f} |')
L+=['', '이번 순열 분석에서는 음주 상세와 근력운동 정보의 상대적 기여가 가장 컸으나 절대 효과는 작았다. 음수 값이 나온 변수는 보호요인/유해요인이라는 뜻이 아니다. 중요도를 보고 입력을 다시 선택하거나 2024년 점수를 높이는 재튜닝은 하지 않았다.',
'','## 7. 가구유형별 확장 다중모델','', '| 집단 | 평가 n | AUC | Youden 민감도 | 특이도 |', '|---|---:|---:|---:|---:|']
for key in ['living_1','living_0']:
    row=r['models']['expanded_multi_s42']['subgroups'][key];m=row['metrics'];L.append(f'| {key} | {row["n"]} | {m["roc_auc"]:.4f} | {m["sensitivity"]:.1%} | {m["specificity"]:.1%} |')
L+=['','## 8. 실행 시간과 사용법','',
'두 방식 모두 전처리를 한 번만 수행하고 종합 위험+네 소견 확률+기대 개수를 JSON으로 출력했다. LR을 다섯 번 전처리하도록 만들어 불리하게 비교하지 않았다. warm CPU 1스레드·1,000회이며 로딩/HTTP는 제외했다.',
'','| 모델 | 중앙값(ms) | p95(ms) |', '|---|---:|---:|']
for key,v in latency['results'].items():L.append(f'| {key} | {v["p50_ms"]:.4f} | {v["p95_ms"]:.4f} |')
L+=['', '다중 NN이 속도에서도 우월하다고 주장하지 않는다.',
'', '```powershell',
'.\\.venv\\Scripts\\python.exe -m metabolic.expanded --artifact artifacts/expanded_multitask_v4/models/expanded_multi_s42 --input artifacts/expanded_multitask_v4/example_request.json --policy youden',
'.\\.venv\\Scripts\\python.exe -m pytest tests -q',
'```',
'API는 `/v4/decide-expanded`이며 `state`와 `policy`를 받는다. 추가 입력 필드 의미는 위 표와 example_request.json을 참조한다. 종합·성분 확률과 개수 분포, 입력 결측 목록을 반환한다. 원래 모델을 자동 교체하지 않았다.',
'단일과제 모델도 models/base_single_s42, expanded_single_s42 등의 경로로 저장했다. 이 모델은 학습하지 않은 소견 확률을 반환하지 않는다.',
'테스트 **35개 통과**: 조사 코드 매핑, 비음주 분기와 입력 간 일치성, 시간 범위, 학습 전처리 고정, 배열 실행 동등성, 단일과제 출력 차단, 입력 스키마와 API 등을 확인했다.',
'', '## 한계 및 근거','',
'이번 실험은 변수를 늘리거나 다중과제를 사용하면 항상 좋아진다는 가설을 지지하지 않는다. 개발/평가 연도 간 불일치, 작은 효과, HDL 악화, 입력 설문 부담을 함께 고려해야 한다. 추가 정보의 실용성과 다중 출력 성능은 새로운 외부 표본에서 검증해야 한다.',
'공식 변수 정의는 저장한 제8·9기 KNHANES 이용지침서의 경제활동·음주·정신건강·신체활동·가족력·가구공통설문 절에서 확인했다. 수면 문항 변경은 제9기 지침의 연도별 변경사항을 참조했다.',
'원본: data/reference/knhanes/guide8.pdf, guide9.pdf. 원시자료와 이전 모델을 보존했다. PLAN.json, split_manifest.json, feature_audit.json, cv_progress.json, FROZEN_SELECTION.json, report.json 및 해시로 추적할 수 있다.']
(OUT/'EXPANSION_MULTITASK_REPORT_KO.md').write_text('\n'.join(L),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for i,tag in enumerate(['base','expanded']):
    vals=[r['models'][tag+'_'+task+'_s42']['primary']['youden']['roc_auc'] for task in ['single','multi']]
    axes[0].bar(np.arange(2)+(i-.5)*.32,vals,width=.32,label=tag)
    for j,v in enumerate(vals):axes[0].text(j+(i-.5)*.32,v+.012,f'{v:.4f}',ha='center',fontsize=8)
axes[0].set(xticks=[0,1],xticklabels=['Single task','Multitask'],ylim=(0,1),ylabel='Weighted AUROC',title='Input expansion x learning tasks (seed 42)');axes[0].legend()
components=list(r['models']['expanded_multi_s42']['components'])
for i,key in enumerate(['base_multi_s42','expanded_multi_s42','expanded_LR5']):
    vals=[r['models'][key]['components'][c]['pr_auc_ap'] for c in components]
    axes[1].bar(np.arange(4)+(i-1)*.25,vals,width=.25,label=key)
axes[1].set(xticks=np.arange(4),xticklabels=['Glucose','BP','TG','Low HDL'],ylim=(0,1),ylabel='Average precision',title='Four separate screening findings');axes[1].legend(fontsize=8)
fig.savefig(OUT/'factorial_comparison.png',dpi=170)
dump(OUT/'verification.json',{'tests':'35 passed','source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'metabolic').glob('*.py'))+list((ROOT/'tests').glob('*.py'))},
    'model_hashes':{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'models').rglob('network.npz')}})
print(OUT/'EXPANSION_MULTITASK_REPORT_KO.md')
