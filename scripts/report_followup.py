"""Render public aggregate follow-up reports without reading participant records."""
import argparse
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    def save(name, lines):
        prefix = Path(os.path.relpath(source, destination)).as_posix()
        text = ('\n'.join(line.rstrip() for line in lines)+'\n').replace('../results/', prefix+'/')
        (destination/name).write_text(text, encoding='utf-8')
    read = lambda name: json.loads((source / name / 'report.json').read_text(encoding='utf-8'))
    screening, household, typed = map(read, ['screening_v5', 'household_v5', 'typed_local_v6_groups'])
    pct = lambda x: f'{100*x:.2f}%'
    groups = [('age20_29_living0', '20–29세 다인가구'), ('age20_29_living1', '20–29세 1인가구'),
              ('age30_39_living0', '30–39세 다인가구'), ('age30_39_living1', '30–39세 1인가구')]
    lines = ['# 조기선별 보완 실험 v5', '',
        '2026-09-21. 목표는 현재 검사 이상 소견의 선별이며 미래 발병 예측이나 진단이 아니다. '
        '2024년 평가자료를 이전 실험에서 반복 확인했으므로 이번 결과 역시 탐색적이다.', '',
        '## 설계', '',
        '학습 1,680명·검증 372명(2021–2022), 확률 보정 481명·컷오프 선정 529명(2023), '
        '평가 1,006명(2024)으로 역할을 분리했다. 보정·컷오프 자료로 재학습하지 않았다. '
        '기본/확장 입력별 NN과 LR을 비교하고 검증자료의 민감도 90%·95% 지점 평균 특이도로 후보를 선정했다. '
        '90/95는 개발자료에서의 목표이며 새 사람이나 하위집단의 민감도 보증이 아니다.', '',
        '층별 전체 PSU 틀(분석 대상이 없는 PSU 포함)을 사용한 300회 재표집으로 보정·컷오프·평가의 변동을 반영했다. '
        '각 층에서 n−1개 PSU를 복원추출하고 n/(n−1) 배율을 적용했다. '
        '모델 학습·선택의 불확실성은 포함하지 않았다. 유한모집단 보정 없는 Rao–Wu 방식의 근사이며 공식 복제 가중치는 아니다.', '',
        '## 2024년 조사 가중 성능', '',
        '| 입력·모델 | 개발 목표 | 실제 민감도 | 특이도 | 검사 의뢰율 | PPV | NPV | 1,000명당 놓침 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, model in screening['models'].items():
        for policy in ['sensitivity90', 'sensitivity95']:
            m = model['weighted'][policy]
            lines.append(f"| {name} | {policy[-2:]}% | {pct(m['sensitivity'])} | {pct(m['specificity'])} | {pct(m['referral_rate'])} | {pct(m['ppv'])} | {pct(m['npv'])} | {m['missed_per_1000']:.2f} |")
    lines += ['', '동일 입력의 NN−LR 차이에 대한 paired 구간은 주요 선별 지표에서 0을 포함했다. '
              '이번 결과로 신경망 우월성을 주장하지 않는다. v4와는 학습/보정 역할 및 모델 선정 기준이 달라 수치 변화 전체를 구조 효과로 해석할 수 없다.', '',
              '### 확장 신경망의 불확실성과 하위집단', '',
              '| 개발 목표 | 민감도 95% 구간 | 검사 의뢰율 95% 구간 |', '|---|---|---|']
    nn = screening['models']['expanded_NN']
    for policy in ['sensitivity90', 'sensitivity95']:
        ci = nn['ci95'][policy]
        fmt = lambda key: '–'.join(pct(x) for x in ci[key]['ci95'])
        lines.append(f"| {policy[-2:]}% | {fmt('sensitivity')} | {fmt('referral_rate')} |")
    lines += ['', '| 평가 집단 | n | 양성 n | 90 목표 실제 민감도 | 95 목표 실제 민감도 | 95 목표 의뢰율 |', '|---|---:|---:|---:|---:|---:|']
    for key, label in groups:
        a, b = (nn['groups'][key]['policies'][p] for p in ['sensitivity90', 'sensitivity95'])
        lines.append(f"| {label} | {a['n']} | {a['positive_n']} | {pct(a['sensitivity'])} | {pct(b['sensitivity'])} | {pct(b['referral_rate'])} |")
    lines += ['', '특히 20대 1인가구는 95 목표에서도 실제 민감도 89.34%였다. '
              '2023년 컷오프 자료의 20대·30대 1인가구 양성은 각각 12명·17명에 불과했다. '
              '개발 집단의 근거 부족을 표시하되 평가자료를 보고 집단별 컷오프를 맞추지 않았다.', '',
              '![검사 의뢰와 놓침](screening_tradeoff.png)', '',
              '## 결정곡선과 민감도 분석', '',
              '공통 위험 임계값 0.05–0.50에서 전원 검사·검사 없음과 순편익을 비교했다. '
              '이는 가정한 위해/이득 비율에 따른 계산이며 실제 임상 개입 효과나 최적 컷오프의 증명이 아니다.', '',
              '![결정곡선](decision_curve.png)', '',
              '8시간 이상 공복 코호트와 미치료 코호트는 동결 모델·동결 컷오프로만 재평가했다. '
              '전체 집계, 비가중 결과, 하위집단 구간, 대체 코호트 결과는 ',
              '[원본 집계](../results/screening_v5/report.json)에 보존했다.', '',
              '방법 참고: [결정곡선 원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC2577036/), '
              '[rescaled bootstrap 설명](https://github.com/dfeehan/surveybootstrap/blob/main/vignettes/rescaled_bootstrap.Rmd).']
    save('SCREENING_V5.md', lines)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    names = list(screening['models'])
    for ax, metric, title in zip(axes, ['referrals_per_1000', 'missed_per_1000'], ['Referred per 1,000', 'Missed findings per 1,000']):
        for j, policy in enumerate(['sensitivity90', 'sensitivity95']):
            ax.bar(np.arange(4)+(j-.5)*.35, [screening['models'][n]['weighted'][policy][metric] for n in names], .35, label=f'Development target {policy[-2:]}%')
        ax.set_xticks(range(4), names, rotation=15); ax.set_title(title); ax.legend(fontsize=7)
    fig.suptitle('Exploratory 2024 evaluation; survey weighted'); fig.savefig(destination/'screening_tradeoff.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    for name in ['refer_none', 'refer_all']+names:
        ax.plot([d['threshold_probability'] for d in screening['decision_curve']], [d[name] for d in screening['decision_curve']], label=name, linestyle='--' if name.startswith('refer') else '-')
    ax.set(xlabel='Hypothetical risk threshold', ylabel='Net benefit', title='Decision curves: hypothetical utilities, not clinical validation'); ax.legend(fontsize=8)
    fig.savefig(destination/'decision_curve.png', dpi=180); plt.close(fig)

    lines = ['# 20·30대 가구유형 보완 연구', '',
        '20–29세와 30–39세를 분리했다(총 3,934명). 19세 134명은 별도 집단이다. '
        '`cfam=1`은 1인가구, 2–6은 다인가구이며 `genertn`과의 분류 충돌은 0건이었다. '
        '**1인가구는 자취의 대리변수이며 룸메이트·기숙사·부모와 별도 거주·취사 여부를 직접 구분하지 못한다.**', '',
        '| 집단 | n | 양성 n | 조사 가중 유병 비율 | 95% 구간 |', '|---|---:|---:|---:|---|']
    for key, label in groups:
        g = household['descriptive']['groups'][key]
        ci = '–'.join(pct(x) for x in g['ci95']['ci95'])
        lines.append(f"| {label} | {g['n']} | {g['positive']} | {pct(g['weighted_prevalence'])} | {ci} |")
    lines += ['', '단순 가중 차이(1인가구−다인가구)의 구간은 두 연령대 모두 0을 포함했다. '
              '미혼자로 제한한 분석과 연령×가구×성별 집계도 저장했다.', '',
              '## 교란요인 조정', '',
              '공통 완전사례 3,718명에 조사 가중 로지스틱 연관 모형을 적합했다. '
              'A는 연령대 내 나이·성별·소득·교육·연도·혼인 경험·취업을 조정했다. '
              'B는 허리둘레·흡연·음주·유산소활동을 추가해 과잉 조정 가능성을 함께 평가했다. '
              '각 연령대의 공통 공변량 분포로 표준화한 예측 유병 비율 차이를 제시한다. '
              '300회 전체 PSU 재표집마다 GLM을 재적합했다. 단순 빈도가중 Wald p값은 사용하지 않았다.', '',
              '| 모형 | 연령 | 표준화 차이 (1인−다인, %p) | 95% 구간 (%p) |', '|---|---|---:|---|']
    for name, model in household['adjusted']['models'].items():
        for age in ['20_29', '30_39']:
            value = model['point'][age]['difference']*100
            ci = ' ~ '.join(f'{v*100:.2f}' for v in model['ci95'][age]['difference']['ci95'])
            lines.append(f'| {name} | {age} | {value:.2f} | {ci} |')
    lines += ['', '**20대의 조정 연관성이 음수여도 자취의 보호 효과를 뜻하지 않는다.** '
              '단면 관찰자료이며 선택편향, 잔여 교란, 혼인·취업 측정 한계가 있다. '
              '연령에 따른 가구 차이의 상호작용 구간도 0을 포함한다. 다중 비교는 탐색적이다.', '',
              '검사 이전 조건을 충족한 4,793명 중 공복·검사·설계정보 조건 후 3,934명(82.08%)이 남았다. '
              '가구별 잔존율이 달라 이 표본 밖 전체 청년에게 그대로 일반화할 수 없다.', '',
              '조기선별 성능은 별도 [v5 보고서](SCREENING_V5.md)를 참고한다. '
              '실제 자취 연구에는 부모 동거 여부, 별도 거주 기간, 기숙사/룸메이트, 식사 준비와 식비 정보를 직접 수집해야 한다. '
              '현재 자료로 이 항목들을 추정해 사실처럼 대체하지 않았다.', '',
              '[전체 집계와 민감도 분석](../results/household_v5/report.json)']
    save('HOUSEHOLD_V5.md', lines)

    lines = ['# 공개 구조 원칙을 적용한 로컬 결정 신경망 v6', '',
        'API 키·외부 추론·교사 모델 출력 없이 동작한다. 기존 v5 확장 신경망 가중치를 유지하고 '
        '공개 문서의 병렬·타입 제한·질문 격리 원칙을 로컬 구현에 적용했다. '
        '**가중치나 교사 출력으로 학습한 지식 증류, 비공개 Jev 구조 복제, RLCD 구현은 아니다.**', '',
        '## 구조와 출력', '',
        '`검증된 상태 → 학습 시 고정한 전처리 → 신경망 1회 → 16개 소견 조합 확률 → 질문별 결정론적 출력 → 별도 선별 정책`', '',
        '- Noul: 학습된 이진 소견의 확률만 반환한다. 별도 confidence 필드를 붙이지 않는다.',
        '- Choice: 사전 지정된 두 범주 또는 16개 조합의 사후 최빈 범주와 분포를 반환한다.',
        '- Score: 0–4개 이상 소견의 기대 개수 또는 명시한 범위로 선형 변환한 점수를 반환한다.',
        '- Choice/Score confidence: 로컬 정의 `1−H(p)/log(K)`이다. 공개되지 않은 TypeSafe 공식과 같다고 주장하지 않으며 정답 확률이나 보정 보증이 아니다.',
        '- 질문 ID·순서·개수는 신경망 입력에 들어가지 않는다. 질문 추가/삭제가 기존 답을 바꾸지 않는다. 소견 사이의 통계적 독립을 가정한다는 뜻은 아니다.',
        '- 임의 자연어 질문은 지원하지 않는다. 학습한 목표만 스키마에서 허용하고 미학습 목표·검사값 입력을 거부한다.',
        '- Choice 최빈 범주와 민감도 중심 선별 판정은 구분한다. 확률 0.3이라도 컷오프 0.2면 선별 양성일 수 있다.', '',
        '## 검증과 제한', '',
        '평가자료 1,006명 중 API 입력 계약을 만족하는 1,002명에서 기존 확률과 최대 차이 1.79×10⁻⁷, '
        '선별 판정 변화 0건이었다. 필요한 신체계측이 부족한 4명은 입력 보완 대상으로 구분했다. '
        '새 성능 향상을 주장하지 않는다. 네트워크 연결을 차단한 단위 테스트와 단일 순전파·질문 격리·API 스키마 테스트를 통과했다.', '',
        '결측, 학습 범위 밖 입력, 낮은 분포 집중도, 컷오프 근처, 개발 하위집단 양성 20명 미만 또는 '
        '개발 집단 민감도 목표 미달을 검토 사유로 표시한다. 모든 근거는 학습·검증·2023년 컷오프 자료에서 계산했다. '
        '다만 하위집단 검토 규칙 자체는 이번 탐색적 감사 후 추가했으므로 사전 검증된 정책이 아니다.', '',
        '| 정책 | 가중 검토 표시율 | 검토 대상을 모두 검사한다고 가정한 총 의뢰율 |', '|---|---:|---:|']
    for policy, values in typed['policies'].items():
        lines.append(f"| {policy} | {pct(values['weighted_review_rate'])} | {pct(values['hypothetical_test_every_review_case']['referral_rate'])} |")
    lines += ['', '**검토 대상을 전부 검사하는 정책은 거의 전원 검사로 이어져 선별 효율을 크게 잃는다.** '
              '이는 비용 점검용 가정이며 권장 자동 의뢰 정책이나 임상 개선 성과가 아니다. '
              '검토 표시는 확률·컷오프를 변경하지 않고, 음성도 의료적 정상 판정이 아니다.', '',
              '## 지연 시간', '', '| 질문 수 | p50 (ms) | p95 (ms) |', '|---|---:|---:|']
    for key, value in typed['latency'].items():
        lines.append(f"| {key} | {value['p50_ms']:.4f} | {value['p95_ms']:.4f} |")
    lines += ['', '워밍업 후 CPU 1스레드, 각 1,000회. 입력 검증·추론·정책·JSON 직렬화 포함, '
              '모델 로딩·HTTP 제외. 순차 측정 블록의 잡음 때문에 7개 질문이 1개보다 빠르다고 해석하지 않는다. '
              '외부 Jev API나 LR과 같은 환경에서 비교한 수치가 아니다.', '',
              '## 로컬 실행', '',
              '체크포인트를 먼저 생성한 뒤 `POST /v6/decide-local`을 사용한다. '
              '[가상 요청](../examples/local_v6_request.json), [가상 응답](../examples/local_v6_response.json), '
              '[집계 결과](../results/typed_local_v6_groups/report.json). 원시 참여자 사례는 공개하지 않는다.', '',
              '공식 원칙 참고: [Introduction](https://docs.typesafe.ai/introduction), '
              '[Primitives](https://docs.typesafe.ai/primitives), [Confidence](https://docs.typesafe.ai/confidence), '
              '[Patterns](https://docs.typesafe.ai/patterns).']
    save('LOCAL_STRUCTURE_V6.md', lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts')
    parser.add_argument('--destination', type=Path, default=ROOT/'docs')
    args = parser.parse_args()
    main(args.source, args.destination)
