# 원자료 확보 및 검증

원자료는 저장소에 재배포하지 않는다. 질병관리청 공식 페이지에서 이용조건을 확인하고 사용자 등록 후 직접 받는다.

- [국민건강영양조사 원시자료 다운로드](https://knhanes.kdca.go.kr/knhanes/rawDataDwnld/rawDataDwnld.do)
- [원시자료 이용지침서](https://knhanes.kdca.go.kr/knhanes/dataAnlsGd/utztnGd.do)

필요한 파일은 **각 연도의 기본DB SPSS ZIP**이다. 식품섭취조사 `24RC` 파일이 아니다.

```text
data/raw/knhanes/
  HN21_ALL(SPSS).zip
  HN22_ALL(SPSS).zip
  HN23_ALL(SPSS).zip
  HN24_ALL(SPSS).zip
```

```powershell
New-Item -ItemType Directory -Force data/raw/knhanes
# 위 폴더에 공식 다운로드 파일을 복사한 뒤 실행
python -m scripts.validate_knhanes
```

검증기는 ZIP CRC와 SPSS 파싱을 확인하고 연도별 `.sav`를 로컬에 추출한다. 결과는 `data/validation/` 및 `data/knhanes_manifest.json`에 생성한다.
이번 실험에 사용한 파일 크기·SHA-256·행/열 수는 [공개 데이터 매니페스트](../results/data_manifest.json)를 참조한다.
사이트에서 파일이 수정되었거나 다른 형식을 받은 경우 수치가 달라질 수 있으므로, 파일명을 바꿔 동일 자료인 것처럼 간주하지 말고 해시·코딩·대상자 수를 확인한다.

## 처리 원칙

- 19–39세, 세 질환 의사진단 없음, 전용 복약 변수로 관련 약물·인슐린 미사용 확인.
- 임신개월수 비기록, 12시간 이상 공복, 목표 검사값 완비, 유효한 건강설문·검진 가중치와 PSU/층을 요구한다.
- 목표 검사값의 결측을 정상으로 처리하지 않는다. 높은 혈당/TG 같은 실제 이상값을 임의 제거하지 않는다.
- 예측변수의 대치·표준화·파생 기저 매듭은 해당 학습 fold에서만 적합한다.
- `cfam=1`은 1인 가구 대리변수다. 자취·룸메이트·부모와 별거 여부의 직접 측정이 아니다.
- 가족력은 `HE_fh`의 만성질환 가족력 범위다. 개인 진단 여부를 예측 입력으로 사용하지 않는다.
- 영양조사 변수를 새로 추가하려면 조사 참여자와 가중치 선택을 다시 검토해야 한다.

원시자료, 개인별 예측값, OOF 예측, 개인별 분할 해시, 체크포인트는 `.gitignore`로 제외된다. 개인정보 등록 이메일을 코드나 설정 파일에 저장할 필요가 없다.
