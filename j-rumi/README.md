# 파운데이션 색상 추천 웹앱

색상카드(Calibrite ColorChecker Classic) + 얼굴을 같이 촬영한 사진에서
카메라/조명 왜곡을 자동 보정하고, 등록된 파운데이션 색상 중 가장 비슷한 것을 추천해주는 앱.

## 구조

- `app.py` — 사용자용 메인 화면 (사진 촬영/업로드 → 추천)
- `pages/1_관리자.py` — 관리자 화면 (비밀번호 보호, 색상 DB 추가/삭제)
- `src/calibration.py` — 색상카드 자동 인식 + 카메라 보정 로직
- `src/skin_extraction.py` — 얼굴 인식 + 피부색 샘플링
- `src/color_match.py` — Lab 변환 + 색상 거리 계산
- `src/recommend.py` — 위 세 가지를 합친 전체 추천 파이프라인
- `src/github_storage.py` — 이 깃허브 레포 자체를 DB로 사용 (별도 DB 서비스 없음)
- `data/foundation_db.json` — 초기 색상 데이터 (배포 후에는 관리자 페이지로 관리하는 게 기본)

## Streamlit Community Cloud 배포 방법

1. share.streamlit.io 접속 → 깃허브 계정으로 로그인
2. "New app" → 이 레포(`kxlxkxlxk/j-rumi`) 선택, main file은 `app.py`
3. 배포되면 "App settings" → "Secrets" 메뉴에 `.streamlit/secrets.toml.example` 내용을 참고해서
   실제 깃허브 토큰이랑 원하는 관리자 비밀번호로 채워서 붙여넣기
   (이 secrets는 깃허브 레포에는 절대 올라가지 않고 Streamlit 서버에만 안전하게 저장됨)
4. 저장하면 앱이 자동으로 다시 시작되면서 적용됨

## 관리자 페이지 사용법

`앱주소/관리자` 로 접속 → 비밀번호 입력 → 브랜드별 색상 목록 확인/삭제 가능,
새 색상은 카드+스와치 사진(최대 5장) 올리고 스와치 부분만 박스로 선택하면 자동으로 평균 계산 후 저장.
