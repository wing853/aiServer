# 🤖 GreenLens AI Server

> GreenLens 분리수거 안내 서비스의 AI 분석 서버  
> YOLO 기반 객체 인식 모델을 Flask 서버로 제공

- 🔗 **메인 레포**: [https://github.com/wing853/recycle](https://github.com/wing853/recycle)
- 🌐 **배포 링크**: [https://aiserver-1-cpjb.onrender.com](https://aiserver-1-cpjb.onrender.com)

<br>

## ⚙️ 기술 스택

| 항목 | 기술 |
|:---:|:---:|
| Server | Flask, Python |
| AI Model | YOLOv8n-seg |
| Fallback | Gemini Vision API |
| Deploy | Render, Docker |

<br>

## 🤖 AI 모델

- **모델**: YOLOv8n-seg (best.pt)
- **데이터셋**: AI Hub 제공 이미지 약 20만장
- **분류 클래스**: 7개
  - 페트병, 플라스틱, 캔류, 유리병, 종이류, 비닐류, 스티로폼
- **학습**: epoch 50, 로컬 GPU 환경
- **정확도**: 초기 65% → 최종 95~99% (클래스별)

<br>

## 📡 API

### 헬스 체크
```
GET /
Response: "✅ AI Server is Live and Model is Loaded!"
```

### 이미지 분석
```
POST /recycle/analyze
Content-Type: multipart/form-data

Request: image (file)
Response: {
  "category": "페트병",
  "confidence": 0.9500,
  "disposal_method": "페트병 전용 수거함에 버려주세요."
}
```

<br>

## 🔥 주요 구현 사항

- **모델 Fallback 전략**: `best_fix.pt` 우선 로드 → 실패 시 `best.pt`로 fallback
- **Lazy Loading**: 전역 모델 인스턴스를 lazy loading 방식으로 관리하여 서버 시작 시 불필요한 로드 방지
- **신뢰도 기반 폴백**: YOLO 신뢰도가 낮을 경우 Gemini Vision API로 폴백하여 일관된 응답 품질 보장
- **임시 파일 처리**: 이미지 분석 후 임시 파일 자동 삭제

<br>

## 🐳 Docker

```dockerfile
# 빌드
docker build -t greenlens-ai .

# 실행
docker run -p 5000:5000 greenlens-ai
```

<br>

## 🔥 트러블 슈팅

### Flask AI 서버 이미지 분석 불가
- **문제**: 배포 환경 변경 후 PyTorch, Ultralytics, Flask 의존성 버전 충돌 및 모델 로드 실패
- **1차 시도 (실패)**: 의존성 버전 명시적 고정 → 문제 해결되지 않음
- **2차 시도 (성공)**: Flask 서버 코드 전면 개선 (lazy loading, fallback 전략, 임시 파일 방식 이미지 처리)

<br>

## 📅 개발 기간

**2025.02 ~ 2025.08**
