<h1 align="center">kim-minji-ai</h1>

<p align="center">
  웨이퍼 결함 탐지 시스템의 <strong>FastAPI 기반 AI 추론 서비스</strong> 레포지토리입니다.
</p>


<p align="center">
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/MinIO-C72E49?style=for-the-badge&logo=minio&logoColor=white"/>
<img src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white"/>
</p>

<br>

## ▍개요

WM-811K 웨이퍼 맵 데이터셋으로 학습한 **CNN 분류 모델을 FastAPI로 서빙**합니다.

외부 AI API에 의존하지 않고 클러스터 내부에서 직접 추론하므로 공정 데이터의 외부 유출이 없습니다. 모델 파일(533MB)은 컨테이너 이미지에 포함하지 않고 pod 기동 시 MinIO에서 다운로드해 메모리에 로드합니다.

<br>

## ▍관련 레포지토리

| Repository | 설명 |
|------------|------|
| [kim-minji-wiki(https://github.com/msp-architect-2026/kim-minji-wiki) | 프로젝트 메인 (Wiki, 칸반보드) |
| [kim-minji-backend](https://github.com/msp-architect-2026/kim-minji-backend) | Spring Boot API 서버 |
| [kim-minji-frontend](https://github.com/msp-architect-2026/kim-minji-frontend) | React 웹 대시보드 |
| [kim-minji-helm](https://github.com/msp-architect-2026/kim-minji-helm) | Kubernetes Helm Chart |
| [kim-minji-infra](https://github.com/msp-architect-2026/kim-minji-infra) | k3s 클러스터 및 GitOps 인프라 |

<br>

## ▍분류 클래스 (9종)

| 클래스 | 설명 |
|--------|------|
| `Center` | 중앙 결함 |
| `Donut` | 도넛형 결함 |
| `Edge-Loc` | 엣지 로컬 결함 |
| `Edge-Ring` | 엣지 링 결함 |
| `Loc` | 로컬 결함 |
| `Near-full` | 거의 전체 결함 |
| `Random` | 무작위 결함 |
| `Scratch` | 스크래치 결함 |
| `none` | 정상 (결함 없음) |

<br>

## ▍API 엔드포인트

### GET `/health`
모델 로딩 완료 여부를 반환합니다. Readiness Probe가 이 엔드포인트를 체크합니다.

- 모델 로딩 완료 → `200 OK` `{ "status": "UP" }`
- 모델 로딩 중 → `503 Service Unavailable`

### POST `/predict`
백엔드(`AiAnalysisService`)가 내부적으로 호출합니다.

```json
// Request
{ "bucket_name": "wafer-images", "object_key": "wafer-images/2026/03/07/uuid.png" }

// Response
{ "prediction": "Edge-Loc", "confidence": 0.9873 }
```

### GET `/metrics`
Prometheus 메트릭 수집 엔드포인트 (`prometheus_fastapi_instrumentator`)

<br>

## ▍모델 Lazy Loading 흐름

```
FastAPI lifespan startup
└─ get_model() 호출
   └─ /tmp/wafer_defect_model.h5 없으면
      └─ MinIO wafer-model 버킷에서 다운로드 (533MB)
         └─ Keras load_model() → 전역 model 변수 싱글턴 캐싱
            └─ model_ready = True
               └─ /health 200 반환 → Readiness Probe 통과 → Ready
```

pod 기동부터 Ready 전환까지 **약 79초** 소요. 이후 요청은 메모리의 모델을 바로 사용합니다.

<br>

## ▍추론 파이프라인

```
POST /predict 수신
└─ MinIO에서 이미지 다운로드 (boto3 s3 client)
   └─ preprocess_image()
      └─ PIL Image.open() → RGB 변환
         → resize(26, 26)
         → np.array / 255.0  (0~1 정규화)
         → reshape(1, 26, 26, 3)
            └─ model.predict()
               └─ argmax → 클래스명, max → confidence 반환
```

<br>

## ▍AI 모델 상세

### 사용 데이터셋
WM-811K (LSWMD), 26×26 크기 웨이퍼 맵 14,366개 샘플

### CNN 분류기 구조

```
Input: (26, 26, 3)
→ Conv2D(16, 3×3, relu, same)
→ Conv2D(64, 3×3, relu, same)
→ Conv2D(128, 3×3, relu, same)
→ Flatten
→ Dense(512, relu)
→ Dense(128, relu)
→ Dense(9, softmax)
Total params: 44,453,259 (169.58 MB)
```

> Pooling 미사용: 26×26 소형 입력에서 Pooling 시 위치 정보 손실 → 결함 패턴 분류 정확도 저하

### 데이터 증강 (AutoEncoder 기반)

일반 flip/rotate 증강 대신 AutoEncoder의 latent space에 가우시안 노이즈를 주입해 synthetic 웨이퍼 이미지를 생성, 클래스 불균형을 완화했습니다.

### 학습 설정

| 항목 | 값 |
|------|----|
| Optimizer | Adam |
| Loss | categorical_crossentropy |
| 검증 | 3-Fold Cross Validation |

<br>

## ▍모델 성능

| 지표 | 값 |
|------|----|
| Accuracy | **99%** |
| Macro avg F1-score | 0.97 |
| Weighted avg F1-score | 0.99 |

| 클래스 | F1 |
|--------|----|
| Center | 0.96 |
| Donut | 1.00 |
| Edge-Loc | 0.95 |
| Edge-Ring | 0.97 |
| Loc | 0.94 |
| Near-full | 1.00 |
| Random | 1.00 |
| Scratch | 0.94 |
| none | 1.00 |

> 단순 grayscale→RGB 복사 방식 사용 시 accuracy 약 2% 저하 발생. 반드시 PIL RGB 변환 + `/255.0` 정규화 방식을 사용해야 합니다.

<br>

## ▍환경변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `MODEL_PATH` | `/tmp/wafer_defect_model.h5` | 모델 파일 로컬 캐시 경로 |
| `MODEL_BUCKET` | `wafer-model` | 모델 파일 MinIO 버킷 |
| `MODEL_KEY` | `wafer_defect_model.h5` | MinIO object key |
| `MINIO_ENDPOINT` | `http://minio-minio.storage.svc.cluster.local:9000` | MinIO 엔드포인트 |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO 액세스 키 |
| `MINIO_SECRET_KEY` | `minioadmin123` | MinIO 시크릿 키 |

<br>

## ▍Dockerfile

```dockerfile
FROM python:3.11-slim
# 단일 스테이지, uvicorn으로 8000 포트 서빙
# --platform=linux/arm64 (맥북 M5 기반 VM 대응)
```

<br>

## ▍Helm Chart 스펙

| 항목 | 값 |
|------|----|
| replicaCount | 1 (HPA 최대 4개) |
| CPU requests / limits | 500m / 1500m |
| Memory requests / limits | 1Gi / 3Gi |
| nodeSelector | `k3s-ai2` (AI 전용 노드 고정) |
| HPA | minReplicas 1 / maxReplicas 4 / CPU 70% 기준 |
| Readiness | `/health`, initialDelay 60s, period 10s, failureThreshold 12 |
| Liveness | `/health`, initialDelay 120s, period 30s, failureThreshold 6 |

<br>

## ▍Readiness Probe 강화

HPA 스케일아웃 시 모델 로딩 완료 전 트래픽이 유입되는 문제를 개선했습니다.

```python
# 기존: 항상 200 반환
@app.get("/health")
def health():
    return {"status": "UP"}

# 개선: model_ready 플래그 체크
@app.get("/health")
def health():
    if not model_ready:
        raise HTTPException(status_code=503)
    return {"status": "UP"}
```

모델 로딩 완료 후에만 `200`을 반환해 Readiness Probe 통과 → Ready 전환 → 트래픽 수신 흐름을 보장합니다.

<br>

## ▍CI/CD 파이프라인

```
git push (main 브랜치)
└─ GitLab CI 트리거
   └─ kaniko ARM64 이미지 빌드
      └─ GitLab Registry push (SHA + latest 태그)
         └─ update-helm.sh → kim-minji-helm values.yaml 태그 업데이트
            └─ ArgoCD 감지 → k3s 자동 배포
```

<br>

## ▍모니터링

`prometheus_fastapi_instrumentator`로 `/metrics` 엔드포인트를 노출합니다.

| 메트릭 | PromQL |
|--------|--------|
| 추론 요청 처리율 | `rate(http_requests_total{job="ai-serving-ai-serving"}[1m])` |
| 평균 응답시간 | `rate(http_request_duration_seconds_sum[1m]) / rate(..._count[1m])` |

<br>

