from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import traceback
from predict import predict_from_minio, get_model
from prometheus_fastapi_instrumentator import Instrumentator

model_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_ready
    print("startup: 모델 로딩 시작")
    get_model()
    model_ready = True
    print("startup: 모델 로딩 완료")
    yield
    print("shutdown")

app = FastAPI(lifespan=lifespan)

# 지연 히스토그램(http_request_duration_seconds, handler별) 버킷. 기본값(0.1, 0.5, 1)으로는 1초 넘는 추론 지연을 구분할 수 없어
# 알림(AIInferenceSlow)·SLO·KEDA 스케일링이 쓸 수 있도록 10초까지 넓힌다
Instrumentator().instrument(app, latency_lowr_buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 10)).expose(app)


class PredictRequest(BaseModel):
    bucket_name: str
    object_key: str


@app.get("/health")
def health():
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "UP"}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        predicted_class, confidence = predict_from_minio(request.bucket_name, request.object_key)
        return {
            "prediction": predicted_class,
            "confidence": confidence
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))