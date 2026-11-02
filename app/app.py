from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import time
import traceback
from predict import predict_from_minio, get_model
from prometheus_fastapi_instrumentator import Instrumentator

model_ready = False

# 장애 주입(Istio Outlier Detection 실험용). 켜진 Pod만 /predict가 503을 반환하고 /health는 그대로 200이라
# Readiness는 정상인데 추론만 실패하는 상황을 재현한다. 상태는 프로세스 메모리에만 있어 호출한 Pod에만 적용된다
FAULT_INJECTION_ENABLED = os.getenv("FAULT_INJECTION_ENABLED", "false").lower() == "true"
fault = {"enabled": False, "delay_seconds": 0.0}

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
# 알림(AIInferenceSlow)·SLO가 쓸 수 있도록 10초까지 넓힌다
Instrumentator().instrument(app, latency_lowr_buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 10)).expose(app)


class PredictRequest(BaseModel):
    bucket_name: str
    object_key: str


class FaultRequest(BaseModel):
    enabled: bool
    # 503을 반환하기 전에 기다릴 시간(초). 0이면 바로 실패
    delay_seconds: float = 0.0


@app.get("/health")
def health():
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "UP"}


@app.get("/admin/fault")
def get_fault():
    if not FAULT_INJECTION_ENABLED:
        raise HTTPException(status_code=404)
    return fault


@app.post("/admin/fault")
def set_fault(request: FaultRequest):
    if not FAULT_INJECTION_ENABLED:
        raise HTTPException(status_code=404)
    fault["enabled"] = request.enabled
    fault["delay_seconds"] = max(0.0, request.delay_seconds)
    print(f"fault injection: {fault}")
    return fault


@app.post("/predict")
def predict(request: PredictRequest):
    if fault["enabled"]:
        if fault["delay_seconds"]:
            time.sleep(fault["delay_seconds"])
        raise HTTPException(status_code=503, detail="Injected fault")
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