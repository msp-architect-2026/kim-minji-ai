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

Instrumentator().instrument(app).expose(app)


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