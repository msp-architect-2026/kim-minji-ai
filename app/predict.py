import os
import io
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import boto3
from botocore.client import Config

MODEL_PATH = os.getenv("MODEL_PATH", "wafer_defect_model.h5")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.storage.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

model = None


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
    return model


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


def predict_from_minio(bucket_name: str, object_key: str):
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    image_bytes = response["Body"].read()

    model_instance = get_model()
    img_array = preprocess_image(image_bytes)
    predictions = model_instance.predict(img_array)

    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return predicted_class, confidence