import os
import io
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import boto3
from botocore.client import Config

MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/wafer_defect_model.h5")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "wafer-model")
MODEL_KEY = os.getenv("MODEL_KEY", "wafer_defect_model.h5")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio.storage.svc.cluster.local:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

model = None


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from MinIO: {MODEL_BUCKET}/{MODEL_KEY}")
        s3 = get_s3_client()
        s3.download_file(MODEL_BUCKET, MODEL_KEY, MODEL_PATH)
        print(f"Model downloaded to {MODEL_PATH}")


def get_model():
    global model
    if model is None:
        download_model_if_needed()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        print(f"Loading model from {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
    return model



def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((26, 26))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 26, 26, 3)
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