FROM --platform=linux/arm64 python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000
# OpenTelemetry 자동 계측(FastAPI·boto3). OTEL_* 환경변수로 켜고 끈다 (OTEL_SDK_DISABLED=true면 계측 없음)
CMD ["opentelemetry-instrument", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]