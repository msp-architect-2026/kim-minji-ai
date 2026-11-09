import io

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = 26
# 웨이퍼 맵 값 0(웨이퍼 밖), 1(양품 다이), 2(불량 다이)를 PNG 그레이스케일로 저장한 규격
PIXEL_LEVELS = np.array([0, 127, 255])
# 웨이퍼 맵은 3단계 값뿐이다. 약간의 인코딩 오차만 허용한다
LEVEL_TOLERANCE = 8


class InvalidWaferImage(ValueError):
    pass


def decode_wafer_map(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidWaferImage("이미지를 읽을 수 없습니다.") from e

    # 보간하면 값이 섞여 3단계가 깨지므로 최근접으로만 줄인다
    pixels = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)).astype(int)
    distances = np.abs(pixels[..., None] - PIXEL_LEVELS)
    if np.any(distances.min(axis=-1) > LEVEL_TOLERANCE):
        raise InvalidWaferImage("웨이퍼 맵 규격(0/127/255 3단계 PNG)이 아닙니다.")
    return distances.argmin(axis=-1)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # 모델은 웨이퍼 맵 값을 채널별로 one-hot한 (26, 26, 3) 입력으로 학습됐다
    wafer_map = decode_wafer_map(image_bytes)
    return np.eye(3)[wafer_map][np.newaxis, ...]
