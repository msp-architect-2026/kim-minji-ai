import os
from pathlib import Path

import pytest

from preprocess import preprocess_image

FIXTURES = Path(__file__).parent / "fixtures"
MODEL_PATH = os.getenv("MODEL_PATH")


# 전처리가 학습 입력과 달라지면 모든 입력이 한 클래스로 몰린다. 실제 모델로 클래스별 샘플을 확인한다 (모델 파일이 있을 때만 실행)
@pytest.mark.skipif(not MODEL_PATH or not os.path.exists(MODEL_PATH), reason="MODEL_PATH에 모델 파일이 없습니다")
@pytest.mark.parametrize("image", sorted(FIXTURES.glob("*.png")), ids=lambda p: p.stem)
def test_model_classifies_fixture(image, model):
    expected = image.stem.removeprefix("sample_")
    from predict import CLASS_NAMES

    prediction = model.predict(preprocess_image(image.read_bytes()), verbose=0)[0]

    assert CLASS_NAMES[int(prediction.argmax())] == expected


@pytest.fixture(scope="module")
def model():
    from tensorflow.keras.models import load_model

    return load_model(MODEL_PATH)
