import io

import numpy as np
import pytest
from PIL import Image

from preprocess import IMAGE_SIZE, InvalidWaferImage, decode_wafer_map, preprocess_image


def png_bytes(array: np.ndarray, mode: str = "L") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode).save(buffer, format="PNG")
    return buffer.getvalue()


def wafer_pixels(size: int = IMAGE_SIZE) -> np.ndarray:
    pixels = np.full((size, size), 127)
    pixels[0, :] = 0
    pixels[size // 2, size // 2] = 255
    return pixels


def test_levels_decode_to_wafer_map_values():
    wafer_map = decode_wafer_map(png_bytes(wafer_pixels()))

    assert wafer_map[0, 0] == 0
    assert wafer_map[1, 1] == 1
    assert wafer_map[IMAGE_SIZE // 2, IMAGE_SIZE // 2] == 2


def test_preprocess_returns_one_hot_channels():
    batch = preprocess_image(png_bytes(wafer_pixels()))

    assert batch.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert np.all(batch.sum(axis=-1) == 1)
    assert batch[0, 0, 0].tolist() == [1, 0, 0]
    assert batch[0, IMAGE_SIZE // 2, IMAGE_SIZE // 2].tolist() == [0, 0, 1]


def test_rgb_png_with_same_levels_is_accepted():
    rgb = np.repeat(wafer_pixels()[..., None], 3, axis=2)

    assert decode_wafer_map(png_bytes(rgb, "RGB"))[1, 1] == 1


def test_larger_map_is_resized_without_mixing_levels():
    pixels = np.full((52, 52), 127)
    pixels[:26, :] = 255

    assert set(np.unique(decode_wafer_map(png_bytes(pixels)))) == {1, 2}


def test_small_encoding_error_is_tolerated():
    pixels = wafer_pixels()
    pixels[5, 5] = 133

    assert decode_wafer_map(png_bytes(pixels))[5, 5] == 1


def test_continuous_grayscale_is_rejected():
    gradient = np.tile(np.linspace(0, 255, IMAGE_SIZE), (IMAGE_SIZE, 1))

    with pytest.raises(InvalidWaferImage):
        decode_wafer_map(png_bytes(gradient))


def test_non_image_bytes_are_rejected():
    with pytest.raises(InvalidWaferImage):
        preprocess_image(b"not an image")
