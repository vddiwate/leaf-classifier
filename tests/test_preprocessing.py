"""Unit Tests for Preprocessing Module (src/preprocessing.py).
Tests input validations, resizing with aspect-ratio preservation,
blurring, CLAHE, and HSV color space conversion.
"""

import numpy as np
import pytest

from src.config import PreprocessingConfig
from src.preprocessing import Preprocessor


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """Creates a sample 3-channel uint8 image (H=100, W=200, C=3)."""
    # Create gradient image with dominant green
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :, 0] = 50   # Blue
    img[:, :, 1] = 180  # Green
    img[:, :, 2] = 40   # Red
    return img


def test_preprocessing_success(sample_bgr_image: np.ndarray):
    """Verifies standard preprocessing flow with expected shapes, scale factors, and HSV ranges."""
    cfg = PreprocessingConfig(target_width=300, blur_kernel=(5, 5), blur_sigma=0.0)
    preprocessor = Preprocessor(cfg)

    resized_bgr, hsv_img, scale_factor = preprocessor.process(sample_bgr_image)

    # 1. Check dimensions
    expected_scale = 300.0 / 200.0  # 1.5
    expected_height = int(round(100 * expected_scale))  # 150
    assert resized_bgr.shape == (expected_height, 300, 3)
    assert hsv_img.shape == (expected_height, 300, 3)
    assert scale_factor == pytest.approx(expected_scale)

    # 2. Check types
    assert resized_bgr.dtype == np.uint8
    assert hsv_img.dtype == np.uint8

    # 3. Check HSV channel value ranges in OpenCV
    # H: 0-180, S: 0-255, V: 0-255
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    assert np.all((h >= 0) & (h <= 180))
    assert np.all((s >= 0) & (s <= 255))
    assert np.all((v >= 0) & (v <= 255))


def test_preprocessing_clahe_enabled(sample_bgr_image: np.ndarray):
    """Verifies preprocessing with CLAHE illumination correction enabled."""
    cfg = PreprocessingConfig(
        target_width=256,
        apply_clahe=True,
        clahe_clip_limit=3.0,
        clahe_grid_size=(8, 8),
    )
    preprocessor = Preprocessor(cfg)
    resized_bgr, hsv_img, scale = preprocessor.process(sample_bgr_image)

    assert resized_bgr.shape[1] == 256
    assert hsv_img.shape == resized_bgr.shape
    assert hsv_img.dtype == np.uint8


def test_invalid_preprocessor_init():
    """Rejects invalid configuration object during Preprocessor initialization."""
    with pytest.raises(TypeError, match="Expected PreprocessingConfig"):
        Preprocessor("not_a_config")


def test_input_validation_none():
    """Rejects None image input with TypeError."""
    preprocessor = Preprocessor(PreprocessingConfig())
    with pytest.raises(TypeError, match="cannot be None"):
        preprocessor.process(None)


def test_input_validation_not_numpy():
    """Rejects non-numpy array inputs with TypeError."""
    preprocessor = Preprocessor(PreprocessingConfig())
    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        preprocessor.process([[1, 2, 3], [4, 5, 6]])


def test_input_validation_wrong_dtype():
    """Rejects non-uint8 images with TypeError."""
    preprocessor = Preprocessor(PreprocessingConfig())
    float_img = np.zeros((50, 50, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="Expected image dtype uint8"):
        preprocessor.process(float_img)


def test_input_validation_empty_array():
    """Rejects empty image array with ValueError."""
    preprocessor = Preprocessor(PreprocessingConfig())
    empty_img = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="is empty"):
        preprocessor.process(empty_img)


def test_input_validation_wrong_dimensions():
    """Rejects 2D grayscale or 4D image arrays with ValueError."""
    preprocessor = Preprocessor(PreprocessingConfig())
    gray_img = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected 3-dimensional"):
        preprocessor.process(gray_img)


def test_input_validation_wrong_channels():
    """Rejects 4-channel BGRA or 1-channel 3D images with ValueError."""
    preprocessor = Preprocessor(PreprocessingConfig())
    bgra_img = np.zeros((50, 50, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected 3-channel BGR"):
        preprocessor.process(bgra_img)
