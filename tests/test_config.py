"""Unit Tests for Configuration Module (src/config.py).
Tests valid configurations, boundary constraints, and invalid input rejections.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config import (
    AppConfig,
    DataConfig,
    PreprocessingConfig,
    SystemConfig,
    load_config,
)


def test_default_config_validity():
    """Verifies that default configuration values are valid and load cleanly."""
    cfg = AppConfig()
    assert cfg.system.log_level == "INFO"
    assert cfg.preprocessing.target_width == 256
    assert cfg.preprocessing.blur_kernel == (5, 5)
    assert cfg.data.min_resolution == (50, 50)


def test_valid_custom_config():
    """Verifies that valid custom overrides are accepted."""
    cfg = AppConfig(
        system=SystemConfig(log_level="DEBUG", random_seed=100),
        preprocessing=PreprocessingConfig(
            target_width=512,
            blur_kernel=(7, 7),
            blur_sigma=1.5,
            apply_clahe=True,
            clahe_clip_limit=3.0,
            clahe_grid_size=(16, 16),
        ),
    )
    assert cfg.system.log_level == "DEBUG"
    assert cfg.preprocessing.target_width == 512
    assert cfg.preprocessing.blur_kernel == (7, 7)
    assert cfg.preprocessing.apply_clahe is True


def test_invalid_log_level():
    """Rejects invalid log levels with informative error."""
    with pytest.raises(ValidationError):
        SystemConfig(log_level="INVALID_LEVEL")


def test_invalid_blur_kernel():
    """Rejects even, negative, or wrong-length blur kernels."""
    # Even dimensions
    with pytest.raises(ValidationError):
        PreprocessingConfig(blur_kernel=(4, 4))

    # Single dimension
    with pytest.raises(ValidationError):
        PreprocessingConfig(blur_kernel=(5,))

    # Negative dimension
    with pytest.raises(ValidationError):
        PreprocessingConfig(blur_kernel=(-5, 5))

    # Kernel exceeding max dimension (99)
    with pytest.raises(ValidationError):
        PreprocessingConfig(blur_kernel=(101, 101))


def test_invalid_target_width():
    """Rejects target_width out of valid range [16, 8192]."""
    with pytest.raises(ValidationError):
        PreprocessingConfig(target_width=0)

    with pytest.raises(ValidationError):
        PreprocessingConfig(target_width=10)

    with pytest.raises(ValidationError):
        PreprocessingConfig(target_width=10000)


def test_invalid_allowed_extensions():
    """Rejects empty extensions or extensions without leading dot."""
    with pytest.raises(ValidationError):
        DataConfig(allowed_extensions=[])

    with pytest.raises(ValidationError):
        DataConfig(allowed_extensions=["jpg", "png"])  # Missing dot


def test_invalid_min_resolution():
    """Rejects non-positive resolution dimensions."""
    with pytest.raises(ValidationError):
        DataConfig(min_resolution=(0, 50))

    with pytest.raises(ValidationError):
        DataConfig(min_resolution=(-10, -10))


def test_invalid_clahe_parameters():
    """Rejects invalid CLAHE parameters."""
    with pytest.raises(ValidationError):
        PreprocessingConfig(clahe_clip_limit=0.0)

    with pytest.raises(ValidationError):
        PreprocessingConfig(clahe_grid_size=(0, 8))


def test_load_config_file_not_found(tmp_path: Path):
    """Raises FileNotFoundError when config file is missing."""
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(str(missing))


def test_load_config_corrupt_yaml(tmp_path: Path):
    """Raises ValueError on YAML syntax errors."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("preprocessing:\n  target_width: [unclosed list", encoding="utf-8")
    with pytest.raises(ValueError, match="Syntax error"):
        load_config(str(bad_yaml))


def test_load_config_non_dict(tmp_path: Path):
    """Raises ValueError when YAML does not contain a mapping."""
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a key-value mapping"):
        load_config(str(list_yaml))


def test_segmentation_config_valid_and_invalid():
    """Verifies that SegmentationConfig validates HSV bounds, morph shapes, and kernel sizes."""
    from src.config import SegmentationConfig

    # Valid default
    seg_cfg = SegmentationConfig()
    assert seg_cfg.green_hsv_lower == (28, 40, 40)
    assert seg_cfg.include_yellow_tones is False
    assert seg_cfg.morph_kernel_shape == "MORPH_ELLIPSE"
    assert seg_cfg.open_kernel_size == (3, 3)
    assert seg_cfg.close_kernel_size == (5, 5)
    assert seg_cfg.open_iterations == 1
    assert seg_cfg.close_iterations == 1
    assert seg_cfg.cluster_solidity_threshold == 0.85
    assert seg_cfg.cluster_area_threshold_px == 25000.0
    assert seg_cfg.enable_watershed is True
    assert seg_cfg.watershed_dist_threshold_ratio == 0.45
    assert seg_cfg.watershed_bg_dilate_iterations == 2

    # Invalid watershed_dist_threshold_ratio (< 0.05 or > 0.95)
    with pytest.raises(ValidationError):
        SegmentationConfig(watershed_dist_threshold_ratio=0.01)

    with pytest.raises(ValidationError):
        SegmentationConfig(watershed_dist_threshold_ratio=0.99)

    # Invalid watershed_bg_dilate_iterations (0 or > 10)
    with pytest.raises(ValidationError):
        SegmentationConfig(watershed_bg_dilate_iterations=0)

    # Invalid Hue (> 180)
    with pytest.raises(ValidationError):
        SegmentationConfig(green_hsv_lower=(190, 40, 40))

    # Invalid lower > upper
    with pytest.raises(ValidationError):
        SegmentationConfig(green_hsv_lower=(90, 40, 40), green_hsv_upper=(50, 40, 40))

    # Invalid morph kernel shape
    with pytest.raises(ValidationError):
        SegmentationConfig(morph_kernel_shape="INVALID_SHAPE")

    # Invalid even morph kernel size
    with pytest.raises(ValidationError):
        SegmentationConfig(morph_kernel_size=(4, 4))

    # Invalid even open / close kernel sizes
    with pytest.raises(ValidationError):
        SegmentationConfig(open_kernel_size=(4, 4))

    with pytest.raises(ValidationError):
        SegmentationConfig(close_kernel_size=(6, 6))

    # Invalid iterations
    with pytest.raises(ValidationError):
        SegmentationConfig(open_iterations=0)

    with pytest.raises(ValidationError):
        SegmentationConfig(close_iterations=15)

    # Invalid cluster solidity threshold (> 1.0 or < 0.0)
    with pytest.raises(ValidationError):
        SegmentationConfig(cluster_solidity_threshold=1.5)

    with pytest.raises(ValidationError):
        SegmentationConfig(cluster_solidity_threshold=-0.1)


def test_classification_config_valid_and_invalid():
    """Verifies that ClassificationConfig validates size threshold, area metric, and labels."""
    from src.config import ClassificationConfig

    # Valid default
    clf_cfg = ClassificationConfig()
    assert clf_cfg.size_threshold_px == 12000.0
    assert clf_cfg.small_label == "Small Leaf"
    assert clf_cfg.large_label == "Large Leaf"
    assert clf_cfg.area_metric == "contour_area"
    assert clf_cfg.min_leaf_area_px == 250.0
    assert clf_cfg.suppress_duplicates is True
    assert clf_cfg.duplicate_iou_threshold == 0.5

    # Valid custom
    custom = ClassificationConfig(
        size_threshold_px=8500.0,
        small_label="Baby Leaf",
        large_label="Mature Leaf",
        area_metric="bbox_area",
        min_leaf_area_px=100.0,
        suppress_duplicates=False,
        duplicate_iou_threshold=0.7,
    )
    assert custom.size_threshold_px == 8500.0
    assert custom.small_label == "Baby Leaf"
    assert custom.large_label == "Mature Leaf"
    assert custom.area_metric == "bbox_area"
    assert custom.min_leaf_area_px == 100.0
    assert custom.suppress_duplicates is False
    assert custom.duplicate_iou_threshold == 0.7

    # Invalid negative or zero threshold
    with pytest.raises(ValidationError):
        ClassificationConfig(size_threshold_px=0.0)

    with pytest.raises(ValidationError):
        ClassificationConfig(size_threshold_px=-50.0)

    # Invalid empty label
    with pytest.raises(ValidationError):
        ClassificationConfig(small_label="")

    with pytest.raises(ValidationError):
        ClassificationConfig(large_label="")

    # Invalid area metric
    with pytest.raises(ValidationError):
        ClassificationConfig(area_metric="invalid_metric")

    # Invalid negative min_leaf_area_px
    with pytest.raises(ValidationError):
        ClassificationConfig(min_leaf_area_px=-10.0)

    # Invalid duplicate_iou_threshold (< 0 or > 1)
    with pytest.raises(ValidationError):
        ClassificationConfig(duplicate_iou_threshold=1.5)

    with pytest.raises(ValidationError):
        ClassificationConfig(duplicate_iou_threshold=-0.1)

    # Valid custom colors
    color_cfg = ClassificationConfig(
        small_color_bgr=(255, 200, 0),
        large_color_bgr=(0, 255, 0),
    )
    assert color_cfg.small_color_bgr == (255, 200, 0)
    assert color_cfg.large_color_bgr == (0, 255, 0)

    # Invalid colors: wrong length or channel out of bounds
    with pytest.raises(ValidationError):
        ClassificationConfig(small_color_bgr=(255, 255))  # type: ignore

    with pytest.raises(ValidationError):
        ClassificationConfig(large_color_bgr=(300, 0, 0))



