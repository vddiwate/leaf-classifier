"""Configuration schemas and validation models for the leafy vegetable pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.logger import get_logger

logger = get_logger("config")


class SystemConfig(BaseModel):
    random_seed: int = Field(ge=0, le=2_147_483_647, default=42)
    log_level: str = "INFO"
    log_dir: str = Field(min_length=1, default="logs")
    log_file_name: str = Field(min_length=1, default="leafy_cv.log")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = v.strip().upper()
        if normalized not in valid_levels:
            raise ValueError(
                f"Invalid log_level '{v}'. Must be one of: {sorted(valid_levels)}"
            )
        return normalized


class DataConfig(BaseModel):
    zip_path: str = Field(min_length=1, default="data/leafy_vegetable_dataset.zip")
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )
    min_resolution: Tuple[int, int] = (50, 50)  # [min_width, min_height]

    @field_validator("allowed_extensions")
    @classmethod
    def validate_extensions(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("allowed_extensions cannot be empty")
        validated = []
        for ext in v:
            ext_clean = ext.strip().lower()
            if not ext_clean.startswith("."):
                raise ValueError(f"Extension '{ext}' must start with '.' (e.g., '.jpg')")
            if len(ext_clean) < 2:
                raise ValueError(f"Extension '{ext}' is too short")
            validated.append(ext_clean)
        return validated

    @field_validator("min_resolution")
    @classmethod
    def validate_min_resolution(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if len(v) != 2:
            raise ValueError(f"min_resolution must contain exactly [width, height], got {v}")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"min_resolution dimensions must be positive integers, got {v}")
        return v


class PreprocessingConfig(BaseModel):
    target_width: int = Field(ge=16, le=8192, default=256)
    blur_kernel: Tuple[int, int] = (5, 5)
    blur_sigma: float = Field(ge=0.0, le=100.0, default=0.0)
    apply_clahe: bool = False
    clahe_clip_limit: float = Field(gt=0.0, le=100.0, default=2.0)
    clahe_grid_size: Tuple[int, int] = (8, 8)

    @field_validator("blur_kernel")
    @classmethod
    def validate_odd_kernel(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if len(v) != 2:
            raise ValueError(f"Blur kernel must have 2 dimensions [width, height], got {v}")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"Blur kernel dimensions must be positive integers, got {v}")
        if v[0] > 99 or v[1] > 99:
            raise ValueError(f"Blur kernel dimensions must be <= 99, got {v}")
        if v[0] % 2 == 0 or v[1] % 2 == 0:
            raise ValueError(f"Blur kernel must have odd dimensions for spatial symmetry, got {v}")
        return v

    @field_validator("clahe_grid_size")
    @classmethod
    def validate_clahe_grid(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if len(v) != 2:
            raise ValueError(f"clahe_grid_size must have 2 dimensions, got {v}")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"clahe_grid_size dimensions must be positive integers, got {v}")
        if v[0] > 128 or v[1] > 128:
            raise ValueError(f"clahe_grid_size dimensions must be <= 128, got {v}")
        return v


class SegmentationConfig(BaseModel):
    green_hsv_lower: Tuple[int, int, int] = (28, 40, 40)
    green_hsv_upper: Tuple[int, int, int] = (85, 255, 255)
    include_yellow_tones: bool = False
    yellow_hsv_lower: Tuple[int, int, int] = (22, 90, 60)
    yellow_hsv_upper: Tuple[int, int, int] = (27, 255, 255)
    morph_kernel_shape: str = "MORPH_ELLIPSE"
    morph_kernel_size: Tuple[int, int] = (5, 5)  # Fallback default
    open_kernel_size: Tuple[int, int] = (3, 3)   # Gentle speck removal without eroding stems
    close_kernel_size: Tuple[int, int] = (5, 5)  # Vein bridging and shadow closing
    open_iterations: int = Field(ge=1, le=10, default=1)
    close_iterations: int = Field(ge=1, le=10, default=1)
    min_contour_area_px: float = Field(ge=1.0, default=150.0)
    max_area_ratio: float = Field(gt=0.0, le=1.0, default=0.95)
    cluster_solidity_threshold: float = Field(ge=0.0, le=1.0, default=0.85)
    cluster_area_threshold_px: float = Field(ge=100.0, default=25000.0)
    enable_watershed: bool = True
    watershed_dist_threshold_ratio: float = Field(ge=0.05, le=0.95, default=0.45)
    watershed_bg_dilate_iterations: int = Field(ge=1, le=10, default=2)

    @field_validator("green_hsv_lower", "green_hsv_upper", "yellow_hsv_lower", "yellow_hsv_upper")
    @classmethod
    def validate_hsv_bounds(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(v) != 3:
            raise ValueError(f"HSV bound must contain 3 elements (H, S, V), got {v}")
        h, s, val = v
        if not (0 <= h <= 180):
            raise ValueError(f"Hue (H) must be in range [0, 180], got {h}")
        if not (0 <= s <= 255):
            raise ValueError(f"Saturation (S) must be in range [0, 255], got {s}")
        if not (0 <= val <= 255):
            raise ValueError(f"Value (V) must be in range [0, 255], got {val}")
        return v

    @field_validator("morph_kernel_shape")
    @classmethod
    def validate_morph_shape(cls, v: str) -> str:
        valid_shapes = {"MORPH_ELLIPSE", "MORPH_RECT", "MORPH_CROSS"}
        normalized = v.strip().upper()
        if normalized not in valid_shapes:
            raise ValueError(
                f"Invalid morph_kernel_shape '{v}'. Must be one of: {sorted(valid_shapes)}"
            )
        return normalized

    @field_validator("morph_kernel_size", "open_kernel_size", "close_kernel_size")
    @classmethod
    def validate_morph_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        if len(v) != 2:
            raise ValueError(f"Kernel size must have 2 dimensions, got {v}")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"Kernel dimensions must be positive integers, got {v}")
        if v[0] > 51 or v[1] > 51:
            raise ValueError(f"Kernel dimensions must be <= 51, got {v}")
        if v[0] % 2 == 0 or v[1] % 2 == 0:
            raise ValueError(f"Kernel dimensions must be odd integers, got {v}")
        return v

    @model_validator(mode="after")
    def validate_hsv_ranges(self) -> "SegmentationConfig":
        for name, lower, upper in [
            ("green", self.green_hsv_lower, self.green_hsv_upper),
            ("yellow", self.yellow_hsv_lower, self.yellow_hsv_upper),
        ]:
            if lower[0] > upper[0] or lower[1] > upper[1] or lower[2] > upper[2]:
                raise ValueError(
                    f"{name}_hsv_lower {lower} must be <= {name}_hsv_upper {upper} across all channels"
                )
        return self


class ClassificationConfig(BaseModel):
    mode: str = "adaptive"  # 'adaptive' (natural ratio comparison) or 'fixed' (absolute threshold)
    relative_ratio_threshold: float = Field(ge=0.10, le=0.99, default=0.80)
    min_large_leaf_area_px: float = Field(ge=100.0, default=3500.0)
    size_threshold_px: float = Field(ge=1.0, default=12000.0)
    small_label: str = Field(min_length=1, default="Small Leaf")
    large_label: str = Field(min_length=1, default="Large Leaf")
    area_metric: str = "contour_area"
    min_leaf_area_px: float = Field(ge=0.0, default=250.0)
    suppress_duplicates: bool = True
    duplicate_iou_threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    small_color_bgr: Tuple[int, int, int] = (255, 215, 0)   # Vivid Cyan in BGR
    large_color_bgr: Tuple[int, int, int] = (50, 225, 50)   # Vivid Lime Green in BGR

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        valid_modes = {"fixed", "adaptive"}
        normalized = v.strip().lower()
        if normalized not in valid_modes:
            raise ValueError(f"Invalid mode '{v}'. Must be one of: {sorted(valid_modes)}")
        return normalized

    @field_validator("small_color_bgr", "large_color_bgr")
    @classmethod
    def validate_bgr_color(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if len(v) != 3:
            raise ValueError(f"Color must be a 3-tuple (B, G, R), got {v}")
        for c in v:
            if not (0 <= c <= 255):
                raise ValueError(f"Color channel values must be in [0, 255], got {c}")
        return v

    @field_validator("area_metric")
    @classmethod
    def validate_area_metric(cls, v: str) -> str:
        valid_metrics = {"contour_area", "bbox_area"}
        normalized = v.strip().lower()
        if normalized not in valid_metrics:
            raise ValueError(
                f"Invalid area_metric '{v}'. Must be one of: {sorted(valid_metrics)}"
            )
        return normalized


class PathsConfig(BaseModel):
    output_dir: str = Field(min_length=1, default="outputs")
    sample_views_dir: str = Field(min_length=1, default="outputs/sample_views")
    segmentation_dir: str = Field(min_length=1, default="outputs/segmentation_views")
    classification_dir: str = Field(min_length=1, default="outputs/classification_views")
    classification_csv: str = Field(min_length=1, default="outputs/leaf_classification_report.csv")
    evaluation_dir: str = Field(min_length=1, default="outputs/evaluation")
    evaluation_csv: str = Field(min_length=1, default="outputs/evaluation/unseen_evaluation_review.csv")
    evaluation_contact_sheet: str = Field(min_length=1, default="outputs/evaluation/unseen_test_contact_sheet.jpg")


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads and validates application configuration from YAML file.

    Args:
        config_path: Path to configuration YAML file. If None, checks default locations.

    Returns:
        AppConfig: Validated Pydantic configuration object.

    Raises:
        FileNotFoundError: If configuration file does not exist.
        ValueError: If YAML is malformed or not a dictionary.
        pydantic.ValidationError: If configuration fails schema validation.
    """
    if config_path is None:
        default_path = Path("config/config.yaml")
        if not default_path.exists():
            default_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        config_path = str(default_path)

    path = Path(config_path)
    if not path.is_file():
        msg = f"Configuration file not found at: {config_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Syntax error in configuration YAML file '{config_path}': {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        msg = f"Configuration file '{config_path}' must contain a key-value mapping, got {type(raw_data).__name__}"
        logger.error(msg)
        raise ValueError(msg)

    try:
        config = AppConfig(**raw_data)
        logger.debug(f"Successfully loaded and validated configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Configuration validation failed for '{config_path}': {e}")
        raise
