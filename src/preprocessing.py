"""Image preprocessing: aspect-ratio resizing, noise smoothing, and HSV conversion."""

from typing import Tuple
import cv2
import numpy as np

from src.config import PreprocessingConfig
from src.logger import get_logger

logger = get_logger("preprocessing")


class Preprocessor:
    """Production preprocessing engine with aspect-ratio preserving resize and color-space conversion."""

    def __init__(self, config: PreprocessingConfig):
        if not isinstance(config, PreprocessingConfig):
            raise TypeError(f"Expected PreprocessingConfig instance, got {type(config).__name__}")
        self.config = config
        logger.debug(
            f"Initialized Preprocessor: target_width={self.config.target_width}, "
            f"blur_kernel={self.config.blur_kernel}, blur_sigma={self.config.blur_sigma}, "
            f"clahe={self.config.apply_clahe}"
        )

    def validate_input(self, image_bgr: np.ndarray) -> Tuple[int, int]:
        """Validates input image dimensions, channel count, and uint8 data type."""
        if image_bgr is None:
            msg = "Input image cannot be None"
            logger.error(msg)
            raise TypeError(msg)

        if not isinstance(image_bgr, np.ndarray):
            msg = f"Expected numpy.ndarray, got {type(image_bgr).__name__}"
            logger.error(msg)
            raise TypeError(msg)

        if image_bgr.size == 0:
            msg = "Input image array is empty (size 0)"
            logger.error(msg)
            raise ValueError(msg)

        if image_bgr.ndim != 3:
            msg = f"Expected 3-dimensional image array (H, W, C), got {image_bgr.ndim} dimensions"
            logger.error(msg)
            raise ValueError(msg)

        if image_bgr.shape[2] != 3:
            msg = f"Expected 3-channel BGR image, got {image_bgr.shape[2]} channels"
            logger.error(msg)
            raise ValueError(msg)

        orig_h, orig_w = image_bgr.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            msg = f"Invalid image dimensions: {orig_w}x{orig_h}"
            logger.error(msg)
            raise ValueError(msg)

        if image_bgr.dtype != np.uint8:
            msg = f"Expected image dtype uint8 (0-255), got {image_bgr.dtype}"
            logger.error(msg)
            raise TypeError(msg)

        return orig_h, orig_w

    def process(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Executes input validation, aspect-preserving resize, Gaussian blur, and HSV conversion."""
        # 1. Input Validation
        orig_h, orig_w = self.validate_input(image_bgr)

        # 2. Aspect-ratio preserving resize
        scale_factor = float(self.config.target_width) / float(orig_w)
        target_height = max(1, int(round(orig_h * scale_factor)))

        interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
        resized_bgr = cv2.resize(
            image_bgr,
            (self.config.target_width, target_height),
            interpolation=interpolation,
        )

        # 3. Gaussian Blur to suppress sensor noise & fine background grain
        ksize = tuple(self.config.blur_kernel)
        blurred_bgr = cv2.GaussianBlur(
            resized_bgr,
            ksize=ksize,
            sigmaX=self.config.blur_sigma,
        )

        # 4. Optional CLAHE on Value (V) channel for uneven shadow handling
        if self.config.apply_clahe:
            hsv_temp = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_temp)
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=tuple(self.config.clahe_grid_size),
            )
            v_clahe = clahe.apply(v)
            hsv_image = cv2.merge([h, s, v_clahe])
        else:
            hsv_image = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2HSV)

        # 5. Output Verification
        expected_shape = (target_height, self.config.target_width, 3)
        if resized_bgr.shape != expected_shape:
            msg = f"Resized BGR shape mismatch: expected {expected_shape}, got {resized_bgr.shape}"
            logger.error(msg)
            raise RuntimeError(msg)

        if hsv_image.shape != expected_shape:
            msg = f"HSV image shape mismatch: expected {expected_shape}, got {hsv_image.shape}"
            logger.error(msg)
            raise RuntimeError(msg)

        if scale_factor <= 0.0:
            msg = f"Calculated invalid scale factor: {scale_factor}"
            logger.error(msg)
            raise RuntimeError(msg)

        logger.debug(
            f"Preprocessed image successfully: ({orig_w}x{orig_h}) -> ({self.config.target_width}x{target_height}), "
            f"scale={scale_factor:.4f}"
        )

        return resized_bgr, hsv_image, scale_factor
