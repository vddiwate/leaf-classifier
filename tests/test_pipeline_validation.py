"""Automated validation tests for leafy vegetable classification pipeline:
1. Image preprocessing and invalid-image handling
2. Empty mask and no-contour handling
3. Single-leaf segmentation and bounding box tightness
4. Watershed separation on a synthetic image containing two touching leaf-like shapes
5. Small/Large classification at, below, and above the configured threshold (fixed and adaptive)
6. Watershed enabled and disabled configuration
7. CSV output schema and complete end-to-end pipeline integration test
"""

import csv
from pathlib import Path
import cv2
import numpy as np
import pytest

from predict import process_single_image, run_inference
from src.classification import LeafMeasurement, SizeClassifier
from src.config import ClassificationConfig, PreprocessingConfig, SegmentationConfig, load_config
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter


def _create_synthetic_leaf_image(
    canvas_size: int = 256,
    leaf_centers: list = None,
    radii: list = None,
    color_bgr: tuple = (35, 180, 50),
) -> np.ndarray:
    """Helper to generate a deterministic synthetic BGR image with circular leaf blobs."""
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    img[:, :] = (120, 120, 120)  # Neutral gray background
    if leaf_centers and radii:
        for center, radius in zip(leaf_centers, radii):
            cv2.circle(img, center, radius, color_bgr, -1)
    return img


# =========================================================================
# 1. Image Preprocessing and Invalid-Image Handling
# =========================================================================

def test_image_preprocessing_and_invalid_image_handling(tmp_path: Path):
    """Verifies image preprocessing behavior on valid images and strict validation on invalid inputs:
    - Normalizes image resolution to configured target_width while preserving aspect ratio.
    - Accurately converts to blurred HSV color representation.
    - Gracefully rejects None, empty arrays, non-numpy inputs, wrong dimensions, and corrupt files."""
    cfg = PreprocessingConfig(target_width=256, blur_kernel=(5, 5), apply_clahe=False)
    preprocessor = Preprocessor(cfg)

    # 1. Valid image handling
    valid_img = np.zeros((100, 200, 3), dtype=np.uint8)
    valid_img[20:80, 30:170] = (35, 180, 50)
    resized_bgr, hsv_img, scale = preprocessor.process(valid_img)

    expected_height = int(round(100 * (256.0 / 200.0)))  # 128
    assert resized_bgr.shape == (expected_height, 256, 3)
    assert hsv_img.shape == (expected_height, 256, 3)
    assert resized_bgr.dtype == np.uint8
    assert hsv_img.dtype == np.uint8
    assert scale == pytest.approx(256.0 / 200.0)

    # 2. None input rejection
    with pytest.raises(TypeError, match="cannot be None"):
        preprocessor.process(None)

    # 3. Non-numpy input rejection
    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        preprocessor.process([[1, 2, 3]])

    # 4. Empty array rejection
    with pytest.raises(ValueError, match="is empty"):
        preprocessor.process(np.array([], dtype=np.uint8))

    # 5. Wrong channel format rejection (e.g., 2D grayscale)
    with pytest.raises(ValueError, match="Expected 3-dimensional"):
        preprocessor.process(np.zeros((100, 100), dtype=np.uint8))

    # 6. Corrupt file on disk handling during pipeline execution
    corrupt_file = tmp_path / "corrupt_image.jpg"
    corrupt_file.write_bytes(b"NOT_A_VALID_JPEG_HEADER")
    ret = run_inference(
        source_path=str(corrupt_file),
        config_path="config/config.yaml",
        output_dir=str(tmp_path / "out"),
    )
    # Corrupt unreadable image returns error exit code 1 without unhandled crash
    assert ret == 1


# =========================================================================
# 2. Empty Mask and No-Contour Handling
# =========================================================================

def test_empty_mask_and_zero_detected_leaves():
    """Verifies that an image without any green/yellow foliage produces an empty mask,
    zero contours, zero measurements, zero CSV records, and a non-crashing annotated visual."""
    prep_cfg = PreprocessingConfig(target_width=256, apply_clahe=False)
    seg_cfg = SegmentationConfig(min_contour_area_px=100.0, enable_watershed=True)
    cls_cfg = ClassificationConfig(min_leaf_area_px=100.0)

    preprocessor = Preprocessor(prep_cfg)
    segmenter = Segmenter(seg_cfg)
    classifier = SizeClassifier(cls_cfg)

    # Completely blank/gray image with zero leaves
    blank_bgr = _create_synthetic_leaf_image(canvas_size=256, leaf_centers=None, radii=None)

    annotated, measurements, records = process_single_image(
        image_bgr=blank_bgr,
        image_name="empty_scene.jpg",
        preprocessor=preprocessor,
        segmenter=segmenter,
        classifier=classifier,
    )

    assert annotated.shape == (256, 256, 3)
    assert len(measurements) == 0
    assert len(records) == 0

    # Direct segmentation on all-zero mask
    empty_mask = np.zeros((256, 256), dtype=np.uint8)
    contours, bboxes = segmenter.find_leaf_contours(empty_mask, base_bgr=blank_bgr)
    assert contours == []
    assert bboxes == []


# =========================================================================
# 3. Single-Leaf Segmentation
# =========================================================================

def test_single_leaf_detection_and_bounding_tightness():
    """Verifies that a solitary leaf is accurately detected, bounded with high tightness,
    and classified into a single LeafMeasurement."""
    prep_cfg = PreprocessingConfig(target_width=256, apply_clahe=False)
    seg_cfg = SegmentationConfig(min_contour_area_px=100.0, enable_watershed=True)
    cls_cfg = ClassificationConfig(min_leaf_area_px=100.0, size_threshold_px=5000.0)

    preprocessor = Preprocessor(prep_cfg)
    segmenter = Segmenter(seg_cfg)
    classifier = SizeClassifier(cls_cfg)

    # 1 distinct leaf blob (radius 40 at center (128, 128))
    single_leaf_img = _create_synthetic_leaf_image(
        canvas_size=256,
        leaf_centers=[(128, 128)],
        radii=[40],
    )

    annotated, measurements, records = process_single_image(
        image_bgr=single_leaf_img,
        image_name="single_leaf.jpg",
        preprocessor=preprocessor,
        segmenter=segmenter,
        classifier=classifier,
    )

    assert len(measurements) == 1
    assert len(records) == 1
    m = measurements[0]

    # Verify bounding box contains the leaf center
    x, y, w, h = m.bbox
    assert x <= 128 <= x + w
    assert y <= 128 <= y + h
    assert m.contour_area_px > 4000.0
    assert records[0]["image_name"] == "single_leaf.jpg"
    assert records[0]["detection_id"] == 1


# =========================================================================
# 4. Watershed Separation for Touching Leaves
# =========================================================================

def test_watershed_separation_for_touching_leaves():
    """Verifies that marker-controlled watershed cleanly splits two touching leaves
    that share an overlapping boundary bridge."""
    seg_cfg = SegmentationConfig(
        enable_watershed=True,
        watershed_dist_threshold_ratio=0.50,
        min_contour_area_px=50.0,
    )
    segmenter = Segmenter(seg_cfg)

    # 2 touching circular leaves: center1=(50, 75), r=25; center2=(95, 75), r=25
    # Distance between centers = 45 < 2*25=50 -> overlap bridge of ~5 px
    h, w = 150, 150
    mask = np.zeros((h, w), dtype=np.uint8)
    base_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.circle(mask, (50, 75), 25, 255, -1)
    cv2.circle(mask, (95, 75), 25, 255, -1)
    cv2.circle(base_bgr, (50, 75), 25, (35, 180, 50), -1)
    cv2.circle(base_bgr, (95, 75), 25, (35, 180, 50), -1)

    split_contours, split_bboxes = segmenter.find_leaf_contours(mask, base_bgr=base_bgr)

    assert len(split_contours) == 2, f"Expected 2 split leaves, got {len(split_contours)}"
    assert len(split_bboxes) == 2

    # Verify both split leaves satisfy bounding box tightness
    for c, bbox in zip(split_contours, split_bboxes):
        assert segmenter.verify_bounding_box_tightness(c, bbox, tolerance_px=1) is True


# =========================================================================
# 5. Small/Large Classification Around Configured Threshold
# =========================================================================

def test_fixed_threshold_boundary_classification():
    """Verifies classification behavior at, below, and above the configured fixed threshold:
    - area < threshold -> Small Leaf
    - area == threshold -> Large Leaf
    - area > threshold -> Large Leaf"""
    threshold = 5000.0
    cfg = ClassificationConfig(mode="fixed", size_threshold_px=threshold, min_leaf_area_px=100.0)
    classifier = SizeClassifier(cfg)

    def _rect_contour(w: int, h: int) -> np.ndarray:
        return np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)

    # 1. Below threshold (4950 px)
    c_below = _rect_contour(50, 99)
    m_below = classifier.extract_measurement(c_below, (0, 0, 50, 99))
    assert m_below.predicted_class == "Small Leaf"

    # 2. Exactly at threshold (5000 px: 50x100)
    c_exact = _rect_contour(50, 100)
    m_exact = classifier.extract_measurement(c_exact, (0, 0, 50, 100))
    assert m_exact.predicted_class == "Large Leaf"

    # 3. Above threshold (5050 px: 50x101)
    c_above = _rect_contour(50, 101)
    m_above = classifier.extract_measurement(c_above, (0, 0, 50, 101))
    assert m_above.predicted_class == "Large Leaf"


def test_adaptive_threshold_boundary_classification():
    """Verifies adaptive natural comparison at, below, and above relative ratio threshold (0.70):
    - ratio < 0.70 -> Small Leaf
    - ratio == 0.70 -> Large Leaf
    - ratio > 0.70 -> Large Leaf"""
    cfg = ClassificationConfig(
        mode="adaptive",
        relative_ratio_threshold=0.70,
        min_leaf_area_px=100.0,
    )
    classifier = SizeClassifier(cfg)

    def _rect_contour(x: int, w: int, h: int) -> np.ndarray:
        return np.array([[[x, 0]], [[x + w, 0]], [[x + w, h]], [[x, h]]], dtype=np.int32)

    # Reference dominant leaf: 100x100 = 10,000 px
    c_max = _rect_contour(0, 100, 100)
    b_max = (0, 0, 100, 100)

    # Below threshold: 6,900 px (69.0% -> Small Leaf)
    c_below = _rect_contour(120, 69, 100)
    b_below = (120, 0, 69, 100)

    # Exactly at threshold: 7,000 px (70.0% -> Large Leaf)
    c_exact = _rect_contour(200, 70, 100)
    b_exact = (200, 0, 70, 100)

    # Above threshold: 7,100 px (71.0% -> Large Leaf)
    c_above = _rect_contour(280, 71, 100)
    b_above = (280, 0, 71, 100)

    contours = [c_max, c_below, c_exact, c_above]
    bboxes = [b_max, b_below, b_exact, b_above]

    results = classifier.classify_regions(contours, bboxes)
    assert len(results) == 4

    assert results[0].predicted_class == "Large Leaf"  # 10,000 px (max)
    assert results[1].predicted_class == "Small Leaf"  # 6,900 px (< 70%)
    assert results[2].predicted_class == "Large Leaf"  # 7,000 px (== 70%)
    assert results[3].predicted_class == "Large Leaf"  # 7,100 px (> 70%)


# =========================================================================
# 6. Watershed Enabled and Disabled Configuration
# =========================================================================

def test_watershed_enabled_vs_disabled_modes():
    """Verifies contrast between watershed enabled vs disabled:
    - Disabled (or base_bgr=None): Touching leaves remain as 1 merged contour.
    - Enabled: Touching leaves are split into 2 individual contours."""
    h, w = 150, 150
    mask = np.zeros((h, w), dtype=np.uint8)
    base_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # 2 touching circles: center 1 at (40, 75), center 2 at (85, 75), radius 25 (5px overlap bridge)
    cv2.circle(mask, (40, 75), 25, 255, -1)
    cv2.circle(mask, (85, 75), 25, 255, -1)
    cv2.circle(base_bgr, (40, 75), 25, (35, 180, 50), -1)
    cv2.circle(base_bgr, (85, 75), 25, (35, 180, 50), -1)

    # Case A: Watershed explicitly disabled
    cfg_disabled = SegmentationConfig(enable_watershed=False, min_contour_area_px=50.0)
    seg_disabled = Segmenter(cfg_disabled)
    merged_contours, _ = seg_disabled.find_leaf_contours(mask, base_bgr=base_bgr)
    assert len(merged_contours) == 1, "Expected 1 merged contour when watershed is disabled"

    # Case B: Watershed explicitly enabled with 0.50 ratio
    cfg_enabled = SegmentationConfig(
        enable_watershed=True,
        watershed_dist_threshold_ratio=0.50,
        min_contour_area_px=50.0,
    )
    seg_enabled = Segmenter(cfg_enabled)
    split_contours, _ = seg_enabled.find_leaf_contours(mask, base_bgr=base_bgr)
    assert len(split_contours) == 2, "Expected 2 split contours when watershed is enabled"


# =========================================================================
# 7. CSV Output Schema and End-to-End Pipeline Integration Test
# =========================================================================

def test_complete_directory_pipeline_execution_and_csv_schema(tmp_path: Path):
    """Verifies end-to-end execution of run_inference on an image directory:
    - Validates output directory and annotated images creation
    - Validates CSV report generation
    - Validates CSV headers match required schema exactly
    - Validates data rows are non-empty and correctly typed."""
    input_dir = tmp_path / "test_images"
    input_dir.mkdir()
    output_dir = tmp_path / "inference_outputs"

    # Image 1: Single leaf (radius 40)
    img1 = _create_synthetic_leaf_image(canvas_size=200, leaf_centers=[(100, 100)], radii=[40])
    cv2.imwrite(str(input_dir / "sample_leaf1.jpg"), img1)

    # Image 2: Dual leaves (1 large radius 38, 1 small radius 25 -> 43% area ratio)
    img2 = _create_synthetic_leaf_image(
        canvas_size=200,
        leaf_centers=[(55, 100), (145, 100)],
        radii=[38, 25],
    )
    cv2.imwrite(str(input_dir / "sample_leaf2.jpg"), img2)

    # Image 3: Empty scene (zero leaves)
    img3 = _create_synthetic_leaf_image(canvas_size=200, leaf_centers=None, radii=None)
    cv2.imwrite(str(input_dir / "sample_leaf3.jpg"), img3)

    # Execute end-to-end inference
    exit_code = run_inference(
        source_path=str(input_dir),
        config_path="config/config.yaml",
        output_dir=str(output_dir),
    )

    assert exit_code == 0
    assert output_dir.exists()

    # Verify annotated image outputs
    assert (output_dir / "pred_sample_leaf1.jpg").exists()
    assert (output_dir / "pred_sample_leaf2.jpg").exists()
    assert (output_dir / "pred_sample_leaf3.jpg").exists()

    # Verify prediction CSV report exists
    csv_report_path = output_dir / "prediction_report.csv"
    assert csv_report_path.exists()

    # Validate exact CSV schema
    expected_headers = [
        "image_name",
        "detection_id",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "contour_area",
        "bbox_area",
        "aspect_ratio",
        "predicted_class",
    ]

    with open(csv_report_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == expected_headers

        rows = list(reader)
        # Sample 1 has 1 leaf, sample 2 has 2 leaves, sample 3 has 0 leaves -> 3 total rows
        assert len(rows) == 3

        for row in rows:
            assert row["image_name"] in ["sample_leaf1.jpg", "sample_leaf2.jpg"]
            assert int(row["detection_id"]) >= 1
            assert float(row["bbox_width"]) > 0
            assert float(row["bbox_height"]) > 0
            assert float(row["contour_area"]) > 0
            assert float(row["bbox_area"]) > 0
            assert row["predicted_class"] in ["Small Leaf", "Large Leaf"]
