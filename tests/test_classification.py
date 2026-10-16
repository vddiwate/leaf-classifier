"""Unit tests for Step 4 Leaf Size Classification module."""

import csv
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.classification import LeafMeasurement, SizeClassifier
from src.config import ClassificationConfig


def _create_synthetic_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Helper to create a rectangular contour polygon."""
    return np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]],
    ], dtype=np.int32)


def test_feature_extraction_mathematical_correctness():
    """Verifies that bounding box width, height, contour area, bbox area,
    and aspect ratio are extracted with exact mathematical correctness."""
    cfg = ClassificationConfig(size_threshold_px=10000.0)
    classifier = SizeClassifier(cfg)

    # 40 wide, 80 high rectangle at (10, 20)
    w, h = 40, 80
    contour = _create_synthetic_contour(10, 20, w, h)
    bbox = (10, 20, w, h)

    measurement = classifier.extract_measurement(contour, bbox, detection_id=1)

    assert measurement.detection_id == 1
    assert measurement.bbox_x == 10
    assert measurement.bbox_y == 20
    assert measurement.bbox_width == 40
    assert measurement.bbox_height == 80
    assert measurement.bbox == (10, 20, 40, 80)
    assert measurement.contour_area_px == pytest.approx(w * h, abs=1.0)
    assert measurement.bbox_area_px == pytest.approx(w * h, abs=0.1)
    assert measurement.measured_area_px == pytest.approx(w * h, abs=1.0)
    assert measurement.aspect_ratio == pytest.approx(0.5, abs=0.01)
    assert measurement.extent == pytest.approx(1.0, abs=0.01)


def test_size_classification_thresholding():
    """Verifies that regions with area < threshold are Small Leaf,
    and regions with area >= threshold are Large Leaf."""
    cfg = ClassificationConfig(size_threshold_px=5000.0)
    classifier = SizeClassifier(cfg)

    # Small leaf: 40x50 = 2000 px < 5000
    c_small = _create_synthetic_contour(0, 0, 40, 50)
    m_small = classifier.extract_measurement(c_small, (0, 0, 40, 50))
    assert m_small.predicted_class == "Small Leaf"
    assert m_small.measured_area_px == pytest.approx(2000.0, abs=1.0)

    # Large leaf: 100x100 = 10000 px >= 5000
    c_large = _create_synthetic_contour(0, 0, 100, 100)
    m_large = classifier.extract_measurement(c_large, (0, 0, 100, 100))
    assert m_large.predicted_class == "Large Leaf"
    assert m_large.measured_area_px == pytest.approx(10000.0, abs=1.0)

    # Exact threshold boundary: 50x100 = 5000 px >= 5000 -> Large Leaf
    c_boundary = _create_synthetic_contour(0, 0, 50, 100)
    m_boundary = classifier.extract_measurement(c_boundary, (0, 0, 50, 100))
    assert m_boundary.predicted_class == "Large Leaf"


def test_size_classification_bbox_area_metric():
    """Verifies that area_metric='bbox_area' correctly uses bounding box area for classification and measured_area."""
    cfg = ClassificationConfig(size_threshold_px=4000.0, area_metric="bbox_area")
    classifier = SizeClassifier(cfg)

    # Bbox area 50x60 = 3000 < 4000 -> Small
    c1 = _create_synthetic_contour(0, 0, 50, 60)
    m1 = classifier.extract_measurement(c1, (0, 0, 50, 60))
    assert m1.predicted_class == "Small Leaf"
    assert m1.measured_area_px == pytest.approx(3000.0, abs=0.1)

    # Bbox area 70x70 = 4900 >= 4000 -> Large
    c2 = _create_synthetic_contour(0, 0, 70, 70)
    m2 = classifier.extract_measurement(c2, (0, 0, 70, 70))
    assert m2.predicted_class == "Large Leaf"
    assert m2.measured_area_px == pytest.approx(4900.0, abs=0.1)


def test_min_leaf_area_filtering():
    """Verifies that tiny noise specks smaller than min_leaf_area_px are filtered out."""
    cfg = ClassificationConfig(min_leaf_area_px=400.0)
    classifier = SizeClassifier(cfg)

    # 10x15 = 150 px < 400 -> should be discarded
    tiny_c = _create_synthetic_contour(0, 0, 10, 15)
    # 30x30 = 900 px >= 400 -> should be retained
    valid_c = _create_synthetic_contour(50, 50, 30, 30)

    results = classifier.classify_regions([tiny_c, valid_c], [(0, 0, 10, 15), (50, 50, 30, 30)])
    assert len(results) == 1
    assert results[0].bbox == (50, 50, 30, 30)
    assert results[0].detection_id == 1


def test_duplicate_fragment_suppression():
    """Verifies that duplicate or heavily contained fragment bounding boxes are suppressed."""
    cfg = ClassificationConfig(suppress_duplicates=True, duplicate_iou_threshold=0.5, min_leaf_area_px=10.0)
    classifier = SizeClassifier(cfg)

    # Main leaf: (10, 10, 100, 100) -> 10,000 px
    main_c = _create_synthetic_contour(10, 10, 100, 100)
    # Heavy overlapping duplicate box: (15, 15, 95, 95) -> IoU > 0.85
    dup_c = _create_synthetic_contour(15, 15, 95, 95)
    # Disjoint separate leaf: (150, 150, 40, 40)
    other_c = _create_synthetic_contour(150, 150, 40, 40)

    results = classifier.classify_regions(
        [main_c, dup_c, other_c],
        [(10, 10, 100, 100), (15, 15, 95, 95), (150, 150, 40, 40)]
    )

    # dup_c should be suppressed because IoU >= 0.5
    assert len(results) == 2
    # Ensure sequential IDs
    assert results[0].detection_id == 1
    assert results[1].detection_id == 2


def test_custom_classification_labels():
    """Verifies that user-defined class labels in configuration are properly applied."""
    cfg = ClassificationConfig(
        size_threshold_px=3000.0,
        small_label="Chhota",
        large_label="Bada",
        min_leaf_area_px=10.0,
    )
    classifier = SizeClassifier(cfg)

    c_small = _create_synthetic_contour(0, 0, 20, 20)
    m_small = classifier.extract_measurement(c_small, (0, 0, 20, 20))
    assert m_small.predicted_class == "Chhota"

    c_large = _create_synthetic_contour(0, 0, 80, 80)
    m_large = classifier.extract_measurement(c_large, (0, 0, 80, 80))
    assert m_large.predicted_class == "Bada"


def test_classify_regions_empty_mask_handling():
    """Verifies that empty masks (0 contours, 0 bboxes) return empty lists gracefully."""
    classifier = SizeClassifier(ClassificationConfig())

    result = classifier.classify_regions([], [])
    assert result == []

    result2 = classifier.classify_regions([], [(0, 0, 10, 10)])
    assert result2 == []


def test_classify_regions_multiple():
    """Verifies classification of multiple contours simultaneously."""
    classifier = SizeClassifier(ClassificationConfig(size_threshold_px=5000.0, min_leaf_area_px=100.0))

    contours = [
        _create_synthetic_contour(0, 0, 20, 30),   # 600 px -> Small
        _create_synthetic_contour(50, 50, 100, 80), # 8000 px -> Large
    ]
    bboxes = [(0, 0, 20, 30), (50, 50, 100, 80)]
    metadata = [{"is_cluster": False}, {"is_cluster": True}]

    results = classifier.classify_regions(contours, bboxes, metadata_list=metadata)
    assert len(results) == 2
    # Small Leaf (600 px) & Large Leaf (8000 px)
    assert any(r.predicted_class == "Small Leaf" for r in results)
    assert any(r.predicted_class == "Large Leaf" for r in results)


def test_render_annotated_image():
    """Verifies that annotated image rendering succeeds, displays boxes and labels, and preserves dimensions."""
    classifier = SizeClassifier(ClassificationConfig(min_leaf_area_px=100.0))
    base_img = np.zeros((200, 256, 3), dtype=np.uint8)

    contours = [_create_synthetic_contour(10, 10, 50, 50)]
    bboxes = [(10, 10, 50, 50)]
    measurements = classifier.classify_regions(contours, bboxes)

    annotated = classifier.render_annotated_image(
        base_bgr=base_img,
        measurements=measurements,
        contours=contours,
        image_name="test_spinach.jpg",
    )

    assert annotated.shape == (200, 256, 3)
    assert annotated.dtype == np.uint8
    assert np.count_nonzero(annotated) > 0


def test_render_annotated_image_empty_graceful():
    """Verifies that rendering an image with zero measurements handles gracefully without crashing."""
    classifier = SizeClassifier(ClassificationConfig())
    base_img = np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = classifier.render_annotated_image(
        base_bgr=base_img,
        measurements=[],
        image_name="empty.jpg",
    )
    assert annotated.shape == (100, 100, 3)
    assert np.count_nonzero(annotated) > 0


def test_export_csv_report(tmp_path: Path):
    """Verifies that CSV report contains all required headers and formatted rows."""
    classifier = SizeClassifier(ClassificationConfig())
    csv_file = tmp_path / "test_report.csv"

    c = _create_synthetic_contour(5, 10, 30, 40)
    m = classifier.extract_measurement(c, (5, 10, 30, 40), detection_id=1)
    record = m.to_dict(image_name="spinach_test.jpg")

    saved_path = classifier.export_csv_report([record], csv_file)
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["image_name"] == "spinach_test.jpg"
    assert row["detection_id"] == "1"
    assert row["bbox_x"] == "5"
    assert row["bbox_y"] == "10"
    assert row["bbox_width"] == "30"
    assert row["bbox_height"] == "40"
    assert float(row["contour_area"]) == pytest.approx(1200.0, abs=1.0)
    assert float(row["bbox_area"]) == pytest.approx(1200.0, abs=0.1)
    assert float(row["aspect_ratio"]) == pytest.approx(0.75, abs=0.01)
    assert row["predicted_class"] == "Small Leaf"


def test_export_csv_report_empty_records(tmp_path: Path):
    """Verifies that exporting an empty list of records generates a valid CSV with headers."""
    classifier = SizeClassifier(ClassificationConfig())
    csv_file = tmp_path / "empty_report.csv"

    saved_path = classifier.export_csv_report([], csv_file)
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

    expected_headers = [
        "image_name", "detection_id", "bbox_x", "bbox_y",
        "bbox_width", "bbox_height", "contour_area", "bbox_area",
        "aspect_ratio", "predicted_class"
    ]
    assert header == expected_headers


def test_invalid_classifier_inputs():
    """Verifies robust exception handling for invalid inputs."""
    with pytest.raises(TypeError):
        SizeClassifier("invalid_config_type")  # type: ignore

    classifier = SizeClassifier(ClassificationConfig())

    with pytest.raises(TypeError):
        classifier.extract_measurement(None, (0, 0, 10, 10))  # type: ignore

    with pytest.raises(ValueError):
        c = _create_synthetic_contour(0, 0, 10, 10)
        classifier.extract_measurement(c, (0, 0, 10))  # type: ignore

    with pytest.raises(ValueError):
        c = _create_synthetic_contour(0, 0, 10, 10)
        classifier.extract_measurement(c, (0, 0, -5, 10))

    with pytest.raises(TypeError):
        classifier.render_annotated_image(None, [])  # type: ignore

    with pytest.raises(ValueError):
        classifier.render_annotated_image(np.zeros((50, 50), dtype=np.uint8), [])


def test_adaptive_natural_size_classification():
    """Verifies that mode='adaptive' compares leaves naturally using relative ratio."""
    cfg = ClassificationConfig(
        mode="adaptive",
        relative_ratio_threshold=0.70,
        min_large_leaf_area_px=3500.0,
        size_threshold_px=6500.0,
    )
    classifier = SizeClassifier(cfg)

    # Case 1: Two leaves of different sizes (10,000 px and 4,000 px -> ratio 0.40 < 0.70)
    c_large = _create_synthetic_contour(0, 0, 100, 100)      # 10,000 px
    c_small = _create_synthetic_contour(150, 150, 40, 100)   # 4,000 px
    m_list1 = classifier.classify_regions([c_large, c_small], [(0, 0, 100, 100), (150, 150, 40, 100)])
    assert m_list1[0].predicted_class == "Large Leaf"
    assert m_list1[1].predicted_class == "Small Leaf"

    # Case 2: Two leaves of comparable large sizes (10,000 px and 8,500 px -> ratio 0.85 >= 0.70)
    c_comp = _create_synthetic_contour(150, 0, 85, 100)      # 8,500 px
    m_list2 = classifier.classify_regions([c_large, c_comp], [(0, 0, 100, 100), (150, 0, 85, 100)])
    assert m_list2[0].predicted_class == "Large Leaf"
    assert m_list2[1].predicted_class == "Large Leaf"

    # Case 3: Solitary single leaf (8,000 px >= 6500 px -> Large Leaf)
    m_list3 = classifier.classify_regions([c_comp], [(150, 0, 85, 100)])
    assert len(m_list3) == 1
    assert m_list3[0].predicted_class == "Large Leaf"

    # Case 4: Tiny seedlings both below min_large_leaf_area_px (1,500 px and 1,000 px < 3500 px)
    c_seed1 = _create_synthetic_contour(0, 0, 30, 50)        # 1,500 px
    c_seed2 = _create_synthetic_contour(50, 50, 20, 50)      # 1,000 px
    m_list4 = classifier.classify_regions([c_seed1, c_seed2], [(0, 0, 30, 50), (50, 50, 20, 50)])
    assert m_list4[0].predicted_class == "Small Leaf"
    assert m_list4[1].predicted_class == "Small Leaf"


def test_run_classification_demonstration():
    """Verifies that the benchmark classification runner script executes successfully."""
    from classification import run_classification_demonstration
    ret = run_classification_demonstration()
    assert ret == 0
