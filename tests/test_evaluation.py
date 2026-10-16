"""Unit tests for Step 6 Evaluation module."""

from pathlib import Path
import csv
import numpy as np
from evaluate import create_contact_sheet, run_evaluation_pipeline


def test_create_contact_sheet():
    """Verifies that create_contact_sheet stitches thumbnails into a valid BGR canvas."""
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    items = [
        ("test_01.jpg", dummy_img, 1),
        ("test_02.jpg", dummy_img, 2),
        ("test_03.jpg", dummy_img, 3),
    ]

    sheet = create_contact_sheet(items, cols=2, thumb_size=(100, 100))

    assert sheet.ndim == 3
    assert sheet.shape[2] == 3
    assert sheet.shape[0] > 100
    assert sheet.shape[1] > 100
    assert np.count_nonzero(sheet) > 0


def test_create_contact_sheet_empty():
    """Verifies graceful handling of empty image list."""
    sheet = create_contact_sheet([])
    assert sheet.shape == (100, 100, 3)


def test_run_evaluation_pipeline_success(tmp_path: Path):
    """Verifies that evaluation pipeline generates review CSV, contact sheet, and markdown report."""
    ret = run_evaluation_pipeline(
        zip_path="data/test_unseen/leafy_vegetable_unseen_test_images.zip",
        config_path="config/config.yaml",
    )
    assert ret == 0

    csv_path = Path("outputs/evaluation/unseen_evaluation_review.csv")
    sheet_path = Path("outputs/evaluation/unseen_test_contact_sheet.jpg")
    report_path = Path("outputs/evaluation/final_evaluation_report.md")

    assert csv_path.exists()
    assert sheet_path.exists()
    assert report_path.exists()

    # Verify that unknown actual counts remain blank (no invented values)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 15
    # test_004 is dense crop -> actual_leaf_count must be blank
    row_004 = next(r for r in rows if r["image_name"] == "test_004.jpg")
    assert row_004["actual_leaf_count"] == ""
    # test_001 is single leaf -> actual_leaf_count is 1
    row_001 = next(r for r in rows if r["image_name"] == "test_001.jpg")
    assert row_001["actual_leaf_count"] == "1"


def test_run_evaluation_pipeline_invalid_zip():
    """Verifies that non-existent zip returns code 1 gracefully."""
    ret = run_evaluation_pipeline(zip_path="non_existent_archive_123.zip")
    assert ret == 1
