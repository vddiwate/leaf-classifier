"""Unit tests for predict.py inference runner."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from predict import process_single_image, run_inference
from src.classification import SizeClassifier
from src.config import load_config
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter


def test_process_single_image_inference():
    """Verifies that process_single_image handles valid and empty frames without error."""
    config = load_config("config/config.yaml")
    preprocessor = Preprocessor(config.preprocessing)
    segmenter = Segmenter(config.segmentation)
    classifier = SizeClassifier(config.classification)

    # Synthetic image
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Draw green leaf blob
    cv2.circle(test_img, (128, 128), 45, (35, 180, 50), -1)

    annotated, measurements, records = process_single_image(
        image_bgr=test_img,
        image_name="synthetic_leaf.jpg",
        preprocessor=preprocessor,
        segmenter=segmenter,
        classifier=classifier,
    )

    assert annotated.shape == (256, 256, 3)
    assert len(measurements) >= 1
    assert len(records) == len(measurements)
    assert records[0]["image_name"] == "synthetic_leaf.jpg"


def test_run_inference_cli_single_image(tmp_path: Path):
    """Verifies end-to-end execution of run_inference on a single image file."""
    img_path = tmp_path / "test_leaf.jpg"
    out_dir = tmp_path / "predictions"
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(test_img, (50, 50), 30, (35, 180, 50), -1)
    cv2.imwrite(str(img_path), test_img)

    ret = run_inference(
        source_path=str(img_path),
        config_path="config/config.yaml",
        output_dir=str(out_dir),
    )

    assert ret == 0
    assert out_dir.exists()
    assert (out_dir / "prediction_report.csv").exists()
    assert (out_dir / "pred_test_leaf.jpg").exists()


def test_run_inference_cli_directory(tmp_path: Path):
    """Verifies end-to-end execution of run_inference on a directory of images."""
    data_dir = tmp_path / "unseen_folder"
    data_dir.mkdir()
    out_dir = tmp_path / "predictions"

    for i in range(2):
        img_file = data_dir / f"leaf_{i}.jpg"
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_img, (50, 50), 30, (35, 180, 50), -1)
        cv2.imwrite(str(img_file), test_img)

    ret = run_inference(
        source_path=str(data_dir),
        config_path="config/config.yaml",
        output_dir=str(out_dir),
    )

    assert ret == 0
    assert (out_dir / "pred_leaf_0.jpg").exists()
    assert (out_dir / "pred_leaf_1.jpg").exists()


def test_run_inference_invalid_source():
    """Verifies that non-existent source path returns error code 1 gracefully."""
    ret = run_inference(source_path="non_existent_path_12345.jpg")
    assert ret == 1
