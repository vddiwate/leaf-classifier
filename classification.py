"""Benchmark demonstration runner for the leaf size classification pipeline."""

from pathlib import Path
import sys
from typing import Any, Dict, List
import cv2
import numpy as np

from src.classification import LeafMeasurement, SizeClassifier
from src.config import load_config
from src.data_loader import ZipDataLoader
from src.logger import get_logger, setup_logger
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter


def run_classification_demonstration() -> int:
    """Executes Step 4 Leaf Size Classification pipeline across dataset images."""
    # 1. Load Central Configuration (Zero Hardcoding)
    try:
        config = load_config("config/config.yaml")
    except Exception as e:
        setup_logger(log_dir="logs", log_file_name="leafy_cv.log")
        logger = get_logger("step4_runner")
        logger.critical(f"Failed to load configuration: {e}")
        return 1

    # 2. Setup Central Logger (Single unified log file + console)
    setup_logger(
        log_dir=config.system.log_dir,
        log_file_name=config.system.log_file_name,
        log_level=config.system.log_level,
    )
    logger = get_logger("step4_runner")
    logger.info("=" * 88)
    logger.info("STEP 4: SMALL VS LARGE LEAF CLASSIFICATION PIPELINE")
    logger.info("=" * 88)
    logger.info(
        f"Active Settings: Size Threshold={config.classification.size_threshold_px:.0f} px "
        f"({config.classification.area_metric}) | Small='{config.classification.small_label}' | "
        f"Large='{config.classification.large_label}'"
    )

    # 3. Load ZIP dataset
    try:
        loader = ZipDataLoader(
            zip_path=config.data.zip_path,
            allowed_extensions=config.data.allowed_extensions,
            min_resolution=config.data.min_resolution,
        )
    except Exception as e:
        logger.error(f"Failed to initialize ZipDataLoader: {e}")
        return 1

    if len(loader) == 0:
        logger.error(f"No valid images found in archive: {config.data.zip_path}")
        return 1

    # 4. Initialize Modular Pipeline Components
    preprocessor = Preprocessor(config.preprocessing)
    segmenter = Segmenter(config.segmentation)
    classifier = SizeClassifier(config.classification)

    # 5. Output directories setup
    out_dir = Path(config.paths.classification_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous files in output directory to keep strictly the top 4 results
    for existing_file in out_dir.glob("*.jpg"):
        try:
            existing_file.unlink()
        except Exception as e:
            logger.warning(f"Could not remove old file {existing_file.name}: {e}")

    csv_path = Path(config.paths.classification_csv)

    # 6. Top 4 representative test cases covering distinct multi-leaf and single-leaf scenarios
    test_indices = [
        (19, "spinach_019.jpg - 3 distinct leaves of different sizes (1 Small, 2 Large)"),
        (10, "spinach_010.jpg - 2 touching leaves on wood separated into 2 Large Leaves"),
        (0, "spinach_001.jpg - Solitary single spinach leaf benchmark (1 Large Leaf)"),
        (12, "spinach_012.jpg - Disjoint leaves of different sizes (1 Large, 1 Small)"),
    ]

    logger.info(f"Processing Top {len(test_indices)} representative images for Step 4 classification")
    logger.info("-" * 92)
    logger.info(
        f"{'Idx':<4} | {'Filename':<16} | {'Detections':<10} | {'Small Leaves':<12} | {'Large Leaves':<12} | {'Area Range (px)'}"
    )
    logger.info("-" * 92)

    all_csv_records: List[Dict[str, Any]] = []
    total_images_processed = 0
    total_detections = 0
    total_small_leaves = 0
    total_large_leaves = 0
    saved_visual_paths: List[Path] = []

    for idx, desc in test_indices:
        if idx >= len(loader):
            continue

        raw_bgr, err, meta = loader.load_image(idx)
        if err or raw_bgr is None:
            logger.warning(f"Skipping index {idx} due to load error: {err}")
            continue

        fname = meta["filename"]

        # Step 2: Preprocess
        resized_bgr, hsv_img, scale = preprocessor.process(raw_bgr)

        # Step 3: Segmentation (with Watershed Instance Separation for touching leaves)
        clean_mask = segmenter.generate_mask(hsv_img)
        contours, bboxes, meta_list = segmenter.find_leaf_contours(
            clean_mask, base_bgr=resized_bgr, return_metadata=True
        )

        # Step 4: Geometric Feature Extraction & Classification
        measurements: List[LeafMeasurement] = classifier.classify_regions(
            contours=contours,
            bboxes=bboxes,
            metadata_list=meta_list,
        )

        num_det = len(measurements)
        total_detections += num_det
        total_images_processed += 1

        small_count = sum(1 for m in measurements if m.predicted_class == config.classification.small_label)
        large_count = sum(1 for m in measurements if m.predicted_class == config.classification.large_label)
        total_small_leaves += small_count
        total_large_leaves += large_count

        if measurements:
            min_area = min(m.measured_area_px for m in measurements)
            max_area = max(m.measured_area_px for m in measurements)
            range_str = f"{min_area:.0f} - {max_area:.0f}"
        else:
            range_str = "N/A (Empty)"

        logger.info(
            f"#{idx+1:03d} | {fname:<16} | {num_det:<10} | {small_count:<12} | {large_count:<12} | {range_str}"
        )

        # Convert measurements to CSV rows
        for m in measurements:
            all_csv_records.append(m.to_dict(image_name=fname))

        # Render and save annotated visualization
        annotated = classifier.render_annotated_image(
            base_bgr=resized_bgr,
            measurements=measurements,
            contours=contours,
            image_name=fname,
        )
        save_path = out_dir / f"classified_{Path(fname).stem}.jpg"
        if cv2.imwrite(str(save_path), annotated):
            saved_visual_paths.append(save_path)

    # 7. Test Empty Mask Handling Gracefully
    logger.info("-" * 92)
    logger.info("Verifying Empty Mask & Zero-Detection Graceful Handling...")
    h, w = config.preprocessing.target_width, config.preprocessing.target_width
    empty_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    empty_measurements = classifier.classify_regions([], [])
    empty_annotated = classifier.render_annotated_image(empty_bgr, empty_measurements, image_name="synthetic_empty.jpg")
    assert empty_annotated.shape == (h, w, 3)
    logger.info("Empty mask handled gracefully: returned 0 detections without exception")

    # 8. Export CSV Report
    logger.info("-" * 92)
    classifier.export_csv_report(all_csv_records, csv_path)

    # 9. Summary Statistics
    logger.info("=" * 92)
    logger.info("[STEP 4 LEAF CLASSIFICATION SUMMARY]")
    logger.info(f"  * Total Images Processed      : {total_images_processed}")
    logger.info(f"  * Total Regions Detected      : {total_detections}")
    logger.info(f"  * Small Leaves Classified     : {total_small_leaves} (Area < {config.classification.size_threshold_px:.0f} px)")
    logger.info(f"  * Large Leaves Classified     : {total_large_leaves} (Area >= {config.classification.size_threshold_px:.0f} px)")
    logger.info(f"  * Annotated Visuals Saved     : {len(saved_visual_paths)} images in {out_dir}")
    logger.info(f"  * Classification CSV Report   : {csv_path} ({len(all_csv_records)} rows)")
    logger.info("=" * 92)

    return 0


if __name__ == "__main__":
    sys.exit(run_classification_demonstration())
