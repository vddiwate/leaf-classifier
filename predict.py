"""Inference runner to process unseen images, directories, or ZIP archives."""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.classification import LeafMeasurement, SizeClassifier
from src.config import load_config
from src.data_loader import ZipDataLoader
from src.logger import get_logger, setup_logger
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter


def process_single_image(
    image_bgr: np.ndarray,
    image_name: str,
    preprocessor: Preprocessor,
    segmenter: Segmenter,
    classifier: SizeClassifier,
) -> Tuple[np.ndarray, List[LeafMeasurement], List[Dict[str, Any]]]:
    """Executes preprocessing, segmentation, and size classification on a single image."""
    # 1. Preprocessing (Resize + HSV conversion)
    resized_bgr, hsv_img, _ = preprocessor.process(image_bgr)

    # 2. Step 3: Segmentation (Color mask + Marker-Controlled Watershed)
    clean_mask = segmenter.generate_mask(hsv_img)
    contours, bboxes, meta_list = segmenter.find_leaf_contours(
        clean_mask, base_bgr=resized_bgr, return_metadata=True
    )

    # 3. Step 4: Size Classification
    measurements = classifier.classify_regions(
        contours=contours,
        bboxes=bboxes,
        metadata_list=meta_list,
    )

    # 4. Render high-contrast visual with distinct class borders
    annotated_bgr = classifier.render_annotated_image(
        base_bgr=resized_bgr,
        measurements=measurements,
        contours=contours,
        image_name=image_name,
    )

    csv_records = [m.to_dict(image_name=image_name) for m in measurements]
    return annotated_bgr, measurements, csv_records


def run_inference(
    source_path: str,
    config_path: str = "config/config.yaml",
    output_dir: Optional[str] = None,
) -> int:
    """Executes prediction on unseen data (single image, folder, or zip)."""
    # 1. Load Central Configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        setup_logger(log_dir="logs", log_file_name="leafy_cv.log")
        logger = get_logger("predict")
        logger.critical(f"Failed to load configuration from {config_path}: {e}")
        return 1

    setup_logger(
        log_dir=config.system.log_dir,
        log_file_name=config.system.log_file_name,
        log_level=config.system.log_level,
    )
    logger = get_logger("predict")

    logger.info("=" * 88)
    logger.info("INFERENCE PIPELINE: EVALUATING UNSEEN DATA")
    logger.info("=" * 88)

    src = Path(source_path)
    if not src.exists():
        logger.error(f"Source path not found: {source_path}")
        return 1

    out_path = Path(output_dir) if output_dir else Path(config.paths.output_dir) / "unseen_predictions"
    out_path.mkdir(parents=True, exist_ok=True)
    csv_report_path = out_path / "prediction_report.csv"

    # Initialize modular components
    preprocessor = Preprocessor(config.preprocessing)
    segmenter = Segmenter(config.segmentation)
    classifier = SizeClassifier(config.classification)

    images_to_process: List[Tuple[str, np.ndarray]] = []
    allowed_exts = set(config.data.allowed_extensions)

    # Case A: Source is a single image file
    if src.is_file() and src.suffix.lower() in allowed_exts:
        img = cv2.imread(str(src))
        if img is None:
            logger.error(f"Failed to read image file: {src}")
            return 1
        images_to_process.append((src.name, img))

    # Case B: Source is a ZIP archive
    elif src.is_file() and src.suffix.lower() == ".zip":
        try:
            loader = ZipDataLoader(
                zip_path=str(src),
                allowed_extensions=config.data.allowed_extensions,
                min_resolution=config.data.min_resolution,
            )
            for idx in range(len(loader)):
                img, err, meta = loader.load_image(idx)
                if img is not None and not err:
                    images_to_process.append((meta["filename"], img))
        except Exception as e:
            logger.error(f"Failed to open ZIP archive {src}: {e}")
            return 1

    # Case C: Source is a directory of images (or contains ZIP archives)
    elif src.is_dir():
        for file_path in sorted(src.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() == ".zip":
                try:
                    loader = ZipDataLoader(
                        zip_path=str(file_path),
                        allowed_extensions=config.data.allowed_extensions,
                        min_resolution=config.data.min_resolution,
                    )
                    for idx in range(len(loader)):
                        img, err, meta = loader.load_image(idx)
                        if img is not None and not err:
                            images_to_process.append((meta["filename"], img))
                except Exception as e:
                    logger.error(f"Failed to open ZIP archive {file_path}: {e}")
            elif file_path.suffix.lower() in allowed_exts:
                img = cv2.imread(str(file_path))
                if img is not None:
                    images_to_process.append((file_path.name, img))
                else:
                    logger.warning(f"Could not decode image: {file_path.name}")

    if not images_to_process:
        logger.warning(f"No valid images found in {source_path}")
        return 1

    logger.info(f"Loaded {len(images_to_process)} unseen image(s) from: {source_path}")
    logger.info("-" * 92)
    logger.info(
        f"{'Filename':<24} | {'Detections':<10} | {'Small Leaves':<12} | {'Large Leaves':<12} | {'Status'}"
    )
    logger.info("-" * 92)

    all_csv_records: List[Dict[str, Any]] = []
    total_small = 0
    total_large = 0

    for name, bgr in images_to_process:
        annotated, measurements, records = process_single_image(
            image_bgr=bgr,
            image_name=name,
            preprocessor=preprocessor,
            segmenter=segmenter,
            classifier=classifier,
        )

        all_csv_records.extend(records)
        small_cnt = sum(1 for m in measurements if m.predicted_class == config.classification.small_label)
        large_cnt = sum(1 for m in measurements if m.predicted_class == config.classification.large_label)
        total_small += small_cnt
        total_large += large_cnt

        save_name = f"pred_{Path(name).stem}.jpg"
        save_file = out_path / save_name
        cv2.imwrite(str(save_file), annotated)

        logger.info(
            f"{name:<24} | {len(measurements):<10} | {small_cnt:<12} | {large_cnt:<12} | Saved {save_name}"
        )

    # Export structured CSV report
    classifier.export_csv_report(all_csv_records, csv_report_path)

    logger.info("=" * 92)
    logger.info("[UNSEEN INFERENCE SUMMARY]")
    logger.info(f"  * Total Images Evaluated   : {len(images_to_process)}")
    logger.info(f"  * Total Leaves Detected    : {len(all_csv_records)}")
    logger.info(f"  * Small Leaves Classified  : {total_small}")
    logger.info(f"  * Large Leaves Classified  : {total_large}")
    logger.info(f"  * Annotated Visuals Saved  : {out_path}")
    logger.info(f"  * Prediction CSV Report    : {csv_report_path}")
    logger.info("=" * 92)

    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run Leaf Size Classification on unseen images, folders, or ZIP archives."
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Path to an unseen image file (.jpg, .png), a folder containing images, or a ZIP dataset.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/unseen_predictions",
        help="Directory to save visual predictions and CSV report (default: outputs/unseen_predictions).",
    )

    args = parser.parse_args()
    return run_inference(
        source_path=args.source,
        config_path=args.config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
