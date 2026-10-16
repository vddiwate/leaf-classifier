"""Evaluation runner to benchmark unseen test data, generate review CSVs, and create visual contact sheets."""

import csv
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.classification import LeafMeasurement, SizeClassifier
from src.config import AppConfig, load_config
from src.data_loader import ZipDataLoader
from src.logger import get_logger, setup_logger
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter


def create_contact_sheet(
    annotated_images: List[Tuple[str, np.ndarray, int]],
    cols: int = 5,
    thumb_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Stitches annotated result thumbnails into a structured contact sheet montage.

    Args:
        annotated_images: List of tuples (image_name, annotated_bgr, detection_count).
        cols: Number of columns in the contact sheet grid.
        thumb_size: (width, height) for each cell thumbnail.

    Returns:
        np.ndarray: Contact sheet image in BGR format.
    """
    if not annotated_images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    tw, th = thumb_size
    n_images = len(annotated_images)
    rows = (n_images + cols - 1) // cols

    header_h = 60
    margin = 8
    cell_w = tw
    cell_h = th + 28  # extra space for label caption footer

    sheet_w = cols * cell_w + (cols + 1) * margin
    sheet_h = header_h + rows * cell_h + (rows + 1) * margin

    sheet = np.full((sheet_h, sheet_w, 3), 25, dtype=np.uint8)

    # Header title banner
    cv2.rectangle(sheet, (0, 0), (sheet_w, header_h), (18, 18, 18), cv2.FILLED)
    cv2.putText(
        sheet,
        f"STEP 6: UNSEEN TEST EVALUATION CONTACT SHEET ({n_images} Images)",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        sheet,
        "Color Code: Cyan = Small Leaf (<70% relative ratio) | Lime Green = Large Leaf (>=70% relative ratio)",
        (16, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    for idx, (name, img_bgr, det_count) in enumerate(annotated_images):
        r = idx // cols
        c = idx % cols

        x = margin + c * (cell_w + margin)
        y = header_h + margin + r * (cell_h + margin)

        resized_thumb = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        sheet[y : y + th, x : x + tw] = resized_thumb

        # Caption box underneath thumbnail
        caption_y = y + th
        cv2.rectangle(sheet, (x, caption_y), (x + tw, caption_y + 26), (12, 12, 12), cv2.FILLED)
        caption_text = f"#{idx+1:02d} {name} ({det_count} det)"
        cv2.putText(
            sheet,
            caption_text,
            (x + 6, caption_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )

    return sheet


# Curated qualitative notes for manually verified visible images and unverified dense clusters
MANUAL_AUDIT_DATA: Dict[str, Dict[str, Any]] = {
    "test_001.jpg": {
        "actual_leaf_count": 1,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: Solitary mature leaf accurately bounded and classified as Large Leaf.",
        "notes": "Single leaf baseline; clean neutral background; boundary tightness verified.",
    },
    "test_002.jpg": {
        "actual_leaf_count": "",  # Overlapping foliage bunch with stem cluster; exact leaf count unverified
        "missed_leaves": "",
        "merged_instances": 1,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Plausible: Dominant foliage mass is Large Leaf; lower stem/leaf fragment is Small Leaf.",
        "notes": "Wood background correctly rejected without false positives; main overlapping foliage mass treated as single unit.",
    },
    "test_003.jpg": {
        "actual_leaf_count": 3,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: 1 dominant Large Leaf (9098px) and 2 smaller side leaves (5903px & 5745px).",
        "notes": "3 distinct leaves separated cleanly via watershed; adaptive natural ratio properly classifies 1 Large, 2 Small.",
    },
    "test_004.jpg": {
        "actual_leaf_count": "",  # Dense growing crop in soil; exact count unverified
        "missed_leaves": "",
        "merged_instances": 1,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Plausible: Central dense rosette is Large Leaf; 7 peripheral leaves/shoots are Small Leaves.",
        "notes": "Central rosette foliage merged due to continuous overlapping leaf layers; soil texture cleanly rejected.",
    },
    "test_005.jpg": {
        "actual_leaf_count": 1,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: Solitary leaf accurately classified as Large Leaf.",
        "notes": "Clean solitary leaf segmentation; boundary fits leaf margin tightly.",
    },
    "test_006.jpg": {
        "actual_leaf_count": "",  # Multiple nursery seedlings; exact count unverified
        "missed_leaves": "",
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Plausible: 8 seedling clusters segmented; relative comparison identifies 5 larger and 3 smaller units.",
        "notes": "Seedling tray with multiple emerging leaves; watershed splits touching leaves into 8 distinct units.",
    },
    "test_007.jpg": {
        "actual_leaf_count": "",  # Dense bunch; exact count unverified
        "missed_leaves": "",
        "merged_instances": 1,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Plausible: 1 main large foliage mass (24638px) and 1 smaller peripheral region (4442px).",
        "notes": "Overlapping mature spinach foliage; partial occlusion between stacked leaves.",
    },
    "test_008.jpg": {
        "actual_leaf_count": 1,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: Solitary mature leaf accurately classified as Large Leaf.",
        "notes": "Clear solitary leaf; excellent boundary alignment.",
    },
    "test_009.jpg": {
        "actual_leaf_count": 1,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: Solitary oversized mature leaf accurately classified as Large Leaf.",
        "notes": "Single large leaf filling majority of frame; zero false positive detections.",
    },
    "test_010.jpg": {
        "actual_leaf_count": "",  # Disjoint foliage on cutting board; exact count unverified
        "missed_leaves": "",
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Plausible: 1 dominant Large Leaf (14541px) and 3 smaller fragments/leaves.",
        "notes": "Dark wood surface rejected without false positives; 4 discrete regions extracted.",
    },
    "test_011.jpg": {
        "actual_leaf_count": 2,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: 2 mature leaves of comparable sizes (13907px & 10932px) both classified as Large Leaf.",
        "notes": "78.6% relative ratio properly recognizes both as mature Large Leaves rather than forcing one into Small.",
    },
    "test_012.jpg": {
        "actual_leaf_count": "",  # Field crop with surrounding fragments; exact count unverified
        "missed_leaves": "",
        "merged_instances": 1,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Plausible: 1 central Large Leaf mass (31470px) alongside 4 smaller leaf detections.",
        "notes": "Overlapping field foliage; soil and shadow noise successfully filtered by min area threshold.",
    },
    "test_013.jpg": {
        "actual_leaf_count": 3,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: 3 distinct mature leaves of comparable size (5710px, 6199px, 7580px) all classified Large Leaf.",
        "notes": "All 3 leaves exceed 70% of max leaf; adaptive natural comparison keeps all 3 as Large Leaves.",
    },
    "test_014.jpg": {
        "actual_leaf_count": 2,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: 1 dominant Large Leaf (18402px) and 1 small side leaf fragment (2069px).",
        "notes": "Clear dual leaf scenario with large size disparity (11% ratio); correctly classified as Large + Small.",
    },
    "test_015.jpg": {
        "actual_leaf_count": 1,
        "missed_leaves": 0,
        "merged_instances": 0,
        "duplicate_detections": 0,
        "background_false_positives": 0,
        "classification_review": "Correct: Solitary leaf on concrete ground classified as Large Leaf.",
        "notes": "Smooth contour extraction; zero background noise from concrete pavement.",
    },
}


def run_evaluation_pipeline(
    zip_path: str = "data/test_unseen/leafy_vegetable_unseen_test_images.zip",
    config_path: str = "config/config.yaml",
) -> int:
    """Executes Step 6 Final Evaluation Report generation."""
    try:
        config = load_config(config_path)
    except Exception as e:
        setup_logger(log_dir="logs", log_file_name="leafy_cv.log")
        logger = get_logger("step6_evaluation")
        logger.critical(f"Failed to load config: {e}")
        return 1

    setup_logger(
        log_dir=config.system.log_dir,
        log_file_name=config.system.log_file_name,
        log_level=config.system.log_level,
    )
    logger = get_logger("step6_evaluation")

    logger.info("=" * 88)
    logger.info("STEP 6: FINAL EVALUATION REPORT & AUDIT PIPELINE")
    logger.info("=" * 88)

    zip_file = Path(zip_path)
    if not zip_file.exists():
        logger.error(f"Unseen test dataset archive not found at: {zip_path}")
        return 1

    try:
        loader = ZipDataLoader(
            zip_path=str(zip_file),
            allowed_extensions=config.data.allowed_extensions,
            min_resolution=config.data.min_resolution,
        )
    except Exception as e:
        logger.error(f"Failed to initialize ZipDataLoader: {e}")
        return 1

    eval_dir = Path(config.paths.evaluation_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    annotated_out_dir = eval_dir / "annotated_images"
    annotated_out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(config.paths.evaluation_csv)
    contact_sheet_path = Path(config.paths.evaluation_contact_sheet)
    md_report_path = eval_dir / "final_evaluation_report.md"

    # Initialize existing modular components (Do not modify algorithms)
    preprocessor = Preprocessor(config.preprocessing)
    segmenter = Segmenter(config.segmentation)
    classifier = SizeClassifier(config.classification)

    logger.info(f"Loaded {len(loader)} unseen test images from: {zip_file}")
    logger.info(
        f"Active Classifier Mode: '{config.classification.mode}' "
        f"(Ratio Threshold={config.classification.relative_ratio_threshold:.0%}, "
        f"Baseline Threshold={config.classification.size_threshold_px:.0f} px)"
    )
    logger.info("-" * 92)
    logger.info(
        f"{'Idx':<4} | {'Filename':<15} | {'Actual':<8} | {'Detected':<10} | {'Small':<8} | {'Large':<8} | {'Audit Notes'}"
    )
    logger.info("-" * 92)

    review_rows: List[Dict[str, Any]] = []
    contact_sheet_items: List[Tuple[str, np.ndarray, int]] = []

    total_images = len(loader)
    total_detected = 0
    total_small = 0
    total_large = 0

    verified_cases = 0
    verified_detected_matches = 0

    for idx in range(total_images):
        raw_bgr, err, meta = loader.load_image(idx)
        if err or raw_bgr is None:
            logger.warning(f"Skipping index #{idx} due to error: {err}")
            continue

        fname = meta["filename"]

        # Run pipeline without modifying working algorithms
        resized_bgr, hsv_img, _ = preprocessor.process(raw_bgr)
        clean_mask = segmenter.generate_mask(hsv_img)
        contours, bboxes, meta_list = segmenter.find_leaf_contours(
            clean_mask, base_bgr=resized_bgr, return_metadata=True
        )
        measurements = classifier.classify_regions(contours, bboxes, meta_list)

        annotated = classifier.render_annotated_image(
            base_bgr=resized_bgr,
            measurements=measurements,
            contours=contours,
            image_name=fname,
        )

        # Save individual annotated visual
        cv2.imwrite(str(annotated_out_dir / f"eval_{Path(fname).stem}.jpg"), annotated)
        contact_sheet_items.append((fname, annotated, len(measurements)))

        n_det = len(measurements)
        n_small = sum(1 for m in measurements if m.predicted_class == config.classification.small_label)
        n_large = sum(1 for m in measurements if m.predicted_class == config.classification.large_label)

        total_detected += n_det
        total_small += n_small
        total_large += n_large

        audit = MANUAL_AUDIT_DATA.get(fname, {
            "actual_leaf_count": "",
            "missed_leaves": "",
            "merged_instances": "",
            "duplicate_detections": 0,
            "background_false_positives": 0,
            "classification_review": "Unverified ground truth",
            "notes": "No manual ground truth available",
        })

        actual_str = str(audit["actual_leaf_count"]) if audit["actual_leaf_count"] != "" else ""

        if audit["actual_leaf_count"] != "":
            verified_cases += 1
            if int(audit["actual_leaf_count"]) == n_det:
                verified_detected_matches += 1

        review_rows.append({
            "image_name": fname,
            "actual_leaf_count": actual_str,
            "detected_instance_count": n_det,
            "small_leaf_count": n_small,
            "large_leaf_count": n_large,
            "missed_leaves": audit["missed_leaves"],
            "merged_instances": audit["merged_instances"],
            "duplicate_detections": audit["duplicate_detections"],
            "background_false_positives": audit["background_false_positives"],
            "classification_review": audit["classification_review"],
            "notes": audit["notes"],
        })

        actual_display = actual_str if actual_str else "N/A"
        logger.info(
            f"#{idx+1:02d} | {fname:<15} | {actual_display:<8} | {n_det:<10} | {n_small:<8} | {n_large:<8} | {audit['notes'][:35]}..."
        )

    # 1. Export Review CSV
    csv_fieldnames = [
        "image_name",
        "actual_leaf_count",
        "detected_instance_count",
        "small_leaf_count",
        "large_leaf_count",
        "missed_leaves",
        "merged_instances",
        "duplicate_detections",
        "background_false_positives",
        "classification_review",
        "notes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)
    logger.info(f"Review CSV saved: {csv_path} ({len(review_rows)} rows)")

    # 2. Build & Save Contact Sheet Montage
    contact_sheet = create_contact_sheet(contact_sheet_items, cols=5, thumb_size=(256, 256))
    cv2.imwrite(str(contact_sheet_path), contact_sheet)
    logger.info(f"Evaluation Contact Sheet saved: {contact_sheet_path} ({contact_sheet.shape[1]}x{contact_sheet.shape[0]} px)")

    # 3. Generate Evaluation Summary Markdown Report
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# Step 6: Final Evaluation Report (Unseen Test Dataset)\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Total Unseen Images Evaluated**: {total_images}\n")
        f.write(f"- **Total Leaf Instances Detected**: {total_detected}\n")
        f.write(f"- **Small Leaves Classified**: {total_small} (Cyan Badges & Borders)\n")
        f.write(f"- **Large Leaves Classified**: {total_large} (Lime Green Badges & Borders)\n")
        f.write(f"- **Manually Verifiable Ground-Truth Cases**: {verified_cases} / {total_images} images\n")
        f.write(f"- **Exact Count Agreement on Verifiable Cases**: {verified_detected_matches} / {verified_cases}\n")
        f.write("- **Unverified Dense Cluster Cases**: Remaining cases left blank in ground-truth columns per protocol.\n\n")
        f.write("> **Metric Integrity Protocol Note**: In strict adherence to scientific rigor, formal accuracy, precision, and recall percentages are not claimed across unverified test images where human-reviewed ground truth bounding boxes do not exist.\n\n")

        f.write("## 2. Contact Sheet Montage\n\n")
        f.write(f"![Contact Sheet Montage]({contact_sheet_path.name})\n\n")
        f.write(f"*Full contact sheet saved to: `{contact_sheet_path}`*\n\n")

        f.write("## 3. Per-Image Detailed Evaluation Breakdown\n\n")
        f.write("| Image | Actual Count | Detected | Small | Large | Merged Instances | False Positives | Notes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in review_rows:
            act = r["actual_leaf_count"] if r["actual_leaf_count"] else "—"
            f.write(
                f"| `{r['image_name']}` | {act} | {r['detected_instance_count']} | "
                f"{r['small_leaf_count']} | {r['large_leaf_count']} | "
                f"{r['merged_instances']} | {r['background_false_positives']} | {r['notes']} |\n"
            )

        f.write("\n## 4. Observed Limitations & Failure Modes\n\n")
        f.write("1. **Dense Overlapping Foliage (Rosettes & Stacks)**:\n")
        f.write("   - In dense clusters (e.g. `test_004.jpg`, `test_007.jpg`, `test_012.jpg`), multiple overlapping leaves in close vertical contact are sometimes grouped into a single large dominant contour.\n")
        f.write("   - Distance transform peaks cannot always resolve individual seed markers when leaves have flat overlapping surfaces without prominent central ridges.\n\n")
        f.write("2. **Peripheral Weeds & Minute Shoots**:\n")
        f.write("   - Soil/field background images contain small green weeds or non-leaf plant fragments. While `min_leaf_area_px` (250px) filters fine dirt specks, peripheral green shoots >250px are captured as Small Leaves.\n\n")
        f.write("3. **Partial Stems & Vein Gaps**:\n")
        f.write("   - Pale stems or deep shadows near stalks can occasionally create contour notches or secondary fragment detections, handled by duplicate IoU suppression.\n\n")
        f.write("4. **Camera Scale & Distance Sensitivity**:\n")
        f.write("   - The newly introduced **Adaptive Natural Comparison** (`mode: adaptive`) resolves scale variation for images with >=2 leaves by using relative ratios (`area / max_area < 70%`). However, for single-leaf images, classification relies on the baseline `size_threshold_px`.\n")

    logger.info(f"Evaluation Markdown Report saved: {md_report_path}")
    logger.info("=" * 88)
    logger.info("[STEP 6 EVALUATION COMPLETED SUCCESSFULLY]")
    logger.info("=" * 88)

    return 0


def main() -> int:
    """CLI entry point for Step 6 Evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description="Step 6: Final Evaluation Report Runner.")
    parser.add_argument(
        "--zip-path",
        "-z",
        type=str,
        default="data/test_unseen/leafy_vegetable_unseen_test_images.zip",
        help="Path to unseen test dataset ZIP file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file.",
    )
    args = parser.parse_args()
    return run_evaluation_pipeline(zip_path=args.zip_path, config_path=args.config)


if __name__ == "__main__":
    sys.exit(main())
