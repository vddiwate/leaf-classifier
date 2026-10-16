"""Leaf size classification using adaptive relative comparison and geometric feature extraction."""

import csv
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import ClassificationConfig
from src.logger import get_logger

logger = get_logger("classification")


@dataclass
class LeafMeasurement:
    """Exact geometric measurements and predicted size class for a detected leaf region."""

    detection_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    contour_area_px: float
    bbox_area_px: float
    aspect_ratio: float
    extent: float
    predicted_class: str
    measured_area_px: float = 0.0  # Exact area used for classification & display based on area_metric
    is_cluster: bool = False

    def to_dict(self, image_name: Optional[str] = None) -> Dict[str, Any]:
        """Converts measurement into a dictionary record suitable for CSV export."""
        rec = {
            "detection_id": self.detection_id,
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "contour_area": round(self.contour_area_px, 1),
            "bbox_area": round(self.bbox_area_px, 1),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "predicted_class": self.predicted_class,
        }
        if image_name is not None:
            rec = {"image_name": image_name, **rec}
        return rec


class SizeClassifier:
    """Modular classifier that extracts geometric properties and predicts leaf size categories."""

    # High-contrast BGR colors for visualization
    COLOR_SMALL_LEAF = (255, 215, 0)     # Vivid Cyan in BGR
    COLOR_LARGE_LEAF = (50, 225, 50)     # Vivid Lime Green in BGR
    COLOR_CLUSTER    = (0, 140, 255)     # Amber / orange in BGR
    COLOR_CONTOUR    = (0, 255, 0)       # Green contour outline

    def __init__(self, config: ClassificationConfig):
        """Initializes SizeClassifier with validated configuration."""
        if not isinstance(config, ClassificationConfig):
            msg = f"config must be an instance of ClassificationConfig, got {type(config).__name__}"
            logger.error(msg)
            raise TypeError(msg)
        self.config = config
        self.color_small = getattr(self.config, "small_color_bgr", self.COLOR_SMALL_LEAF)
        self.color_large = getattr(self.config, "large_color_bgr", self.COLOR_LARGE_LEAF)
        logger.debug(
            f"Initialized SizeClassifier: threshold={self.config.size_threshold_px} px, "
            f"metric='{self.config.area_metric}', min_area={self.config.min_leaf_area_px} px, "
            f"small='{self.config.small_label}', large='{self.config.large_label}', "
            f"suppress_duplicates={self.config.suppress_duplicates}"
        )

    @staticmethod
    def compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculates Intersection-over-Union (IoU) between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = float(inter_w * inter_h)
        area1 = float(w1 * h1)
        area2 = float(w2 * h2)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def compute_containment(bbox_inner: Tuple[int, int, int, int], bbox_outer: Tuple[int, int, int, int]) -> float:
        """Calculates the fraction of bbox_inner that is contained inside bbox_outer."""
        x1, y1, w1, h1 = bbox_inner
        x2, y2, w2, h2 = bbox_outer
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = float(inter_w * inter_h)
        inner_area = float(w1 * h1)
        return inter_area / inner_area if inner_area > 0 else 0.0

    def extract_measurement(
        self,
        contour: np.ndarray,
        bbox: Tuple[int, int, int, int],
        detection_id: int = 1,
        is_cluster: bool = False,
    ) -> LeafMeasurement:
        """Extracts required geometric features from a contour and bounding box,
        and assigns size classification according to configurable thresholds.

        Extracted properties:
        - Bounding box width and height
        - Contour area in pixels
        - Bounding box area in pixels
        - Aspect ratio (width / height)
        - Extent (contour area / bounding box area)
        - Measured area (contour area or bbox area based on area_metric)

        Returns:
            LeafMeasurement dataclass with all metrics and predicted size class.
        """
        if contour is None or not isinstance(contour, np.ndarray):
            msg = "Contour must be a non-None numpy.ndarray"
            logger.error(msg)
            raise TypeError(msg)

        if len(bbox) != 4:
            msg = f"Bounding box must be (x, y, w, h), got {bbox}"
            logger.error(msg)
            raise ValueError(msg)

        bx, by, bw, bh = [int(v) for v in bbox]
        if bw < 0 or bh < 0:
            msg = f"Bounding box dimensions cannot be negative: ({bw}, {bh})"
            logger.error(msg)
            raise ValueError(msg)

        # 1. Contour area in pixels
        contour_area = float(cv2.contourArea(contour))

        # 2. Bounding box area in pixels
        bbox_area = float(bw * bh)

        # 3. Aspect ratio: width / height (handles height=0 gracefully)
        aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

        # 4. Extent: contour area / bounding box area
        extent = contour_area / bbox_area if bbox_area > 0 else 0.0

        # 5. Classify Small vs Large using configurable metric and threshold
        # Consistently records measured_area_px for display and threshold verification
        if self.config.area_metric == "bbox_area":
            measured_area = bbox_area
        else:
            measured_area = contour_area

        if measured_area >= self.config.size_threshold_px:
            predicted_class = self.config.large_label
        else:
            predicted_class = self.config.small_label

        return LeafMeasurement(
            detection_id=int(detection_id),
            bbox=(bx, by, bw, bh),
            bbox_x=bx,
            bbox_y=by,
            bbox_width=bw,
            bbox_height=bh,
            contour_area_px=contour_area,
            bbox_area_px=bbox_area,
            aspect_ratio=aspect_ratio,
            extent=extent,
            predicted_class=predicted_class,
            measured_area_px=measured_area,
            is_cluster=bool(is_cluster),
        )

    def classify_regions(
        self,
        contours: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[LeafMeasurement]:
        """Classifies a list of detected contours and bounding boxes.
        Applies configurable minimum-area filtering to suppress noise specks,
        and optionally filters heavily overlapping/duplicate fragments.
        Handles empty masks / zero detected regions gracefully.

        Returns:
            List[LeafMeasurement]: List of classified leaf measurements (empty list if no contours).
        """
        if not contours or not bboxes:
            logger.debug("No contours or bounding boxes provided for classification (empty mask)")
            return []

        if len(contours) != len(bboxes):
            logger.warning(
                f"Mismatch between contours count ({len(contours)}) and bounding boxes count ({len(bboxes)})"
            )

        raw_measurements: List[LeafMeasurement] = []
        for idx, (c, bbox) in enumerate(zip(contours, bboxes), start=1):
            is_cluster = False
            if metadata_list and idx - 1 < len(metadata_list):
                is_cluster = bool(metadata_list[idx - 1].get("is_cluster", False))

            measurement = self.extract_measurement(
                contour=c,
                bbox=bbox,
                detection_id=idx,
                is_cluster=is_cluster,
            )

            # Minimum area filtering: discard tiny background noise specks and minute fragments
            if measurement.measured_area_px < self.config.min_leaf_area_px:
                logger.debug(
                    f"Filtered tiny speck detection #{idx}: area={measurement.measured_area_px:.0f}px "
                    f"< min_leaf_area_px={self.config.min_leaf_area_px:.0f}px"
                )
                continue

            raw_measurements.append(measurement)

        # Duplicate / Fragment Suppression if enabled
        if self.config.suppress_duplicates and len(raw_measurements) > 1:
            filtered: List[LeafMeasurement] = []
            # Sort by area descending so dominant leaf regions take precedence over duplicate fragments
            sorted_measurements = sorted(raw_measurements, key=lambda m: m.measured_area_px, reverse=True)
            for m in sorted_measurements:
                is_dup = False
                for keep in filtered:
                    iou = self.compute_iou(m.bbox, keep.bbox)
                    if iou >= self.config.duplicate_iou_threshold:
                        logger.debug(
                            f"Suppressed duplicate fragment #{m.detection_id} ({m.measured_area_px:.0f}px) "
                            f"overlapping with #{keep.detection_id} (IoU={iou:.2f})"
                        )
                        is_dup = True
                        break
                if not is_dup:
                    filtered.append(m)
            # Re-sort by spatial position (top-to-bottom)
            raw_measurements = sorted(filtered, key=lambda m: (m.bbox_y, m.bbox_x))

        # Re-index sequential detection IDs
        for new_id, m in enumerate(raw_measurements, start=1):
            m.detection_id = new_id

        # Natural / Adaptive Size Classification when mode == "adaptive"
        if getattr(self.config, "mode", "fixed") == "adaptive":
            if len(raw_measurements) >= 2:
                max_area = max(m.measured_area_px for m in raw_measurements)
                # If all detected leaves in the frame are below min_large_leaf_area_px, all are Small Leaves
                if max_area < self.config.min_large_leaf_area_px:
                    for m in raw_measurements:
                        m.predicted_class = self.config.small_label
                else:
                    for m in raw_measurements:
                        ratio = m.measured_area_px / max_area
                        if ratio < self.config.relative_ratio_threshold:
                            m.predicted_class = self.config.small_label
                        else:
                            m.predicted_class = self.config.large_label
            elif len(raw_measurements) == 1:
                # Solitary single leaf: evaluate against baseline size_threshold_px
                m = raw_measurements[0]
                if m.measured_area_px >= self.config.size_threshold_px:
                    m.predicted_class = self.config.large_label
                else:
                    m.predicted_class = self.config.small_label

        small_count = sum(1 for m in raw_measurements if m.predicted_class == self.config.small_label)
        large_count = sum(1 for m in raw_measurements if m.predicted_class == self.config.large_label)
        logger.debug(
            f"Classified {len(raw_measurements)} regions ({getattr(self.config, 'mode', 'fixed')} mode): "
            f"{small_count} {self.config.small_label}, {large_count} {self.config.large_label}"
        )
        return raw_measurements

    def render_annotated_image(
        self,
        base_bgr: np.ndarray,
        measurements: List[LeafMeasurement],
        contours: Optional[List[np.ndarray]] = None,
        image_name: Optional[str] = None,
    ) -> np.ndarray:
        """Generates an annotated visualization showing:
        - Contour outlines
        - Color-coded bounding boxes per class (Cyan for Small, Lime Green for Large)
        - Labels displaying detection ID, predicted size class, and the EXACT measured area used for classification
        - High-contrast background tags to guarantee readability
        - Summary header banner

        Returns:
            np.ndarray: Annotated BGR image.
        """
        if base_bgr is None or not isinstance(base_bgr, np.ndarray):
            msg = "base_bgr must be a valid numpy.ndarray"
            logger.error(msg)
            raise TypeError(msg)

        if base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
            msg = f"base_bgr must have shape (H, W, 3), got {base_bgr.shape}"
            logger.error(msg)
            raise ValueError(msg)

        annotated = base_bgr.copy()
        h, w = annotated.shape[:2]

        # Handle empty mask / no detections gracefully
        if not measurements:
            banner_text = f"{image_name + ': ' if image_name else ''}No leaves detected (Empty Mask / Below Min Area)"
            cv2.rectangle(annotated, (0, 0), (w, 28), (30, 30, 30), cv2.FILLED)
            cv2.putText(
                annotated,
                banner_text,
                (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return annotated

        # 1. Draw top summary header bar FIRST so it never overwrites leaf badges
        small_cnt = sum(1 for m in measurements if m.predicted_class == self.config.small_label)
        large_cnt = sum(1 for m in measurements if m.predicted_class == self.config.large_label)
        mode_tag = (
            f"Adaptive (ratio<{self.config.relative_ratio_threshold:.0%})"
            if getattr(self.config, "mode", "fixed") == "adaptive"
            else f"Thresh={self.config.size_threshold_px:.0f}px"
        )
        header_text = f"Step 4: {small_cnt} Small, {large_cnt} Large | {mode_tag}"
        if image_name:
            header_text = f"[{image_name}] {header_text}"

        header_h = 24
        cv2.rectangle(annotated, (0, 0), (w, header_h), (20, 20, 20), cv2.FILLED)
        cv2.putText(
            annotated,
            header_text,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 2. Draw contour outlines with distinct border colors per class (Small vs Large)
        if contours:
            for i, c in enumerate(contours):
                if i < len(measurements):
                    m = measurements[i]
                    col = self.color_small if m.predicted_class == self.config.small_label else self.color_large
                else:
                    col = self.color_large
                cv2.drawContours(annotated, [c], -1, color=col, thickness=2)

        # 3. Draw bounding boxes with matching distinct border colors per class
        for m in measurements:
            bx, by, bw, bh = m.bbox
            col = self.color_small if m.predicted_class == self.config.small_label else self.color_large
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), col, 2)

        # 4. Anti-collision solid badge labels with bold crisp text
        placed_boxes = [(0, 0, w, header_h)]  # Reserve top header banner area
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        font_thick = 1

        for m in measurements:
            label = f"#{m.detection_id} {m.predicted_class} ({m.measured_area_px:.0f}px)"
            col = self.color_small if m.predicted_class == self.config.small_label else self.color_large
            (tw, th), bl = cv2.getTextSize(label, font, font_scale, font_thick)
            badge_w = tw + 8
            badge_h = th + bl + 6
            bx, by, bw, bh = m.bbox

            # Evaluate candidate anchor locations in order of preference
            candidates = [
                (bx, by - badge_h - 2),               # Above bounding box
                (bx, by + 3),                          # Top-left inside bounding box
                (bx + bw - badge_w, by + 3),          # Top-right inside bounding box
                (bx, by + bh - badge_h - 3),         # Bottom-left inside bounding box
                (bx, by + bh + 2),                     # Below bounding box
            ]

            best_pos = None
            for cx, cy in candidates:
                cx = max(2, min(cx, w - badge_w - 2))
                cy = max(header_h + 2, min(cy, h - badge_h - 2))
                collides = False
                for px, py, pw, ph in placed_boxes:
                    if not (cx + badge_w <= px or cx >= px + pw or cy + badge_h <= py or cy >= py + ph):
                        collides = True
                        break
                if not collides:
                    best_pos = (cx, cy)
                    break

            if best_pos is None:
                # Scan vertically for a clear non-overlapping slot
                for step_y in range(header_h + 4, h - badge_h - 2, 8):
                    cx = max(2, min(bx, w - badge_w - 2))
                    collides = False
                    for px, py, pw, ph in placed_boxes:
                        if not (cx + badge_w <= px or cx >= px + pw or step_y + badge_h <= py or step_y >= py + ph):
                            collides = True
                            break
                    if not collides:
                        best_pos = (cx, step_y)
                        break

            if best_pos is None:
                best_pos = (max(2, min(bx, w - badge_w - 2)), max(header_h + 2, min(by + 3, h - badge_h - 2)))

            placed_boxes.append((best_pos[0], best_pos[1], badge_w, badge_h))

            # Draw solid high-contrast pill badge with black text
            cv2.rectangle(
                annotated,
                (best_pos[0], best_pos[1]),
                (best_pos[0] + badge_w, best_pos[1] + badge_h),
                col,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (best_pos[0] + 4, best_pos[1] + th + 2),
                font,
                font_scale,
                (10, 10, 10),
                font_thick,
                cv2.LINE_AA,
            )

        return annotated

    @staticmethod
    def export_csv_report(
        records: List[Dict[str, Any]],
        csv_path: Union[str, Path],
    ) -> Path:
        """Exports classification results to a structured CSV file.
        Contains:
        image_name, detection_id, bbox_x, bbox_y, bbox_width, bbox_height,
        contour_area, bbox_area, aspect_ratio, predicted_class.

        Args:
            records: List of dictionaries matching the required schema.
            csv_path: Output CSV file path.

        Returns:
            Path: Path to saved CSV file.
        """
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
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

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for rec in records:
                    writer.writerow(rec)
            logger.info(f"Successfully exported {len(records)} classification records to {path}")
            return path
        except Exception as e:
            msg = f"Failed to export CSV report to '{csv_path}': {e}"
            logger.error(msg)
            raise IOError(msg) from e
