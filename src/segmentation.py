"""Leaf segmentation, morphological cleanup, and touching-leaf watershed separation."""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import SegmentationConfig
from src.logger import get_logger

logger = get_logger("segmentation")


class Segmenter:
    """Production segmentation engine for leafy vegetables."""

    def __init__(self, config: SegmentationConfig):
        if not isinstance(config, SegmentationConfig):
            raise TypeError(f"Expected SegmentationConfig instance, got {type(config).__name__}")
        self.config = config

        # Pre-compute HSV boundary numpy arrays
        self.green_lower = np.array(self.config.green_hsv_lower, dtype=np.uint8)
        self.green_upper = np.array(self.config.green_hsv_upper, dtype=np.uint8)

        if self.config.include_yellow_tones:
            self.yellow_lower = np.array(self.config.yellow_hsv_lower, dtype=np.uint8)
            self.yellow_upper = np.array(self.config.yellow_hsv_upper, dtype=np.uint8)

        # Map kernel shape name to OpenCV constant
        shape_map = {
            "MORPH_ELLIPSE": cv2.MORPH_ELLIPSE,
            "MORPH_RECT": cv2.MORPH_RECT,
            "MORPH_CROSS": cv2.MORPH_CROSS,
        }
        morph_shape = shape_map[self.config.morph_kernel_shape]

        # Decoupled morphological kernels:
        # 1. Opening kernel: Smaller (e.g. 3x3) to remove fine isolated noise specks without eroding thin petioles/stems
        # 2. Closing kernel: Larger (e.g. 5x5) to bridge leaf veins, join margins, and fill internal leaf shadows
        open_size = getattr(self.config, "open_kernel_size", self.config.morph_kernel_size)
        close_size = getattr(self.config, "close_kernel_size", self.config.morph_kernel_size)

        self.open_kernel = cv2.getStructuringElement(morph_shape, tuple(open_size))
        self.close_kernel = cv2.getStructuringElement(morph_shape, tuple(close_size))
        self.kernel = self.close_kernel  # Backward compatibility reference

        logger.debug(
            f"Initialized Segmenter: green_bounds=[{self.config.green_hsv_lower} - {self.config.green_hsv_upper}], "
            f"yellow_tones={self.config.include_yellow_tones}, "
            f"open_kernel={open_size} (iters={self.config.open_iterations}), "
            f"close_kernel={close_size} (iters={self.config.close_iterations}), "
            f"min_area={self.config.min_contour_area_px}px"
        )

    def validate_hsv_image(self, hsv_image: np.ndarray) -> Tuple[int, int]:
        """Validates that input HSV array meets production criteria."""
        if hsv_image is None:
            msg = "HSV input image cannot be None"
            logger.error(msg)
            raise TypeError(msg)

        if not isinstance(hsv_image, np.ndarray):
            msg = f"Expected numpy.ndarray, got {type(hsv_image).__name__}"
            logger.error(msg)
            raise TypeError(msg)

        if hsv_image.size == 0 or hsv_image.ndim != 3:
            msg = f"Expected non-empty 3-dimensional array, got shape {getattr(hsv_image, 'shape', None)}"
            logger.error(msg)
            raise ValueError(msg)

        if hsv_image.shape[2] != 3:
            msg = f"Expected 3 channels (H, S, V), got {hsv_image.shape[2]} channels"
            logger.error(msg)
            raise ValueError(msg)

        if hsv_image.dtype != np.uint8:
            msg = f"Expected uint8 dtype, got {hsv_image.dtype}"
            logger.error(msg)
            raise TypeError(msg)

        return hsv_image.shape[:2]

    def generate_raw_and_clean_masks(
        self, hsv_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates both raw (unfiltered) and cleaned (morphologically refined) masks.

        Returns:
            Tuple[clean_mask, raw_mask]: Both 2D uint8 arrays (0 or 255).
        """
        h, w = self.validate_hsv_image(hsv_image)

        # 1. Primary green foliage mask
        mask_green = cv2.inRange(hsv_image, self.green_lower, self.green_upper)

        # 2. Optional yellow foliage mask (combined via bitwise OR)
        if self.config.include_yellow_tones:
            mask_yellow = cv2.inRange(hsv_image, self.yellow_lower, self.yellow_upper)
            raw_mask = cv2.bitwise_or(mask_green, mask_yellow)
        else:
            raw_mask = mask_green

        # 3. Decoupled Morphological Opening (Erosion -> Dilation with open_kernel)
        # Gentle opening removes isolated single-pixel background noise without eroding delicate leaf stems
        opened_mask = cv2.morphologyEx(
            raw_mask, cv2.MORPH_OPEN, self.open_kernel, iterations=self.config.open_iterations
        )

        # 4. Decoupled Morphological Closing (Dilation -> Erosion with close_kernel)
        # Closing bridges leaf vein gaps and seals internal shadow holes
        clean_mask = cv2.morphologyEx(
            opened_mask, cv2.MORPH_CLOSE, self.close_kernel, iterations=self.config.close_iterations
        )

        # Verify output
        if clean_mask.shape != (h, w) or clean_mask.dtype != np.uint8:
            msg = f"Output mask shape/type mismatch: expected {(h, w)} uint8, got {clean_mask.shape} {clean_mask.dtype}"
            logger.error(msg)
            raise RuntimeError(msg)

        return clean_mask, raw_mask

    def generate_mask(self, hsv_image: np.ndarray) -> np.ndarray:
        """Generates a cleaned binary segmentation mask from an HSV image.

        Returns:
            np.ndarray: Binary mask (2D array, uint8, values 0 or 255).
        """
        clean_mask, _ = self.generate_raw_and_clean_masks(hsv_image)
        return clean_mask

    @staticmethod
    def compute_tight_bounding_box(contour: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculates a tight bounding box directly from contour point coordinates.

        Returns:
            Tuple[int, int, int, int]: (x, y, w, h) strictly derived from contour min/max coordinates.
        """
        pts = contour.reshape(-1, 2)
        min_x = int(np.min(pts[:, 0]))
        max_x = int(np.max(pts[:, 0]))
        min_y = int(np.min(pts[:, 1]))
        max_y = int(np.max(pts[:, 1]))
        w = max_x - min_x + 1
        h = max_y - min_y + 1
        return (min_x, min_y, w, h)

    @staticmethod
    def verify_bounding_box_tightness(
        contour: np.ndarray,
        bbox: Tuple[int, int, int, int],
        tolerance_px: int = 1,
    ) -> bool:
        """Verifies that the bounding box tightly encloses all detected contour coordinates.

        Verifies:
        1. All contour vertices reside inside or on the bounding box boundaries.
        2. At least one contour vertex touches each of the 4 bounding box borders within tolerance.

        Returns:
            bool: True if bounding box is tightly fitted, False otherwise.
        """
        pts = contour.reshape(-1, 2)
        min_x = int(np.min(pts[:, 0]))
        max_x = int(np.max(pts[:, 0]))
        min_y = int(np.min(pts[:, 1]))
        max_y = int(np.max(pts[:, 1]))

        bx, by, bw, bh = bbox

        # Check edge alignment
        left_tight = abs(bx - min_x) <= tolerance_px
        top_tight = abs(by - min_y) <= tolerance_px
        right_tight = abs((bx + bw - 1) - max_x) <= tolerance_px
        bottom_tight = abs((by + bh - 1) - max_y) <= tolerance_px

        # Check all points inside
        all_inside = bool(
            np.all(pts[:, 0] >= bx - tolerance_px)
            and np.all(pts[:, 0] <= bx + bw - 1 + tolerance_px)
            and np.all(pts[:, 1] >= by - tolerance_px)
            and np.all(pts[:, 1] <= by + bh - 1 + tolerance_px)
        )

        return bool(left_tight and top_tight and right_tight and bottom_tight and all_inside)

    def classify_contour(self, contour: np.ndarray, area: float) -> Tuple[bool, float, str]:
        """Classifies whether a contour is an isolated single leaf or a merged foliage cluster.

        Criteria for cluster classification:
        1. Solidity < cluster_solidity_threshold: Indicates deep concave indentations between touching leaves.
        2. Area >= cluster_area_threshold_px: Foliage mass exceeds the bounds of an individual leaf.

        Returns:
            Tuple[is_cluster, solidity, label_str]:
            - is_cluster: True if multiple touching/overlapping leaves form this contour.
            - solidity: Ratio of contour area to convex hull area.
            - label_str: "Cluster" or "Leaf"
        """
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 0 else 1.0

        is_cluster = bool(
            solidity < self.config.cluster_solidity_threshold
            or area >= self.config.cluster_area_threshold_px
        )
        label_str = "Cluster" if is_cluster else "Leaf"
        return is_cluster, round(solidity, 3), label_str

    def separate_instances_watershed(
        self, binary_mask: np.ndarray, base_bgr: np.ndarray
    ) -> List[np.ndarray]:
        """Separates touching or overlapping leaves into individual leaf instances
        using Euclidean Distance Transform and Marker-Controlled Watershed.

        Args:
            binary_mask: Cleaned 2D uint8 segmentation mask.
            base_bgr: 3D uint8 BGR image.

        Returns:
            List[np.ndarray]: List of separated individual leaf contours.
        """
        if binary_mask is None or base_bgr is None:
            return []

        h, w = binary_mask.shape[:2]
        if base_bgr.shape[:2] != (h, w) or base_bgr.dtype != np.uint8:
            return []

        # 1. Exact Euclidean Distance Transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        max_dist = float(dist_transform.max())
        if max_dist <= 0:
            return []

        # 2. Sure foreground seeds from distance peaks
        thresh_val = float(self.config.watershed_dist_threshold_ratio * max_dist)
        _, sure_fg = cv2.threshold(dist_transform, thresh_val, 255, cv2.THRESH_BINARY)
        sure_fg = np.uint8(sure_fg)

        # 3. Sure background from binary mask dilation
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(
            binary_mask, kernel, iterations=self.config.watershed_bg_dilate_iterations
        )

        # 4. Unknown boundary zone between foreground and background
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 5. Connected components marker labeling
        num_labels, markers = cv2.connectedComponents(sure_fg)
        if num_labels <= 1:
            return []

        # Offset markers by 1 so sure background is labeled 1
        markers = markers + 1
        markers[unknown == 255] = 0

        # 6. Apply Watershed to split along touching contact seams
        markers = cv2.watershed(base_bgr, markers)

        instance_contours: List[np.ndarray] = []
        unique_markers = [m for m in np.unique(markers) if m > 1]

        for m_id in unique_markers:
            leaf_mask = np.uint8(markers == m_id) * 255
            cnts, _ = cv2.findContours(
                leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                area = float(cv2.contourArea(c))
                if area >= self.config.min_contour_area_px:
                    instance_contours.append(c)

        return instance_contours

    def find_leaf_contours(
        self,
        binary_mask: np.ndarray,
        base_bgr: Optional[np.ndarray] = None,
        return_metadata: bool = False,
    ) -> Union[
        Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]],
        Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], List[Dict[str, any]]],
    ]:
        """Finds leaf contours in a binary mask, optionally applying Marker-Controlled
        Watershed to split touching foliage into individual leaves, applies area filtering,
        derives tight bounding boxes directly from contour coordinates, and evaluates cluster status.

        Args:
            binary_mask: Cleaned 2D uint8 segmentation mask.
            base_bgr: Optional BGR image used for watershed instance segmentation.
            return_metadata: If True, returns additional geometric metadata per region.

        Returns:
            If return_metadata is False:
                valid_contours, bboxes
            If return_metadata is True:
                valid_contours, bboxes, metadata_list
        """
        if binary_mask is None or not isinstance(binary_mask, np.ndarray):
            msg = "Mask must be a valid numpy.ndarray"
            logger.error(msg)
            raise TypeError(msg)

        if binary_mask.ndim != 2 or binary_mask.dtype != np.uint8:
            msg = f"Mask must be 2D uint8 array, got shape {binary_mask.shape} and dtype {binary_mask.dtype}"
            logger.error(msg)
            raise ValueError(msg)

        h, w = binary_mask.shape
        frame_area = float(h * w)
        max_allowed_area = frame_area * self.config.max_area_ratio

        # Determine raw contours: use Watershed instance separation if enabled and base_bgr is provided
        raw_contours: List[np.ndarray] = []
        if self.config.enable_watershed and base_bgr is not None:
            raw_contours = self.separate_instances_watershed(binary_mask, base_bgr)

        # Fallback to standard external contour extraction if watershed returned 0 or wasn't invoked
        if not raw_contours:
            raw_contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

        valid_contours: List[np.ndarray] = []
        bboxes: List[Tuple[int, int, int, int]] = []
        metadata_list: List[Dict[str, any]] = []

        for c in raw_contours:
            area = float(cv2.contourArea(c))

            # Discard small specks / dust
            if area < self.config.min_contour_area_px:
                continue

            # Discard massive background artifacts
            if area > max_allowed_area:
                logger.warning(
                    f"Discarded oversized contour: area={area:.0f} px "
                    f"exceeds max_area_ratio={self.config.max_area_ratio:.0%}"
                )
                continue

            # Calculate tight bounding box directly from contour coordinates
            bbox = self.compute_tight_bounding_box(c)

            # Mathematically verify bounding box tightness
            is_tight = self.verify_bounding_box_tightness(c, bbox, tolerance_px=1)
            if not is_tight:
                logger.error(
                    f"Bounding box {bbox} failed tightness check for contour with area {area:.1f}px"
                )
                raise RuntimeError(f"Bounding box calculation failed tightness verification: {bbox}")

            is_cluster, solidity, label_str = self.classify_contour(c, area)

            valid_contours.append(c)
            bboxes.append(bbox)
            metadata_list.append({
                "area": area,
                "solidity": solidity,
                "is_cluster": is_cluster,
                "type": label_str,
                "bbox": bbox,
            })

        logger.debug(
            f"Extracted {len(valid_contours)} valid regions ({sum(1 for m in metadata_list if m['is_cluster'])} clusters, "
            f"{sum(1 for m in metadata_list if not m['is_cluster'])} single leaves)"
        )

        if return_metadata:
            return valid_contours, bboxes, metadata_list
        return valid_contours, bboxes

    def render_overlay(
        self,
        base_bgr: np.ndarray,
        contours: List[np.ndarray],
        bboxes: List[Tuple[int, int, int, int]],
        metadata_list: Optional[List[Dict[str, any]]] = None,
    ) -> np.ndarray:
        """Draws contour outlines and bounding boxes on the base image.
        Distinguishes single isolated leaves (Cyan) from merged foliage clusters (Orange).
        """
        overlay = base_bgr.copy()

        # Draw green contour boundaries
        cv2.drawContours(overlay, contours, -1, color=(0, 255, 0), thickness=2)

        # Draw bounding boxes and ID tags
        for idx, (x, y, bw, bh) in enumerate(bboxes, start=1):
            area = float(cv2.contourArea(contours[idx - 1]))

            if metadata_list and idx - 1 < len(metadata_list):
                meta = metadata_list[idx - 1]
                is_cluster = meta["is_cluster"]
                tag_type = meta["type"]
            else:
                is_cluster, _, tag_type = self.classify_contour(contours[idx - 1], area)

            # Cyan for isolated Single Leaf, Orange/Amber for Merged Cluster
            bbox_color = (0, 165, 255) if is_cluster else (255, 255, 0)

            # Rectangle enclosing region
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), bbox_color, 2)

            # Text label clearly indicating [Leaf] vs [Cluster]
            label = f"#{idx} [{tag_type}: {area:.0f}px]"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(y - 4, th + 4)
            cv2.rectangle(overlay, (x, ty - th - 2), (x + tw + 4, ty + bl), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                overlay,
                label,
                (x + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                bbox_color,
                1,
                cv2.LINE_AA,
            )

        return overlay

    def create_comparison_panel(
        self,
        original_bgr: np.ndarray,
        binary_mask: np.ndarray,
        overlay_bgr: np.ndarray,
        raw_mask: np.ndarray = None,
    ) -> np.ndarray:
        """Stitches a comparison image:
        If raw_mask is provided (4-panel): [1. Input | 2. Raw Mask (Before) | 3. Clean Mask (After) | 4. Detected Contours & BBoxes]
        If raw_mask is None (3-panel):     [1. Input | 2. Binary Mask | 3. Detected Contours & BBoxes]
        """
        h, w = original_bgr.shape[:2]

        def add_banner(img: np.ndarray, title: str) -> np.ndarray:
            canvas = img.copy()
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                canvas, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1, cv2.LINE_AA
            )
            return canvas

        p1 = add_banner(original_bgr, "1. Input Resized")

        # Convert clean mask to 3-channel
        mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        m_resized = cv2.resize(mask_3ch, (w, h), interpolation=cv2.INTER_NEAREST)

        o_resized = cv2.resize(overlay_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        if raw_mask is not None:
            raw_3ch = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
            raw_resized = cv2.resize(raw_3ch, (w, h), interpolation=cv2.INTER_NEAREST)
            p_raw = add_banner(raw_resized, "2. Raw Mask (Before)")
            p_clean = add_banner(m_resized, "3. Clean Mask (After)")
            p_det = add_banner(o_resized, "4. Contours & Tight BBoxes")
            return np.hstack([p1, p_raw, p_clean, p_det])
        else:
            p2 = add_banner(m_resized, "2. Binary Mask (0/255)")
            p3 = add_banner(o_resized, "3. Detected Contours & BBoxes")
            return np.hstack([p1, p2, p3])
