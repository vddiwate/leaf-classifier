"""Unit Tests for Leaf Segmentation Module (src/segmentation.py).
Tests binary mask generation, multi-tone color ranges, morphological operations,
contour filtering, and input boundary validations.
"""

import cv2
import numpy as np
import pytest

from src.config import SegmentationConfig
from src.segmentation import Segmenter


@pytest.fixture
def synthetic_hsv_scene() -> np.ndarray:
    """Creates a synthetic HSV image (H=100, W=100, C=3) with:
    - Neutral background (H=0, S=0, V=100: gray)
    - 1 Green leaf patch (H=50, S=150, V=180, area = 30x30 = 900 px)
    - 1 Yellow leaf patch (H=20, S=150, V=180, area = 20x20 = 400 px)
    - 1 Tiny noise dot (H=50, S=150, V=180, area = 2x2 = 4 px)
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 100  # Gray background (V=100)

    # Green leaf patch: [20:50, 20:50]
    img[20:50, 20:50, 0] = 50   # Green Hue
    img[20:50, 20:50, 1] = 150  # Saturation
    img[20:50, 20:50, 2] = 180  # Value

    # Yellow leaf patch: [60:80, 60:80]
    img[60:80, 60:80, 0] = 20   # Yellow Hue
    img[60:80, 60:80, 1] = 150  # Saturation
    img[60:80, 60:80, 2] = 180  # Value

    # Tiny noise dot: [5:7, 5:7] (2x2)
    img[5:7, 5:7, 0] = 50
    img[5:7, 5:7, 1] = 150
    img[5:7, 5:7, 2] = 180

    return img


def test_segmentation_with_yellow_tones(synthetic_hsv_scene: np.ndarray):
    """Verifies that green and yellow leaf patches are both captured when yellow_tones=True."""
    cfg = SegmentationConfig(
        include_yellow_tones=True,
        yellow_hsv_lower=(15, 40, 40),
        yellow_hsv_upper=(25, 255, 255),
        min_contour_area_px=100.0,
    )
    segmenter = Segmenter(cfg)
    mask = segmenter.generate_mask(synthetic_hsv_scene)

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8

    # Both green (20:50, 20:50) and yellow (60:80, 60:80) patches should be 255
    assert np.all(mask[25:45, 25:45] == 255)
    assert np.all(mask[65:75, 65:75] == 255)

    # Background should be 0
    assert np.all(mask[0:10, 50:60] == 0)

    # Contours should detect 2 regions (noise dot removed by morphology or min_area)
    contours, bboxes = segmenter.find_leaf_contours(mask)
    assert len(contours) == 2
    assert len(bboxes) == 2


def test_segmentation_without_yellow_tones(synthetic_hsv_scene: np.ndarray):
    """Verifies that only green foliage is captured when include_yellow_tones=False."""
    cfg = SegmentationConfig(
        include_yellow_tones=False,
        min_contour_area_px=100.0,
    )
    segmenter = Segmenter(cfg)
    mask = segmenter.generate_mask(synthetic_hsv_scene)

    # Green patch is 255
    assert np.all(mask[25:45, 25:45] == 255)

    # Yellow patch is excluded (0)
    assert np.all(mask[65:75, 65:75] == 0)

    contours, _ = segmenter.find_leaf_contours(mask)
    assert len(contours) == 1


def test_contour_area_filtering():
    """Verifies that contours smaller than min_contour_area_px are discarded."""
    cfg = SegmentationConfig(min_contour_area_px=200.0)
    segmenter = Segmenter(cfg)

    # Create artificial binary mask with one small square (10x10 = 100px) and one large square (20x20 = 400px)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 255  # 100 px (below threshold)
    mask[50:70, 50:70] = 255  # 400 px (above threshold)

    contours, bboxes = segmenter.find_leaf_contours(mask)
    assert len(contours) == 1
    x, y, w, h = bboxes[0]
    assert w == 20 and h == 20


def test_invalid_segmenter_init():
    """Rejects non-SegmentationConfig initialization."""
    with pytest.raises(TypeError, match="Expected SegmentationConfig"):
        Segmenter("not_a_config")


def test_invalid_hsv_inputs():
    """Rejects None, non-numpy, empty, or wrong-channel HSV images."""
    segmenter = Segmenter(SegmentationConfig())

    with pytest.raises(TypeError, match="cannot be None"):
        segmenter.generate_mask(None)

    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        segmenter.generate_mask([[1, 2, 3]])

    with pytest.raises(TypeError, match="Expected uint8 dtype"):
        segmenter.generate_mask(np.zeros((50, 50, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="3-dimensional array"):
        segmenter.generate_mask(np.zeros((50, 50), dtype=np.uint8))

    with pytest.raises(ValueError, match="Expected 3 channels"):
        segmenter.generate_mask(np.zeros((50, 50, 4), dtype=np.uint8))


def test_invalid_mask_inputs():
    """Rejects invalid mask inputs in find_leaf_contours."""
    segmenter = Segmenter(SegmentationConfig())

    with pytest.raises(TypeError, match="must be a valid numpy.ndarray"):
        segmenter.find_leaf_contours(None)

    with pytest.raises(ValueError, match="Mask must be 2D uint8 array"):
        segmenter.find_leaf_contours(np.zeros((50, 50, 3), dtype=np.uint8))


def test_panel_creation():
    """Verifies that create_comparison_panel stitches 3 images or 4 images horizontally."""
    segmenter = Segmenter(SegmentationConfig())
    orig = np.zeros((100, 150, 3), dtype=np.uint8)
    mask = np.zeros((100, 150), dtype=np.uint8)
    overlay = np.zeros((100, 150, 3), dtype=np.uint8)

    # 3-panel mode (backward compatibility)
    panel_3 = segmenter.create_comparison_panel(orig, mask, overlay)
    assert panel_3.shape == (100, 450, 3)

    # 4-panel before/after mode
    raw_mask = np.zeros((100, 150), dtype=np.uint8)
    panel_4 = segmenter.create_comparison_panel(orig, mask, overlay, raw_mask=raw_mask)
    assert panel_4.shape == (100, 600, 3)


def test_bounding_box_tightness_verification():
    """Verifies that bounding boxes tightly fit detected contour coordinates."""
    # Synthetic L-shaped contour
    contour = np.array([
        [[15, 20]],
        [[35, 20]],
        [[35, 60]],
        [[25, 60]],
        [[25, 40]],
        [[15, 40]],
    ], dtype=np.int32)

    bbox = Segmenter.compute_tight_bounding_box(contour)
    assert bbox == (15, 20, 21, 41)  # min_x=15, min_y=20, w=35-15+1=21, h=60-20+1=41

    # Tightness verification should pass
    assert Segmenter.verify_bounding_box_tightness(contour, bbox, tolerance_px=0) is True

    # Padded or shifted bbox should fail tightness check
    padded_bbox = (10, 15, 30, 50)
    assert Segmenter.verify_bounding_box_tightness(contour, padded_bbox, tolerance_px=0) is False

    shifted_bbox = (16, 20, 20, 41)
    assert Segmenter.verify_bounding_box_tightness(contour, shifted_bbox, tolerance_px=0) is False


def test_decoupled_morphology_stem_preservation():
    """Verifies that 3x3 open preserves thin stem structures that 5x5 open destroys."""
    # Create an image with a realistic 4-pixel wide petiole / stem
    hsv_stem = np.zeros((60, 60, 3), dtype=np.uint8)
    hsv_stem[:, :, 2] = 200  # Bright background (white, S=0)
    # Green thin stem at x in [28:32], y in [10:50]
    hsv_stem[10:50, 28:32, 0] = 50   # Green Hue
    hsv_stem[10:50, 28:32, 1] = 120  # Green Saturation
    hsv_stem[10:50, 28:32, 2] = 180  # Value

    cfg_gentle = SegmentationConfig(open_kernel_size=(3, 3), close_kernel_size=(5, 5))
    seg_gentle = Segmenter(cfg_gentle)
    mask_gentle = seg_gentle.generate_mask(hsv_stem)

    cfg_aggressive = SegmentationConfig(open_kernel_size=(5, 5), close_kernel_size=(5, 5))
    seg_aggressive = Segmenter(cfg_aggressive)
    mask_aggressive = seg_aggressive.generate_mask(hsv_stem)

    # Gentle 3x3 preserves stem pixels, aggressive 5x5 erodes them completely
    assert np.sum(mask_gentle > 0) > 0
    assert np.sum(mask_aggressive > 0) == 0


def test_cluster_classification():
    """Verifies that convex contours are classified as 'Leaf' and indented/concave contours as 'Cluster'."""
    cfg = SegmentationConfig(
        cluster_solidity_threshold=0.85,
        cluster_area_threshold_px=5000.0,
    )
    seg = Segmenter(cfg)

    # 1. Convex rectangle contour (area = 400px, solidity = 1.0) -> Single Leaf
    rect_contour = np.array([[[10, 10]], [[30, 10]], [[30, 30]], [[10, 30]]], dtype=np.int32)
    is_cluster, solidity, label = seg.classify_contour(rect_contour, area=400.0)
    assert is_cluster is False
    assert solidity == 1.0
    assert label == "Leaf"

    # 2. Notched U-shaped contour (area = 250px, hull area = 400px, solidity = 0.625 < 0.85) -> Merged Cluster
    u_contour = np.array([
        [[10, 10]], [[30, 10]], [[30, 30]], [[25, 30]], [[25, 15]], [[15, 15]], [[15, 30]], [[10, 30]]
    ], dtype=np.int32)
    is_cluster_u, solidity_u, label_u = seg.classify_contour(u_contour, area=250.0)
    assert is_cluster_u is True
    assert solidity_u < 0.85
    assert label_u == "Cluster"

    # 3. Oversized contour (area = 6000px >= cluster_area_threshold_px 5000px) -> Merged Cluster
    is_cluster_big, _, label_big = seg.classify_contour(rect_contour, area=6000.0)
    assert is_cluster_big is True
    assert label_big == "Cluster"


def test_find_leaf_contours_with_metadata():
    """Verifies that find_leaf_contours returns detailed metadata when return_metadata=True."""
    cfg = SegmentationConfig(min_contour_area_px=100.0)
    seg = Segmenter(cfg)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 20:60] = 255  # 40x40 = 1521 px contour area

    contours, bboxes, meta = seg.find_leaf_contours(mask, return_metadata=True)
    assert len(contours) == 1
    assert len(bboxes) == 1
    assert len(meta) == 1
    assert meta[0]["area"] == 1521.0
    assert meta[0]["is_cluster"] is False
    assert meta[0]["type"] == "Leaf"


def test_empty_mask_handling():
    """Verifies that empty masks and scenes with no foliage return empty lists gracefully."""
    cfg = SegmentationConfig()
    seg = Segmenter(cfg)

    # 1. Pure all-zero binary mask
    empty_mask = np.zeros((120, 120), dtype=np.uint8)
    contours, bboxes = seg.find_leaf_contours(empty_mask)
    assert len(contours) == 0
    assert len(bboxes) == 0

    # 2. find_leaf_contours with metadata on empty mask
    c_empty, b_empty, meta_empty = seg.find_leaf_contours(empty_mask, return_metadata=True)
    assert len(c_empty) == 0
    assert len(b_empty) == 0
    assert len(meta_empty) == 0

    # 3. HSV scene with zero green foliage (e.g. pure white background: S=0, V=255)
    white_hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    white_hsv[:, :, 2] = 255
    generated_mask = seg.generate_mask(white_hsv)
    assert np.all(generated_mask == 0)

    contours_gen, bboxes_gen = seg.find_leaf_contours(generated_mask)
    assert len(contours_gen) == 0
    assert len(bboxes_gen) == 0


def test_noisy_background_suppression():
    """Verifies that random background noise specks are eliminated by morphology and area filtering."""
    cfg = SegmentationConfig(
        min_contour_area_px=150.0,
        open_kernel_size=(3, 3),
        close_kernel_size=(5, 5),
    )
    seg = Segmenter(cfg)

    # Create synthetic scene (100x100) with neutral gray background
    noisy_hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    noisy_hsv[:, :, 2] = 100  # Gray background

    # 1. Inject isolated 1px and 2px green noise specks at random locations
    np.random.seed(42)
    noise_y = np.random.randint(0, 100, size=40)
    noise_x = np.random.randint(0, 100, size=40)
    noisy_hsv[noise_y, noise_x, 0] = 45   # Green Hue
    noisy_hsv[noise_y, noise_x, 1] = 150  # Saturation
    noisy_hsv[noise_y, noise_x, 2] = 180  # Value

    # Generate mask on pure noisy background
    clean_mask = seg.generate_mask(noisy_hsv)
    contours, bboxes = seg.find_leaf_contours(clean_mask)
    # Morphology (3x3 open) should destroy 1px specks, and area filter catches any residual
    assert len(contours) == 0
    assert len(bboxes) == 0

    # 2. Add one valid leaf patch (30x30 = 900px) surrounded by noise
    noisy_hsv[30:60, 30:60, 0] = 50   # Green Hue
    noisy_hsv[30:60, 30:60, 1] = 160  # Saturation
    noisy_hsv[30:60, 30:60, 2] = 180  # Value

    clean_mask_with_leaf = seg.generate_mask(noisy_hsv)
    leaf_contours, leaf_bboxes = seg.find_leaf_contours(clean_mask_with_leaf)

    # Exactly 1 leaf should be detected; all noise specks must be suppressed
    assert len(leaf_contours) == 1
    assert len(leaf_bboxes) == 1
    assert cv2.contourArea(leaf_contours[0]) > 500.0


def test_watershed_instance_separation_touching_leaves():
    """Verifies that Marker-Controlled Watershed separates touching overlapping leaves
    into individual leaf contours with tight bounding boxes."""
    cfg = SegmentationConfig(
        enable_watershed=True,
        watershed_dist_threshold_ratio=0.50,
        min_contour_area_px=100.0,
    )
    seg = Segmenter(cfg)

    # Synthetic image with 2 touching circular leaf shapes (5px overlap bridge)
    h, w = 150, 150
    mask = np.zeros((h, w), dtype=np.uint8)
    base_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # Circle 1 at (40, 75), radius 25
    cv2.circle(mask, (40, 75), 25, 255, -1)
    cv2.circle(base_bgr, (40, 75), 25, (35, 180, 50), -1)

    # Circle 2 at (85, 75), radius 25 -> touching circle 1 at the border
    cv2.circle(mask, (85, 75), 25, 255, -1)
    cv2.circle(base_bgr, (85, 75), 25, (35, 180, 50), -1)

    # Without watershed (base_bgr=None): Merged into 1 single contour
    merged_contours, _ = seg.find_leaf_contours(mask, base_bgr=None)
    assert len(merged_contours) == 1

    # With watershed (base_bgr provided): Separated into 2 distinct individual leaves!
    split_contours, split_bboxes = seg.find_leaf_contours(mask, base_bgr=base_bgr)
    assert len(split_contours) == 2
    assert len(split_bboxes) == 2

    # Verify bounding box tightness on both split leaves
    for c, bbox in zip(split_contours, split_bboxes):
        assert seg.verify_bounding_box_tightness(c, bbox, tolerance_px=1) is True


def test_watershed_empty_mask_handling():
    """Verifies that watershed on empty mask gracefully returns empty lists."""
    cfg = SegmentationConfig(enable_watershed=True)
    seg = Segmenter(cfg)

    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    empty_bgr = np.zeros((100, 100, 3), dtype=np.uint8)

    contours, bboxes = seg.find_leaf_contours(empty_mask, base_bgr=empty_bgr)
    assert contours == []
    assert bboxes == []




