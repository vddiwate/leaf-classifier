"""In-memory dataset loader for ZIP archives and directories with format validation."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zipfile
import cv2
import numpy as np

from src.logger import get_logger

logger = get_logger("data_loader")


class ZipDataLoader:
    """Loads and validates images directly from a ZIP archive in memory without disk extraction."""

    def __init__(
        self,
        zip_path: Union[str, Path],
        allowed_extensions: Optional[List[str]] = None,
        min_resolution: Tuple[int, int] = (50, 50),
    ):
        self.zip_path = Path(zip_path)
        self.allowed_extensions = allowed_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        self.min_width, self.min_height = min_resolution
        self.seen_hashes: Dict[str, str] = {}  # hash -> first_seen_filename

        if not self.zip_path.exists() or not self.zip_path.is_file():
            raise FileNotFoundError(f"ZIP archive not found at: {self.zip_path}")

        # Index all image entries inside the zip
        with zipfile.ZipFile(self.zip_path, "r") as z:
            all_entries = z.namelist()
            self.image_entries = [
                e for e in all_entries
                if not e.endswith("/") and Path(e).suffix.lower() in self.allowed_extensions
            ]

        logger.info(
            f"Initialized ZipDataLoader: Found {len(self.image_entries)} images in {self.zip_path.name}"
        )

    def __len__(self) -> int:
        return len(self.image_entries)

    def get_image_list(self) -> List[str]:
        """Returns the list of all image paths inside the zip archive."""
        return list(self.image_entries)

    def load_image(
        self, index_or_name: Union[int, str]
    ) -> Tuple[Optional[np.ndarray], Optional[str], Dict[str, any]]:
        """Loads and decodes an image directly from the zip archive into memory.

        Returns:
            Tuple[image_bgr, error_message, metadata_dict]
        """
        # Resolve entry name
        if isinstance(index_or_name, int):
            if index_or_name < 0 or index_or_name >= len(self.image_entries):
                msg = f"Index {index_or_name} out of bounds (0 to {len(self.image_entries)-1})"
                logger.warning(msg)
                return None, msg, {}
            entry_name = self.image_entries[index_or_name]
        else:
            entry_name = index_or_name
            if entry_name not in self.image_entries:
                msg = f"Image entry '{entry_name}' not found in archive"
                logger.warning(msg)
                return None, msg, {}

        # 1. Read raw binary bytes from ZIP
        try:
            with zipfile.ZipFile(self.zip_path, "r") as z:
                raw_bytes = z.read(entry_name)
        except Exception as e:
            msg = f"Failed to read bytes for '{entry_name}': {e}"
            logger.error(msg)
            return None, msg, {}

        file_size_kb = len(raw_bytes) / 1024.0

        # 2. Duplicate Detection via MD5 Hash
        img_hash = hashlib.md5(raw_bytes).hexdigest()
        is_duplicate = False
        original_file = None

        if img_hash in self.seen_hashes:
            is_duplicate = True
            original_file = self.seen_hashes[img_hash]
            logger.warning(f"Duplicate detected: '{entry_name}' has same content as '{original_file}'")
        else:
            self.seen_hashes[img_hash] = entry_name

        # 3. In-memory decode into OpenCV NumPy array
        np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if decoded_img is None or decoded_img.size == 0:
            msg = f"Corrupted image data in '{entry_name}'"
            logger.warning(msg)
            return None, msg, {}

        # 4. Normalize to 3-Channel BGR
        if len(decoded_img.shape) == 2:
            bgr_img = cv2.cvtColor(decoded_img, cv2.COLOR_GRAY2BGR)
        elif len(decoded_img.shape) == 3 and decoded_img.shape[2] == 4:
            bgr_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGRA2BGR)
        elif len(decoded_img.shape) == 3 and decoded_img.shape[2] == 3:
            bgr_img = decoded_img
        else:
            msg = f"Unsupported shape {decoded_img.shape} for '{entry_name}'"
            logger.warning(msg)
            return None, msg, {}

        # 5. Minimum resolution check
        h, w = bgr_img.shape[:2]
        if w < self.min_width or h < self.min_height:
            msg = f"Resolution ({w}x{h}) below minimum threshold ({self.min_width}x{self.min_height})"
            logger.warning(msg)
            return None, msg, {}

        metadata = {
            "entry_name": entry_name,
            "filename": Path(entry_name).name,
            "file_size_kb": round(file_size_kb, 2),
            "width": w,
            "height": h,
            "channels": bgr_img.shape[2],
            "is_duplicate": is_duplicate,
            "duplicate_of": original_file,
            "md5_hash": img_hash,
        }

        return bgr_img, None, metadata
