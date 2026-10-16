"""Interactive Web Application server for Leafy Vegetable Detection & Size Classification.
Serves a modern, responsive web UI with drag-and-drop image upload, instant segmentation,
and CSV export using Python's built-in HTTP server (zero extra dependencies).
"""

import base64
import csv
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import mimetypes
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.classification import LeafMeasurement, SizeClassifier
from src.config import load_config
from src.data_loader import ZipDataLoader
from src.logger import get_logger, setup_logger
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter

# Base directories
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
UNSEEN_ZIP = BASE_DIR / "data" / "test_unseen" / "leafy_vegetable_unseen_test_images.zip"

# Initialize global components
setup_logger(log_dir="logs", log_file_name="leafy_cv.log")
logger = get_logger("web_app")
config = load_config(str(CONFIG_PATH))
preprocessor = Preprocessor(config.preprocessing)
segmenter = Segmenter(config.segmentation)
classifier = SizeClassifier(config.classification)
_last_config_mtime: float = 0.0


def get_pipeline_components():
    """Returns pipeline components, automatically reloading if config.yaml changes."""
    global config, preprocessor, segmenter, classifier, _last_config_mtime
    try:
        if CONFIG_PATH.exists():
            mtime = CONFIG_PATH.stat().st_mtime
            if mtime != _last_config_mtime:
                config = load_config(str(CONFIG_PATH))
                preprocessor = Preprocessor(config.preprocessing)
                segmenter = Segmenter(config.segmentation)
                classifier = SizeClassifier(config.classification)
                _last_config_mtime = mtime
                logger.info(
                    f"Config loaded/updated: ratio_threshold={config.classification.relative_ratio_threshold:.2f}"
                )
    except Exception as e:
        logger.warning(f"Could not check/reload config: {e}")
    return preprocessor, segmenter, classifier, config


# Cache for benchmark zip loader
_zip_loader: Optional[ZipDataLoader] = None


def get_zip_loader() -> Optional[ZipDataLoader]:
    """Lazy loader for sample unseen dataset."""
    global _zip_loader
    if _zip_loader is None and UNSEEN_ZIP.exists():
        try:
            _zip_loader = ZipDataLoader(str(UNSEEN_ZIP))
        except Exception as e:
            logger.warning(f"Could not initialize ZipDataLoader: {e}")
    return _zip_loader


def bgr_to_base64(bgr_img: np.ndarray, quality: int = 90) -> str:
    """Converts a BGR image array into a base64-encoded JPEG data URL."""
    success, buffer = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def decode_base64_image(b64_data_url: str) -> Optional[np.ndarray]:
    """Decodes a base64 data URL into an OpenCV BGR image array."""
    try:
        if "," in b64_data_url:
            _, b64_str = b64_data_url.split(",", 1)
        else:
            b64_str = b64_data_url
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def execute_pipeline(image_bgr: np.ndarray, image_name: str) -> Dict[str, Any]:
    """Executes the full pipeline and packages JSON response with base64 visual and CSV data."""
    prep, seg, clf, cfg = get_pipeline_components()
    start_time = time.perf_counter()

    # Preprocessing
    resized_bgr, hsv_img, _ = prep.process(image_bgr)

    # Segmentation & Watershed
    clean_mask = seg.generate_mask(hsv_img)
    contours, bboxes, meta_list = seg.find_leaf_contours(
        clean_mask, base_bgr=resized_bgr, return_metadata=True
    )

    # Size Classification
    measurements = clf.classify_regions(
        contours=contours,
        bboxes=bboxes,
        metadata_list=meta_list,
    )

    # Annotated Image Visual
    annotated_bgr = clf.render_annotated_image(
        base_bgr=resized_bgr,
        measurements=measurements,
        contours=contours,
        image_name=image_name,
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 1)

    # Summary counts
    n_small = sum(1 for m in measurements if m.predicted_class == cfg.classification.small_label)
    n_large = sum(1 for m in measurements if m.predicted_class == cfg.classification.large_label)

    # Build CSV text
    csv_buffer = io.StringIO()
    csv_writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
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
        ],
    )
    csv_writer.writeheader()
    for m in measurements:
        csv_writer.writerow(m.to_dict(image_name=image_name))

    # Automatically persist to disk in outputs/web_predictions/
    save_dir = BASE_DIR / "outputs" / "web_predictions"
    save_dir.mkdir(parents=True, exist_ok=True)

    clean_stem = Path(image_name).stem
    save_img_path = save_dir / f"pred_{clean_stem}.jpg"
    cv2.imwrite(str(save_img_path), annotated_bgr)

    # Append to cumulative CSV report
    csv_report_path = save_dir / "prediction_report.csv"
    file_exists = csv_report_path.exists()
    with open(csv_report_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        if not file_exists:
            writer.writeheader()
        for m in measurements:
            writer.writerow(m.to_dict(image_name=image_name))

    logger.info(f"Persisted visual: {save_img_path} and updated {csv_report_path}")

    # Measurements list for frontend table
    measurements_data = [
        {
            "detection_id": m.detection_id,
            "predicted_class": m.predicted_class,
            "measured_area_px": round(m.measured_area_px, 1),
            "contour_area_px": round(m.contour_area_px, 1),
            "bbox": list(m.bbox),
            "aspect_ratio": round(m.aspect_ratio, 2),
            "extent": round(m.extent, 2),
        }
        for m in measurements
    ]

    return {
        "image_name": image_name,
        "total_leaves": len(measurements),
        "small_leaves": n_small,
        "large_leaves": n_large,
        "latency_ms": elapsed_ms,
        "annotated_base64": bgr_to_base64(annotated_bgr),
        "original_base64": bgr_to_base64(image_bgr),
        "measurements": measurements_data,
        "csv_data": csv_buffer.getvalue(),
    }


class WebAppHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving web UI and prediction API."""

    def log_message(self, format: str, *args: Any) -> None:
        """Redirects default access logs to centralized logger."""
        logger.debug(f"{self.address_string()} - {format % args}")

    def send_json(self, data: Dict[str, Any], status_code: int = 200) -> None:
        """Sends a JSON HTTP response."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        """Handles static file requests."""
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            target_file = WEB_DIR / "index.html"
        elif path.startswith("/web/"):
            rel = path[len("/web/") :]
            target_file = WEB_DIR / rel
        else:
            self.send_error(404, "File Not Found")
            return

        if not target_file.exists() or not target_file.is_file():
            self.send_error(404, "File Not Found")
            return

        mime_type, _ = mimetypes.guess_type(str(target_file))
        if not mime_type:
            mime_type = "application/octet-stream"

        try:
            with open(target_file, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            logger.error(f"Error serving {target_file}: {e}")
            self.send_error(500, "Internal Server Error")

    def do_POST(self) -> None:
        """Handles prediction API requests."""
        if self.path == "/api/predict":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len).decode("utf-8")
                payload = json.loads(body)

                # Case 1: Uploaded custom image
                if "image_base64" in payload:
                    img_name = payload.get("image_name", "uploaded_leaf.jpg")
                    bgr_img = decode_base64_image(payload["image_base64"])
                    if bgr_img is None:
                        self.send_json({"error": "Invalid or unreadable image data."}, status_code=400)
                        return
                    result = execute_pipeline(bgr_img, img_name)
                    self.send_json(result)
                    return

                # Case 2: Pre-selected sample from test archive
                elif "sample_name" in payload:
                    sample_name = payload["sample_name"]
                    loader = get_zip_loader()
                    if not loader:
                        self.send_json({"error": "Sample archive data not available."}, status_code=500)
                        return

                    target_entry = None
                    for entry in loader.image_entries:
                        if Path(entry).name == sample_name or entry == sample_name:
                            target_entry = entry
                            break

                    if target_entry is None:
                        self.send_json({"error": f"Sample {sample_name} not found in archive."}, status_code=404)
                        return

                    bgr_img, err, _ = loader.load_image(target_entry)
                    if err or bgr_img is None:
                        self.send_json({"error": f"Failed to load sample: {err}"}, status_code=500)
                        return

                    result = execute_pipeline(bgr_img, sample_name)
                    self.send_json(result)
                    return

                else:
                    self.send_json({"error": "No image_base64 or sample_name provided."}, status_code=400)
                    return

            except Exception as e:
                logger.error(f"Inference exception: {e}", exc_info=True)
                self.send_json({"error": str(e)}, status_code=500)
        else:
            self.send_error(404, "API Endpoint Not Found")


def start_server(host: str = "127.0.0.1", port: int = 5000) -> None:
    """Starts the HTTP web server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, WebAppHandler)
    logger.info("=" * 72)
    logger.info(f"LeafyCV Web UI Server Running at: http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server.")
    logger.info("=" * 72)
    print(f"\n=======================================================")
    print(f"[+] LeafyCV Web UI is LIVE at: http://{host}:{port}")
    print(f"Open your browser to upload and test spinach leaf images!")
    print(f"Press Ctrl+C to terminate.")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Web UI Server...")
        httpd.server_close()
        logger.info("Server stopped.")


if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    start_server(port=port)
