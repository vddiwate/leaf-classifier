# Leaf Classifier

An explainable, production-ready Computer Vision pipeline built with Python, OpenCV, and NumPy for automated individual leaf segmentation and natural size grading (**Large Leaf** vs. **Small Leaf**).

---

## Architecture & Pipeline Flow

```text
[Input Image]
      │
      ▼
1. Preprocessing       ➔ Resize (256px, aspect-preserved) + 5x5 Gaussian noise filter
      │
      ▼
2. Color Conversion    ➔ BGR to HSV color space (chlorophyll isolation)
      │
      ▼
3. Segmentation        ➔ Green HSV thresholding + Decoupled morphology (3x3 open / 5x5 close)
      │
      ▼
4. Instance Separation ➔ Euclidean Distance Transform + Marker-Controlled Watershed
      │
      ▼
5. Detection           ➔ Contour extraction + IoU duplicate suppression (IoU > 0.50)
      │
      ▼
6. Feature Extraction  ➔ Contour Area (Green's theorem), Bounding Box, Aspect Ratio, Extent
      │
      ▼
7. Size Classification ➔ Adaptive natural contrast (80% relative ratio) vs. baseline cutoff
      │
      ▼
8. Output Generation   ➔ Color-coded visuals (Lime Green / Cyan) + Tabular CSV logging
```

---

## Key Features

- **Touching Leaf Separation**: Uses Marker-Controlled Watershed to split overlapping foliage along natural seams.
- **Adaptive Size Grading**: Automatically compares leaves in multi-leaf frames using relative area ratios (`area / max_area < 80%` $\to$ Small Leaf, otherwise Large Leaf).
- **Interactive Web UI**: Modern dark-theme browser interface with drag-and-drop, instant latency (~29ms), and CSV export.
- **Pure Classical OpenCV**: Zero black-box neural networks, zero GPU required, runs at 30+ FPS on CPU.
- **Automated Test Suite**: 68 comprehensive unit and integration tests passing in ~0.5s.

---

## Quickstart

### 1. Installation
```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Web Application
```powershell
python main.py --ui
```
Open **`http://127.0.0.1:5000`** in your browser to upload and test images interactively.

### 3. Run via CLI
```powershell
# Test a single image
python main.py --source "path/to/leaf.jpg"

# Test a folder or ZIP archive
python main.py --source "data/test_unseen/leafy_vegetable_unseen_test_images.zip"

# Run 15-image benchmark evaluation
python main.py --evaluate
```

### 4. Run Automated Tests
```powershell
pytest tests/ -v
```

---

## Project Structure

```text
leaf-classifier/
├── main.py                     # Unified CLI entrypoint (--ui, --source, --evaluate)
├── app.py                      # Web application server (http://127.0.0.1:5000)
├── predict.py                  # CLI inference engine
├── evaluate.py                 # Benchmark evaluation runner
├── config/config.yaml          # Central configuration (Zero hardcoding)
├── src/                        # Core modular library (preprocessing, segmentation, classification)
├── web/                        # Responsive dark-theme Web UI (HTML, CSS, JS)
├── tests/                      # 68 Automated unit & integration tests
├── outputs/                    # Visual predictions and audit CSV reports
└── requirements.txt            # Production dependencies
```
