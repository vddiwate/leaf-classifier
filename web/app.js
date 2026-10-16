/* Frontend controller for LeafyCV Web UI */

let currentCsvData = "";
let currentAnnotatedDataUrl = "";

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const samplePills = document.querySelectorAll(".pill-btn");
  const downloadImgBtn = document.getElementById("downloadImgBtn");
  const downloadCsvBtn = document.getElementById("downloadCsvBtn");

  // Load default initial sample
  loadSample("test_011.jpg");

  // Drag and drop event listeners
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  });

  dropzone.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Sample buttons
  samplePills.forEach((btn) => {
    btn.addEventListener("click", () => {
      samplePills.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const sampleName = btn.getAttribute("data-sample");
      loadSample(sampleName);
    });
  });

  // Download buttons
  downloadImgBtn.addEventListener("click", () => {
    if (!currentAnnotatedDataUrl) return;
    const link = document.createElement("a");
    link.href = currentAnnotatedDataUrl;
    link.download = `leafy_annotated_${Date.now()}.jpg`;
    link.click();
  });

  downloadCsvBtn.addEventListener("click", () => {
    if (!currentCsvData) return;
    const blob = new Blob([currentCsvData], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `leaf_prediction_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  });
});

async function handleFileUpload(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid image file (JPG, PNG, WebP).");
    return;
  }

  document.querySelectorAll(".pill-btn").forEach((b) => b.classList.remove("active"));
  document.getElementById("imageNameLabel").textContent = file.name;

  const reader = new FileReader();
  reader.onload = async (e) => {
    const originalBase64 = e.target.result;
    document.getElementById("originalImg").src = originalBase64;
    setLoading(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_name: file.name,
          image_base64: originalBase64,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Inference failed");
      }
      renderResults(data);
    } catch (err) {
      alert(`Detection Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  reader.readAsDataURL(file);
}

async function loadSample(sampleName) {
  document.getElementById("imageNameLabel").textContent = sampleName;
  setLoading(true);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_name: sampleName }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to load sample");
    }

    document.getElementById("originalImg").src = data.original_base64;
    renderResults(data);
  } catch (err) {
    console.error("Sample Error:", err);
  } finally {
    setLoading(false);
  }
}

function renderResults(data) {
  currentAnnotatedDataUrl = data.annotated_base64;
  currentCsvData = data.csv_data;

  // Set annotated visual
  document.getElementById("annotatedImg").src = data.annotated_base64;

  // Set stats
  animateValue("statTotal", data.total_leaves);
  animateValue("statLarge", data.large_leaves);
  animateValue("statSmall", data.small_leaves);
  document.getElementById("statLatency").textContent = `${data.latency_ms} ms`;

  // Render table rows
  const tbody = document.getElementById("measurementsBody");
  tbody.innerHTML = "";

  if (!data.measurements || data.measurements.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-muted); padding: 24px;">No leaf contours detected</td></tr>`;
    return;
  }

  data.measurements.forEach((m) => {
    const tr = document.createElement("tr");
    const isLarge = m.predicted_class === "Large Leaf";
    const badgeClass = isLarge ? "large" : "small";

    tr.innerHTML = `
      <td><strong>#${m.detection_id}</strong></td>
      <td><span class="badge-tag ${badgeClass}">${m.predicted_class}</span></td>
      <td><strong>${Math.round(m.measured_area_px).toLocaleString()} px</strong></td>
      <td>${Math.round(m.contour_area_px).toLocaleString()} px</td>
      <td><code>(${m.bbox[0]}, ${m.bbox[1]}, ${m.bbox[2]}, ${m.bbox[3]})</code></td>
      <td>${m.aspect_ratio.toFixed(2)}</td>
      <td>${m.extent.toFixed(2)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function setLoading(isLoading) {
  const overlay = document.getElementById("loadingOverlay");
  if (isLoading) {
    overlay.classList.add("active");
  } else {
    overlay.classList.remove("active");
  }
}

function animateValue(elementId, targetValue) {
  const el = document.getElementById(elementId);
  el.textContent = targetValue;
}
