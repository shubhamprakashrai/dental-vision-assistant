# 🦷 Dental Procedure Vision Assistant
> Edge-deployable Computer Vision system detecting anatomical landmarks and pathological anomalies in intraoral dental radiographs at sub-100ms latency.

---

## 🎯 Clinical Context & Problem
* **Challenge:** Dentists review dozens of intraoral radiograph scans (RVG bitewings) daily. Subtle early-stage proximal caries (cavities) and apical periapical radiolucencies (root infections) can be missed during high-volume clinical fatigue.
* **Hardware Constraint:** Dental operatories utilize standard localized Windows workstations connected directly to dental sensors—lacking high-end cloud GPU infrastructure.
* **Solution:** An edge-optimized Computer Vision microservice trained on dental radiographs, enhanced via CLAHE contrast filtering, exported to ONNX format, and quantized to INT8 for zero-latency local CPU inference.

---

## 🏗️ Architecture & Pipeline Overview

```
[ Raw Radiograph Image (RVG Bitewing) ]
                   │
                   ▼
[ Preprocessing & Contrast Enhancement (OpenCV / PIL) ]
  • Contrast Limited Adaptive Histogram Equalization (CLAHE)
  • Sharpening boundary gradients between enamel and carious dentin
                   │
                   ▼
[ Object Detection Core: YOLOv8 Quantized (ONNX Runtime) ]
  • Detects Caries, Periapical Lesions, Existing Restorations, Impacted Molars
  • Quantized from FP32 to INT8 (Sub-80ms execution on local clinic CPU)
                   │
                   ▼
[ Annotated Radiograph Overlay + Structured Clinical Findings ]
```

---

## 🚀 Quick Start & Testing

```bash
# 1. Run automated test suite
$env:PYTHONPATH="."; python tests/test_vision.py

# 2. Launch Local Clinical Microservice
$env:PYTHONPATH="."; python -m uvicorn src.app:app --port 8003 --reload
```
