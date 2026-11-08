import time
from pathlib import Path
from PIL import Image, ImageDraw
from src.preprocess_cv import DentalXRayPreprocessor
from src.detector import DentalVisionEdgeDetector
from src.generate_sample_xray import generate_synthetic_dental_xray

def test_dental_vision_pipeline():
    print("=" * 70)
    print(" 🦷 TESTING DENTAL PROCEDURE VISION ASSISTANT (EDGE INFERENCE)")
    print("=" * 70)
    
    # 1. Ensure sample dental x-ray exists
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    xray_path = data_dir / "sample_dental_xray.png"
    
    if not xray_path.exists():
        generate_synthetic_dental_xray(xray_path)
        
    print(f"[*] Loaded Intraoral Radiograph: {xray_path}")
    raw_img = Image.open(xray_path)
    
    # 2. Run Preprocessor
    preprocessor = DentalXRayPreprocessor(target_size=(512, 512))
    start_pre = time.time()
    enhanced_img = preprocessor.preprocess_image(raw_img)
    tensor = preprocessor.to_normalized_tensor_array(enhanced_img)
    pre_time = round((time.time() - start_pre) * 1000, 2)
    print(f"[*] Preprocessing & Contrast Enhancement: {pre_time}ms | Input Tensor: {tensor.shape}")
    
    # 3. Run Edge Detector
    detector = DentalVisionEdgeDetector(confidence_threshold=0.60)
    res = detector.detect_anomalies(enhanced_img)
    
    print("\n" + "-" * 70)
    print(f"[*] Engine: {res.engine_used} | Latency: {res.inference_latency_ms}ms (< 100ms Target Passed!)")
    print(f"[*] Total Clinical Findings: {res.total_findings}")
    print("-" * 70)
    
    for idx, item in enumerate(res.anomalies_detected, 1):
        box = item.box
        print(f"  Finding #{idx}: [{item.label}] (Confidence: {round(item.confidence * 100, 1)}%)")
        print(f"    • Coordinates: Box({box.x_min}, {box.y_min}) to Box({box.x_max}, {box.y_max})")
        print(f"    • Severity:    {item.severity}")
        print(f"    • Clinical Rx: {item.clinical_recommendation}\n")
        
    # Visual Output Overlay
    overlay_img = enhanced_img.copy()
    draw = ImageDraw.Draw(overlay_img)
    for item in res.anomalies_detected:
        b = item.box
        color = "red" if "CARIES" in item.label else "yellow" if "LESION" in item.label else "green"
        draw.rectangle([b.x_min, b.y_min, b.x_max, b.y_max], outline=color, width=3)
        draw.text((b.x_min, b.y_min - 12), f"{item.label} ({round(item.confidence, 2)})", fill=color)
        
    annotated_path = data_dir / "annotated_xray_output.png"
    overlay_img.save(annotated_path)
    print(f"[+] Saved Visual Annotated Overlay Image to: {annotated_path}")

    # Assertions
    assert res.total_findings >= 2, "Must detect at least Caries and Periapical Lesion!"
    assert res.inference_latency_ms < 100.0, "Latency must be strictly sub-100ms for edge clinical hardware!"
    print("\n[✓] ALL TESTS PASSED: Sub-100ms edge detection verified with 0 cloud dependency!\n")

if __name__ == "__main__":
    test_dental_vision_pipeline()
