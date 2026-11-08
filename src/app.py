import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from src.preprocess_cv import DentalXRayPreprocessor
from src.detector import DentalVisionEdgeDetector, DetectionResponse

app = FastAPI(
    title="Dental Procedure Vision Assistant API",
    description="Local Clinical Edge AI microservice for intraoral radiograph anomaly detection.",
    version="1.0.0"
)

preprocessor = DentalXRayPreprocessor(target_size=(512, 512))
detector = DentalVisionEdgeDetector(confidence_threshold=0.50)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Dental Procedure Vision Assistant",
        "runtime": "ONNX-Runtime-Edge-Quantized",
        "target_hardware": "Clinic Local Workstation CPU"
    }

@app.post("/detect-anomalies", response_model=DetectionResponse)
async def detect_dental_anomalies(file: UploadFile = File(...)):
    """
    Accepts raw intraoral dental X-ray (PNG/JPG/DICOM-extracted),
    applies CLAHE preprocessing, and outputs localized bounding boxes with clinical advice.
    """
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        
        # 1. Preprocess & Contrast Enhance
        enhanced_img = preprocessor.preprocess_image(pil_img)
        
        # 2. Edge Inference
        results = detector.detect_anomalies(enhanced_img)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision Pipeline Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8003, reload=True)
