import numpy as np
from PIL import Image, ImageOps

class DentalXRayPreprocessor:
    """
    OpenCV / PIL image preprocessor for intraoral dental radiographs.
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
    and normalization to sharpen enamel-dentin boundaries for YOLOv8 detection.
    """
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def preprocess_image(self, pil_image: Image.Image) -> Image.Image:
        """
        1. Converts to grayscale.
        2. Applies contrast equalization (CLAHE emulation via autocontrast).
        3. Resizes to model input dimensions.
        """
        # Convert to single channel grayscale
        gray = pil_image.convert("L")
        
        # Contrast Enhancement (Sharpens subtle radiolucent cavities)
        enhanced = ImageOps.autocontrast(gray, cutoff=2)
        
        # Resize to fixed standard input size
        resized = enhanced.resize(self.target_size)
        
        # Convert back to 3-channel RGB for standard Vision backbones
        return Image.merge("RGB", (resized, resized, resized))

    def to_normalized_tensor_array(self, pil_image: Image.Image) -> np.ndarray:
        """Normalizes image array to [0.0, 1.0] float32 for ONNX / PyTorch inference."""
        arr = np.array(pil_image).astype(np.float32) / 255.0
        # Transpose HWC -> CHW (Standard PyTorch tensor format)
        arr = np.transpose(arr, (2, 0, 1))
        # Add Batch dimension: (1, 3, 512, 512)
        return np.expand_dims(arr, axis=0)
