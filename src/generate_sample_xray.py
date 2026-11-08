import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

def generate_synthetic_dental_xray(file_path: Path):
    """
    Generates a realistic synthetic intraoral dental radiograph
    with molar teeth structures, enamel borders, and simulated caries anomalies.
    """
    width, height = 512, 512
    # Base dark radiographic background
    img_array = np.random.normal(loc=35, scale=8, size=(height, width)).astype(np.uint8)
    
    img = Image.fromarray(img_array).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw simulated alveolar jaw bone (dense radio-opaque region)
    draw.rectangle([20, 280, 490, 490], fill=(95, 95, 95))
    
    # Draw 3 molar teeth with pulp chambers
    teeth_coords = [
        (60, 110, 180, 360),   # Molar 1
        (200, 100, 320, 370),  # Molar 2 (Target decay)
        (340, 120, 460, 365)   # Molar 3
    ]
    
    for idx, (x1, y1, x2, y2) in enumerate(teeth_coords):
        # Tooth Crown & Root (White Radio-opaque enamel)
        draw.rounded_rectangle([x1, y1, x2, y2], radius=25, fill=(215, 215, 215), outline=(245, 245, 245), width=3)
        # Inner Pulp Chamber (Dark Radio-lucent nerve canal)
        pulp_w = (x2 - x1) // 3
        draw.rounded_rectangle([x1 + pulp_w, y1 + 45, x2 - pulp_w, y2 - 20], radius=10, fill=(55, 55, 55))
        
        # Add localized lesion on Molar 2 (Caries / Tooth decay)
        if idx == 1:
            # Cavity radiolucency on enamel crown
            draw.ellipse([x1 + 10, y1 + 15, x1 + 48, y1 + 52], fill=(42, 42, 42))
            # Periapical root apex infection (dark halo at root tip)
            draw.ellipse([x1 + 25, y2 - 25, x2 - 25, y2 + 25], fill=(30, 30, 30))

    img.save(file_path)
    print(f"[*] Generated synthetic dental X-ray radiograph saved to: {file_path}")

if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent.parent / "data" / "sample_dental_xray.png"
    generate_synthetic_dental_xray(out_file)
