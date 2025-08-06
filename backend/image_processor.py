import numpy as np
import cv2
from PIL import Image, ImageDraw
import base64

# Current model expects 64x64 (HxW)
MODEL_IMG_SIZE = (64, 64)

def calculate_min_max_from_datacube(numerical_datacube, modalities):
    """
    Calculates the min and max value for each modality channel within a single datacube.
    """
    modality_min_max = {}
    num_modalities = numerical_datacube.shape[3]

    for mod_idx in range(num_modalities):
        modality_name = modalities[mod_idx]
        modality_data = numerical_datacube[:, :, :, mod_idx]
        
        if np.all(np.isnan(modality_data)) or not np.any(np.isfinite(modality_data)):
             min_val, max_val = 0, 1
        else:
            min_val = np.nanmin(modality_data[np.isfinite(modality_data)])
            max_val = np.nanmax(modality_data[np.isfinite(modality_data)])
        
        modality_min_max[modality_name] = (min_val, max_val)
        
    return modality_min_max


def normalize_slice(image_slice, min_val, max_val):
    """
    Normalizes a single 50x50 slice of data to an 8-bit BGR image.
    """
    if abs(max_val - min_val) < 1e-10:
        gray_slice = np.full(image_slice.shape, 128, dtype=np.uint8)
    else:
        normalized = (image_slice - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        normalized = np.nan_to_num(normalized, nan=0.0)
        gray_slice = (normalized * 255).astype(np.uint8)
    
    # Convert grayscale to 3 channel BGR
    bgr_slice = cv2.cvtColor(gray_slice, cv2.COLOR_GRAY2BGR)
    return bgr_slice


def convert_datacube_to_images(numerical_datacube, config, tier):
    """
    Converts a raw numerical datacube into a final sequence of concatenated images.
    """
    modalities = config['modalities']
    print("Converting numerical datacube to image format...")
    
    local_min_max = calculate_min_max_from_datacube(numerical_datacube, modalities)
    print(f"Calculated local min/max ranges: {local_min_max}")

    height, width, num_days, num_modalities = numerical_datacube.shape
    
    # This will hold the final sequence of n images
    final_sequence = []

    for mod_idx in range(num_modalities):
        daily_modality_images = []
        for day_idx in range(num_days):
            modality_name = modalities[mod_idx]
            min_val, max_val = local_min_max[modality_name]
            
            numerical_slice = numerical_datacube[:, :, day_idx, mod_idx]
            
            # Normalize to a (50, 50, mod) BGR image
            bgr_slice = normalize_slice(numerical_slice, min_val, max_val)
            
            # Resize to a (64, 64, mod) BGR image
            if tier == 'free':
                resized_bgr_slice = cv2.resize(bgr_slice, MODEL_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
                resized_bgr_slice = resized_bgr_slice.astype('float32') / 255.0
            
            elif tier in ['tier1', 'tier2', 'admin']:
                img = Image.fromarray(bgr_slice).resize(MODEL_IMG_SIZE).convert("L")
                resized_bgr_slice = np.array(img)
                
            daily_modality_images.append(resized_bgr_slice)
                    
        final_sequence.append(daily_modality_images)
        
    if tier == 'free':
        return np.array(final_sequence)
        
    # Stack the num_days daily images to create the final sequence
    final_sequence_array = np.stack(final_sequence, axis=-1)
    
    print(f"Image conversion and resizing complete. Final shape: {final_sequence_array.shape}")
    return final_sequence_array

def draw_detections_on_image_and_save(input_img: Image.Image, detections: list):
    # Make a copy for drawing
    print('Creating BOUNDING BOX ::::::::::::::')
    image = input_img.copy()
    draw = ImageDraw.Draw(image)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline='blue', width=4)
        
    print('BOUNDING BOX CREATED ::::::::::::::')

    # Ensure the save directory exists
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')

    # Convert to base64 for frontend output
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_img
