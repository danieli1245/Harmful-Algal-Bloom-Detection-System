import base64
import io
from PIL import Image, ImageDraw, ImageFont

def draw_detections_on_image_and_save(input_img: Image.Image, detections: list):
    image = input_img.copy()
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence = round(det.get("confidence", 0) * 100, 2)
        label = det.get("label", "")

        draw.rectangle([x1, y1, x2, y2], outline='red', width=4)

        if label:
            text = f"   {label} ({confidence}%)   "
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if y1 - text_height < 0:
                # Draw label inside the bounding box at the top
                label_rect = [x1, y1, x1 + text_width, y1 + text_height]
                text_pos = (x1, y1)
            else:
                # Draw label above the bounding box
                label_rect = [x1, y1 - text_height, x1 + text_width, y1]
                text_pos = (x1, y1 - text_height)
            draw.rectangle(label_rect, fill='red')
            draw.text(text_pos, text, fill="white", font=font)

    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    base64_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_img