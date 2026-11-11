from pathlib import Path

import cv2
import numpy as np

from bias_indicators import analyze_caption


def analyze_image(image_file):
    """
    Extract descriptive image features.
    These measurements do not determine political ideology.
    """
    image_file.seek(0)
    file_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
    image_file.seek(0)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))

    # Average color composition, expressed as RGB percentages.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    average_rgb = np.mean(rgb, axis=(0, 1))

    # Approximate dominant color using a small color palette.
    resized = cv2.resize(rgb, (100, 100))
    pixels = resized.reshape(-1, 3)
    quantized = (pixels // 64) * 64
    colors, counts = np.unique(quantized, axis=0, return_counts=True)

    dominant_index = int(np.argmax(counts))
    dominant_rgb = colors[dominant_index].tolist()

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 3),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "saturation": round(saturation, 2),
        "average_rgb": [round(float(x), 1) for x in average_rgb],
        "approx_dominant_rgb": dominant_rgb,
        "disclaimer": (
            "Brightness, contrast, saturation, and color are descriptive "
            "visual measurements. They do not independently establish "
            "political bias or ideology."
        ),
    }


def analyze_post(image_file, caption):
    image_results = analyze_image(image_file)
    caption_results = analyze_caption(caption)

    return {
        "image": image_results,
        "caption": caption_results,
    }
