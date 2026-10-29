import pandas as pd
import cv2
import math
from PIL import Image, ImageDraw

METADATA = "data/face_metadata.csv"
MODEL = "models/face_detection_yunet_2023mar.onnx"
OUTPUT = "data/primary_face_review.jpg"

CONFIDENCE_THRESHOLD = 0.85

df = pd.read_csv(METADATA)

detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    CONFIDENCE_THRESHOLD,
    0.3,
    5000
)

COLS = 4
CELL_W = 400
CELL_H = 330
LABEL_H = 40

ROWS = math.ceil(len(df) / COLS)

sheet = Image.new(
    "RGB",
    (COLS * CELL_W, ROWS * CELL_H),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, (_, row) in enumerate(df.iterrows()):

    image_id = row["image_id"]
    image_path = row["image_path"]

    img = cv2.imread(image_path)

    if img is None:
        continue

    height, width = img.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        # Mark images where YuNet found no face
        cv2.putText(
            img,
            "NO FACE DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3
        )
        primary_number = "NONE"

    else:

        # Find largest face
        face_areas = []

        for face_number, face in enumerate(faces, start=1):

            x, y, w, h = face[:4]

            area = w * h

            face_areas.append(
                (area, face_number, x, y, w, h)
            )

        primary = max(face_areas, key=lambda item: item[0])

        area, primary_number, x, y, w, h = primary

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # Draw PRIMARY face box
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            8
        )

        # Label selected face
        cv2.putText(
            img,
            f"PRIMARY FACE: {primary_number}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 0, 0),
            3
        )

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img)

    pil_img.thumbnail(
        (CELL_W - 20, CELL_H - LABEL_H - 10)
    )

    cell_x = (i % COLS) * CELL_W
    cell_y = (i // COLS) * CELL_H

    img_x = cell_x + (CELL_W - pil_img.width) // 2
    img_y = cell_y + 5

    sheet.paste(pil_img, (img_x, img_y))

    label = f"{image_id} | primary={primary_number}"

    draw.text(
        (cell_x + 10, cell_y + CELL_H - LABEL_H),
        label,
        fill="black"
    )

sheet.save(OUTPUT, quality=90)

print()
print("=== Primary Face Review ===")
print(f"Images reviewed: {len(df)}")
print(f"Saved to: {OUTPUT}")