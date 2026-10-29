import pandas as pd
import cv2
import math
from PIL import Image, ImageDraw

METADATA = "data/clean_metadata.csv"
MODEL = "models/face_detection_yunet_2023mar.onnx"
OUTPUT = "data/face_review.jpg"

# Load usable images
df = pd.read_csv(METADATA)
df = df[df["usable"] == True].copy()

# YuNet face detector
detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.85,
    0.3,
    5000
)

COLS = 4
CELL_W = 400
CELL_H = 330
LABEL_H = 35

ROWS = math.ceil(len(df) / COLS)

sheet = Image.new(
    "RGB",
    (COLS * CELL_W, ROWS * CELL_H),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, (_, row) in enumerate(df.iterrows()):

    path = row["image_path"]

    try:
        img = cv2.imread(path)

        if img is None:
            print(f"Could not open: {path}")
            continue

        height, width = img.shape[:2]

        # Tell YuNet the actual image size
        detector.setInputSize((width, height))

        # Detect ALL faces
        _, faces = detector.detect(img)

        if faces is None:
            faces = []

        # Draw every detected face
        for face_number, face in enumerate(faces, start=1):

            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            # Draw rectangle
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                5
            )

            # Label face
            cv2.putText(
                img,
                str(face_number),
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3
            )

        # Add total face count
        cv2.putText(
            img,
            f"Faces: {len(faces)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3
        )

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img)

        # Resize for contact sheet
        pil_img.thumbnail(
            (CELL_W - 20, CELL_H - LABEL_H - 10)
        )

        cell_x = (i % COLS) * CELL_W
        cell_y = (i // COLS) * CELL_H

        img_x = cell_x + (CELL_W - pil_img.width) // 2
        img_y = cell_y + 5

        sheet.paste(pil_img, (img_x, img_y))

        # Image label
        label = f"{row['image_id']} | faces={len(faces)}"

        draw.text(
            (cell_x + 10, cell_y + CELL_H - LABEL_H),
            label,
            fill="black"
        )

    except Exception as e:
        print(f"Error processing {path}: {e}")

sheet.save(OUTPUT, quality=90)

print()
print("=== Face Review Sheet ===")
print(f"Images reviewed: {len(df)}")
print(f"Saved to: {OUTPUT}")