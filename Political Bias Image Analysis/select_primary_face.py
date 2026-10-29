import pandas as pd
import cv2

METADATA = "data/clean_metadata.csv"
MODEL = "models/face_detection_yunet_2023mar.onnx"
OUTPUT = "data/face_metadata.csv"

CONFIDENCE_THRESHOLD = 0.85

# --------------------------------------------------
# Manual corrections
# --------------------------------------------------

PRIMARY_FACE_OVERRIDES = {
    "bbc_0002": 2
}

# --------------------------------------------------
# Load usable images
# --------------------------------------------------

df = pd.read_csv(METADATA)

# Make sure usable values are interpreted correctly
df["usable"] = df["usable"].astype(str).str.lower().eq("true")

df = df[df["usable"]].copy()

print(f"Usable images loaded: {len(df)}")

# --------------------------------------------------
# Load YuNet
# --------------------------------------------------

detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    CONFIDENCE_THRESHOLD,
    0.3,
    5000
)

results = []

# --------------------------------------------------
# Detect faces and select primary face
# --------------------------------------------------

for _, row in df.iterrows():

    image_id = row["image_id"]
    image_path = row["image_path"]

    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not open: {image_path}")
        continue

    height, width = img.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(img)

    if faces is None:
        faces = []

    face_count = len(faces)

    # --------------------------------------------------
    # No faces detected
    # --------------------------------------------------

    if face_count == 0:

        results.append({
            "image_id": image_id,
            "image_path": image_path,
            "source": row["source"],
            "face_count": 0,
            "primary_face": None,
            "primary_x": None,
            "primary_y": None,
            "primary_width": None,
            "primary_height": None,
            "primary_area": None,
            "primary_confidence": None
        })

        continue

    # --------------------------------------------------
    # Collect face information
    # --------------------------------------------------

    face_data = []

    for face_number, face in enumerate(faces, start=1):

        x, y, w, h = face[:4]
        confidence = float(face[14])

        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        area = w * h

        face_data.append({
            "face_number": face_number,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "area": area,
            "confidence": confidence
        })

    # --------------------------------------------------
    # Default: largest face
    # --------------------------------------------------

    primary = max(face_data, key=lambda f: f["area"])

    # --------------------------------------------------
    # Apply manual override if needed
    # --------------------------------------------------

    if image_id in PRIMARY_FACE_OVERRIDES:

        selected_face_number = PRIMARY_FACE_OVERRIDES[image_id]

        matching_faces = [
            f for f in face_data
            if f["face_number"] == selected_face_number
        ]

        if matching_faces:
            primary = matching_faces[0]

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------

    results.append({
        "image_id": image_id,
        "image_path": image_path,
        "source": row["source"],
        "face_count": face_count,
        "primary_face": primary["face_number"],
        "primary_x": primary["x"],
        "primary_y": primary["y"],
        "primary_width": primary["width"],
        "primary_height": primary["height"],
        "primary_area": primary["area"],
        "primary_confidence": primary["confidence"]
    })

# --------------------------------------------------
# Save metadata
# --------------------------------------------------

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT, index=False)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("=== Primary Face Selection ===")
print(f"Images processed: {len(result_df)}")
print(f"Images with faces: {(result_df['face_count'] > 0).sum()}")
print(f"Images with multiple faces: {(result_df['face_count'] > 1).sum()}")
print()
print(f"Saved to: {OUTPUT}")