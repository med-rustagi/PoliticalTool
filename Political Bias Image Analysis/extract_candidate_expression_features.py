from pathlib import Path
import sys

import cv2
import mediapipe as mp
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extract_expression_features import get_features

INPUT = ROOT / "data" / "external_political_candidates" / "candidate_image_metadata.csv"
OUTPUT = ROOT / "data" / "external_political_candidates" / "candidate_expression_features.csv"
MODEL = ROOT / "models" / "face_landmarker.task"

if not INPUT.exists():
    raise SystemExit(f"Missing candidate metadata: {INPUT}")
if not MODEL.exists():
    raise SystemExit(f"Missing model file: {MODEL}")

metadata = pd.read_csv(INPUT)
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)

results = []

with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
    for _, item in metadata.iterrows():
        path = ROOT / str(item["image_path"])
        image = cv2.imread(str(path))

        row = item.to_dict()
        if image is None:
            row["face_detected"] = False
            row["extraction_status"] = "unreadable_image"
            results.append(row)
            print(f"SKIP unreadable image: {path.name}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        features = get_features(detector.detect(mp_image))

        row["face_detected"] = features is not None
        row["extraction_status"] = (
            "features_extracted" if features else "no_face_landmarks"
        )
        if features:
            row.update(features)

        results.append(row)
        print(f"{path.name}: {row['extraction_status']}")

pd.DataFrame(results).to_csv(OUTPUT, index=False)
print(f"\nSaved {len(results)} rows to {OUTPUT}")
print("Face landmarks detected:", sum(r["face_detected"] for r in results))
print("No landmarks:", sum(not r["face_detected"] for r in results))
