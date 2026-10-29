from pathlib import Path
import pandas as pd
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean_metadata.csv"
FACES = ROOT / "data" / "face_metadata.csv"
EXPRESSIONS = ROOT / "data" / "expression_features.csv"

OUTPUT = ROOT / "data" / "image_composition_analysis.csv"
SUMMARY = ROOT / "data" / "image_composition_summary.csv"

clean = pd.read_csv(CLEAN)
faces = pd.read_csv(FACES)
expressions = pd.read_csv(EXPRESSIONS)

# Use the cleaned metadata as the master list.
df = clean.merge(
    faces.drop(columns=["image_path", "source"], errors="ignore"),
    on="image_id",
    how="left",
    suffixes=("", "_face"),
)

# Recover image dimensions from the original image files.
def image_dimensions(path):
    image = cv2.imread(str(path))
    if image is None:
        return pd.Series({"image_width_px": np.nan, "image_height_px": np.nan})
    height, width = image.shape[:2]
    return pd.Series({"image_width_px": width, "image_height_px": height})

dimensions = df["image_path"].apply(image_dimensions)
df = pd.concat([df, dimensions], axis=1)

# Face area divided by full image area.
df["face_area_ratio"] = (
    pd.to_numeric(df["primary_width"], errors="coerce")
    * pd.to_numeric(df["primary_height"], errors="coerce")
    / (df["image_width_px"] * df["image_height_px"])
)

# Normalize face center to a 0–1 scale across the image.
df["face_center_x_ratio"] = (
    pd.to_numeric(df["primary_x"], errors="coerce")
    + pd.to_numeric(df["primary_width"], errors="coerce") / 2
) / df["image_width_px"]

df["face_center_y_ratio"] = (
    pd.to_numeric(df["primary_y"], errors="coerce")
    + pd.to_numeric(df["primary_height"], errors="coerce") / 2
) / df["image_height_px"]

df["face_detection_confidence"] = pd.to_numeric(
    df["primary_confidence"], errors="coerce"
)

# Attach the expression signals without duplicating their column names.
signal_columns = [
    "image_id", "smile_signal", "frown_signal",
    "brow_raise_signal", "eye_widening_signal",
    "jaw_open_signal", "mouth_open_geometry", "head_roll_degrees"
]
if all(c in expressions.columns for c in signal_columns):
    df = df.merge(
        expressions[signal_columns],
        on="image_id",
        how="left",
        suffixes=("", "_expression"),
    )

df.to_csv(OUTPUT, index=False)

# Summarize composition by outlet; standard deviation shows variation.
metrics = [
    "face_area_ratio",
    "face_center_x_ratio",
    "face_center_y_ratio",
    "face_count",
    "face_detection_confidence",
]
summary = df.groupby("source")[metrics].agg(["count", "mean", "std"])
summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(SUMMARY, index=False)

print(f"Metadata rows: {len(df)}")
print(f"Images with readable files: {df['image_width_px'].notna().sum()}")
print(f"Images with face-area measurements: {df['face_area_ratio'].notna().sum()}")
print("\nMean composition measurements by outlet:")
print(df.groupby("source")[metrics].mean().round(4).to_string())
print(f"\nSaved detailed results: {OUTPUT}")
print(f"Saved outlet summary: {SUMMARY}")
print(
    "\nCaution: composition differences are descriptive, not proof of "
    "political slant. Subjects, events, image selection, and photo quality "
    "may explain differences."
)
