from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "expression_features.csv"
OUTPUT = ROOT / "data" / "visual_signal_analysis.csv"
SUMMARY = ROOT / "data" / "visual_signal_summary.csv"

if not INPUT.exists():
    raise SystemExit(f"Missing input file: {INPUT}")

df = pd.read_csv(INPUT)

if df.empty:
    raise SystemExit("The input CSV has no data rows.")

def get_col(name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")

def average_columns(names):
    available = [name for name in names if name in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[available].apply(pd.to_numeric, errors="coerce").mean(axis=1)

# Descriptive blendshape indicators, not emotion or ideology diagnoses.
df["smile_signal"] = average_columns([
    "blend_mouthSmileLeft", "blend_mouthSmileRight"
])
df["frown_signal"] = average_columns([
    "blend_mouthFrownLeft", "blend_mouthFrownRight"
])
df["brow_raise_signal"] = average_columns([
    "blend_browInnerUp",
    "blend_browOuterUpLeft",
    "blend_browOuterUpRight",
])
df["eye_widening_signal"] = average_columns([
    "blend_eyeWideLeft", "blend_eyeWideRight"
])
df["jaw_open_signal"] = get_col("blend_jawOpen")
df["mouth_open_geometry"] = get_col("mouth_open_ratio")
df["head_roll_degrees"] = get_col("head_roll_degrees")

# Preserve existing data and save the new derived features separately.
df.to_csv(OUTPUT, index=False)

signals = [
    "smile_signal",
    "frown_signal",
    "brow_raise_signal",
    "eye_widening_signal",
    "jaw_open_signal",
    "mouth_open_geometry",
    "head_roll_degrees",
]

valid = df.copy()
if "face_detected" in valid.columns:
    valid = valid[
        valid["face_detected"].astype(str).str.lower().isin(["true", "1"])
    ]

summary = valid.groupby("source")[signals].agg(["count", "mean", "std"])
summary.columns = [
    f"{feature}_{stat}" for feature, stat in summary.columns
]
summary = summary.reset_index()
summary.to_csv(SUMMARY, index=False)

print(f"Rows in input: {len(df)}")
print(f"Rows with detected faces: {len(valid)}")
print("\nMean descriptive signals by outlet:")
print(valid.groupby("source")[signals].mean().round(4).to_string())
print(f"\nDetailed results saved to: {OUTPUT}")
print(f"Outlet summary saved to: {SUMMARY}")
print(
    "\nInterpretation warning: these are model-estimated visual signals. "
    "They do not establish emotion, political ideology, or media bias."
)
