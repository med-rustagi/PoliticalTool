from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "visual_signal_analysis.csv"
OUTPUT = DATA / "expression_report.csv"

SIGNALS = [
    "smile_signal",
    "frown_signal",
    "brow_raise_signal",
    "eye_widening_signal",
    "jaw_open_signal",
]

def main():
    df = pd.read_csv(INPUT)

    required = ["image_id", "image_path", "face_detected"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in SIGNALS:
        if col not in df.columns:
            raise ValueError(f"Missing expression signal: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    detected = df["face_detected"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    report = df[required + SIGNALS].copy()
    report["analysis_status"] = "analyzed"
    report.loc[~detected, "analysis_status"] = "no_face_detected"

    for signal in SIGNALS:
        # Calculate thresholds using detected faces only.
        reference = df.loc[detected, signal].dropna()
        flag_col = f"{signal}_relatively_high"

        if len(reference) < 5:
            report[flag_col] = pd.NA
            continue

        threshold = reference.quantile(0.75)
        report[flag_col] = (
            detected
            & df[signal].notna()
            & (df[signal] > threshold)
        )

        print(f"{signal}: upper-quartile threshold = {threshold:.4f}")

    report.to_csv(OUTPUT, index=False)

    print(f"\nImages processed: {len(report)}")
    print(f"Images with detected faces: {int(detected.sum())}")
    print(f"Images without detected faces: {int((~detected).sum())}")
    print(f"Saved report: {OUTPUT}")

if __name__ == "__main__":
    main()
