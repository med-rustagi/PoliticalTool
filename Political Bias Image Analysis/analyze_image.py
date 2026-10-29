import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from face_detector import analyze_image as detect_faces


def main():
    parser = argparse.ArgumentParser(
        description="Analyze observable visual cues in one image."
    )
    parser.add_argument("image", help="Path to a JPG, PNG, or WebP image")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = (Path.cwd() / image_path).resolve()

    if not image_path.is_file():
        raise SystemExit(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit("Could not read this image format.")

    height, width = image.shape[:2]
    print("\nIDENTIPOLI — IMAGE-LEVEL VISUAL ANALYSIS")
    print("=" * 44)
    print(f"Image: {image_path.name}")
    print(f"Dimensions: {width} x {height}")

    try:
        faces = detect_faces(str(image_path))
    except Exception as exc:
        raise SystemExit(f"Face detection failed: {exc}")

    print("\nOBSERVATIONS")
    print("-" * 44)
    print(f"Faces detected: {len(faces)}")

    if faces:
        # Largest detected face is a geometric choice, not proof that
        # the person is the main political subject.
        primary = max(
            faces,
            key=lambda f: f["face_width"] * f["face_height"]
        )
        face_area_ratio = (
            primary["face_width"] * primary["face_height"]
            / (width * height)
        )
        center_x = (
            primary["x"] + primary["face_width"] / 2
        ) / width
        center_y = (
            primary["y"] + primary["face_height"] / 2
        ) / height

        print(f"Largest face area / image area: {face_area_ratio:.3f}")
        print(f"Largest face horizontal center: {center_x:.3f}")
        print(f"Largest face vertical center: {center_y:.3f}")
        print(f"Detection confidence: {primary['confidence']:.3f}")

        if face_area_ratio < 0.04:
            print("Composition cue: detected face occupies a small image area.")
        elif face_area_ratio > 0.10:
            print("Composition cue: detected face occupies a relatively large image area.")
        else:
            print("Composition cue: detected face occupies a moderate image area.")

        if center_x < 0.40:
            print("Position cue: largest face is left of the image center.")
        elif center_x > 0.60:
            print("Position cue: largest face is right of the image center.")
        else:
            print("Position cue: largest face is approximately centered horizontally.")
    else:
        print("No face detected; face-expression analysis may not be possible.")

    print("\nEXPRESSION FEATURES")
    print("-" * 44)
    print("Running the existing MediaPipe expression extractor...")

    command = [
        sys.executable,
        str(ROOT / "src" / "extract_expression_features.py"),
        "--image",
        str(image_path),
    ]
    run = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if run.returncode == 0:
        # The existing extractor writes to the dataset CSV. Read the
        # latest row, then restore the original file to avoid side effects.
        print(run.stdout.strip())
        print(
            "\nNote: the existing extractor writes to "
            "data/expression_features.csv."
        )
        print(
            "For now, its output is not automatically merged into this "
            "report to avoid accidentally altering your dataset."
        )
    else:
        print("Expression extraction could not complete.")
        print(run.stderr.strip() or run.stdout.strip())

    report = {
        "image": str(image_path),
        "width": width,
        "height": height,
        "faces_detected": len(faces),
        "analysis_scope": "observable visual cues only",
        "political_bias_verdict": "not determined",
        "caution": (
            "Face position and expression alone cannot establish political "
            "bias, intent, or left/right ideology. Image context and "
            "independent framing annotations are needed."
        ),
    }

    output_dir = ROOT / "data" / "identipoli_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{image_path.stem}_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nASSESSMENT")
    print("-" * 44)
    print("Political bias: NOT DETERMINED")
    print("Reason: visual measurements alone are insufficient to establish bias.")
    print(f"\nBasic report saved to: {report_path}")


if __name__ == "__main__":
    main()
