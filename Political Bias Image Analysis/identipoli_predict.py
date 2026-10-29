import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "src" / "extract_expression_features.py"
FEATURES_CSV = ROOT / "data" / "expression_features.csv"

# These are illustrative phrases, not a comprehensive political dictionary.
# A phrase match is only a weak signal and must be checked in context.
STANCE_TERMS = {
    "liberal-leaning": [
        "pro-choice",
        "abortion rights",
        "gun control",
        "universal healthcare",
        "universal health care",
        "climate action",
        "raise the minimum wage",
        " LGBTQ rights ",
        "same-sex marriage",
        "wealth tax",
        "expand social programs",
    ],
    "conservative-leaning": [
        "pro-life",
        "gun rights",
        "tax cuts",
        "lower taxes",
        "border security",
        "traditional family values",
        "school choice",
        "deregulation",
        "religious freedom",
        "stronger border enforcement",
    ],
}

FEATURE_ALIASES = {
    "mouth_width_ratio": ["mouth_width_ratio"],
    "mouth_open_ratio": ["mouth_open_ratio"],
    "left_eye_open_ratio": [
        "left_eye_open_ratio", "eye_open_left_ratio"
    ],
    "right_eye_open_ratio": [
        "right_eye_open_ratio", "eye_open_right_ratio"
    ],
    "left_brow_eye_distance": [
        "left_brow_eye_distance", "eyebrow_eye_left_ratio"
    ],
    "right_brow_eye_distance": [
        "right_brow_eye_distance", "eyebrow_eye_right_ratio"
    ],
    "head_roll_degrees": ["head_roll_degrees"],
}


def find_phrase_matches(text, terms):
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    padded = " " + normalized + " "

    matches = []
    for term in terms:
        normalized_term = re.sub(
            r"[^a-z0-9]+", " ", term.lower()
        ).strip()
        if normalized_term and " " + normalized_term + " " in padded:
            matches.append(term.strip())

    return matches


def analyze_context(text):
    matches = {
        category: find_phrase_matches(text, terms)
        for category, terms in STANCE_TERMS.items()
    }

    left_count = len(matches["liberal-leaning"])
    right_count = len(matches["conservative-leaning"])

    if left_count == 0 and right_count == 0:
        prediction = "neutral / unclear"
        explanation = (
            "No listed stance phrases were found. This does not prove "
            "the article is neutral; the text may discuss politics "
            "without using these phrases."
        )
    elif left_count == right_count:
        prediction = "neutral / unclear"
        explanation = (
            "The simple phrase counts are tied. The text needs "
            "contextual review."
        )
    elif abs(left_count - right_count) < 2:
        prediction = "uncertain"
        explanation = (
            "The phrase-count difference is small. No reliable "
            "viewpoint conclusion can be drawn from this heuristic."
        )
    elif left_count > right_count:
        prediction = "tentatively liberal-leaning"
        explanation = (
            "More phrases from the illustrative liberal-associated "
            "list were found. Matches may be quoted, criticized, "
            "or discussed without endorsement."
        )
    else:
        prediction = "tentatively conservative-leaning"
        explanation = (
            "More phrases from the illustrative conservative-associated "
            "list were found. Matches may be quoted, criticized, "
            "or discussed without endorsement."
        )

    return {
        "prediction": prediction,
        "status": "experimental_keyword_heuristic",
        "phrase_matches": matches,
        "explanation": explanation,
    }


def extract_face_features(image_path):
    if not EXTRACTOR.exists():
        raise FileNotFoundError(f"Extractor not found: {EXTRACTOR}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Feature CSV not found: {FEATURES_CSV}")

    # Preserve the full dataset because the existing extractor writes
    # single-image results to expression_features.csv.
    original_csv = FEATURES_CSV.read_bytes()

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(EXTRACTOR),
                "--image",
                str(image_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Face extraction failed:\n" + result.stderr
            )

        single = pd.read_csv(FEATURES_CSV)

        if single.empty:
            return {"face_detected": False, "features": {}}

        row = single.iloc[-1]
        detected = str(row.get("face_detected", "")).lower() in {
            "true", "1", "yes"
        }

        features = {}
        if detected:
            for output_name, candidates in FEATURE_ALIASES.items():
                for column in candidates:
                    if column in row.index and pd.notna(row[column]):
                        features[output_name] = round(
                            float(row[column]), 5
                        )
                        break

        return {
            "face_detected": detected,
            "features": features,
        }

    finally:
        # Restore the original 67-image dataset even if extraction fails.
        FEATURES_CSV.write_bytes(original_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Identipoli experimental context and face analyzer"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--text",
        required=True,
        help="Paste the article headline and/or article text",
    )
    parser.add_argument("--output", default="data/identipoli_report.json")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()

    face = extract_face_features(image_path)
    context = analyze_context(args.text)

    report = {
        "image": str(image_path),
        "context_analysis": context,
        "facial_analysis": {
            **face,
            "interpretation": (
                "Descriptive geometry only. Facial measurements do not "
                "determine political ideology and are not used to assign "
                "the context category."
            ),
        },
        "overall_status": (
            "Experimental prototype; not validated for political-bias "
            "classification."
        ),
    }

    output = json.dumps(report, indent=2)
    print(output)

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output + "\n", encoding="utf-8")
    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
