import argparse
import csv
import math
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_IMAGE_DIR = ROOT_DIR / "data" / "images"
DEFAULT_IMAGE_PATH = DEFAULT_IMAGE_DIR / "bbc_0001.jpg"
DEFAULT_METADATA_PATH = ROOT_DIR / "data" / "metadata.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FEATURE_COLUMNS = [
    "face_count",
    "face_size",
    "face_position",
    "face_x",
    "face_y",
    "face_width",
    "face_height",
    "confidence",
]


def resolve_image_path(image_path: str) -> Path:
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()
    return path


def update_metadata_csv(image_file: Path, face_records: list[dict]) -> None:
    csv_path = DEFAULT_METADATA_PATH
    if not csv_path.exists():
        return

    relative_path = image_file.relative_to(ROOT_DIR).as_posix()

    with csv_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        existing_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    base_fieldnames = [
        "image_id",
        "image_path",
        "source",
        "source_ideology",
        "candidate",
        "article_date",
        "article_url",
        "image_url",
    ]
    fieldnames = []
    for name in base_fieldnames + existing_fieldnames:
        if name not in fieldnames:
            fieldnames.append(name)
    for column in FEATURE_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    output_rows = []
    matched = False
    for row in rows:
        normalized = {field: row.get(field, "") for field in fieldnames}
        row_path = (normalized.get("image_path") or "").strip().replace("\\", "/")
        if row_path == relative_path:
            matched = True
            if face_records:
                primary = face_records[0]
                normalized["face_count"] = str(len(face_records))
                normalized["face_size"] = f"{primary['face_size']:.4f}"
                normalized["face_position"] = f"{primary['face_position']:.4f}"
                normalized["face_x"] = f"{primary['x']:.1f}"
                normalized["face_y"] = f"{primary['y']:.1f}"
                normalized["face_width"] = f"{primary['face_width']:.1f}"
                normalized["face_height"] = f"{primary['face_height']:.1f}"
                normalized["confidence"] = f"{primary['confidence']:.3f}"
            else:
                normalized["face_count"] = "0"
                normalized["face_size"] = ""
                normalized["face_position"] = ""
                normalized["face_x"] = ""
                normalized["face_y"] = ""
                normalized["face_width"] = ""
                normalized["face_height"] = ""
                normalized["confidence"] = ""
        output_rows.append(normalized)

    if not matched:
        new_row = {field: "" for field in fieldnames}
        new_row["image_id"] = image_file.stem
        new_row["image_path"] = relative_path
        if face_records:
            primary = face_records[0]
            new_row["face_count"] = str(len(face_records))
            new_row["face_size"] = f"{primary['face_size']:.4f}"
            new_row["face_position"] = f"{primary['face_position']:.4f}"
            new_row["face_x"] = f"{primary['x']:.1f}"
            new_row["face_y"] = f"{primary['y']:.1f}"
            new_row["face_width"] = f"{primary['face_width']:.1f}"
            new_row["face_height"] = f"{primary['face_height']:.1f}"
            new_row["confidence"] = f"{primary['confidence']:.3f}"
        else:
            new_row["face_count"] = "0"
        output_rows.append(new_row)

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def analyze_image(image_path: str) -> list[dict]:
    image_file = resolve_image_path(image_path)
    image = cv2.imread(str(image_file))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")

    height, width = image.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        str(DEFAULT_MODEL_PATH),
        "",
        (width, height),
        0.7,
        0.3,
        5000,
    )

    _, faces = detector.detect(image)

    print(f"Image: {image_file}")
    print(f"Image size: {width} x {height}")

    if faces is None or len(faces) == 0:
        print("Faces detected: 0")
        update_metadata_csv(image_file, [])
        return []

    print(f"Faces detected: {len(faces)}")

    image_area = width * height
    image_center_x = width / 2
    image_center_y = height / 2
    image_diagonal = math.sqrt(width**2 + height**2)
    face_records = []

    for i, face in enumerate(faces):
        x, y, face_width, face_height = face[:4]
        confidence = face[-1]

        face_center_x = x + face_width / 2
        face_center_y = y + face_height / 2
        face_area = face_width * face_height
        face_size = face_area / image_area

        distance = math.sqrt(
            (face_center_x - image_center_x) ** 2 +
            (face_center_y - image_center_y) ** 2
        )

        face_position = 1 - (2 * distance / image_diagonal)
        record = {
            "x": x,
            "y": y,
            "face_width": face_width,
            "face_height": face_height,
            "confidence": confidence,
            "face_size": face_size,
            "face_position": face_position,
        }
        face_records.append(record)

        print(f"\nFace {i + 1}")
        print("--------------------")
        print("Bounding box:")
        print(f"  x = {x:.1f}")
        print(f"  y = {y:.1f}")
        print(f"  width = {face_width:.1f}")
        print(f"  height = {face_height:.1f}")
        print("\nDetection confidence:")
        print(f"  {confidence:.3f}")
        print("\nResearch-derived features:")
        print(f"  Face size = {face_size:.4f}")
        print(f"  Face position = {face_position:.4f}")

    update_metadata_csv(image_file, face_records)
    return face_records


def get_default_image_paths() -> list[str]:
    if not DEFAULT_IMAGE_DIR.exists():
        return [str(DEFAULT_IMAGE_PATH)]

    paths = []
    for image_path in sorted(DEFAULT_IMAGE_DIR.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(str(image_path))

    return paths or [str(DEFAULT_IMAGE_PATH)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect faces in one or more images.")
    parser.add_argument(
        "images",
        nargs="*",
        help="One or more image files or folders to analyze. If omitted, all images in data/images are processed.",
    )
    args = parser.parse_args()

    if not args.images:
        image_paths = get_default_image_paths()
    else:
        image_paths = []
        for raw_path in args.images:
            path = resolve_image_path(raw_path)
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                        image_paths.append(str(child))
            elif path.is_file():
                image_paths.append(str(path))
            else:
                raise FileNotFoundError(f"Image or folder not found: {raw_path}")

    for image_path in image_paths:
        analyze_image(image_path)


if __name__ == "__main__":
    main()