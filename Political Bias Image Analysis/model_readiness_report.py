from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "model_ready_features.csv")
ann = pd.read_csv(DATA / "framing_annotations.csv")
audit = pd.read_csv(DATA / "model_feature_audit.csv")

# Use annotation labels as the authoritative target.
# Drop any existing copy from the feature dataset before merging.
if "visual_framing_label" in df.columns:
    df = df.drop(columns=["visual_framing_label"])

merged = df.merge(
    ann[["image_id", "visual_framing_label"]],
    on="image_id",
    how="left",
    validate="one_to_one"
)

labels = (
    merged["visual_framing_label"]
    .astype("string").str.strip().str.lower()
)
valid = labels.isin(["liberal", "conservative", "neutral"])

summary = [
    ("Images in feature dataset", len(df)),
    ("Unique image IDs", df["image_id"].nunique()),
    ("Images with valid framing labels", int(valid.sum())),
    ("Images awaiting labels or marked indeterminate", int((~valid).sum())),
    ("Expression features", len([
        x for x in (DATA / "expression_model_features.txt")
        .read_text().splitlines() if x.strip()
    ])),
    ("Composition features", len([
        x for x in (DATA / "composition_model_features.txt")
        .read_text().splitlines() if x.strip()
    ])),
    ("Feature columns with missing values", int((audit["missing_count"] > 0).sum())),
]

print("IDENTIPOLI — MODEL READINESS REPORT")
print("=" * 42)
for name, value in summary:
    print(f"{name}: {value}")

print("\nClass balance among valid labels:")
if valid.any():
    print(labels[valid].value_counts().to_string())
else:
    print("No valid labels yet.")

print("\nResearch safeguards:")
print("- Outlet ideology is not used as the target label.")
print("- Indeterminate images are excluded from supervised classes.")
print("- Duplicate images and same-article images must not cross train/test splits.")
print("- Missing facial measurements need explicit handling during training.")

if valid.sum() == 0:
    status = "NOT READY: add independently assigned framing labels."
elif labels[valid].nunique() < 3:
    status = "PARTIALLY READY: one or more target classes are absent."
else:
    status = "LABELS PRESENT: assess class counts and grouped evaluation."

print(f"\nSTATUS: {status}")

out = DATA / "model_readiness_report.txt"
out.write_text(
    "IDENTIPOLI — MODEL READINESS REPORT\n"
    + "\n".join(f"{name}: {value}" for name, value in summary)
    + f"\n\nSTATUS: {status}\n"
)
print(f"Saved: {out.relative_to(ROOT)}")
