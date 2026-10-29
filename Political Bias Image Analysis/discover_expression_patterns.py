from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "expression_features.csv"
OUTPUT = ROOT / "data" / "expression_patterns.csv"

def main():
    df = pd.read_csv(INPUT)

    # Use numeric facial measurements only.
    # IDs, filenames, sources, and labels must not become model features.
    feature_cols = [
        col for col in df.columns
        if col.startswith("blend_")
        or col in {
            "mouth_width_ratio",
            "mouth_open_ratio",
            "eye_open_left_ratio",
            "eye_open_right_ratio",
            "eyebrow_eye_left_ratio",
            "eyebrow_eye_right_ratio",
            "head_roll_degrees",
            "smile_signal",
            "frown_signal",
            "brow_raise_signal",
            "eye_widening_signal",
            "jaw_open_signal",
            "mouth_open_geometry",
        }
    ]

    if not feature_cols:
        raise ValueError("No facial-expression feature columns were found.")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Exclude features with no usable variation.
    usable_cols = [
        col for col in X.columns
        if X[col].notna().sum() > 0 and X[col].nunique(dropna=True) > 1
    ]
    X = X[usable_cols]

    # Keep rows with at least some measured facial information.
    valid_rows = X.notna().any(axis=1)
    result = df.loc[valid_rows].copy()
    X = X.loc[valid_rows]

    if len(X) < 6 or X.shape[1] < 2:
        raise ValueError(
            f"Not enough usable data: {len(X)} images, {X.shape[1]} features."
        )

    # Fill individual missing measurements, then standardize scales.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    # Explore a few possible cluster counts and choose by silhouette score.
    # This is exploratory, not proof that these are natural expression types.
    max_k = min(5, len(result) - 1)
    scores = {}

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)

        if len(set(labels)) > 1:
            scores[k] = silhouette_score(X_scaled, labels)

    if not scores:
        raise ValueError("Could not find a valid clustering configuration.")

    best_k = max(scores, key=scores.get)
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    result["expression_cluster"] = final_model.fit_predict(X_scaled)

    # Save the cluster assignments with image identifiers and measurements.
    result.to_csv(OUTPUT, index=False)

    print(f"Images analyzed: {len(result)}")
    print(f"Expression features used: {len(usable_cols)}")
    print(f"Selected exploratory clusters: {best_k}")
    print("\nSilhouette scores by cluster count:")
    for k, score in scores.items():
        print(f"  {k} clusters: {score:.3f}")

    print("\nImages per cluster:")
    print(result["expression_cluster"].value_counts().sort_index())

    print(f"\nSaved results to: {OUTPUT}")

if __name__ == "__main__":
    main()
