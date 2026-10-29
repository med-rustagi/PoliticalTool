from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "expression_features.csv")
original = pd.read_csv(DATA / "expression_patterns.csv")

features = [
    "blend_mouthSmileLeft",
    "blend_mouthSmileRight",
    "blend_jawOpen",
    "blend_mouthFrownLeft",
    "blend_mouthFrownRight",
    "blend_browDownLeft",
    "blend_browDownRight",
    "blend_browInnerUp",
    "blend_eyeWideLeft",
    "blend_eyeWideRight",
    "head_roll_degrees",
    "mouth_open_ratio",
]

features = [c for c in features if c in df.columns]
X = df[features].apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), -float("inf")], float("nan"))

valid = X.notna().any(axis=1)
X = X.loc[valid]
ids = df.loc[valid, "image_id"].astype(str)

X = SimpleImputer(strategy="median").fit_transform(X)
X = StandardScaler().fit_transform(X)

model = KMeans(n_clusters=2, random_state=42, n_init=30)
labels = model.fit_predict(X)

print("Images analyzed:", len(labels))
print("Features used:", len(features))
print("Features:", features)
print("Cluster sizes:")
print(pd.Series(labels).value_counts().sort_index().to_string())
print("Silhouette score:", round(silhouette_score(X, labels), 3))

# Compare against the previous clustering for the same image IDs.
old = original.set_index("image_id")["expression_cluster"]
comparison = pd.DataFrame({
    "image_id": ids.values,
    "new_cluster": labels,
})
comparison["old_cluster"] = comparison["image_id"].map(old)
matched = comparison.dropna(subset=["old_cluster"])

if len(matched) > 1:
    print(
        "Agreement with original clustering (adjusted Rand index):",
        round(
            adjusted_rand_score(
                matched["old_cluster"].astype(int),
                matched["new_cluster"].astype(int),
            ),
            3,
        ),
    )

comparison.to_csv(DATA / "expression_cluster_comparison.csv", index=False)
print("Saved: data/expression_cluster_comparison.csv")
