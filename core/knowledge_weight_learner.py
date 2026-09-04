import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

FEATURES = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct", "water_pct"]

AR_FEATURES = {
    "grass_pct": "العشب",
    "trees_pct": "الأشجار",
    "flowers_pct": "الزهور",
    "ground_pct": "الأرض المكشوفة",
    "water_pct": "الماء",
}

def safe_corr(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = (~x.isna()) & (~y.isna())
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])

def _out_of_sample_importance(X, y, features, random_state, n_splits=5):
    """
    Cross-validated (out-of-sample) permutation importance: fit the model
    on each fold's training split, then permute on that fold's HELD-OUT
    test split. Averaging across folds gives an honest estimate of which
    features actually help the model generalize — computing importance
    on the same data the model was fit on would be optimistically biased.
    """
    n_splits = min(n_splits, len(y)) if len(y) >= 2 else 1
    if n_splits < 2:
        model = RandomForestRegressor(n_estimators=300, random_state=random_state, min_samples_leaf=2)
        model.fit(X, y)
        perm = permutation_importance(model, X, y, n_repeats=30, random_state=random_state,
                                       scoring="neg_mean_absolute_error")
        return perm.importances_mean

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    per_fold = []

    for tr, te in kf.split(X):
        model = RandomForestRegressor(n_estimators=300, random_state=random_state, min_samples_leaf=2)
        model.fit(X[tr], y[tr])
        perm = permutation_importance(
            model, X[te], y[te],
            n_repeats=30,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        per_fold.append(perm.importances_mean)

    return np.mean(per_fold, axis=0)

def learn_human_knowledge_weights(dataset_clean_path, out_dir, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(dataset_clean_path)

    if "mean_score" not in df.columns:
        raise ValueError("dataset_clean.xlsx must contain mean_score")

    features = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=features + ["mean_score"]).copy()

    X = df[features].values.astype(float)
    y = df["mean_score"].values.astype(float)

    # Final model, fit on all data, used only for the returned feature
    # rankings' underlying model quality — NOT for computing importance.
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        min_samples_leaf=2
    )
    model.fit(X, y)

    importances_mean = _out_of_sample_importance(X, y, features, random_state)

    raw_importance = np.maximum(importances_mean, 0)
    total = raw_importance.sum()
    weights = raw_importance / total if total > 0 else np.ones(len(features)) / len(features)

    rows = []
    high = df[df["mean_score"] >= 4.0]

    for i, f in enumerate(features):
        corr = safe_corr(df[f], df["mean_score"])

        if len(high):
            preferred_min = float(high[f].min())
            preferred_avg = float(high[f].mean())
            preferred_max = float(high[f].max())
        else:
            preferred_min = preferred_avg = preferred_max = np.nan

        rows.append({
            "feature": f,
            "feature_ar": AR_FEATURES.get(f, f),
            "human_weight": float(weights[i]),
            "importance_raw": float(raw_importance[i]),
            "correlation_with_human_score": corr,
            "preferred_min_high_beauty": preferred_min,
            "preferred_avg_high_beauty": preferred_avg,
            "preferred_max_high_beauty": preferred_max,
            "sample_count": len(df),
            "high_beauty_count": len(high),
        })

    weights_df = pd.DataFrame(rows).sort_values("human_weight", ascending=False)

    xlsx_path = os.path.join(out_dir, "human_knowledge_weights.xlsx")
    json_path = os.path.join(out_dir, "human_knowledge_weights.json")

    weights_df.to_excel(xlsx_path, index=False)

    data = {
        "metadata": {
            "source": "dataset_clean.xlsx",
            "sample_count": int(len(df)),
            "high_beauty_count": int(len(high)),
            "method": "RandomForest permutation importance + correlation + high-beauty preferred ranges",
        },
        "weights": weights_df.to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return weights_df, xlsx_path, json_path