# core/trainer.py
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

def safe_corr(a, b):
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def train_mlp_kfold(
    dataset_xlsx: str,
    out_dir: str,
    use_water_feature: bool = False,          # ✅
    include_context_features: bool = False,   # اختياري للتجربة فقط
    hidden=(32, 16),
    random_state=42
):

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(dataset_xlsx)

    base_features = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct"]

    if use_water_feature and "water_pct" in df.columns:
        base_features = base_features + ["water_pct"]

    features = base_features


    # ✅ ميزات سياق (للتجربة فقط - غير مفضلة للدكتوراه)
    if include_context_features:
        for c in ["sky_pct", "building_pct", "wall_pct"]:
            if c in df.columns:
                features.append(c)

    target = "mean_score"

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df2 = df.dropna(subset=features + [target]).copy()
    X = df2[features].values.astype(np.float32)
    y = df2[target].values.astype(np.float32)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=4000,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=40,
            validation_fraction=0.15
        ))
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    fold_rows = []
    all_true, all_pred, all_name = [], [], []
    importance_rows = []  # collected per-fold, out-of-sample

    for fold, (tr, te) in enumerate(kf.split(X), 1):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        pred = np.clip(pred, 1.0, 5.0)

        mae = mean_absolute_error(y[te], pred)
        rmse = np.sqrt(mean_squared_error(y[te], pred))
        corr = safe_corr(y[te], pred)

        fold_rows.append({"fold": fold, "mae": mae, "rmse": rmse, "corr": corr})

        all_true.extend(list(y[te]))
        all_pred.extend(list(pred))
        if "image_name" in df2.columns:
            all_name.extend(list(df2.iloc[te]["image_name"].values))
        else:
            all_name.extend([""] * len(te))

        # Permutation importance computed OUT-OF-SAMPLE: using this fold's
        # held-out test set with the model fitted only on the training
        # split, so the importance reflects generalization — not what the
        # model memorized from the data it was trained on.
        try:
            r = permutation_importance(
                model, X[te], y[te],
                n_repeats=30,
                random_state=random_state,
                scoring="neg_mean_absolute_error"
            )
            for feat, imp_mean in zip(features, r.importances_mean):
                importance_rows.append({"fold": fold, "feature": feat, "importance": imp_mean})
        except Exception:
            pass

    folds_df = pd.DataFrame(fold_rows)
    summary = {
        "mae_mean": float(folds_df["mae"].mean()),
        "mae_std": float(folds_df["mae"].std()),
        "rmse_mean": float(folds_df["rmse"].mean()),
        "rmse_std": float(folds_df["rmse"].std()),
        "corr_mean": float(folds_df["corr"].mean()),
        "corr_std": float(folds_df["corr"].std()),
    }

    features_path = os.path.join(out_dir, "training_features.txt")
    with open(features_path, "w", encoding="utf-8") as f:
        for c in features:
            f.write(c + "\n")

    # Train final model on all data (for deployment / inference use)
    model.fit(X, y)
    model_path = os.path.join(out_dir, "mlp_model.joblib")
    joblib.dump(model, model_path)

    pred_df = pd.DataFrame({
        "image_name": all_name,
        "y_true": all_true,
        "y_pred": all_pred
    })
    pred_path = os.path.join(out_dir, "mlp_predictions_cv.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # Cross-validated (out-of-sample) permutation importance: average and
    # std of each feature's importance across the 5 held-out test folds.
    imp_path = None
    if importance_rows:
        imp_all = pd.DataFrame(importance_rows)
        imp = (
            imp_all.groupby("feature")["importance"]
            .agg(importance_mean="mean", importance_std="std")
            .reset_index()
            .sort_values("importance_mean", ascending=False)
        )
        imp_path = os.path.join(out_dir, "mlp_feature_importance_permutation.csv")
        imp.to_csv(imp_path, index=False, encoding="utf-8-sig")

    return {
        "features": features,
        "folds": folds_df,
        "summary": summary,
        "model_path": model_path,
        "pred_path": pred_path,
        "imp_path": imp_path,
        "features_path": features_path

    }