import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

FEATURES = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct"]

CANDIDATE_K = [3, 5, 7, 9, 11, 15]


def _loo_knn_predictions(X, y, k, weighting="inv_dist"):
    """
    Leave-One-Out weighted-KNN predictions.
    X: standardized feature matrix, y: real survey scores (mean_score).
    """
    n = len(y)
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)

    preds = np.zeros(n)
    for i in range(n):
        order = np.argsort(D[i])[:k]
        d = D[i, order]
        w = 1.0 / (1.0 + d) if weighting == "inv_dist" else np.ones_like(d)
        preds[i] = np.average(y[order], weights=w)

    return preds


def _mae_rmse(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def learn_calibration_weights(
    dataset_clean_path: str,
    out_dir: str,
    mlp_cv_predictions_path: str = None,
    feature_cols=None,
    candidate_k=None,
):
    """
    Learns, from real survey-labeled cases, how much to trust a
    case-based ("knowledge") score versus the MLP score.

    Method (fully data-driven, no hand-picked constants):
    1) For each candidate K, run Leave-One-Out weighted-KNN over the
       125 real survey cases and measure MAE/RMSE against the true
       survey mean_score. Pick the K with the lowest LOO-MAE.
    2) Compare that best LOO-MAE against the MLP's own cross-validated
       MAE (from mlp_predictions_cv.csv, produced by Stage 5 training).
    3) Fuse the two estimators using inverse-variance (precision)
       weighting, a standard meta-analysis technique:
           w_i ∝ 1 / MAE_i^2
       This gives more global weight to whichever estimator is
       empirically more accurate, instead of an arbitrary 50/50 split.
    """
    os.makedirs(out_dir, exist_ok=True)
    feature_cols = feature_cols or FEATURES
    candidate_k = candidate_k or CANDIDATE_K

    df = pd.read_excel(dataset_clean_path)
    features = [f for f in feature_cols if f in df.columns]
    df = df.dropna(subset=features + ["mean_score"]).reset_index(drop=True)

    if len(df) < max(candidate_k) + 1:
        candidate_k = [k for k in candidate_k if k < len(df)]

    X_raw = df[features].values.astype(float)
    y = df["mean_score"].values.astype(float)

    mean_ = X_raw.mean(axis=0)
    std_ = X_raw.std(axis=0)
    std_safe = np.where(std_ == 0, 1.0, std_)
    X = (X_raw - mean_) / std_safe

    sweep_rows = []
    best_k, best_mae, best_rmse = None, np.inf, np.inf

    for k in candidate_k:
        preds = _loo_knn_predictions(X, y, k)
        mae, rmse = _mae_rmse(y, preds)
        sweep_rows.append({"k": k, "loo_mae": mae, "loo_rmse": rmse})
        if mae < best_mae:
            best_k, best_mae, best_rmse = k, mae, rmse

    sweep_df = pd.DataFrame(sweep_rows)

    # ---- MLP's own cross-validated accuracy, as the comparison baseline ----
    mae_mlp, rmse_mlp = None, None
    if mlp_cv_predictions_path and os.path.isfile(mlp_cv_predictions_path):
        cv = pd.read_csv(mlp_cv_predictions_path)
        if {"y_true", "y_pred"}.issubset(cv.columns):
            mae_mlp, rmse_mlp = _mae_rmse(
                cv["y_true"].values.astype(float), cv["y_pred"].values.astype(float)
            )

    if mae_mlp is None:
        # No MLP CV file available: fall back to an even split so the
        # system still works, but flag it clearly.
        w_mlp, w_knowledge = 0.5, 0.5
        fusion_note = "MLP CV predictions file not found — used an even 50/50 fallback split."
    else:
        prec_mlp = 1.0 / (mae_mlp ** 2)
        prec_knn = 1.0 / (best_mae ** 2)
        total = prec_mlp + prec_knn
        w_mlp = prec_mlp / total
        w_knowledge = prec_knn / total
        fusion_note = (
            f"Inverse-variance (precision) weighting from cross-validated MAE: "
            f"MLP MAE={mae_mlp:.3f} vs Knowledge(K={best_k}) LOO-MAE={best_mae:.3f}."
        )

    weights = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_count": int(len(df)),
            "features": features,
            "method": "LOO weighted-KNN K-sweep + inverse-variance fusion with MLP CV accuracy",
        },
        "k_used": int(best_k),
        "k_sweep": sweep_df.to_dict(orient="records"),
        "mae_knowledge": float(best_mae),
        "rmse_knowledge": float(best_rmse),
        "mae_mlp": float(mae_mlp) if mae_mlp is not None else None,
        "rmse_mlp": float(rmse_mlp) if rmse_mlp is not None else None,
        "w_mlp": float(w_mlp),
        "w_knowledge": float(w_knowledge),
        "fusion_note": fusion_note,
        "scaler_mean": mean_.tolist(),
        "scaler_std": std_safe.tolist(),
    }

    json_path = os.path.join(out_dir, "calibration_weights.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

    sweep_path = os.path.join(out_dir, "calibration_k_sweep.xlsx")
    sweep_df.to_excel(sweep_path, index=False)

    return weights, sweep_df, json_path, sweep_path
