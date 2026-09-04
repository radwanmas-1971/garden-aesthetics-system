import os
import json

import numpy as np

from core.reasoning.decision_analyzer import analyze_decision

FEATURES = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct"]

# Fallback used only if calibration_weights.json was never built (Stage 6
# not run yet). These specific numbers come from the LOO analysis performed
# on the 125-case survey dataset (K=7, LOO-MAE=1.083 vs MLP 5-fold MAE=1.252).
_FALLBACK = {
    "k_used": 7,
    "w_mlp": 0.428,
    "w_knowledge": 0.572,
    "mae_mlp": 1.252,
    "mae_knowledge": 1.083,
    "scaler_mean": None,
    "scaler_std": None,
}


def to_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_calibration_weights(weights_path):
    if weights_path and os.path.isfile(weights_path):
        with open(weights_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(_FALLBACK)


def _case_matrix(repo, feature_cols):
    cases = repo.get("case_memory", [])
    rows, names, ids, scores = [], [], [], []

    for c in cases:
        score = c.get("mean_score", None)
        if score is None:
            continue
        vec = [to_float(c.get(f, 0.0)) for f in feature_cols]
        rows.append(vec)
        names.append(c.get("image_name", ""))
        ids.append(c.get("case_id", ""))
        scores.append(to_float(score))

    if not rows:
        return None, None, None, None

    return np.array(rows, dtype=float), np.array(scores, dtype=float), names, ids


def knowledge_score_from_cases(feats, repo, feature_cols, k, scaler_mean=None, scaler_std=None):
    """
    Finds the K nearest real survey-labeled cases to the new image and
    returns a similarity-weighted average of their real (human) scores,
    plus the neighbor list and agreement diagnostics.
    """
    X, y, names, ids = _case_matrix(repo, feature_cols)
    if X is None:
        return None

    x = np.array([to_float(feats.get(f, 0.0)) for f in feature_cols], dtype=float)

    if scaler_mean is not None and scaler_std is not None:
        mean_ = np.array(scaler_mean, dtype=float)
        std_ = np.array(scaler_std, dtype=float)
        std_ = np.where(std_ == 0, 1.0, std_)
        Xs = (X - mean_) / std_
        xs = (x - mean_) / std_
    else:
        Xs, xs = X, x

    dist = np.linalg.norm(Xs - xs, axis=1)
    k = min(k, len(dist))
    order = np.argsort(dist)[:k]

    d = dist[order]
    sim = 1.0 / (1.0 + d)
    neighbor_scores = y[order]

    knowledge_score = float(np.average(neighbor_scores, weights=sim))
    avg_similarity = float(np.mean(sim))

    score_spread = max(1e-6, (5.0 - 1.0) / 2.0)
    consistency = float(np.clip(1.0 - (neighbor_scores.std() / score_spread), 0.0, 1.0))

    neighbors = [
        {
            "case_id": ids[j],
            "image_name": names[j],
            "mean_score": round(float(y[j]), 2),
            "similarity": round(float(sim[i]), 4),
        }
        for i, j in enumerate(order)
    ]

    return {
        "knowledge_score": knowledge_score,
        "avg_similarity": avg_similarity,
        "consistency": consistency,
        "neighbors": neighbors,
        "k_used": k,
    }


def calibrate_score(mlp_score, feats, repo, feature_cols=None, weights_path=None):
    """
    Combines the raw MLP score with a survey-grounded case-based score.

    Fusion weight on the knowledge estimate is the globally learned
    w_knowledge (from calibration_weights.json, precision-weighted against
    the MLP's own CV accuracy), further discounted per-image by how much
    the nearest real cases actually agree with each other:

        w_knowledge_final = w_knowledge_global * consistency
        w_mlp_final       = 1 - w_knowledge_final

    So when nearby real survey cases strongly disagree on the score, the
    system automatically leans back on the MLP instead of an unreliable
    local neighborhood.
    """
    feature_cols = [f for f in (feature_cols or FEATURES) if f != "water_pct"]
    weights = load_calibration_weights(weights_path)

    result = knowledge_score_from_cases(
        feats=feats,
        repo=repo,
        feature_cols=feature_cols,
        k=int(weights.get("k_used", 7)),
        scaler_mean=weights.get("scaler_mean"),
        scaler_std=weights.get("scaler_std"),
    )

    mlp_score = float(np.clip(mlp_score, 1.0, 5.0))

    if result is None:
        # No case memory available yet — cannot ground in the survey.
        decision = analyze_decision(mlp_score)
        return {
            "mlp_score": mlp_score,
            "knowledge_score": None,
            "calibrated_score": mlp_score,
            "beauty_level": decision["beauty_level"],
            "k_used": int(weights.get("k_used", 7)),
            "avg_similarity": 0.0,
            "consistency": 0.0,
            "w_knowledge": 0.0,
            "w_mlp": 1.0,
            "neighbors": [],
            "note_ar": "لا توجد حالات في قاعدة الخبرة بعد؛ تم استخدام درجة MLP الخام بدون معايرة.",
        }

    w_knowledge_global = to_float(weights.get("w_knowledge", 0.5))
    w_knowledge_final = w_knowledge_global * result["consistency"]
    w_mlp_final = 1.0 - w_knowledge_final

    calibrated = w_mlp_final * mlp_score + w_knowledge_final * result["knowledge_score"]
    calibrated = float(np.clip(calibrated, 1.0, 5.0))

    decision = analyze_decision(calibrated)

    note_ar = (
        f"اعتمدت المعايرة على أقرب {result['k_used']} حالات حقيقية من الاستبيان "
        f"(تشابه متوسط {result['avg_similarity']*100:.1f}٪، اتساق بينها {result['consistency']*100:.1f}٪)، "
        f"فحصلت درجة المعرفة على وزن {w_knowledge_final*100:.1f}٪ مقابل {w_mlp_final*100:.1f}٪ لدرجة MLP."
    )

    return {
        "mlp_score": mlp_score,
        "knowledge_score": round(result["knowledge_score"], 3),
        "calibrated_score": round(calibrated, 3),
        "beauty_level": decision["beauty_level"],
        "k_used": result["k_used"],
        "avg_similarity": result["avg_similarity"],
        "consistency": result["consistency"],
        "w_knowledge": w_knowledge_final,
        "w_mlp": w_mlp_final,
        "neighbors": result["neighbors"],
        "note_ar": note_ar,
    }
