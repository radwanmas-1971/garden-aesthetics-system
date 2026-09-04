import numpy as np

FEATURES = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct", "water_pct"]

AR_FEATURES = {
    "grass_pct": "العشب",
    "trees_pct": "الأشجار",
    "flowers_pct": "الزهور",
    "ground_pct": "الأرض المكشوفة",
    "water_pct": "الماء",
}


def to_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def feature_vector(row, cols):
    return np.array([to_float(row.get(c, 0.0)) for c in cols], dtype=float)


def get_high_profile(repo):
    for r in repo.get("beauty_range_summary", []):
        level = str(r.get("beauty_level", ""))
        if "4-5" in level or "عالي" in level:
            return r
    return {}


def get_human_weights(repo):
    rows = repo.get("human_knowledge_weights", [])
    weights = {}

    for r in rows:
        f = r.get("feature")
        if not f:
            continue

        weights[f] = {
            "human_weight": to_float(r.get("human_weight", 0.0)),
            "preferred_min": to_float(r.get("preferred_min_high_beauty", None), None),
            "preferred_avg": to_float(r.get("preferred_avg_high_beauty", None), None),
            "preferred_max": to_float(r.get("preferred_max_high_beauty", None), None),
            "correlation": to_float(r.get("correlation_with_human_score", 0.0)),
        }

    return weights


def find_similar_cases(feats, repo, feature_cols, top_k=3):
    cases = repo.get("case_memory", [])
    if not cases:
        return []

    cols = [c for c in feature_cols if c in feats and c != "water_pct"]
    if not cols:
        return []

    x = feature_vector(feats, cols)
    rows = []

    for case in cases:
        y = feature_vector(case, cols)
        dist = float(np.linalg.norm(x - y))
        sim = 1.0 / (1.0 + dist)

        rows.append({
            "case_id": case.get("case_id", ""),
            "image_name": case.get("image_name", ""),
            "mean_score": case.get("mean_score", None),
            "beauty_level": case.get("beauty_level", ""),
            "similarity": sim,
            "case_story_ar": case.get("case_story_ar", ""),
        })

    return sorted(rows, key=lambda r: r["similarity"], reverse=True)[:top_k]


def _condition_satisfied(rule, value_lookup):
    """
    يتحقق فعليًا من أن قيم الصورة الحالية (value_lookup) تُحقّق شرط
    القاعدة (نص condition + threshold المخزَّنين)، بدل افتراض أن أي
    قاعدة "قوية إحصائيًا" عن هذه الميزة تنطبق تلقائيًا على هذه الصورة.
    يدعم كلًا من قواعد الميزة المفردة وقواعد التفاعل الثنائي (AND).
    """
    condition = str(rule.get("condition", ""))
    threshold_raw = rule.get("threshold", None)

    parts = [p.strip() for p in condition.split("AND")]

    if isinstance(threshold_raw, str) and "," in threshold_raw:
        try:
            thresholds = [float(t.strip()) for t in threshold_raw.split(",")]
        except ValueError:
            return False
    else:
        t = to_float(threshold_raw, None)
        if t is None:
            return False
        thresholds = [t] * len(parts)

    if len(thresholds) != len(parts):
        return False

    for part, thr in zip(parts, thresholds):
        if "<=" in part:
            feat = part.split("<=")[0].strip()
            val = value_lookup.get(feat)
            if val is None or not (val <= thr):
                return False
        elif ">=" in part:
            feat = part.split(">=")[0].strip()
            val = value_lookup.get(feat)
            if val is None or not (val >= thr):
                return False
        else:
            return False

    return True


def best_rule_for_feature(repo, feature, value_lookup=None):
    value_lookup = value_lookup or {}

    candidates = [
        r for r in repo.get("knowledge_rules", [])
        if r.get("feature_1") == feature
        # نقتصر على قواعد الميزة المفردة فقط عند تفسير ميزة واحدة بمفردها
        and r.get("rule_type", "single_feature") == "single_feature"
        # الشرط الحاسم: يجب أن تُحقّق قيمة هذه الصورة تحديدًا شرط القاعدة
        and _condition_satisfied(r, value_lookup)
    ]
    if not candidates:
        return None

    def strength(r):
        support = to_float(r.get("support_count", 0))
        high = to_float(r.get("high_beauty_ratio_score_ge_4", 0))
        low = to_float(r.get("low_beauty_ratio_score_lt_3", 0))
        return support * max(high, low)

    best = sorted(candidates, key=strength, reverse=True)[0]

    return {
        "feature": feature,
        "feature_ar": best.get("feature_1_ar", AR_FEATURES.get(feature, feature)),
        "condition": best.get("condition", ""),
        "support_count": int(to_float(best.get("support_count", 0))),
        "support_percent": to_float(best.get("support_percent", 0)),
        "high_ratio": to_float(best.get("high_beauty_ratio_score_ge_4", 0)),
        "low_ratio": to_float(best.get("low_beauty_ratio_score_lt_3", 0)),
        "conclusion_ar": best.get("conclusion_ar", ""),
    }


def collect_evidence(feats, repo, feature_cols, similar_cases):
    high_profile = get_high_profile(repo)
    human_weights = get_human_weights(repo)

    # fallback weights if stage 3 weights were not built
    usable_features = [f for f in feature_cols if f in feats]
    if not human_weights:
        base_w = 1.0 / max(len(usable_features), 1)
        human_weights = {f: {"human_weight": base_w} for f in usable_features}

    collected = []

    # قاموس القيم الفعلية لهذه الصورة تحديدًا، لاستخدامه في التحقق من
    # انطباق شرط أي قاعدة معرفية قبل اعتمادها كدليل.
    value_lookup = {f: to_float(feats.get(f, 0.0)) for f in usable_features}

    for f in usable_features:
        value = to_float(feats.get(f, 0.0))
        ar = AR_FEATURES.get(f, f)

        hw = human_weights.get(f, {})
        preferred_min = hw.get("preferred_min")
        preferred_avg = hw.get("preferred_avg")
        preferred_max = hw.get("preferred_max")

        if preferred_avg is None:
            preferred_min = to_float(high_profile.get(f"{f}_min", None), None)
            preferred_avg = to_float(high_profile.get(f"{f}_avg", None), None)
            preferred_max = to_float(high_profile.get(f"{f}_max", None), None)

        rule = best_rule_for_feature(repo, f, value_lookup)

        low_cases = 0
        usable_cases = 0
        for c in similar_cases:
            score = c.get("mean_score", None)
            if score is None:
                continue
            usable_cases += 1
            if float(score) < 3:
                low_cases += 1

        similar_low_ratio = low_cases / usable_cases if usable_cases else 0.0

        collected.append({
            "feature": f,
            "feature_ar": ar,
            "current_value": value,
            "preferred_min": preferred_min,
            "preferred_avg": preferred_avg,
            "preferred_max": preferred_max,
            "human_weight": to_float(hw.get("human_weight", 0.0)),
            "correlation": to_float(hw.get("correlation", 0.0)),
            "rule": rule,
            "similar_low_ratio": similar_low_ratio,
        })

    return collected