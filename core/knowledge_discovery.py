import os
import json
import itertools
import numpy as np
import pandas as pd

FEATURES = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct", "water_pct"]

AR_FEATURES = {
    "grass_pct": "العشب",
    "trees_pct": "الأشجار",
    "flowers_pct": "الزهور",
    "ground_pct": "الأرض المكشوفة",
    "water_pct": "الماء",
}

RANGES = [
    (1.0, 2.0, "1-2 منخفض جدًا"),
    (2.0, 3.0, "2-3 منخفض"),
    (3.0, 4.0, "3-4 متوسط"),
    (4.0, 5.01, "4-5 عالي"),
]


def assign_beauty_level(score):
    for lo, hi, label in RANGES:
        if lo <= float(score) < hi:
            return label
    return "غير محدد"


def safe_corr(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = (~x.isna()) & (~y.isna())
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def build_knowledge_discovery(dataset_clean_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(dataset_clean_path)

    if "image_name" not in df.columns:
        raise ValueError("dataset_clean.xlsx must contain image_name")
    if "mean_score" not in df.columns:
        raise ValueError("dataset_clean.xlsx must contain mean_score")

    features = [c for c in FEATURES if c in df.columns]
    df = df.dropna(subset=features + ["mean_score"]).copy()
    df["beauty_level"] = df["mean_score"].apply(assign_beauty_level)

    # =========================
    # 1) Feature trends by level
    # =========================
    trend_rows = []
    for _, _, level in RANGES:
        sub = df[df["beauty_level"] == level]
        row = {
            "beauty_level": level,
            "image_count": len(sub),
            "mean_score_avg": sub["mean_score"].mean() if len(sub) else np.nan,
        }
        for f in features:
            row[f"{f}_avg"] = sub[f].mean() if len(sub) else np.nan
            row[f"{f}_min"] = sub[f].min() if len(sub) else np.nan
            row[f"{f}_max"] = sub[f].max() if len(sub) else np.nan
        trend_rows.append(row)

    feature_trends_df = pd.DataFrame(trend_rows)

    # =========================
    # 2) Correlation with human ratings
    # =========================
    corr_rows = []
    for f in features:
        corr = safe_corr(df[f], df["mean_score"])

        direction = "موجب" if corr > 0 else "سالب" if corr < 0 else "ضعيف/غير واضح"
        strength = (
            "قوي" if abs(corr) >= 0.60 else
            "متوسط" if abs(corr) >= 0.35 else
            "ضعيف"
        )

        corr_rows.append({
            "feature": f,
            "feature_ar": AR_FEATURES.get(f, f),
            "correlation_with_mean_score": corr,
            "direction_ar": direction,
            "strength_ar": strength,
        })

    feature_correlations_df = pd.DataFrame(corr_rows).sort_values(
        "correlation_with_mean_score",
        key=lambda s: s.abs(),
        ascending=False
    )

    # =========================
    # 3) High vs low pattern differences
    # =========================
    high = df[df["mean_score"] >= 4.0]
    low = df[df["mean_score"] < 3.0]

    pattern_rows = []
    for f in features:
        high_avg = high[f].mean() if len(high) else np.nan
        low_avg = low[f].mean() if len(low) else np.nan
        diff = high_avg - low_avg if not pd.isna(high_avg) and not pd.isna(low_avg) else np.nan

        if f == "ground_pct":
            interpretation = "ارتفاعها غالبًا يرتبط بانخفاض الجمال" if diff < 0 else "العلاقة غير واضحة"
        else:
            interpretation = "ارتفاعها غالبًا يرتبط بارتفاع الجمال" if diff > 0 else "العلاقة غير واضحة"

        pattern_rows.append({
            "feature": f,
            "feature_ar": AR_FEATURES.get(f, f),
            "low_beauty_avg_score_lt_3": low_avg,
            "high_beauty_avg_score_ge_4": high_avg,
            "difference_high_minus_low": diff,
            "interpretation_ar": interpretation,
        })

    beauty_patterns_df = pd.DataFrame(pattern_rows)

    # =========================
    # 4) Rule discovery - single feature rules
    # =========================
    rule_rows = []
    min_support = max(2, int(len(df) * 0.10))

    for f in features:
        values = df[f].dropna()
        if len(values) < 4:
            continue

        q25 = float(values.quantile(0.25))
        q50 = float(values.quantile(0.50))
        q75 = float(values.quantile(0.75))

        tests = [
            ("low", f"{f} <= Q25", df[f] <= q25, q25),
            ("high", f"{f} >= Q75", df[f] >= q75, q75),
        ]

        for rule_type, cond_text, mask, threshold in tests:
            sub = df[mask]
            if len(sub) < min_support:
                continue

            support_percent = len(sub) / len(df) * 100

            # فحص عام: إذا كانت نسبة الدعم قريبة جدًا من 100% (أو من الصفر)،
            # فهذا يعني أن الشرط لا يُميّز فعليًا بين مجموعتين — غالبًا بسبب
            # توزيع منحاز بشدة (قيم متطابقة عند حد الربيع)، كما يحدث مثلاً
            # حين تكون معظم قيم الميزة صفرًا (نادرة الحدوث) أو ثابتة (معطّلة
            # في هذا التشغيل). في هذه الحالة تكون "القاعدة" مجرد متوسط
            # العينة العام معاد تسميته خطأً كنتيجة عن هذه الميزة، فنتجاهلها.
            if support_percent >= 95.0 or support_percent <= 0.0:
                continue

            avg_score = sub["mean_score"].mean()
            high_ratio = (sub["mean_score"] >= 4.0).mean()
            low_ratio = (sub["mean_score"] < 3.0).mean()

            if high_ratio >= 0.50:
                conclusion = "يرتبط غالبًا بجمال عالٍ"
            elif low_ratio >= 0.50:
                conclusion = "يرتبط غالبًا بجمال منخفض"
            else:
                conclusion = "لا يعطي حكمًا واضحًا وحده"

            rule_rows.append({
                "rule_type": "single_feature",
                "feature_1": f,
                "feature_1_ar": AR_FEATURES.get(f, f),
                "feature_2": "",
                "condition": cond_text,
                "threshold": threshold,
                "support_count": len(sub),
                "support_percent": len(sub) / len(df) * 100,
                "avg_mean_score": avg_score,
                "high_beauty_ratio_score_ge_4": high_ratio,
                "low_beauty_ratio_score_lt_3": low_ratio,
                "conclusion_ar": conclusion,
            })

    # =========================
    # 5) Pair interaction rules
    # =========================
    for f1, f2 in itertools.combinations(features, 2):
        q1 = float(df[f1].quantile(0.75))
        q2 = float(df[f2].quantile(0.75))

        mask = (df[f1] >= q1) & (df[f2] >= q2)
        sub = df[mask]

        if len(sub) >= min_support:
            support_percent = len(sub) / len(df) * 100

            # نفس الفحص العام المطبَّق على القواعد المفردة: نتجاهل أي
            # تفاعل ثنائي يكون دعمه قريبًا جدًا من كامل العينة، لأنه في
            # هذه الحالة لا يعكس فعليًا "تقاء" ميزتين نادرتين/مرتفعتين،
            # بل مجرد صدفة أن كل الصور (أو أغلبها) تحقق الشرطين معًا.
            if support_percent >= 95.0:
                continue

            # === فحص التكرار الإحصائي (جديد) ===
            # إذا كان عدد الصور المحققة للشرطين معًا (f1 AND f2) يساوي
            # تمامًا عدد الصور المحققة لأحد الشرطين وحده، فهذا يعني أن
            # الشرط الآخر لم يُقيّد العيّنة إطلاقًا (غالبًا لأن Q75 لتلك
            # الميزة = 0 بسبب سيطرة القيم الصفرية على أغلب البيانات، كما
            # في flowers_pct أو water_pct). في هذه الحالة تكون القاعدة
            # الزوجية نسخة مكرّرة رياضيًا من قاعدة السمة المفردة، فنتجاهلها
            # بدل تسجيلها كتفاعل ذي معنى.
            single_f1_high_count = int((df[f1] >= q1).sum())
            single_f2_high_count = int((df[f2] >= q2).sum())
            if len(sub) == single_f1_high_count or len(sub) == single_f2_high_count:
                continue

            avg_score = sub["mean_score"].mean()
            high_ratio = (sub["mean_score"] >= 4.0).mean()
            low_ratio = (sub["mean_score"] < 3.0).mean()

            if high_ratio >= 0.50:
                conclusion = "اجتماعهما يرتبط غالبًا بجمال عالٍ"
            elif low_ratio >= 0.50:
                conclusion = "اجتماعهما يرتبط غالبًا بجمال منخفض"
            else:
                conclusion = "اجتماعهما لا يعطي حكمًا واضحًا"

            rule_rows.append({
                "rule_type": "pair_interaction",
                "feature_1": f1,
                "feature_1_ar": AR_FEATURES.get(f1, f1),
                "feature_2": f2,
                "feature_2_ar": AR_FEATURES.get(f2, f2),
                "condition": f"{f1} >= Q75 AND {f2} >= Q75",
                "threshold": f"{q1:.2f}, {q2:.2f}",
                "support_count": len(sub),
                "support_percent": len(sub) / len(df) * 100,
                "avg_mean_score": avg_score,
                "high_beauty_ratio_score_ge_4": high_ratio,
                "low_beauty_ratio_score_lt_3": low_ratio,
                "conclusion_ar": conclusion,
            })

    knowledge_rules_df = pd.DataFrame(rule_rows)

    if len(knowledge_rules_df):
        knowledge_rules_df = knowledge_rules_df.sort_values(
            by=["high_beauty_ratio_score_ge_4", "low_beauty_ratio_score_lt_3", "support_count"],
            ascending=[False, False, False]
        )

    # =========================
    # 6) Contradictory / exceptional cases
    # Similar features but different human scores
    # =========================
    exception_rows = []
    X = df[features].values.astype(float)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            feature_dist = float(np.linalg.norm(X[i] - X[j]))
            score_diff = abs(float(df.iloc[i]["mean_score"]) - float(df.iloc[j]["mean_score"]))

            if score_diff >= 1.0:
                exception_rows.append({
                    "image_1": df.iloc[i]["image_name"],
                    "score_1": df.iloc[i]["mean_score"],
                    "image_2": df.iloc[j]["image_name"],
                    "score_2": df.iloc[j]["mean_score"],
                    "feature_distance": feature_dist,
                    "score_difference": score_diff,
                    "note_ar": "حالتان قد تكونان متقاربتين في بعض السمات لكن تقييم البشر مختلف؛ تحتاج مراجعة بصرية."
                })

    exceptions_df = pd.DataFrame(exception_rows)
    if len(exceptions_df):
        exceptions_df = exceptions_df.sort_values(
            by=["feature_distance", "score_difference"],
            ascending=[True, False]
        ).head(30)

    # =========================
    # Save
    # =========================
    out_trends = os.path.join(out_dir, "feature_trends.xlsx")
    out_corr = os.path.join(out_dir, "feature_correlations.xlsx")
    out_patterns = os.path.join(out_dir, "beauty_level_patterns.xlsx")
    out_rules = os.path.join(out_dir, "knowledge_rules.xlsx")
    out_exceptions = os.path.join(out_dir, "exceptional_cases.xlsx")
    out_json = os.path.join(out_dir, "knowledge_discovery.json")

    feature_trends_df.to_excel(out_trends, index=False)
    feature_correlations_df.to_excel(out_corr, index=False)
    beauty_patterns_df.to_excel(out_patterns, index=False)
    knowledge_rules_df.to_excel(out_rules, index=False)
    exceptions_df.to_excel(out_exceptions, index=False)

    data = {
        "features": features,
        "feature_trends": feature_trends_df.to_dict(orient="records"),
        "feature_correlations": feature_correlations_df.to_dict(orient="records"),
        "beauty_level_patterns": beauty_patterns_df.to_dict(orient="records"),
        "knowledge_rules": knowledge_rules_df.to_dict(orient="records"),
        "exceptional_cases": exceptions_df.to_dict(orient="records"),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "feature_trends": feature_trends_df,
        "feature_correlations": feature_correlations_df,
        "beauty_level_patterns": beauty_patterns_df,
        "knowledge_rules": knowledge_rules_df,
        "exceptional_cases": exceptions_df,
        "paths": {
            "feature_trends": out_trends,
            "feature_correlations": out_corr,
            "beauty_level_patterns": out_patterns,
            "knowledge_rules": out_rules,
            "exceptional_cases": out_exceptions,
            "json": out_json,
        }
    }