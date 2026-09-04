import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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


def confidence_from_std(std):
    if pd.isna(std):
        return "غير محدد"
    std = float(std)
    if std <= 0.50:
        return "ثقة عالية: اتفاق واضح بين المقيمين"
    elif std <= 0.90:
        return "ثقة متوسطة: يوجد اختلاف بسيط بين المقيمين"
    else:
        return "ثقة منخفضة: اختلاف واضح بين المقيمين"


def describe_case_strengths(row, high_profile, features, zero_variance_features=None):
    strengths = []
    weaknesses = []
    zero_variance_features = zero_variance_features or set()

    for f in features:
        # نتجاهل أي ميزة ثابتة القيمة عبر كامل مجموعة البيانات (تباين = 0)،
        # لأن أي مقارنة "أعلى/أقل من النمط العالي" ستكون تساويًا دائمًا (0
        # مقابل 0 مثلاً)، فتوليد جملة "ارتفاع/انخفاض" في هذه الحالة مضلِّل
        # وفارغ المعنى تمامًا — لا يوجد فرق حقيقي ليُقاس أصلًا.
        if f in zero_variance_features:
            continue

        val = float(row.get(f, 0.0))
        avg_high = high_profile.get(f"{f}_avg", np.nan)

        if pd.isna(avg_high):
            continue

        ar = AR_FEATURES.get(f, f)

        if f == "ground_pct":
            if val <= avg_high:
                strengths.append(f"انخفاض {ar} مقارنة بنمط الجمال العالي")
            else:
                weaknesses.append(f"ارتفاع {ar} مقارنة بنمط الجمال العالي")
        else:
            if val >= avg_high:
                strengths.append(f"ارتفاع نسبة {ar} مقارنة بنمط الجمال العالي")
            else:
                weaknesses.append(f"انخفاض نسبة {ar} مقارنة بنمط الجمال العالي")

    return strengths, weaknesses


def make_case_story(row, strengths, weaknesses):
    case_id = row["case_id"]
    image_name = row["image_name"]
    score = float(row["mean_score"])
    level = row["beauty_level"]
    confidence = row.get("confidence_level", "")

    s_text = "، ".join(strengths[:3]) if strengths else "لا توجد نقاط قوة واضحة مقارنة بالنمط العالي."
    w_text = "، ".join(weaknesses[:3]) if weaknesses else "لا توجد فجوات واضحة مقارنة بالنمط العالي."

    base = (
        f"الحالة {case_id} للصورة {image_name} حصلت على متوسط جمال {score:.2f} "
        f"وتقع ضمن مستوى {level}. "
        f"أبرز نقاط القوة: {s_text}. أبرز الفجوات: {w_text}."
    )

    # نُضيف جملة مستوى الثقة فقط عند توفر بيانات فعلية عن تشتّت آراء
    # المقيّمين (std_score)؛ عند غيابها لا نكرر جملة غامضة في كل قصة.
    if confidence and not confidence.startswith("غير متاح"):
        base = (
            f"الحالة {case_id} للصورة {image_name} حصلت على متوسط جمال {score:.2f} "
            f"وتقع ضمن مستوى {level}. مستوى الثقة في تقييم البشر: {confidence}. "
            f"أبرز نقاط القوة: {s_text}. أبرز الفجوات: {w_text}."
        )

    return base


def build_case_memory(dataset_clean_path, out_dir, top_k=5):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_excel(dataset_clean_path)

    if "image_name" not in df.columns:
        raise ValueError("dataset_clean.xlsx must contain image_name")
    if "mean_score" not in df.columns:
        raise ValueError("dataset_clean.xlsx must contain mean_score")

    features = [c for c in FEATURES if c in df.columns]
    df = df.dropna(subset=features + ["mean_score"]).copy()

    df["case_id"] = [f"CASE_{i+1:03d}" for i in range(len(df))]
    df["beauty_level"] = df["mean_score"].apply(assign_beauty_level)

    if "std_score" in df.columns:
        df["confidence_level"] = df["std_score"].apply(confidence_from_std)
    else:
        # لا تتوفر درجات المقيّمين كلٌّ على حدة (فقط المتوسط النهائي)،
        # لذا لا يمكن حساب تشتّت الآراء إطلاقًا. نُصرّح بذلك بوضوح
        # بدل استخدام قيمة غامضة موحية بأن الحساب "لم يتم بعد".
        df["confidence_level"] = "غير متاح (لا تتوفر درجات المقيّمين منفردة، فقط المتوسط النهائي)"

    # =========================
    # Normalize features
    # =========================
    X = df[features].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # =========================
    # Similarity matrix
    # =========================
    sim = cosine_similarity(Xs)
    sim_df = pd.DataFrame(sim, index=df["image_name"], columns=df["image_name"])

    # =========================
    # Nearest similar cases
    # =========================
    similar_cases = []
    similarity_percent = []

    for i in range(len(df)):
        order = np.argsort(sim[i])[::-1]
        order = [j for j in order if j != i][:top_k]

        names = df.iloc[order]["image_name"].astype(str).tolist()
        vals = [f"{sim[i, j] * 100:.1f}%" for j in order]

        similar_cases.append(" | ".join(names))
        similarity_percent.append(" | ".join(vals))

    df["similar_cases"] = similar_cases
    df["similarity_percent"] = similarity_percent

    # =========================
    # Beauty range summary
    # =========================
    range_rows = []

    for _, _, level in RANGES:
        sub = df[df["beauty_level"] == level]

        row = {
            "beauty_level": level,
            "image_count": len(sub),
            "mean_score_avg": sub["mean_score"].mean() if len(sub) else np.nan,
            "mean_score_min": sub["mean_score"].min() if len(sub) else np.nan,
            "mean_score_max": sub["mean_score"].max() if len(sub) else np.nan,
        }

        for f in features:
            row[f"{f}_avg"] = sub[f].mean() if len(sub) else np.nan
            row[f"{f}_min"] = sub[f].min() if len(sub) else np.nan
            row[f"{f}_max"] = sub[f].max() if len(sub) else np.nan

        range_rows.append(row)

    range_summary_df = pd.DataFrame(range_rows)

    # high beauty profile
    high_row = range_summary_df[range_summary_df["beauty_level"] == "4-5 عالي"]
    if len(high_row):
        high_profile = high_row.iloc[0].to_dict()
    else:
        high_profile = {}

    # =========================
    # Representative cases
    # =========================
    rep_rows = []
    for level in df["beauty_level"].unique():
        sub = df[df["beauty_level"] == level]
        idxs = sub.index.tolist()

        if len(idxs) == 1:
            best_idx = idxs[0]
            avg_sim = 1.0
        else:
            local_sim = sim[np.ix_(idxs, idxs)]
            avg_sims = local_sim.mean(axis=1)
            best_pos = int(np.argmax(avg_sims))
            best_idx = idxs[best_pos]
            avg_sim = float(avg_sims[best_pos])

        rep_rows.append({
            "beauty_level": level,
            "representative_case_id": df.loc[best_idx, "case_id"],
            "image_name": df.loc[best_idx, "image_name"],
            "mean_score": df.loc[best_idx, "mean_score"],
            "avg_similarity_inside_level": avg_sim,
        })

    representative_df = pd.DataFrame(rep_rows)
    df["is_representative"] = df["case_id"].isin(
        representative_df["representative_case_id"].tolist()
    )

    # =========================
    # Outlier detection
    # =========================
    avg_sim_all = []
    for i in range(len(df)):
        others = [j for j in range(len(df)) if j != i]
        avg_sim_all.append(float(sim[i, others].mean()) if others else 1.0)

    df["avg_similarity_to_all"] = avg_sim_all
    threshold = np.percentile(avg_sim_all, 15) if len(avg_sim_all) > 3 else 0
    df["is_outlier"] = df["avg_similarity_to_all"] < threshold

    # =========================
    # Case strengths / weaknesses / stories
    # =========================
    # نحسب مسبقًا أي ميزة ثابتة القيمة تمامًا عبر كل العينة (مثل water_pct
    # حين تكون معطّلة)، لتجاهلها عند توليد نقاط القوة/الضعف لكل حالة.
    zero_variance_features = {f for f in features if df[f].std(ddof=0) == 0}

    strengths_list = []
    weaknesses_list = []
    stories = []

    for _, row in df.iterrows():
        strengths, weaknesses = describe_case_strengths(
            row, high_profile, features, zero_variance_features
        )

        strengths_list.append(" | ".join(strengths))
        weaknesses_list.append(" | ".join(weaknesses))

        row2 = row.copy()
        row2["confidence_level"] = row.get("confidence_level", "غير محدد")

        stories.append(make_case_story(row2, strengths, weaknesses))

    df["strength_points_ar"] = strengths_list
    df["weakness_points_ar"] = weaknesses_list
    df["case_story_ar"] = stories

    # =========================
    # Nearest high beauty cases
    # =========================
    high_cases = df[df["beauty_level"] == "4-5 عالي"].index.tolist()
    nearest_high_names = []
    nearest_high_similarity = []

    for i in range(len(df)):
        if not high_cases:
            nearest_high_names.append("")
            nearest_high_similarity.append("")
            continue

        ordered_high = sorted(high_cases, key=lambda j: sim[i, j], reverse=True)
        names = df.loc[ordered_high, "image_name"].astype(str).tolist()[:top_k]
        vals = [f"{sim[i, j] * 100:.1f}%" for j in ordered_high[:top_k]]

        nearest_high_names.append(" | ".join(names))
        nearest_high_similarity.append(" | ".join(vals))

    df["nearest_high_cases"] = nearest_high_names
    df["nearest_high_similarity"] = nearest_high_similarity

    # =========================
    # Final case memory columns
    # =========================
    case_cols = [
        "case_id",
        "image_name",
        "beauty_level",
        "mean_score",
    ]

    if "std_score" in df.columns:
        case_cols.append("std_score")
    if "n_raters" in df.columns:
        case_cols.append("n_raters")

    case_cols += features + [
        "confidence_level",
        "is_representative",
        "is_outlier",
        "avg_similarity_to_all",
        "similar_cases",
        "similarity_percent",
        "nearest_high_cases",
        "nearest_high_similarity",
        "strength_points_ar",
        "weakness_points_ar",
        "case_story_ar",
    ]

    case_memory_df = df[case_cols].copy()

    # separate stories table
    case_stories_df = df[
        [
            "case_id",
            "image_name",
            "beauty_level",
            "mean_score",
            "confidence_level",
            "strength_points_ar",
            "weakness_points_ar",
            "case_story_ar",
        ]
    ].copy()

    # =========================
    # Save outputs
    # =========================
    out_case = os.path.join(out_dir, "case_memory.xlsx")
    out_sim = os.path.join(out_dir, "similarity_matrix.xlsx")
    out_rep = os.path.join(out_dir, "representative_cases.xlsx")
    out_range = os.path.join(out_dir, "beauty_range_summary.xlsx")
    out_stories = os.path.join(out_dir, "case_stories.xlsx")
    out_json = os.path.join(out_dir, "experience_memory.json")

    case_memory_df.to_excel(out_case, index=False)
    sim_df.to_excel(out_sim)
    representative_df.to_excel(out_rep, index=False)
    range_summary_df.to_excel(out_range, index=False)
    case_stories_df.to_excel(out_stories, index=False)

    memory = {
        "features": features,
        "beauty_ranges": [
            {"min": lo, "max": hi, "label": label} for lo, hi, label in RANGES
        ],
        "case_memory": case_memory_df.to_dict(orient="records"),
        "representative_cases": representative_df.to_dict(orient="records"),
        "beauty_range_summary": range_summary_df.to_dict(orient="records"),
        "case_stories": case_stories_df.to_dict(orient="records"),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

    return {
        "case_memory": case_memory_df,
        "similarity_matrix": sim_df,
        "representatives": representative_df,
        "range_summary": range_summary_df,
        "case_stories": case_stories_df,
        "paths": {
            "case_memory": out_case,
            "similarity_matrix": out_sim,
            "representatives": out_rep,
            "range_summary": out_range,
            "case_stories": out_stories,
            "json": out_json,
        },
    }