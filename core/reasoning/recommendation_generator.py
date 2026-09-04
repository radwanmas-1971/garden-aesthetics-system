def generate_recommendations(root_causes):
    recs = []

    for c in root_causes:
        f = c["feature"]
        missing = float(c.get("missing_amount", 0.0))

        if f == "flowers_pct":
            recs.append({
                "priority": 1,
                "feature": f,
                "recommendation_ar": f"زيادة الزهور بحوالي {missing:.1f}٪ للوصول إلى المجال المفضل.",
                "missing_amount": missing,
            })
        elif f == "ground_pct":
            recs.append({
                "priority": 1,
                "feature": f,
                "recommendation_ar": f"تقليل الأرض المكشوفة بحوالي {missing:.1f}٪ باستخدام عشب أو غطاء أرضي.",
                "missing_amount": missing,
            })
        elif f == "grass_pct":
            recs.append({
                "priority": 1,
                "feature": f,
                "recommendation_ar": f"تحسين المسطح الأخضر بحوالي {missing:.1f}٪.",
                "missing_amount": missing,
            })
        elif f == "trees_pct":
            recs.append({
                "priority": 1,
                "feature": f,
                "recommendation_ar": f"زيادة الأشجار أو الشجيرات بحوالي {missing:.1f}٪.",
                "missing_amount": missing,
            })
        elif f == "water_pct":
            recs.append({
                "priority": 1,
                "feature": f,
                "recommendation_ar": f"إضافة عنصر ماء بسيط بحوالي {missing:.1f}٪ إذا كان مناسبًا.",
                "missing_amount": missing,
            })

    # remove duplicates
    clean = []
    seen = set()
    for r in recs:
        if r["recommendation_ar"] not in seen:
            clean.append(r)
            seen.add(r["recommendation_ar"])

    for i, r in enumerate(clean, 1):
        r["priority"] = i

    return clean
