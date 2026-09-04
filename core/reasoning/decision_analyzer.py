def beauty_level(score):
    score = float(score)
    if score < 2:
        return "1-2 منخفض جدًا"
    if score < 3:
        return "2-3 منخفض"
    if score < 4:
        return "3-4 متوسط"
    return "4-5 عالي"


def analyze_decision(predicted_score):
    score = float(predicted_score)
    level = beauty_level(score)

    if score < 3:
        msg = "التقييم منخفض، لذلك يبحث النظام عن الأسباب الأكثر تأثيرًا التي خفّضت درجة الجمال."
    elif score < 4:
        msg = "التقييم متوسط؛ توجد عناصر جيدة، لكن هناك عوامل تمنع الصورة من الوصول إلى مستوى الجمال العالي."
    else:
        msg = "التقييم عالٍ نسبيًا؛ الصورة قريبة من النمط الذي فضّله المشاركون، مع إمكانية تحسينات محدودة."

    return {
        "predicted_score": score,
        "beauty_level": level,
        "decision_ar": msg,
    }
