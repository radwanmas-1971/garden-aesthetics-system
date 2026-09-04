def stars(score):
    score = float(score)
    if score >= 0.80:
        return "★★★★★"
    if score >= 0.60:
        return "★★★★☆"
    if score >= 0.40:
        return "★★★☆☆"
    if score >= 0.20:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def generate_explanation(decision, similar_cases, root_causes, strengths, all_evidence):
    lines = []
    score = decision["predicted_score"]
    level = decision["beauty_level"]

    lines.append(f"القرار النهائي: درجة الجمال المتوقعة هي {score:.2f} من 5، وتصنيف الصورة هو: {level}.")
    lines.append(decision["decision_ar"])

    if similar_cases:
        sim_txt = "، ".join([
            f"{c.get('image_name', '')} بدرجة {float(c.get('mean_score', 0)):.2f}"
            for c in similar_cases
        ])
        lines.append(f"اعتمد النظام على حالات مشابهة في قاعدة الخبرة، أهمها: {sim_txt}.")

    if root_causes:
        lines.append("أهم الأسباب بعد دمج الأدلة المختلفة:")
        for i, c in enumerate(root_causes, 1):
            pref = ""
            if c.get("preferred_avg") is not None:
                pref = (
                    f" المجال المفضل في الصور عالية الجمال: "
                    f"{c.get('preferred_min', 0):.2f}٪ - {c.get('preferred_max', 0):.2f}٪ "
                    f"(المتوسط {c.get('preferred_avg', 0):.2f}٪)."
                )

            rule_txt = ""
            rule = c.get("rule")
            if rule:
                rule_txt = f" توجد قاعدة معرفية مدعومة بعدد {rule.get('support_count', 0)} حالة."

            lines.append(
                f"{i}) {c['feature_ar']}: القيمة الحالية {c['current_value']:.2f}٪، "
                f"والحالة: {c['status_ar']}.{pref} "
                f"الفجوة التقريبية: {c['missing_amount']:.2f}٪. "
                f"درجة الدليل المدمج: {stars(c['evidence_score'])}، والثقة: {c['confidence_percent']:.1f}٪."
                f"{rule_txt}"
            )

        # مقارنة مختصرة بين أفضل سبب وما بعده
        if len(root_causes) >= 2:
            lines.append(
                f"تم اختيار {root_causes[0]['feature_ar']} كسبب رئيسي لأنه حصل على أعلى درجة دليل "
                f"({root_causes[0]['confidence_percent']:.1f}٪) مقارنةً بـ {root_causes[1]['feature_ar']} "
                f"({root_causes[1]['confidence_percent']:.1f}٪)."
            )
        else:
            main = root_causes[0]
            other_scores = [x for x in all_evidence if x["feature"] != main["feature"]]
            if other_scores:
                best_other = other_scores[0]
                lines.append(
                    f"تم اختيار {main['feature_ar']} كسبب رئيسي لأنه حصل على أعلى درجة دليل "
                    f"({main['confidence_percent']:.1f}٪)، بينما أقرب سمة أخرى كانت {best_other['feature_ar']} "
                    f"بدرجة {best_other['confidence_percent']:.1f}٪."
                )
    else:
        lines.append("لم يجد النظام سببًا سلبيًا قويًا بعد دمج الأدلة المختلفة.")

    if strengths:
        lines.append("عناصر القوة في الصورة:")
        for s in strengths:
            lines.append(
                f"- {s['feature_ar']}: القيمة الحالية {s['current_value']:.2f}٪ قريبة من النمط المفضل، لذلك لا تعد سببًا رئيسيًا للانخفاض."
            )

    if root_causes:
        main = root_causes[0]
        conclusion = (
            f"الاستنتاج الخبيري: بعد دمج الأدلة المختلفة، يتضح أن العامل الأكثر تفسيرًا للنتيجة هو "
            f"{main['feature_ar']} بدرجة ثقة {main['confidence_percent']:.1f}٪. "
            f"لذلك يجب أن تبدأ خطة التحسين به أولًا."
        )
    else:
        conclusion = (
            "الاستنتاج الخبيري: لا توجد فجوة رقمية قوية في السمات الحالية، وقد يكون الحكم مرتبطًا بعوامل بصرية أخرى مثل التكوين أو التوزيع أو الإضاءة."
        )

    lines.append(conclusion)
    return lines, conclusion
