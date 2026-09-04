from .evidence_collector import to_float

# موثقة في الرسالة:
# Confidence = 0.35*HumanWeight + 0.25*RuleEvidence + 0.20*RangeGap + 0.15*SimilarCases + 0.05*Sensitivity
FUSION_WEIGHTS = {
    "human_weight": 0.35,
    "rule_evidence": 0.25,
    "range_gap": 0.20,
    "similar_cases": 0.15,
    "sensitivity": 0.05,
}


def normalize_human_weight(value, max_weight):
    if max_weight <= 0:
        return 0.0
    return min(1.0, max(0.0, float(value) / max_weight))


def range_gap_score(feature, current, preferred_avg):
    if preferred_avg is None:
        return 0.0, 0.0, "غير محدد"

    current = float(current)
    preferred_avg = float(preferred_avg)

    if feature == "ground_pct":
        gap = max(0.0, current - preferred_avg)
        score = min(1.0, gap / 20.0)
        status = "مرتفع" if gap > 0 else "ضمن المجال المقبول"
        return score, gap, status

    if feature == "flowers_pct" and current <= 0.5:
        gap = max(0.0, preferred_avg - current)
        return 1.0, gap, "منخفض جدًا"

    gap = max(0.0, preferred_avg - current)
    score = min(1.0, gap / 20.0)
    status = "منخفض" if gap > 0 else "ضمن المجال المقبول"
    return score, gap, status


def rule_evidence_score(rule):
    if not rule:
        return 0.0

    support = to_float(rule.get("support_count", 0))
    low_ratio = to_float(rule.get("low_ratio", 0))
    high_ratio = to_float(rule.get("high_ratio", 0))

    confidence = max(low_ratio, high_ratio)
    support_norm = min(1.0, support / 100.0)
    return 0.60 * confidence + 0.40 * support_norm


def sensitivity_score(feature, sensitivity_rows=None):
    if not sensitivity_rows:
        return 0.0

    for r in sensitivity_rows:
        if r.get("feature") == feature:
            delta = abs(to_float(r.get("score_change", 0.0)))
            return min(1.0, delta / 0.25)

    return 0.0


def score_and_rank_evidence(collected, sensitivity_rows=None):
    max_weight = max([to_float(r.get("human_weight", 0.0)) for r in collected] + [0.0])
    scored = []

    for r in collected:
        f = r["feature"]
        gap, missing, status = range_gap_score(
            feature=f,
            current=r["current_value"],
            preferred_avg=r.get("preferred_avg"),
        )

        hw = normalize_human_weight(r.get("human_weight", 0.0), max_weight)
        rs = rule_evidence_score(r.get("rule"))
        ss = to_float(r.get("similar_low_ratio", 0.0))
        ls = sensitivity_score(f, sensitivity_rows)

        # إذا لا توجد فجوة في المجال، لا نعتبرها سببًا حتى لو عندها وزن
        if gap <= 0.05:
            evidence_score = 0.0
            cause_type = "strength"
        else:
            evidence_score = (
                FUSION_WEIGHTS["human_weight"] * hw +
                FUSION_WEIGHTS["rule_evidence"] * rs +
                FUSION_WEIGHTS["range_gap"] * gap +
                FUSION_WEIGHTS["similar_cases"] * ss +
                FUSION_WEIGHTS["sensitivity"] * ls
            )
            cause_type = "root_cause"

        out = dict(r)
        out.update({
            "range_gap_score": gap,
            "missing_amount": missing,
            "status_ar": status,
            "human_weight_norm": hw,
            "rule_score": rs,
            "similar_case_score": ss,
            "sensitivity_score": ls,
            "evidence_score": evidence_score,
            "confidence_percent": round(evidence_score * 100, 1),
            "cause_type": cause_type,
            "fusion_formula": "0.35*HumanWeight + 0.25*RuleEvidence + 0.20*RangeGap + 0.15*SimilarCases + 0.05*Sensitivity",
        })
        scored.append(out)

    scored = sorted(scored, key=lambda x: x["evidence_score"], reverse=True)
    root_causes = [x for x in scored if x["cause_type"] == "root_cause" and x["evidence_score"] > 0][:3]
    strengths = [x for x in scored if x["cause_type"] == "strength"][:2]
    return scored, root_causes, strengths
