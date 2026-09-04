from core.reasoning.decision_analyzer import analyze_decision
from core.reasoning.evidence_collector import find_similar_cases, collect_evidence
from core.reasoning.evidence_fusion import score_and_rank_evidence, FUSION_WEIGHTS
from core.reasoning.explanation_generator import generate_explanation
from core.reasoning.recommendation_generator import generate_recommendations
import json


def load_knowledge_repository(repo_path):
    with open(repo_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_reasoning(feats, predicted_score, repo, feature_cols=None, sensitivity_rows=None):
    """
    Clean Architecture Reasoning Engine.

    Confidence formula:
    Confidence = 0.35*HumanWeight + 0.25*RuleEvidence + 0.20*RangeGap + 0.15*SimilarCases + 0.05*Sensitivity

    feature_cols controls which features are allowed in reasoning.
    If water_pct was not used in training, it must not appear in reasoning.
    """
    if feature_cols is None:
        feature_cols = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct"]

    feature_cols = [c for c in feature_cols if c in feats]

    decision = analyze_decision(predicted_score)

    similar_cases = find_similar_cases(
        feats=feats,
        repo=repo,
        feature_cols=feature_cols,
        top_k=3,
    )

    collected = collect_evidence(
        feats=feats,
        repo=repo,
        feature_cols=feature_cols,
        similar_cases=similar_cases,
    )

    feature_evidence, root_causes, strengths = score_and_rank_evidence(
        collected=collected,
        sensitivity_rows=sensitivity_rows,
    )

    reasoning_ar, conclusion = generate_explanation(
        decision=decision,
        similar_cases=similar_cases,
        root_causes=root_causes,
        strengths=strengths,
        all_evidence=feature_evidence,
    )

    recommendations_structured = generate_recommendations(root_causes)
    recommendations_ar = [r["recommendation_ar"] for r in recommendations_structured]

    matched_rules = []
    for c in root_causes:
        rule = c.get("rule")
        if rule:
            matched_rules.append(rule)

    return {
        "predicted_score": float(predicted_score),
        "beauty_level": decision["beauty_level"],
        "decision": decision,
        "similar_cases": similar_cases,
        "feature_evidence": feature_evidence,
        "root_causes": root_causes,
        "strengths": strengths,
        "matched_rules": matched_rules,
        "reasoning_ar": reasoning_ar,
        "recommendations_ar": recommendations_ar,
        "recommendations_structured": recommendations_structured,
        "expert_conclusion_ar": conclusion,
        "confidence_formula": "Confidence = 0.35*HumanWeight + 0.25*RuleEvidence + 0.20*RangeGap + 0.15*SimilarCases + 0.05*Sensitivity",
        "fusion_weights": FUSION_WEIGHTS,
        "used_features": feature_cols,
    }
