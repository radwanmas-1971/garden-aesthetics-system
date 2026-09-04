import os
import json
from datetime import datetime
import pandas as pd

def read_excel_if_exists(path):
    if os.path.isfile(path):
        return pd.read_excel(path).to_dict(orient="records")
    return []

def build_knowledge_repository(experience_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    repo = {
        "metadata": {
            "version": "1.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": experience_dir,
        },
        "case_memory": read_excel_if_exists(os.path.join(experience_dir, "case_memory.xlsx")),
        "case_stories": read_excel_if_exists(os.path.join(experience_dir, "case_stories.xlsx")),
        "representative_cases": read_excel_if_exists(os.path.join(experience_dir, "representative_cases.xlsx")),
        "beauty_range_summary": read_excel_if_exists(os.path.join(experience_dir, "beauty_range_summary.xlsx")),
        "feature_trends": read_excel_if_exists(os.path.join(experience_dir, "feature_trends.xlsx")),
        "feature_correlations": read_excel_if_exists(os.path.join(experience_dir, "feature_correlations.xlsx")),
        "beauty_level_patterns": read_excel_if_exists(os.path.join(experience_dir, "beauty_level_patterns.xlsx")),
        "knowledge_rules": read_excel_if_exists(os.path.join(experience_dir, "knowledge_rules.xlsx")),
        "exceptional_cases": read_excel_if_exists(os.path.join(experience_dir, "exceptional_cases.xlsx")),
        "human_knowledge_weights": read_excel_if_exists(os.path.join(experience_dir, "human_knowledge_weights.xlsx")),
    }

    out_json = os.path.join(out_dir, "knowledge_repository.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(repo, f, ensure_ascii=False, indent=2)

    summary = {
        "case_memory_count": len(repo["case_memory"]),
        "case_stories_count": len(repo["case_stories"]),
        "representative_cases_count": len(repo["representative_cases"]),
        "knowledge_rules_count": len(repo["knowledge_rules"]),
        "exceptional_cases_count": len(repo["exceptional_cases"]),
        "human_knowledge_weights_count": len(repo["human_knowledge_weights"]),
    }

    summary_path = os.path.join(out_dir, "knowledge_repository_summary.xlsx")
    pd.DataFrame([summary]).to_excel(summary_path, index=False)

    return repo, out_json, summary_path, summary