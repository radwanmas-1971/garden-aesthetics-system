# core/dataset_builder.py
import os
import re
import numpy as np
import pandas as pd
import cv2

def normalize_image_name(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\\", "/")
    if re.fullmatch(r"\d+", s):
        return f"{s}.jpg"
    if re.fullmatch(r"\d+\.0", s):
        return f"{s.split('.')[0]}.jpg"
    if re.search(r"\.(jpg|jpeg|png|bmp|tif|tiff)$", s, re.IGNORECASE):
        return os.path.basename(s)
    return f"{s}.jpg"

def compute_quality_metrics(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return np.nan, np.nan
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    return blur, bright

_SCORE_NAME_HINTS = [
    "mean_score", "mean score", "average", "avg", "score", "rating",
    "متوسط", "الدرجة", "درجة", "تقييم", "الجمال",
]


def _find_score_column(df: pd.DataFrame) -> str:
    """
    Picks the beauty-score column out of the survey sheet, instead of
    blindly assuming it is "whatever column comes right after image_name".
    That assumption silently breaks (wrong column merged in as the score,
    with no error) if the sheet ever has an extra column — e.g. notes,
    voter count, a timestamp — between image_name and the real score.

    Strategy, in order:
      1) Prefer a column whose name matches common score-related wording
         (English or Arabic) — this is how a human would find it.
      2) Otherwise, fall back to the first remaining column that is
         "mostly numeric" (>=80% of non-empty values parse as numbers)
         AND whose values plausibly look like a beauty rating (median
         between 0 and 10 — covers both 1-5 and 1-10 scales).
      3) If nothing qualifies, raise a clear error instead of guessing.
    """
    candidates = [c for c in df.columns if c != "image_name"]
    if not candidates:
        raise ValueError("لم يتم العثور على أي عمود آخر غير image_name في ملف الاستبيان")

    lowered = {c: str(c).strip().lower() for c in candidates}

    for c in candidates:
        name = lowered[c]
        if any(hint in name for hint in _SCORE_NAME_HINTS):
            return c

    for c in candidates:
        numeric = pd.to_numeric(df[c], errors="coerce")
        non_null = numeric.notna().sum()
        total = df[c].notna().sum()
        if total == 0:
            continue
        numeric_ratio = non_null / total
        if numeric_ratio >= 0.8 and 0 <= numeric.median() <= 10:
            return c

    raise ValueError(
        "تعذّر تحديد عمود درجة الجمال في ملف الاستبيان تلقائيًا. "
        "الأعمدة المتاحة: " + ", ".join(map(str, candidates)) +
        " — يرجى إعادة تسمية العمود المطلوب إلى شيء يحتوي على "
        "'score' أو 'mean_score' أو 'درجة' أو 'تقييم'."
    )


def load_survey_scores(survey_xlsx_path, sheet_name=0):
    import pandas as pd

    raw = pd.read_excel(survey_xlsx_path, sheet_name=sheet_name, header=None)

    header_row = None
    for i in range(len(raw)):
        row_values = [str(x).strip() for x in raw.iloc[i].tolist()]
        if "image_name" in row_values:
            header_row = i
            break

    if header_row is None:
        raise ValueError("لم يتم العثور على عمود image_name داخل ملف الاستبيان")

    df = pd.read_excel(survey_xlsx_path, sheet_name=sheet_name, header=header_row)

    df = df.dropna(how="all").copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "image_name" not in df.columns:
        raise ValueError("ملف الاستبيان يجب أن يحتوي على عمود image_name")

    score_col = _find_score_column(df)

    df = df[["image_name", score_col]].copy()
    df = df.rename(columns={score_col: "mean_score"})

    def fix_image_name(x):
        x = str(x).strip()
        if x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".png"):
            return x
        if x.replace(".", "", 1).isdigit():
            return str(int(float(x))) + ".jpg"
        return x

    df["image_name"] = df["image_name"].apply(fix_image_name)
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")

    df = df.dropna(subset=["image_name", "mean_score"])

    return df


def apply_keep_rules(row, thresholds):
    reasons = []
    if not np.isnan(row["blur_score"]) and row["blur_score"] < thresholds["blur_min"]:
        reasons.append(f"blur<{thresholds['blur_min']}")
    if not np.isnan(row["brightness_mean"]) and row["brightness_mean"] < thresholds["bright_min"]:
        reasons.append(f"dark<{thresholds['bright_min']}")
    if not np.isnan(row["brightness_mean"]) and row["brightness_mean"] > thresholds["bright_max"]:
        reasons.append(f"bright>{thresholds['bright_max']}")
    if not np.isnan(row["other_pct"]) and row["other_pct"] > thresholds["other_max"]:
        reasons.append(f"other>{thresholds['other_max']}%")

    keep = "yes" if len(reasons) == 0 else "no"
    return keep, "; ".join(reasons)

def build_dataset(
    image_folder: str,
    results_csv_path: str,
    survey_xlsx_path: str,
    out_dir: str,
    thresholds: dict,
    apply_exclusion: bool = True
):
    os.makedirs(out_dir, exist_ok=True)

    feats = pd.read_csv(results_csv_path)
    if "image_name" not in feats.columns:
        raise ValueError("results.csv must include image_name")

    for c in ["grass_pct", "trees_pct", "flowers_pct", "ground_pct"]:
        if c not in feats.columns:
            raise ValueError(f"results.csv missing column: {c}")

    if "water_pct" not in feats.columns:
        feats["water_pct"] = 0.0

    feats["other_pct"] = 100.0 - (
        feats["grass_pct"].fillna(0) +
        feats["trees_pct"].fillna(0) +
        feats["flowers_pct"].fillna(0) +
        feats["ground_pct"].fillna(0) +
        feats["water_pct"].fillna(0)
    )

    scores = load_survey_scores(survey_xlsx_path, sheet_name=0)

    df = feats.merge(scores, on="image_name", how="inner")

    # compute blur/brightness
    blur_list, bright_list = [], []
    for name in df["image_name"].tolist():
        p = os.path.join(image_folder, name)
        b, br = compute_quality_metrics(p)
        blur_list.append(b)
        bright_list.append(br)

    df["blur_score"] = blur_list
    df["brightness_mean"] = bright_list

    keeps, reasons = [], []
    for _, r in df.iterrows():
        k, reason = apply_keep_rules(r, thresholds)
        keeps.append(k)
        reasons.append(reason)
    df["keep"] = keeps
    df["drop_reason"] = reasons

    all_path = os.path.join(out_dir, "dataset_all_with_flags.xlsx")
    clean_path = os.path.join(out_dir, "dataset_clean.xlsx")

    df.to_excel(all_path, index=False)

    if apply_exclusion:
        df_clean = df[df["keep"] == "yes"].copy()
    else:
        df_clean = df.copy()

    df_clean.to_excel(clean_path, index=False)

    return df, df_clean, all_path, clean_path