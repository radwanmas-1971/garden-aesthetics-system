import os
import json
import re
import numpy as np
import pandas as pd

FEATURES = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct", "water_pct"]

LEVELS = [
    (1.0, 3.0, "منخفض"),
    (3.0, 4.0, "متوسط"),
    (4.0, 5.01, "مرتفع"),
]


def assign_level(score):
    score = float(score)
    for lo, hi, label in LEVELS:
        if lo <= score < hi:
            return label
    return "غير محدد"


def image_number(name):
    m = re.search(r"\d+", str(name))
    return int(m.group()) if m else 999999


def representative_image(sub, features):
    if len(sub) == 0:
        return ""

    center = sub[features].mean().values.astype(float)
    X = sub[features].values.astype(float)
    dists = np.linalg.norm(X - center, axis=1)
    idx = int(np.argmin(dists))
    return sub.iloc[idx]["image_name"]


def build_beauty_level_analysis(dataset_clean_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_excel(dataset_clean_path)

    if "image_name" not in df.columns:
        raise ValueError("dataset_clean.xlsx must contain image_name")
    if "mean_score" not in df.columns:
        raise ValueError("dataset_clean.xlsx must contain mean_score")

    features=[f for f in FEATURES if f in df.columns]
    df=df.dropna(subset=["image_name","mean_score"]).copy()
    df["beauty_level_3"]=df["mean_score"].apply(assign_level)
    df["image_num"]=df["image_name"].apply(image_number)

    summary_rows=[]
    stats_rows=[]

    for _,_,level in LEVELS:
        sub=df[df["beauty_level_3"]==level].copy().sort_values("image_num")
        imgs=sub["image_name"].astype(str).tolist()
        nums=[str(image_number(x)) for x in imgs]
        range_text={"منخفض":"1 إلى أقل من 3","متوسط":"3 إلى أقل من 4","مرتفع":"4 إلى 5"}[level]

        summary_rows.append({
            "مستوى الجمال":level,
            "مدى الدرجة":range_text,
            "عدد الصور":len(sub),
            "أرقام الصور":", ".join(nums),
            "متوسط التقييم":round(sub["mean_score"].mean(),2) if len(sub) else None,
            "العشب (%)":round(sub["grass_pct"].mean(),2) if "grass_pct" in sub.columns else None,
            "الأشجار (%)":round(sub["trees_pct"].mean(),2) if "trees_pct" in sub.columns else None,
            "الزهور (%)":round(sub["flowers_pct"].mean(),2) if "flowers_pct" in sub.columns else None,
            "الأرض (%)":round(sub["ground_pct"].mean(),2) if "ground_pct" in sub.columns else None,
            "الماء (%)":round(sub["water_pct"].mean(),2) if "water_pct" in sub.columns else 0,
            "الصورة الأكثر تمثيلاً":representative_image(sub,features)
        })

        for f in features:
            stats_rows.append({
                "Beauty Level":level,
                "Feature":f,
                "Min":round(sub[f].min(),2) if len(sub) else None,
                "Average":round(sub[f].mean(),2) if len(sub) else None,
                "Max":round(sub[f].max(),2) if len(sub) else None,
                "Std":round(sub[f].std(),2) if len(sub)>1 else 0,
                "Count":len(sub)
            })

    summary_df=pd.DataFrame(summary_rows)
    total={
        "مستوى الجمال":"الإجمالي","مدى الدرجة":"","عدد الصور":len(df),"أرقام الصور":"",
        "متوسط التقييم":round(df["mean_score"].mean(),2),
        "العشب (%)":round(df["grass_pct"].mean(),2) if "grass_pct" in df.columns else None,
        "الأشجار (%)":round(df["trees_pct"].mean(),2) if "trees_pct" in df.columns else None,
        "الزهور (%)":round(df["flowers_pct"].mean(),2) if "flowers_pct" in df.columns else None,
        "الأرض (%)":round(df["ground_pct"].mean(),2) if "ground_pct" in df.columns else None,
        "الماء (%)":round(df["water_pct"].mean(),2) if "water_pct" in df.columns else 0,
        "الصورة الأكثر تمثيلاً":""
    }
    summary_df=pd.concat([summary_df,pd.DataFrame([total])],ignore_index=True)
    stats_df=pd.DataFrame(stats_rows)

    image_cols=["image_name","image_num","beauty_level_3","mean_score"]+features
    images_df=df[image_cols].sort_values(["beauty_level_3","image_num"]).rename(columns={
        "image_name":"اسم الصورة","image_num":"رقم الصورة","beauty_level_3":"مستوى الجمال","mean_score":"متوسط التقييم"
    })

    out_xlsx=os.path.join(out_dir,"beauty_level_master_table.xlsx")
    stats_path=os.path.join(out_dir,"beauty_level_statistics.xlsx")
    out_json=os.path.join(out_dir,"beauty_level_master_table.json")

    with pd.ExcelWriter(out_xlsx,engine="openpyxl") as writer:
        summary_df.to_excel(writer,sheet_name="Summary",index=False)
        images_df.to_excel(writer,sheet_name="All Images",index=False)

    stats_df.to_excel(stats_path,index=False)

    with open(out_json,"w",encoding="utf-8") as f:
        json.dump({"levels":summary_df.to_dict(orient="records"),
                   "images":images_df.to_dict(orient="records")},
                  f,ensure_ascii=False,indent=2)

    return summary_df, images_df, out_xlsx, out_json