# core/garden_integration.py

import math
import numpy as np
import pandas as pd

# استيراد من مرحلة 8 لتوحيد المفردات والأسلوب بين المرحلتين (نفس مكتبة
# الإثراءات الوصفية غير المُقيَّمة من النموذج: إضاءة/جلوس/حواف)، بدل
# نسخة محلية منفصلة كانت تقتصر على الإضاءة فقط. لا استيراد عكسي —
# core/budget_scenarios.py لا يستورد من هذا الملف إطلاقًا، فلا تعارض دوري.
from core.budget_scenarios import NARRATIVE_ENRICHMENTS, NARRATIVE_DISCLAIMER_AR, _enrichment_pool


def _tier_to_bs_key(tier):
    """يُطابق تسميات المستوى في هذا الملف (economic/balanced/ideal وأسماء
    بديلة) مع مفاتيح مكتبة مرحلة 8 (low/medium/open)."""
    t = (tier or "").lower()
    if t in ("economic", "low"):
        return "low"
    if t in ("ideal", "open", "high"):
        return "open"
    return "medium"


# ============================================================
# Feature configuration
# ============================================================

TARGET_FEATURES = [
    "grass_pct",
    "trees_pct",
    "flowers_pct",
    "ground_pct",
    "water_pct",
]


AR_FEATURE_NAMES = {
    "grass_pct": "المسطحات الخضراء",
    "trees_pct": "الأشجار والغطاء النباتي المرتفع",
    "flowers_pct": "أحواض الزهور",
    "ground_pct": "الممرات والأرضيات والعناصر غير النباتية",
    "water_pct": "العناصر المائية",
}


# ============================================================
# مكتبة أسماء وصفية للماء والزهور (Action Library)
# ============================================================
# ملاحظة منهجية مهمة: هذه أسماء **وصفية فقط** تُضفي طابعًا سرديًا أغنى
# على التقرير — النموذج (MLP) لا يعرف الفرق بين "نافورة" و"شلال"، فهو
# يرى فقط رقم water_pct. اختيار الاسم يعتمد على حجم النسبة (ومستوى
# السيناريو للعنصر الأكثر تميّزًا)، وليس على تقييم فعلي من النموذج لنوع
# العنصر تحديدًا. العشوائية محدَّدة (seeded) بمجموع السمات نفسها لضمان
# نفس الاسم دائمًا لنفس التركيبة بالضبط (قابلية تكرار أكاديمية).
# ============================================================

import random as _random


def _seeded_choice(options, seed_key):
    return _random.Random(seed_key).choice(options)


# ============================================================
# بوابة الجدوى الفيزيائية (Physical Feasibility Gate)
# ============================================================
# مرحلة 9 هي الوحيدة التي تملك أبعادًا حقيقية للحديقة (متر/م²) — مرحلة 8
# تعمل فقط على نِسَب مستخرجة من صورة بلا معرفة بالحجم الفعلي، لذلك هذه
# البوابة مطبَّقة هنا حصرًا. الفكرة: عناصر "الواجهة" الكبيرة (نافورة راقصة
# موسيقية، تقسيم Garden Rooms متعدد) لا تُقترَح إلا إن كانت المساحة الحقيقية
# الموحَّدة كافية فعليًا لاستيعابها، بدل افتراض أنها تناسب أي حديقة.
# العتبات تقديرية معقولة (لا معيار أكاديمي موحَّد لحجم "ساحة نافورة")، وقابلة
# للتعديل بسهولة من مكان واحد.
# ============================================================
FLAGSHIP_WATER_MIN_AREA_M2 = 150.0   # مساحة دنيا لساحة نافورة راقصة/موسيقية
FLAGSHIP_WATER_MIN_DIM_M = 8.0       # أصغر بُعد (عرض أو طول) يجب ألا يقل عنه
MULTI_ROOM_MIN_AREA_M2 = 80.0        # تحتها: لا تُقترَح Garden Rooms منفصلة


def space_allows_flagship_water(geometry_or_area, length=None):
    """
    يقبل إمّا قاموس geometry كامل (فيه area/width/length)، أو مساحة رقمية
    مباشرة + طول اختياري. يُعيد True فقط إن كانت المساحة والأبعاد كافية
    فعليًا لعنصر مائي "بارز" (نافورة راقصة موسيقية).
    """
    if isinstance(geometry_or_area, dict):
        area = float(geometry_or_area.get("area", 0.0))
        width = float(geometry_or_area.get("width", 0.0))
        length_ = float(geometry_or_area.get("length", 0.0))
        min_dim = min(width, length_) if (width > 0 and length_ > 0) else 0.0
    else:
        area = float(geometry_or_area or 0.0)
        min_dim = float(length) if length is not None else FLAGSHIP_WATER_MIN_DIM_M
    return area >= FLAGSHIP_WATER_MIN_AREA_M2 and min_dim >= FLAGSHIP_WATER_MIN_DIM_M


def space_allows_multi_room(geometry_or_area):
    if isinstance(geometry_or_area, dict):
        area = float(geometry_or_area.get("area", 0.0))
    else:
        area = float(geometry_or_area or 0.0)
    return area >= MULTI_ROOM_MIN_AREA_M2


WATER_FEATURE_LIBRARY = {
    "tiny": [
        "نافورة صغيرة هادئة",
        "بركة صغيرة مع نباتات محيطة",
    ],
    "small": [
        "نافورة دائرية حديثة تحيط بها حلقة من الزهور والنباتات",
        "شلال صخري صغير يتدفق بين الصخور والنباتات",
        "مجرى مائي قصير",
    ],
    "medium": [
        "نافورة مركزية كلاسيكية مناسبة للتصميم المتماثل",
        "مجرى مائي متعرج ينتهي ببركة صغيرة",
        "جدار مائي مناسب للمساحات المحدودة",
    ],
    "flagship": [
        "عنصر مائي تفاعلي موسيقي (نافورة راقصة مع تأثيرات ماء وإضاءة متزامنة)",
    ],
}


def describe_water_feature(water_pct, tier="medium", space_ok=True):
    """
    يُعيد اسمًا وصفيًا لعنصر مائي بناءً على حجم water_pct، مع السماح
    بالعنصر الأبرز (النافورة التفاعلية الموسيقية) فقط عند تحقّق ثلاثة
    شروط معًا: نسبة ماء كبيرة نسبيًا (>=8%)، مستوى سيناريو "مثالي"،
    و space_ok=True (بوابة الجدوى الفيزيائية — مساحة الحديقة الحقيقية
    الموحَّدة كافية فعليًا، انظر space_allows_flagship_water أعلاه).
    وليس افتراضيًا لأي عنصر مائي كما كانت المذكرة تنتقد ("لا نجعل النظام
    يقترح دائمًا بركة ماء فقط"، والآن أيضًا: "لا نقترح نافورة راقصة في
    حديقة صغيرة لا تتّسع لها فعليًا").
    """
    water_pct = float(water_pct)
    if water_pct <= 0.05:
        return None

    seed_key = f"water|{water_pct:.2f}|{tier}"

    if water_pct >= 8.0 and tier in ("ideal", "open", "high") and space_ok:
        # نصف الحالات فقط تحصل على العنصر الأبرز، لتفادي تكراره في كل
        # سيناريو مثالي بلا استثناء — قرار محدَّد (seeded) لكل حالة.
        if _random.Random(seed_key + "|flagship_gate").random() < 0.5:
            return _seeded_choice(WATER_FEATURE_LIBRARY["flagship"], seed_key)
        return _seeded_choice(WATER_FEATURE_LIBRARY["medium"], seed_key)
    if water_pct >= 5.0:
        return _seeded_choice(WATER_FEATURE_LIBRARY["medium"], seed_key)
    if water_pct >= 2.0:
        return _seeded_choice(WATER_FEATURE_LIBRARY["small"], seed_key)
    return _seeded_choice(WATER_FEATURE_LIBRARY["tiny"], seed_key)


FLOWER_STYLE_LIBRARY = {
    "minimal": [
        "أحواض حدودية متقطعة على أطراف الحديقة (بدل صف عشوائي واحد)",
    ],
    "moderate_with_trees": [
        "أحواض دائرية حول قواعد الأشجار",
    ],
    "moderate_no_trees": [
        "زهور موزَّعة على امتداد الممرات الرئيسية",
    ],
    "rich": [
        "جزر زهرية موسمية موزَّعة في مواقع مختارة",
        "مناطق زهرية بتناسق لوني (مثل: وردي وأبيض، أو أصفر وبرتقالي)",
    ],
}


def describe_flower_style(flowers_pct, trees_pct=0.0):
    """
    يُعيد نمط أحواض زهور وصفيًا بناءً على حجم flowers_pct ووجود أشجار
    كافية (لأن "أحواض دائرية حول الأشجار" لا معنى لها بلا أشجار فعلية).
    """
    flowers_pct = float(flowers_pct)
    if flowers_pct <= 0.05:
        return None

    seed_key = f"flowers|{flowers_pct:.2f}|{trees_pct:.2f}"

    if flowers_pct >= 8.0:
        return _seeded_choice(FLOWER_STYLE_LIBRARY["rich"], seed_key)
    if flowers_pct >= 3.0:
        if float(trees_pct) >= 10.0:
            return _seeded_choice(FLOWER_STYLE_LIBRARY["moderate_with_trees"], seed_key)
        return _seeded_choice(FLOWER_STYLE_LIBRARY["moderate_no_trees"], seed_key)
    return _seeded_choice(FLOWER_STYLE_LIBRARY["minimal"], seed_key)


# ============================================================
# محرك قواعد: يصف مستوى كل سمة نسبةً لمعايير مرجعية (منخفض جدًا/
# منخفض/متوسط/جيد)، مبنية على متوسطات الصور "عالية الجمال" الحقيقية
# في بيانات الاستبيان (مرحلة 4: grass≈22%, trees≈44%) — وليست عتبات
# تخمينية. يُستخدَم لبناء جمل تقرير مخصَّصة لكل حديقة حسب نقصها الفعلي،
# بدل قالب ثابت لكل مستوى (بالضبط كما اقترحت المذكرة).
# ============================================================

def _level_ar(value, low, high):
    value = float(value)
    if value < low * 0.5:
        return "منخفض جدًا"
    if value < low:
        return "منخفض"
    if value <= high:
        return "متوسط"
    return "جيد"


# صيغة المؤنث لنفس مستويات _level_ar — لازمة عند وصف سمة اسمها مؤنث في
# العربية (مثل "أحواض الزهور" جمع تكسير يُعامَل معاملة المؤنث)، تفاديًا
# لخطأ مطابقة نحوية مثل "أحواض الزهور كانت منخفض" (رُصِد فعليًا).
_LEVEL_AR_FEMININE = {
    "منخفض جدًا": "منخفضة جدًا",
    "منخفض": "منخفضة",
    "متوسط": "متوسطة",
    "جيد": "جيدة",
}


def _feminize_level(level):
    return _LEVEL_AR_FEMININE.get(level, level)


def describe_deficiencies_ar(composition):
    """
    يُعيد قاموسًا {feature: مستوى_نصي} لكل سمة من التركيبة، لاستخدامه في
    بناء جمل تقرير مخصَّصة (محرك القواعد).
    """
    grass = composition.get("grass_pct", 0.0)
    trees = composition.get("trees_pct", 0.0)
    flowers = composition.get("flowers_pct", 0.0)
    ground = composition.get("ground_pct", 0.0)
    water = composition.get("water_pct", 0.0)

    return {
        "grass_pct": _level_ar(grass, 15.0, 35.0),
        "trees_pct": _level_ar(trees, 15.0, 35.0),
        "flowers_pct": _level_ar(flowers, 3.0, 8.0),
        "ground_empty": "كبيرة" if float(ground) > 35.0 else "معتدلة",
        "water_pct": "غير موجود" if float(water) <= 0.5 else "موجود",
    }


def deficiency_rule_sentences_ar(composition, scenario_key="economic"):
    """
    يبني جملًا تفسيرية مخصَّصة بناءً على نقاط الضعف الفعلية في composition
    (وليس قالبًا ثابتًا لكل سيناريو) — هذا هو "محرك القواعد" الذي اقترحته
    المذكرة: "المشكلة: grass منخفض جدًا، flowers منخفضة..." يُترجَم مباشرة
    لجملة اقتراح مخصَّصة لهذه الحالة تحديدًا.
    """
    levels = describe_deficiencies_ar(composition)
    lines = []

    grass_level = levels["grass_pct"]
    if grass_level in ("منخفض جدًا", "منخفض"):
        if scenario_key == "economic":
            lines.append(
                "بما أن المسطح الأخضر " + grass_level + "، يُنصح بمعالجة "
                "التربة وتحسين الري وإعادة بذر المناطق الفارغة قبل أي تدخل آخر."
            )
        else:
            lines.append(
                "تم إعطاء أولوية لتوسيع وتحسين المسطح الأخضر لأنه كان " + grass_level + " أصلًا."
            )

    trees_level = levels["trees_pct"]
    if trees_level in ("منخفض جدًا", "منخفض") and scenario_key != "economic":
        lines.append(
            "الغطاء الشجري كان " + trees_level + "، لذلك أُضيفت مجموعات أشجار "
            "وشجيرات جديدة لمعالجة هذا النقص تحديدًا."
        )
    elif trees_level == "جيد" and scenario_key != "economic":
        lines.append(
            "الغطاء الشجري كان جيدًا أصلًا، فرُكِّز التحسين على عناصر أخرى بدل زيادته أكثر."
        )

    flowers_level = levels["flowers_pct"]
    if flowers_level in ("منخفض جدًا", "منخفض"):
        lines.append(
            "أحواض الزهور كانت " + _feminize_level(flowers_level) + "، فأُضيفت زهور "
            + ("محدودة في المواقع الأكثر ظهورًا (كلفة منخفضة)." if scenario_key == "economic"
               else "بأسلوب أوسع لزيادة التنوع اللوني.")
        )

    if levels["ground_empty"] == "كبيرة":
        lines.append(
            "الفراغات الظاهرة (الأرض المكشوفة) كانت كبيرة نسبيًا، وقد "
            "استهدف التحسين تقليلها وزيادة الإحساس بالاكتمال البصري."
        )

    return lines


# ============================================================
# Geometry
# ============================================================

def garden_area(width, length):
    return float(width) * float(length)


def build_garden_geometry(gardens, layout="grid_2x2"):
    """
    gardens:
    [
        {"name": "Garden 1", "width": 10, "length": 15},
        ...
    ]

    Returns unified geometry and positions.

    Supported layouts:
        row
        column
        grid_2x2
    """

    if not gardens:
        raise ValueError("No gardens were provided.")

    gardens = [
        {
            "name": str(g.get("name", f"Garden {i+1}")),
            "width": float(g["width"]),
            "length": float(g["length"]),
        }
        for i, g in enumerate(gardens)
    ]

    if any(g["width"] <= 0 or g["length"] <= 0 for g in gardens):
        raise ValueError("All garden dimensions must be greater than zero.")

    positions = []

    # --------------------------------------------------------
    # Horizontal row
    # --------------------------------------------------------
    if layout == "row":

        x = 0.0
        max_length = max(g["length"] for g in gardens)

        for g in gardens:
            positions.append({
                **g,
                "x": x,
                "y": 0.0,
                "area": garden_area(g["width"], g["length"]),
            })
            x += g["width"]

        total_width = x
        total_length = max_length

    # --------------------------------------------------------
    # Vertical column
    # --------------------------------------------------------
    elif layout == "column":

        y = 0.0
        max_width = max(g["width"] for g in gardens)

        for g in gardens:
            positions.append({
                **g,
                "x": 0.0,
                "y": y,
                "area": garden_area(g["width"], g["length"]),
            })
            y += g["length"]

        total_width = max_width
        total_length = y

    # --------------------------------------------------------
    # 2 x 2 grid
    # --------------------------------------------------------
    else:

        # Pad to four positions for a simple rectangular grid.
        grid = list(gardens)

        while len(grid) < 4:
            grid.append({
                "name": "",
                "width": 0.0,
                "length": 0.0,
            })

        # Row 1
        w1, w2 = grid[0]["width"], grid[1]["width"]
        l1, l2 = grid[0]["length"], grid[1]["length"]

        # Row 2
        w3, w4 = grid[2]["width"], grid[3]["width"]
        l3, l4 = grid[2]["length"], grid[3]["length"]

        top_width = w1 + w2
        bottom_width = w3 + w4

        total_width = max(top_width, bottom_width)

        top_length = max(l1, l2)
        bottom_length = max(l3, l4)

        total_length = top_length + bottom_length

        # Top-left
        if grid[0]["name"]:
            positions.append({
                **grid[0],
                "x": 0.0,
                "y": bottom_length,
                "area": garden_area(w1, l1),
            })

        # Top-right
        if grid[1]["name"]:
            positions.append({
                **grid[1],
                "x": w1,
                "y": bottom_length,
                "area": garden_area(w2, l2),
            })

        # Bottom-left
        if grid[2]["name"]:
            positions.append({
                **grid[2],
                "x": 0.0,
                "y": 0.0,
                "area": garden_area(w3, l3),
            })

        # Bottom-right
        if grid[3]["name"]:
            positions.append({
                **grid[3],
                "x": w3,
                "y": 0.0,
                "area": garden_area(w4, l4),
            })

    total_area = sum(g["area"] for g in positions)

    bounding_area = total_width * total_length

    return {
        "gardens": positions,
        "layout": layout,
        "width": float(total_width),
        "length": float(total_length),
        "area": float(total_area),
        "bounding_area": float(bounding_area),
    }


# ============================================================
# MLP scoring
# ============================================================

def predict_beauty(model, feature_cols, feature_values):
    """
    Predict beauty using the current trained MLP.

    Missing features are filled with zero.

    ملاحظة مهمة (مُتّسقة مع إصلاح مرحلة 8 في app.py): بدل القص الحاد
    (np.clip(score, 1, 5)) الذي يُسطِّح أي تنبؤ خام يتجاوز 5 (وقد يكون
    فعليًا 7 أو 10 داخليًا لتوليفات غنية بالماء، غير موثوقة إحصائيًا خارج
    نطاق التدريب) إلى نفس الرقم 5.0 بالضبط دائمًا — نستخدم "سقفًا ليّنًا"
    يسمح للدرجة بالاقتراب من 4.8 دون الوصول إليها أبدًا تمامًا، فيعكس
    عدم اليقين المتزايد بدل إخفائه خلف رقم صناعي متكرر. بدون هذا الإصلاح،
    أي هدف قريب من 4.5+ سيبحث عن توليفات مائية تُعطي "5.00" مصطنعة دائمًا.
    """
    row = [
        float(feature_values.get(col, 0.0))
        for col in feature_cols
    ]

    x = np.array([row], dtype=np.float32)

    raw = float(model.predict(x)[0])

    SOFT_CEILING_START = 4.5
    SOFT_CEILING_MAX = 4.8

    if raw <= SOFT_CEILING_START:
        return float(np.clip(raw, 1.0, 5.0))

    excess = raw - SOFT_CEILING_START
    span = SOFT_CEILING_MAX - SOFT_CEILING_START
    compressed = SOFT_CEILING_START + span * (1.0 - np.exp(-excess / 2.0))
    return float(compressed)


# ============================================================
# Feature composition
# ============================================================

def normalize_composition(composition):
    """
    Normalize the five landscape percentages to exactly 100%.
    """

    out = {}

    for key in TARGET_FEATURES:
        out[key] = max(0.0, float(composition.get(key, 0.0)))

    total = sum(out.values())

    if total <= 0:
        return {
            "grass_pct": 40.0,
            "trees_pct": 20.0,
            "flowers_pct": 10.0,
            "ground_pct": 30.0,
            "water_pct": 0.0,
        }

    for key in out:
        out[key] = out[key] * 100.0 / total

    return out


def composition_to_area(composition, total_area):
    """
    Convert percentages into real square metres.
    """

    composition = normalize_composition(composition)

    result = {}

    for key, pct in composition.items():
        result[key] = float(total_area) * float(pct) / 100.0

    return result


# ============================================================
# Candidate generation
# ============================================================

def _frange(start, stop, step):
    values = []

    value = float(start)

    while value <= float(stop) + 1e-9:
        values.append(round(value, 4))
        value += float(step)

    return values


def generate_candidate_compositions(
    water_allowed=True,
    step=5.0,
):
    """
    Generate valid landscape compositions.

    All values are percentages and sum to 100.
    """

    candidates = []

    grass_values = _frange(20, 70, step)
    tree_values = _frange(5, 35, step)
    flower_values = _frange(0, 25, step)

    water_values = (
        _frange(0, 15, step)
        if water_allowed
        else [0.0]
    )

    for grass in grass_values:
        for trees in tree_values:
            for flowers in flower_values:
                for water in water_values:

                    ground = 100.0 - (
                        grass +
                        trees +
                        flowers +
                        water
                    )

                    # Realistic design constraints.
                    if ground < 10.0 or ground > 45.0:
                        continue

                    candidate = {
                        "grass_pct": grass,
                        "trees_pct": trees,
                        "flowers_pct": flowers,
                        "ground_pct": ground,
                        "water_pct": water,
                    }

                    candidates.append(
                        normalize_composition(candidate)
                    )

    return candidates


# ============================================================
# Target beauty optimisation
# ============================================================

def find_best_design_for_target(
    model,
    feature_cols,
    target_score,
    water_allowed=True,
    step=5.0,
):
    """
    Search candidate feature compositions and find the one that:
      1. reaches the target score if possible
      2. otherwise gets as close as possible
      3. prefers simpler designs when scores are tied
    """

    target_score = float(np.clip(target_score, 1.0, 5.0))

    candidates = generate_candidate_compositions(
        water_allowed=water_allowed,
        step=step,
    )

    results = []

    for composition in candidates:

        score = predict_beauty(
            model=model,
            feature_cols=feature_cols,
            feature_values=composition,
        )

        distance = abs(score - target_score)

        # Complexity is used only as a tie-breaker.
        complexity = (
            composition["trees_pct"] +
            composition["flowers_pct"] +
            composition["water_pct"]
        )

        results.append({
            "composition": composition,
            "predicted_score": score,
            "distance": distance,
            "complexity": complexity,
        })

    if not results:
        raise RuntimeError("No valid design candidates were generated.")

    # Prefer designs that meet/exceed target.
    above_target = [
        r for r in results
        if r["predicted_score"] >= target_score
    ]

    if above_target:

        # Smallest score above target, then simplest design.
        best = sorted(
            above_target,
            key=lambda r: (
                r["predicted_score"] - target_score,
                r["complexity"],
            )
        )[0]

    else:

        # If the model cannot reach the target,
        # return the highest predicted design.
        best = sorted(
            results,
            key=lambda r: (
                -r["predicted_score"],
                r["complexity"],
            )
        )[0]

    return {
        **best,
        "target_score": target_score,
        "target_reached": (
            best["predicted_score"] >= target_score
        ),
        "searched_candidates": len(results),
    }


# ============================================================
# Layout planner
# ============================================================

def build_design_zones(
    unified_width,
    unified_length,
    composition,
    tier="medium",
):
    """
    Create conceptual design zones.

    Coordinates are relative to the unified rectangular bounding box.
    This is a planning representation, not a CAD drawing.

    تحديث: بدل منطقة "أشجار" مسطحة واحدة تغطي كل الحديقة تقريبًا، نبني
    طبقتين (حدّية للأشجار الطويلة، وداخلية للشجيرات) — تجسيدًا مبسَّطًا
    لفكرة "الطبقات النباتية" (أشجار خلفية → شجيرات → نباتات منخفضة →
    زهور) المقترحة. كل منطقة تحمل أيضًا مفتاح "room" اختياريًا (تصنيف
    Garden Rooms) للاستخدام المستقبلي دون كسر التوافق مع app.py الحالي
    (الذي يقرأ فقط type/label_ar/x/y/width/length).
    """

    composition = normalize_composition(composition)

    W = float(unified_width)
    L = float(unified_length)
    area = W * L

    # بوابة الجدوى الفيزيائية: حديقة صغيرة (< MULTI_ROOM_MIN_AREA_M2) لا
    # تُقسَّم إلى Garden Rooms منفصلة اسميًا — تبقى تصميمًا متكاملاً واحدًا،
    # بدل الإيحاء بوجود "غرف" متعددة داخل مساحة لا تتّسع فعليًا لذلك.
    rooms_allowed = space_allows_multi_room({"area": area})
    space_ok_flagship = space_allows_flagship_water({"area": area, "width": W, "length": L})

    def _room(name):
        return name if rooms_allowed else None

    zones = []

    # --------------------------------------------------------
    # الطبقة الخارجية: أشجار خلفية (حدّية، على طول محيط الحديقة)
    # --------------------------------------------------------
    zones.append({
        "type": "trees_border",
        "label_ar": "أشجار خلفية على المحيط",
        "x": W * 0.02,
        "y": L * 0.02,
        "width": W * 0.96,
        "length": L * 0.96,
        "room": _room("Quiet Green Zone"),
    })

    # --------------------------------------------------------
    # الطبقة الوسطى: شجيرات وأشجار متوسطة (حلقة داخلية أضيق)
    # --------------------------------------------------------
    zones.append({
        "type": "shrubs_mid",
        "label_ar": "شجيرات وأشجار متوسطة",
        "x": W * 0.10,
        "y": L * 0.10,
        "width": W * 0.80,
        "length": L * 0.80,
        "room": _room("Quiet Green Zone"),
    })

    # --------------------------------------------------------
    # المسطح الأخضر الرئيسي (المركز)
    # --------------------------------------------------------
    zones.append({
        "type": "grass",
        "label_ar": "المسطح الأخضر الرئيسي",
        "x": W * 0.22,
        "y": L * 0.22,
        "width": W * 0.56,
        "length": L * 0.56,
        "room": _room("Quiet Green Zone"),
    })

    # --------------------------------------------------------
    # أحواض الزهور — بنمط وصفي حسب النسبة (وليس تسمية ثابتة دائمًا)
    # --------------------------------------------------------
    if composition["flowers_pct"] > 0.1:
        flower_style = describe_flower_style(
            composition["flowers_pct"], composition.get("trees_pct", 0.0)
        )
        zones.append({
            "type": "flowers",
            "label_ar": flower_style or "أحواض الزهور",
            "x": W * 0.12,
            "y": L * 0.08,
            "width": W * 0.76,
            "length": L * 0.12,
            "room": _room("Floral Zone"),
        })

    # --------------------------------------------------------
    # عنصر مائي — بتسمية وصفية حسب الحجم والمستوى (وليس "بركة" دائمًا)،
    # مع بوابة الجدوى الفيزيائية للعنصر الأبرز (نافورة راقصة موسيقية).
    # --------------------------------------------------------
    if composition["water_pct"] > 0.1:
        side = min(W, L) * 0.15
        water_name = describe_water_feature(
            composition["water_pct"], tier=tier, space_ok=space_ok_flagship
        )

        zones.append({
            "type": "water",
            "label_ar": water_name or "عنصر مائي مركزي",
            "x": (W - side) / 2.0,
            "y": (L - side) / 2.0,
            "width": side,
            "length": side,
            "room": _room("Water Zone"),
        })

    # --------------------------------------------------------
    # الممرات الرئيسية
    # --------------------------------------------------------
    zones.append({
        "type": "paths",
        "label_ar": "الممرات الرئيسية",
        "x": W * 0.05,
        "y": (L * 0.50) - (L * 0.04),
        "width": W * 0.90,
        "length": L * 0.08,
        "room": _room("Resting Zone"),
    })

    return zones


# ============================================================
# Design report
# ============================================================

def _largest_elements(composition, top_n=3):
    ordered = sorted(
        composition.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return ordered[:top_n]


def generate_design_report(
    geometry,
    design_result,
    tier="medium",
    existing_compositions=None,
):
    """
    Generate an Arabic explanatory report.

    tier: "economic" | "balanced"/"medium" | "ideal"/"open" — يُستخدَم فقط
    لتسمية العنصر المائي وصفيًا (انظر describe_water_feature) ولإضافة
    ملاحظة الإضاءة الاختيارية عند "ideal".

    existing_compositions: قائمة اختيارية من قواميس (تركيبة حالية لكل
    حديقة مُدخَلة، إن أدخلها المستخدم) — إن وُجدت، يُضاف "محرك القواعد"
    (جمل مخصَّصة حسب نقص كل حديقة فعليًا)، بدل الاكتفاء بوصف التصميم
    النهائي فقط.
    """

    composition = design_result["composition"]
    predicted = design_result["predicted_score"]
    target = design_result["target_score"]
    reached = design_result["target_reached"]

    areas = composition_to_area(
        composition,
        geometry["area"],
    )

    rooms_allowed = space_allows_multi_room(geometry)
    space_ok_flagship = space_allows_flagship_water(geometry)

    largest = _largest_elements(composition, top_n=3)

    main_names = [
        AR_FEATURE_NAMES.get(key, key)
        for key, _ in largest
    ]

    lines = []

    lines.append(
        f"تم دمج الحدائق في مساحة موحدة مقدارها "
        f"{geometry['area']:.2f} م² ضمن أبعاد كلية تقارب "
        f"{geometry['width']:.2f} × {geometry['length']:.2f} متر."
    )

    lines.append(
        f"استهدف التصميم الوصول إلى درجة جمال مقدارها "
        f"{target:.2f} من 5."
    )

    lines.append(
        "اعتمد البحث على اختبار عدد من التركيبات الممكنة "
        "لنسب العناصر باستخدام نموذج MLP المدرب في المنظومة."
    )

    # --------------------------------------------------------
    # محرك القواعد: جمل مخصَّصة حسب نقص كل حديقة فعليًا (إن أُدخلت
    # تركيبتها الحالية) — بدل قالب عام فقط.
    # --------------------------------------------------------
    if existing_compositions:
        for idx, comp in enumerate(existing_compositions, 1):
            if not comp:
                continue
            garden_lines = deficiency_rule_sentences_ar(comp, scenario_key=tier)
            if garden_lines:
                lines.append(f"بخصوص الحديقة رقم {idx} تحديدًا:")
                lines.extend(garden_lines)

    lines.append(
        f"تشكل العناصر الرئيسية في التصميم المقترح "
        f"{'، '.join(main_names)}، بهدف تحقيق تنوع وتوازن بصري مناسبين."
    )

    if composition["grass_pct"] > 0:
        lines.append(
            f"يوصى بتخصيص نحو {areas['grass_pct']:.1f} م² "
            f"للمسطحات الخضراء."
        )

    if composition["trees_pct"] > 0:
        lines.append(
            f"وحوالي {areas['trees_pct']:.1f} م² "
            f"للأشجار والشجيرات، موزَّعة على طبقتين (خلفية ووسطى) "
            f"لعمق بصري أفضل."
        )

    if composition["flowers_pct"] > 0:
        flower_style = describe_flower_style(
            composition["flowers_pct"], composition.get("trees_pct", 0.0)
        )
        style_txt = f" بأسلوب {flower_style}" if flower_style else ""
        lines.append(
            f"مع تخصيص قرابة {areas['flowers_pct']:.1f} م² "
            f"لأحواض الزهور{style_txt} لزيادة التنوع اللوني والحيوية البصرية."
        )

    if composition["water_pct"] > 0:
        water_name = describe_water_feature(
            composition["water_pct"], tier=tier, space_ok=space_ok_flagship
        )
        water_txt = water_name or "عنصر مائي"
        lines.append(
            f"كما يقترح التصميم تخصيص نحو {areas['water_pct']:.1f} م² "
            f"لـ{water_txt} يعمل كنقطة تركيز بصرية."
        )
        if tier in ("ideal", "open", "high") and not space_ok_flagship:
            lines.append(
                f"🔎 ملاحظة: مساحة الحديقة الموحَّدة الحالية ({geometry['area']:.0f} م²) "
                f"أقل من الحد الأدنى المفترَض لعنصر مائي بارز كنافورة راقصة "
                f"موسيقية (~{FLAGSHIP_WATER_MIN_AREA_M2:.0f} م² وبُعد أدنى "
                f"{FLAGSHIP_WATER_MIN_DIM_M:.0f} م)، لذلك اقتُرح عنصر مائي أصغر "
                f"أكثر واقعية لهذه المساحة."
            )

    if not rooms_allowed:
        lines.append(
            f"🔎 ملاحظة: بمساحة موحَّدة قدرها {geometry['area']:.0f} م² (أقل من "
            f"{MULTI_ROOM_MIN_AREA_M2:.0f} م²)، لم يُقترَح تقسيم الحديقة إلى "
            f"مناطق (Garden Rooms) منفصلة — يبقى التصميم متكاملاً كمساحة واحدة "
            f"متجانسة بدل الإيحاء بغرف متعددة لا تتّسع لها المساحة الفعلية."
        )

    if reached:
        lines.append(
            f"أظهر النموذج أن التصميم يحقق درجة متوقعة قدرها "
            f"{predicted:.2f} من 5، وهي عند أو أعلى من الهدف المحدد."
        )
    else:
        lines.append(
            f"أعلى درجة وجدها البحث ضمن القيود الحالية هي "
            f"{predicted:.2f} من 5، وهي أقل من الهدف المطلوب، "
            f"لذلك قد يلزم توسيع خيارات التصميم أو تعديل الهدف."
        )

    # --------------------------------------------------------
    # إثراءات وصفية غير مُقيَّمة من النموذج (إضاءة/جلوس/حواف) — نفس مكتبة
    # مرحلة 8 (NARRATIVE_ENRICHMENTS) بدل نسخة محلية كانت تقتصر على
    # الإضاءة فقط ولـ"ideal" حصرًا. الآن متوازن أيضًا يحصل على إثراء خفيف،
    # ومُوسَّمة بصراحة أنها اقتراحات تصميمية لم يُقيِّمها النموذج (لا توجد
    # سمات كالإضاءة أو الجلوس في بيانات التدريب)، تفاديًا لأي إيحاء
    # مضلِّل بأن الذكاء الاصطناعي "قيّم" هذا الجانب.
    # --------------------------------------------------------
    bs_tier = _tier_to_bs_key(tier)
    has_water_final = composition.get("water_pct", 0.0) > 0.5
    enrich_pool = _enrichment_pool(bs_tier, has_water=has_water_final)
    if enrich_pool and reached:
        import random as _rnd
        seed_key = f"enrich|{bs_tier}|{predicted:.4f}|{geometry['area']:.1f}"
        n_enrich = 1 if bs_tier == "medium" else 2
        rng = _rnd.Random(seed_key)
        pool = list(enrich_pool)
        rng.shuffle(pool)
        for e in pool[:n_enrich]:
            lines.append(f"✨ {e}")
        lines.append(NARRATIVE_DISCLAIMER_AR)

    return lines


# ============================================================
# Main integrated function
# ============================================================

def create_unified_garden_design(
    gardens,
    layout,
    model,
    feature_cols,
    target_score,
    water_allowed=True,
    step=5.0,
):
    """
    Main function used by Streamlit.

    ملاحظة: لا يوجد مفهوم "مستوى ثابت" (اقتصادي/متوازن/مثالي) هنا كما في
    مخطط التحسين (مرحلة 8) — هذه الدالة تبحث عن تركيبة واحدة تحقق
    target_score. نستنتج "مستوى وصفي" تقريبي من قيمة target_score نفسها
    (فقط لأغراض تسمية العنصر المائي وصفيًا وإظهار ملاحظة الإضاءة)، وليس
    كتصنيف رسمي مطابق لمرحلة 8.

    gardens: كل عنصر قد يحمل اختياريًا "existing_composition" (قاموس
    grass_pct/trees_pct/flowers_pct/ground_pct/water_pct لحالة الحديقة
    الفعلية الحالية) — إن وُجد، يُفعَّل "محرك القواعد" لتلك الحديقة في
    التقرير النهائي.
    """

    if target_score < 3.0:
        tier = "economic"
    elif target_score < 4.0:
        tier = "balanced"
    else:
        tier = "ideal"

    existing_compositions = [
        g.get("existing_composition") for g in gardens
        if isinstance(g, dict) and g.get("existing_composition")
    ]

    geometry = build_garden_geometry(
        gardens=gardens,
        layout=layout,
    )

    design_result = find_best_design_for_target(
        model=model,
        feature_cols=feature_cols,
        target_score=target_score,
        water_allowed=water_allowed,
        step=step,
    )

    composition = design_result["composition"]

    areas = composition_to_area(
        composition=composition,
        total_area=geometry["area"],
    )

    zones = build_design_zones(
        unified_width=geometry["width"],
        unified_length=geometry["length"],
        composition=composition,
        tier=tier,
    )

    report_lines = generate_design_report(
        geometry=geometry,
        design_result=design_result,
        tier=tier,
        existing_compositions=existing_compositions,
    )

    area_table = pd.DataFrame([
        {
            "feature": key,
            "element_ar": AR_FEATURE_NAMES[key],
            "percentage": composition[key],
            "area_m2": areas[key],
        }
        for key in TARGET_FEATURES
    ])

    return {
        "geometry": geometry,
        "design": design_result,
        "areas": areas,
        "zones": zones,
        "area_table": area_table,
        "report_lines": report_lines,
    }