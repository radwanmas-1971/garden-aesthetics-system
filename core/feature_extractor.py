# core/feature_extractor.py
import os
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from skimage.morphology import disk, dilation

MODEL_NAME_DEFAULT = "nvidia/segformer-b2-finetuned-ade-512-512"

ADE20K_CLASSES = [
    'wall','building','sky','floor','tree','ceiling','road','bed','windowpane','grass',
    'cabinet','sidewalk','person','earth','door','table','mountain','plant','curtain','chair',
    'car','water','painting','sofa','shelf','house','sea','mirror','rug','field',
    'armchair','seat','fence','desk','rock','wardrobe','lamp','bathtub','railing','cushion',
    'base','box','column','signboard','chest of drawers','counter','sand','sink','skyscraper','fireplace',
    'refrigerator','grandstand','path','stairs','runway','case','pool table','pillow','screen door','stairway',
    'river','bridge','bookcase','blind','coffee table','toilet','flower','book','hill','bench',
    'countertop','stove','palm','kitchen island','computer','swivel chair','boat','bar','arcade machine','hovel',
    'bus','towel','light','truck','tower','chandelier','awning','streetlight','booth','television receiver',
    'airplane','dirt track','apparel','pole','land','bannister','escalator','ottoman','bottle','buffet',
    'poster','stage','van','ship','fountain','conveyer belt','canopy','washer','plaything','swimming pool',
    'stool','barrel','basket','waterfall','tent','bag','minibike','cradle','oven','ball',
    'food','step','tank','trade name','microwave','pot','animal','bicycle','lake','dishwasher',
    'screen','blanket','sculpture','hood','sconce','vase','traffic light','tray','ashcan','fan',
    'pier','crt screen','plate','monitor','bulletin board','shower','radiator','glass','clock','flag'
]

TARGET_LABELS = {
    "grass":   ["grass"],
    "trees":   ["tree", "palm", "plant"],
    "flowers": ["flower"],
    "water":   ["water", "sea", "lake", "river", "swimming pool", "waterfall", "fountain"],
    "ground":  ["earth", "land", "field", "sand", "dirt track", "floor",
                "road", "sidewalk", "path", "runway", "rock"]
}

EXCLUDE_FROM_DENOM_DEFAULT = ["sky", "building", "wall"]

COLORS = {
    "ground":  [255, 255, 0],    # yellow
    "grass":   [0, 200, 0],      # green
    "trees":   [0, 120, 0],      # dark green
    "flowers": [200, 0, 200],    # purple
    "water":   [0, 0, 255],      # blue
}

# =====================================================================
# تصحيحات مخصَّصة للتصوير الجوي (aerial / drone / nadir)
# =====================================================================
# نموذج SegFormer-ADE20K مُدرَّب أساسًا على مشاهد بمنظور أرضي (شوارع/مبانٍ
# من مستوى الإنسان)، وليس مناظر جوية من الأعلى. هذا يسبب أخطاء تصنيف
# منهجية متكررة عند التصوير الجوي تحديدًا (الترقيم يطابق خطوات
# apply_aerial_corrections فعليًا):
#   1-2) عشب جاف/مصفرّ يُصنَّف "أرض" أو "مبنى"                (يعمل دائمًا)
#   3)   نوافير/مسطحات مائية صغيرة تُصنَّف فئات متنوعة (ظل داكن أو غيره) —
#        بلا قائمة بيضاء لمصدر الفئة، عتبة لونية صارمة فقط    (جوي فقط)
#   4)   سياج/أشجار محيطية كثيفة الظل تُصنَّف "حائط"           (جوي فقط)
#   5)   ممرات مرصوفة فاتحة اللون تُصنَّف "حائط/مبنى"          (جوي فقط)
#
# تصحيحات 1-2 و4-5 تستهدف فقط الفئات القابلة للمراجعة (soft ground /
# building / wall) ولا تمسّ road/sidewalk/path/floor/runway/rock (تبقى
# ground دومًا) ولا trees/grass الصحيحة أصلًا. تصحيح 3 (الماء) وحده بلا
# قيد على فئة المصدر — انظر التعليق عند تطبيقه للتفصيل والسبب.
# =====================================================================
SOFT_GROUND_CLASS_NAMES = ["earth", "field", "land", "sand", "dirt track"]
REVIEWABLE_EXCLUDE_CLASS_NAMES = ["building", "wall"]

# تفعيل/تعطيل كل التصحيحات الجوية بسهولة من مكان واحد
ENABLE_AERIAL_CORRECTIONS = True


def grass_hsv_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    قناع للبكسلات التي لونها يشبه عشبًا (أخضر إلى أخضر-مصفرّ/زيتوني،
    يغطي أيضًا العشب الجاف في الخريف/الصيف). سطوع متوسط إلى مرتفع فقط
    (v>=40) — العشب المضاء بوضوح، وليس الظل الداكن (انظر tree_dark_hsv_mask).
    image_rgb: مصفوفة uint8 بصيغة RGB.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (h >= 20) & (h <= 95) & (s >= 35) & (v >= 40) & (v <= 245)
    return mask


def tree_dark_hsv_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    قناع أخضر داكن/مظلَّل — نطاق لوني مشابه للعشب لكن بسطوع أقل بكثير
    (v<40، بخلاف grass_hsv_mask الذي يتطلب v>=40). يستهدف تحديدًا الأسيجة
    والأشجار الكثيفة المحيطية التي تُظهر تظليلًا ذاتيًا قويًا عند التصوير
    الجوي المباشر من الأعلى، وتُخطئ نماذج مُدرَّبة على مشاهد أرضية بتصنيفها
    "حائط" بدل "أشجار".
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (h >= 25) & (h <= 100) & (s >= 15) & (v >= 8) & (v < 40)
    return mask


def pavement_gray_hsv_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    قناع رمادي/بيج فاتح منخفض إلى متوسط التشبع اللوني — يستهدف الممرات
    والأرصفة المرصوفة (حجر بلاط رمادي محايد، أو بيج/تان دافئ) التي قد
    تُصنَّف خطأً "مبنى/حائط" عند التصوير الجوي، بدل "أرض مكشوفة".
    عتبة التشبع (s<=50) مُعايَرة على قياس فعلي لرصيف بيج حقيقي (تشبع
    22-45 عبر عيّنات متعددة) — العتبة القديمة (s<=30) كانت تُفوّت جزءًا
    كبيرًا منه.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (s <= 50) & (v >= 90) & (v <= 235)
    return mask


def water_hsv_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    قناع فيروزي/أزرق مميّز للمسطحات المائية والنوافير عند التصوير الجوي.
    تشبع لوني مرتفع (s>=150) — مُعاير تجريبيًا على عينة حقيقية (4 نوافير
    مؤكدة مقابل 20 صورة بلا ماء): ظل الأشجار الداكن على حواف الصور له
    نفس نطاق اللون تقريبًا (h 85-135) لكن تشبعًا أقل بكثير (~85 مقابل
    135-208 للماء الحقيقي)، فرفع العتبة يفصل الحالتين بدقة تامة تقريبًا.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (h >= 85) & (h <= 135) & (s >= 150) & (v >= 40)
    return mask


def flower_hsv_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    قناع للألوان الزاهية غير الخضراء وغير الزرقاء (وردي/أرجواني/أحمر) —
    يستهدف أزهارًا ملوَّنة قد لا يكتشفها النموذج إطلاقًا كفئة "flower".
    تشبع لوني مرتفع (s>=180) — مُعاير تجريبيًا: المقاعد الخشبية البنية
    (لون دافئ مشابه) لها تشبع أقل بكثير (~100-150) من الزهور الزاهية
    الحقيقية (~180-255)؛ رفع العتبة يفصل الحالتين بدقة (اختُبر على 4 صور
    مؤكَّد خلوّها من الزهور مقابل 5 صور مؤكَّد وجود زهور فيها).
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((h < 15) | (h > 140)) & (s >= 180) & (v >= 60)
    return mask


def apply_aerial_corrections(pred: np.ndarray, image_rgb: np.ndarray, is_aerial: bool = False):
    """
    يطبّق تصحيحات التصنيف على مصفوفة pred الخام (قبل حساب أي نسب).

    - الخطوتان 1-2 (عشب): تُطبَّقان دائمًا، بغض النظر عن is_aerial — آمنتان
      لأي زاوية تصوير.

    - الخطوات 3-5 (ماء، سياج/أشجار داكنة، ممرات رمادية): تُطبَّق فقط إن
      كان is_aerial=True، لأنها تفترض ضمنًا أن أي "مبنى/حائط" مكتشَف هو
      خطأ تصنيف — صحيح للصور الجوية لحدائق، خاطئ وخطير للصور الجانبية
      حيث المباني/الجدران حقيقية.

    ملاحظة: تصحيح الزهور اللوني لم يعد جزءًا من هذه الدالة — يُطبَّق بمعزل
    تام في analyze_one_image كطبقة إضافية بعد التمدد، وليس قبله (انظر
    التعليق أدناه لسبب هذا الفصل).

    يعيد (pred المصحَّحة، إحصاءات كل خطوة تصحيح).
    """
    grass_id = names_to_ids(["grass"])[0]
    tree_id = names_to_ids(["tree"])[0]
    path_id = names_to_ids(["path"])[0]
    water_id = names_to_ids(["water"])[0]

    soft_ground_ids = names_to_ids(SOFT_GROUND_CLASS_NAMES)
    reviewable_exclude_ids = names_to_ids(REVIEWABLE_EXCLUDE_CLASS_NAMES)  # building, wall
    trees_source_ids = names_to_ids(["tree", "palm", "plant"])
    all_ground_ids = names_to_ids(TARGET_LABELS["ground"])

    grass_mask = grass_hsv_mask(image_rgb)

    corrected = pred.copy()
    stats = {
        "from_ground_to_grass_px": 0,
        "from_building_wall_to_grass_px": 0,
        "from_building_wall_trees_ground_to_water_px": 0,
        "from_building_wall_to_trees_px": 0,
        "from_building_wall_to_ground_px": 0,
        "removed_small_buildings": 0,
        "removed_small_walls": 0,
        "border_wall_removed": 0,
    }

    # 1) (يعمل دومًا) أرض رخوة ذات لون عشب واضح → عشب
    m = np.isin(corrected, soft_ground_ids) & grass_mask
    corrected[m] = grass_id
    stats["from_ground_to_grass_px"] = int(m.sum())

    # 2) (يعمل دومًا) مبنى/حائط ذو لون عشب واضح → عشب
    m = np.isin(corrected, reviewable_exclude_ids) & grass_mask
    corrected[m] = grass_id
    stats["from_building_wall_to_grass_px"] = int(m.sum())

    # ملاحظة: تصحيح الزهور اللوني لم يعد يُطبَّق هنا (على pred مباشرة) —
    # نُفِّذ بمعزل تام في analyze_one_image كطبقة إضافية منفصلة، لتفادي
    # تغذية آلية expand_flowers_with_foliage (التمدد بنصف قطر 18 بكسل)
    # ببذور متفرقة (كبكسل مقعد خشبي واحد)، والتي كانت تُضخِّم flowers_pct
    # بشكل غير متناسب مع اللون الفعلي، وقد تبتلع بكسلات ماء مجاورة خطأً.

    if is_aerial:
        tree_dark_mask = tree_dark_hsv_mask(image_rgb)
        pavement_mask = pavement_gray_hsv_mask(image_rgb)
        water_mask = water_hsv_mask(image_rgb)
        flower_id_protect = names_to_ids(["flower"])[0]

        # 3) (جوي فقط) — بلا قائمة بيضاء لمصدر الفئة. جُرِّبت قائمة محدَّدة
        #    (مبنى/حائط/أشجار/أرض/عشب) سابقًا، لكن بعض النوافير الحقيقية
        #    ظلت غير مُكتشَفة (فحص مباشر أثبت وجود تطابق لوني قوي في
        #    مناطقها، لكن بفئة مصدر لم تكن ضمن القائمة ولم نستطع تحديدها
        #    يقينًا). الحل: الاعتماد كليًا على العتبة اللونية الصارمة
        #    (s>=150، مُختبَرة سابقًا على فصل الماء الحقيقي عن ظل الأشجار
        #    بدقة) كحارس وحيد، بدل تخمين فئة المصدر. يُستثنى فقط "زهور"
        #    (نطاقا اللون متنافيان أصلًا فلا خطر عمليًا، لكن استبعاد صريح
        #    للسلامة) — أي بكسل آخر مطابق للعتبة الصارمة يُصبح ماءً.
        m = water_mask & (corrected != flower_id_protect)
        corrected[m] = water_id
        stats["from_building_wall_trees_ground_to_water_px"] = int(m.sum())


        # 4) (جوي فقط) مبنى/حائط داكن اللون أخضر (سياج/أشجار محيطية مُظلَّلة) → أشجار
        m = np.isin(corrected, reviewable_exclude_ids) & tree_dark_mask
        corrected[m] = tree_id
        stats["from_building_wall_to_trees_px"] = int(m.sum())

        # 5) (جوي فقط) مبنى/حائط رمادي فاتح (ممرات مرصوفة) → أرض
        m = np.isin(corrected, reviewable_exclude_ids) & pavement_mask
        corrected[m] = path_id
        stats["from_building_wall_to_ground_px"] = int(m.sum())

        # --------------------------------------------------
        # 6) تنظيف مورفولوجي: إزالة كتل "مبنى" الصغيرة (ضجيج) → أرض
        # --------------------------------------------------
        # مكمِّل للتصحيحات اللونية أعلاه: أي كتلة متصلة من "مبنى" مساحتها
        # أقل من 300 بكسل على الأرجح ضجيج تصنيف (لا مبنى حقيقي بهذا الحجم
        # الصغير)، فتُعاد لفئة "أرض" (path_id) بدل حائط. ملاحظة تسمية
        # مهمة: cv2.connectedComponentsWithStats تُعيد مصفوفة NumPy تحمل
        # نفس اسم "stats" — استُخدم هنا اسم مختلف (cc_stats_building)
        # صراحة لتفادي استبدال قاموس stats الذي يجمع كل إحصاءات الدالة.
        building_id = names_to_ids(["building"])[0]
        building_mask = (corrected == building_id)
        num_labels, labels, cc_stats_building, _ = cv2.connectedComponentsWithStats(
            building_mask.astype(np.uint8), connectivity=8
        )
        removed_buildings = 0
        for i in range(1, num_labels):
            area = cc_stats_building[i, cv2.CC_STAT_AREA]
            if area < 300:
                corrected[labels == i] = path_id
                removed_buildings += area
        stats["removed_small_buildings"] = int(removed_buildings)

        # --------------------------------------------------
        # 7) تنظيف مورفولوجي: إزالة كتل "حائط" الصغيرة (ضجيج) → أشجار
        # --------------------------------------------------
        wall_id = names_to_ids(["wall"])[0]
        wall_mask = (corrected == wall_id)
        num_labels, labels, cc_stats_wall, _ = cv2.connectedComponentsWithStats(
            wall_mask.astype(np.uint8), connectivity=8
        )
        removed_walls = 0
        for i in range(1, num_labels):
            area = cc_stats_wall[i, cv2.CC_STAT_AREA]
            if area < 400:
                corrected[labels == i] = tree_id
                removed_walls += area
        stats["removed_small_walls"] = int(removed_walls)

        # --------------------------------------------------
        # 8) إزالة "حائط" الملاصق لحواف الصورة (غالبًا قصّ/إطار وليس مبنى)
        # --------------------------------------------------
        border = np.zeros_like(corrected, dtype=bool)
        border[:20, :] = True
        border[-20:, :] = True
        border[:, :20] = True
        border[:, -20:] = True
        wall_mask = (corrected == wall_id)
        border_wall = wall_mask & border
        corrected[border_wall] = tree_id
        stats["border_wall_removed"] = int(border_wall.sum())

    stats["total_px"] = int(pred.size)
    return corrected, stats


def names_to_ids(names):
    s = set(names)
    return [i for i, n in enumerate(ADE20K_CLASSES) if n in s]

def crop_pil(image: Image.Image, crop_top_ratio: float) -> Image.Image:
    w, h = image.size
    top = int(h * crop_top_ratio)
    top = max(0, min(top, h-1))
    return image.crop((0, top, w, h))

def expand_flowers_with_foliage(pred, valid_mask, flower_ids, foliage_ids, radius: int):
    seed = (np.isin(pred, flower_ids) & valid_mask)
    if seed.sum() == 0:
        return seed
    selem = disk(max(1, radius))
    dil = dilation(seed, selem)
    foliage = (np.isin(pred, foliage_ids) & valid_mask)
    return seed | (dil & foliage)

def load_model(model_name=MODEL_NAME_DEFAULT, device="cpu"):
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model

def analyze_one_image(
    image_path: str,
    processor,
    model,
    out_dir: str,
    crop_top_ratio: float = 0.35,
    flower_expand_radius: int = 18,
    disable_water: bool = False,
    is_aerial: bool = False,
    exclude_from_denom = None,
    device="cpu"
):
    if exclude_from_denom is None:
        exclude_from_denom = EXCLUDE_FROM_DENOM_DEFAULT

    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert("RGB")
    # لا نعمل قص للصورة نهائيًا (حتى يكون overlay بنفس حجم الأصل)
      # image = crop_pil(image, crop_top_ratio=crop_top_ratio)


    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    up = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred = up.argmax(dim=1)[0].detach().cpu().numpy()

    # --- تصحيحات التصنيف (عشب دائمًا، وماء/سياج/ممرات فقط إن is_aerial=True) ---
    correction_stats = None
    if ENABLE_AERIAL_CORRECTIONS:
        pred, correction_stats = apply_aerial_corrections(pred, np.array(image), is_aerial=is_aerial)

    total_px = pred.size

    sky_id = names_to_ids(["sky"])
    building_id = names_to_ids(["building"])
    wall_id = names_to_ids(["wall"])

    sky_pct = round((np.isin(pred, sky_id).sum() * 100.0) / total_px, 2)
    building_pct = round((np.isin(pred, building_id).sum() * 100.0) / total_px, 2)
    wall_pct = round((np.isin(pred, wall_id).sum() * 100.0) / total_px, 2)

    exclude_ids = names_to_ids(exclude_from_denom)
    valid_mask = ~np.isin(pred, exclude_ids)
    denom = valid_mask.sum() if valid_mask.sum() > 0 else pred.size

    # flowers expansion (الآلية الأصلية: تمدد من بذور "flower" الأصلية
    # للنموذج فقط — بلا أي تغيير عن السابق)
    flower_ids = names_to_ids(["flower"])
    foliage_ids = names_to_ids(["plant", "grass"])
    flowers_expanded = expand_flowers_with_foliage(
        pred, valid_mask, flower_ids, foliage_ids, radius=flower_expand_radius
    )

    # طبقة إضافية منفصلة تمامًا: بكسلات بلون زاهٍ مميّز (وردي/أرجواني/أحمر)
    # لم يكتشفها النموذج كـ"flower" أصلًا، فلم تُشارك في أي بذرة للتمدد
    # أعلاه. تُضاف مباشرة (بلا أي تمدد إضافي) لتفادي التضخيم غير المتناسب.
    #
    # مُقيَّدة بفئات مصدر محدَّدة (وليس كل الصورة) — إغفال هذا القيد سابقًا
    # كان يسمح لفئات مشروعة تمامًا (مثل "مقعد" bench بلونه البني الدافئ)
    # بالتسرّب خطأً إلى "زهور". نفس فئات المصدر المستخدمة في تصحيحات أخرى:
    # أشجار/نباتات، مبنى/حائط، كل فئات الأرض، وعشب.
    flower_source_ids = (
        names_to_ids(["tree", "palm", "plant"])
        + names_to_ids(REVIEWABLE_EXCLUDE_CLASS_NAMES)
        + names_to_ids(TARGET_LABELS["ground"])
        + names_to_ids(["grass"])
    )
    flower_color_extra = (
        flower_hsv_mask(np.array(image))
        & valid_mask
        & (~flowers_expanded)
        & np.isin(pred, flower_source_ids)
    )
    flowers_final = flowers_expanded | flower_color_extra
    flower_color_extra_px = int(flower_color_extra.sum())

    results = {}
    cat_ids = {}

    for cat, names in TARGET_LABELS.items():
        ids = names_to_ids(names)
        cat_ids[cat] = ids

        if cat == "flowers":
            m = flowers_final
        else:
            m = (np.isin(pred, ids) & valid_mask & (~flowers_final))

        results[cat] = round(m.sum() * 100.0 / denom, 2)

    if disable_water:
        results["water"] = 0.0

    # overlay image
    h, w = pred.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    def paint(cat, mask, color):
        overlay[mask] = color

    # build masks again for overlay (visual)
    masks = {}
    for cat in ["ground", "grass", "trees", "flowers"]:
        if cat == "flowers":
            masks[cat] = flowers_final
        else:
            ids = cat_ids[cat]
            masks[cat] = (np.isin(pred, ids) & valid_mask & (~flowers_final))

    # water mask للعرض المرئي (يُرسم فقط إن لم يكن معطَّلًا)
    if not disable_water:
        water_ids = cat_ids.get("water", names_to_ids(TARGET_LABELS["water"]))
        masks["water"] = (np.isin(pred, water_ids) & valid_mask & (~flowers_final))

    paint("ground", masks["ground"], COLORS["ground"])
    paint("grass", masks["grass"], COLORS["grass"])
    paint("trees", masks["trees"], COLORS["trees"])
    paint("flowers", masks["flowers"], COLORS["flowers"])
    if not disable_water:
        paint("water", masks["water"], COLORS["water"])

    base = np.array(image).astype(np.float32)
    mask_rgb = overlay.astype(np.float32)
    alpha = 0.45
    blended = (base * (1 - alpha) + mask_rgb * alpha).clip(0, 255).astype(np.uint8)

    overlay_path = os.path.join(out_dir, f"{base_name}_overlay.png")
    plt.imsave(overlay_path, blended)


    results["sky"] = sky_pct
    results["building"] = building_pct
    results["wall"] = wall_pct

    # يُسجَّل دائمًا (ليس فقط عند is_aerial) لأنه طبقة مستقلة تعمل في كل الحالات
    results["_corr_flower_color_extra_px"] = flower_color_extra_px

    if correction_stats is not None:
        results["_corr_from_ground_to_grass"] = correction_stats["from_ground_to_grass_px"]
        results["_corr_from_building_wall_to_grass"] = correction_stats["from_building_wall_to_grass_px"]
        results["_corr_from_building_wall_trees_ground_to_water"] = correction_stats["from_building_wall_trees_ground_to_water_px"]
        results["_corr_from_building_wall_to_trees"] = correction_stats["from_building_wall_to_trees_px"]
        results["_corr_from_building_wall_to_ground"] = correction_stats["from_building_wall_to_ground_px"]
        results["_removed_small_buildings"] = correction_stats.get("removed_small_buildings", 0)
        results["_removed_small_walls"] = correction_stats.get("removed_small_walls", 0)
        results["_border_wall_removed"] = correction_stats.get("border_wall_removed", 0)

    return results, overlay_path, image.size

def run_feature_extraction_on_folder(
    image_folder: str,
    outputs_dir: str,
    model_name: str = MODEL_NAME_DEFAULT,
    crop_top_ratio: float = 0.35,
    flower_expand_radius: int = 18,
    disable_water: bool = False,
    is_aerial: bool = False,
    exclude_from_denom=None,
    max_images: int = None,
    device: str = "cpu"
):
    """
    is_aerial: حدِّد True إن كانت كل صور هذا المجلد (الدُفعة) جوية
    (Nadir/Drone) — عندها تُفعَّل تصحيحات الماء/السياج/الممرات الإضافية.
    اترك False للصور الجانبية (منظور أرضي عادي) حيث المباني/الجدران
    الحقيقية موجودة فعلًا ولا يجوز إعادة تصنيفها.

    ملاحظة: هذا إعداد واحد لكل الصور في هذا الاستدعاء (دُفعة واحدة).
    إن كان لديك مزيج من صور جوية وجانبية، شغّل الدالة مرتين منفصلتين
    (مرة لكل مجموعة في مجلدها الخاص) بقيمة is_aerial المناسبة لكل مجموعة.
    """
    os.makedirs(outputs_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png")
    images = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(exts)]
    if max_images is not None:
        images = images[:max_images]

    processor, model = load_model(model_name=model_name, device=device)

    rows = []
    for i, fname in enumerate(images, 1):
        path = os.path.join(image_folder, fname)
        res, overlay_path, _ = analyze_one_image(
            path, processor, model, outputs_dir,
            crop_top_ratio=crop_top_ratio,
            flower_expand_radius=flower_expand_radius,
            disable_water=disable_water,
            is_aerial=is_aerial,
            exclude_from_denom=exclude_from_denom,
            device=device
        )

        row = {"image_name": fname}
        row.update({
            "grass_pct": res["grass"],
            "trees_pct": res["trees"],
            "flowers_pct": res["flowers"],
            "ground_pct": res["ground"],
            "water_pct": res.get("water", 0.0),
            "sky_pct": res.get("sky", 0.0),
            "building_pct": res.get("building", 0.0),
            "wall_pct": res.get("wall", 0.0),
            "is_aerial": is_aerial,
            "corrected_px_from_ground": res.get("_corr_from_ground_to_grass", 0),
            "corrected_px_from_building_wall": res.get("_corr_from_building_wall_to_grass", 0),
            "corrected_px_any_to_flower": res.get("_corr_flower_color_extra_px", 0),
            "corrected_px_building_wall_trees_ground_to_water": res.get("_corr_from_building_wall_trees_ground_to_water", 0),
            "corrected_px_building_wall_to_trees": res.get("_corr_from_building_wall_to_trees", 0),
            "corrected_px_building_wall_to_ground": res.get("_corr_from_building_wall_to_ground", 0),
            "removed_small_buildings_px": res.get("_removed_small_buildings", 0),
            "removed_small_walls_px": res.get("_removed_small_walls", 0),
            "border_wall_removed_px": res.get("_border_wall_removed", 0),
            "overlay_path": overlay_path
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outputs_dir, "results.csv")
    xlsx_path = os.path.join(outputs_dir, "results.xlsx")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)

    return df, csv_path, xlsx_path
