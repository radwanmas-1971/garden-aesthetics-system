import itertools
import random
import numpy as np
import pandas as pd

AR_ACTION = {
    "Add flowers bed": "إضافة أحواض زهور",
    "Improve lawn/soil quality": "تحسين المسطح الأخضر والتربة",
    "Plant trees and shrubs": "زراعة أشجار وشجيرات",
    "Add small water feature": "إضافة عنصر مائي صغير",
    "Add medium water feature": "إضافة عنصر مائي متوسط",
    "Reduce bare ground (mulch/cover)": "تقليل الأرض المكشوفة",
    "Add/Enhance water element": "إضافة/تعزيز عنصر الماء",
    "Increase green spaces": "زيادة المساحات الخضراء",
    "Add shrubs": "إضافة شجيرات",
    "Extensive lawn/soil renovation": "توسيع تجديد المسطح الأخضر والتربة",
    "Additional canopy growth": "زيادة إضافية في الغطاء الشجري",
    "Add water pond/fountain": "إضافة بركة/نافورة ماء",
    "Improve lawn/soil quality (adaptive)": "تحسين المسطح الأخضر والتربة (تكيّفي)",
    "Extensive lawn/soil renovation (adaptive)": "توسيع تجديد المسطح الأخضر والتربة (تكيّفي)",
}

SCENARIO_INFO = {
    "low": {
        "scenario": "Economic Scenario",
        "scenario_ar": "🟢 السيناريو الاقتصادي",
        "description_ar": "مناسب عندما تكون الميزانية محدودة، ويقتصر على إجراءات ذات كلفة منخفضة فقط.",
    },
    "medium": {
        "scenario": "Balanced Scenario",
        "scenario_ar": "🟡 السيناريو المتوازن",
        "description_ar": "مناسب عندما نريد توازنًا بين الكلفة والتحسن، ويسمح بإجراءات منخفضة ومتوسطة الكلفة معًا.",
    },
    "open": {
        "scenario": "Ideal Scenario",
        "scenario_ar": "🔴 السيناريو المثالي",
        "description_ar": "مناسب عندما لا يوجد قيد على الكلفة، ويطبّق أفضل حزمة تحسينات ممكنة بغض النظر عن مستوى كلفتها.",
    },
}

# ============================================================
# تقارير تفسيرية متغيرة للسيناريوهات
# ============================================================


REPORT_VARIANTS_AR = {

    "low": {
        "openings": [
            "يعتمد السيناريو الاقتصادي على تحسينات محدودة ومدروسة تحافظ على انخفاض الكلفة.",
            "يركز السيناريو الاقتصادي على تحقيق أكبر تحسن بصري ممكن باستخدام تدخلات منخفضة الكلفة.",
            "تم توجيه السيناريو الاقتصادي نحو التحسينات البسيطة التي تحقق أثراً جمالياً واضحاً دون رفع مستوى الكلفة.",
            "يعطي هذا السيناريو الأولوية للتدخلات الاقتصادية التي يمكن تنفيذها بسهولة مع المحافظة على الطابع العام للحديقة."
        ],

        "closings": [
            "وبذلك يمكن رفع مستوى الجمال تدريجياً مع المحافظة على الموارد المتاحة.",
            "وهذا يحقق تحسناً عملياً دون الحاجة إلى تدخلات مكلفة أو تغييرات جذرية.",
            "لذلك يمثل السيناريو خياراً مناسباً عندما تكون الميزانية هي العامل المحدد.",
            "ويُعد هذا المستوى مناسباً عندما يكون الهدف تحسين المظهر بأقل تكلفة ممكنة."
        ]
    },

    "medium": {
        "openings": [
            "يعتمد السيناريو المتوازن على مزيج من التحسينات منخفضة ومتوسطة الكلفة لتحقيق تطور واضح في المشهد العام.",
            "تم تصميم السيناريو المتوازن لتحقيق توازن عملي بين حجم الاستثمار ومستوى التحسن الجمالي.",
            "يركز هذا السيناريو على معالجة أكثر من جانب من جوانب الحديقة دون الوصول إلى تدخلات عالية الكلفة.",
            "يوفر السيناريو المتوازن مستوى أكبر من التحسين مقارنة بالحل الاقتصادي مع المحافظة على كلفة معقولة."
        ],

        "closings": [
            "وبذلك يتحقق تحسن ملحوظ مع المحافظة على توازن جيد بين الكلفة والنتيجة.",
            "وهذا يجعل السيناريو مناسباً للحالات التي تتطلب تحسناً واضحاً دون اعتماد خطة مثالية مرتفعة الكلفة.",
            "لذلك يمثل هذا الخيار حلاً وسطاً بين الاقتصاد في التنفيذ والرغبة في تحسين المظهر العام.",
            "وبهذه الطريقة يتم تحقيق تطوير أكثر شمولاً مع بقاء مستوى الاستثمار ضمن حدود متوسطة."
        ]
    },

    "open": {
        "openings": [
            "يقدم السيناريو المثالي خطة تحسين شاملة تهدف إلى رفع القيمة الجمالية للحديقة إلى أعلى مستوى ممكن.",
            "يركز السيناريو المثالي على تحقيق أكبر تحسن جمالي ممكن دون فرض قيود على مستوى الكلفة.",
            "تم تصميم هذا السيناريو للاستفادة من أوسع مجموعة ممكنة من تدخلات تحسين المشهد الطبيعي.",
            "يمثل هذا السيناريو الحالة الأكثر شمولاً، حيث يسمح بتنفيذ التحسينات اللازمة بغض النظر عن مستوى كلفتها."
        ],

        "closings": [
            "وبذلك تصبح الحديقة أكثر غنى وتنوعاً وتماسكاً بصرياً.",
            "والنتيجة المتوقعة هي مشهد أكثر تنوعاً وتوازناً مع نقطة تركيز واضحة وتفاصيل نباتية أكثر غنى.",
            "وهذا يهدف إلى الوصول إلى أعلى مستوى جمالي يمكن أن تدعمه خطة التحسين المقترحة.",
            "لذلك يمثل هذا السيناريو المرجع الأعلى للمقارنة مع السيناريوهين الاقتصادي والمتوازن."
        ]
    }
}


ACTION_EFFECT_AR = {
    "Add flowers bed":
        "إضافة أحواض الزهور تزيد التنوع اللوني وتمنح المشهد النباتي مظهراً أكثر حيوية.",

    "Improve lawn/soil quality":
        "تحسين جودة المسطح الأخضر والتربة يساعد على جعل المساحة العشبية أكثر تجانساً وجاذبية.",

    "Plant trees and shrubs":
        "إضافة الأشجار والشجيرات تزيد التنوع النباتي وتساعد على توزيع العناصر الخضراء بصورة أكثر توازناً.",

    "Add small water feature":
        "إضافة عنصر مائي صغير توفر نقطة اهتمام إضافية دون إحداث تغيير كبير في بنية الحديقة.",

    "Add medium water feature":
        "إضافة عنصر مائي متوسط تقوي نقطة التركيز وتضيف تنوعاً بصرياً إلى التصميم.",

    "Reduce bare ground (mulch/cover)":
        "تقليل مساحات الأرض المكشوفة يساعد على زيادة الإحساس بالامتلاء والاستمرارية البصرية.",

    "Add/Enhance water element":
        "تعزيز عنصر الماء يرفع من أهمية نقطة التركيز ويضيف تنوعاً إلى المشهد.",

    "Increase green spaces":
        "زيادة المساحات الخضراء تدعم استمرارية المشهد الطبيعي وتقلل من الإحساس بالفراغ.",

    "Add shrubs":
        "إضافة الشجيرات تساعد على زيادة الطبقات النباتية وإثراء التكوين البصري.",

    "Extensive lawn/soil renovation":
        "تجديد المسطح الأخضر والتربة على نطاق واسع يرفع من تجانس وجودة المساحات العشبية.",

    "Additional canopy growth":
        "زيادة الغطاء الشجري تضيف عمقاً وتنوعاً إلى المشهد وتساعد على تحقيق توزيع نباتي أكثر توازناً.",

    "Add water pond/fountain":
        "إضافة بركة أو نافورة توفر نقطة تركيز مركزية قوية وتزيد من التنوع البصري.",

    "Improve lawn/soil quality (adaptive)":
        "التحسين التكيفي للمسطح الأخضر يعالج جودة العشب وفقاً لخصائص الصورة الحالية.",

    "Extensive lawn/soil renovation (adaptive)":
        "التجديد التكيفي الواسع للمسطح الأخضر يستهدف تحسين المناطق التي تحتاج إلى تدخل أكبر."
}


def _seeded_choice_ar(options, seed_key):
    return random.Random(seed_key).choice(options)


# مكتبة أسماء وصفية للماء (نسخة مستقلة، متّسقة مع core/garden_integration.py)
_WATER_LIB = {
    "tiny": ["نافورة صغيرة هادئة", "بركة صغيرة مع نباتات محيطة"],
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
    "flagship": ["عنصر مائي تفاعلي موسيقي (نافورة راقصة مع تأثيرات ماء وإضاءة متزامنة)"],
}


def describe_water_feature(water_pct, tier="medium"):
    water_pct = float(water_pct)
    if water_pct <= 0.05:
        return None
    seed_key = f"water|{water_pct:.2f}|{tier}"
    if water_pct >= 8.0 and tier in ("ideal", "open", "high"):
        if random.Random(seed_key + "|flagship_gate").random() < 0.5:
            return _seeded_choice_ar(_WATER_LIB["flagship"], seed_key)
        return _seeded_choice_ar(_WATER_LIB["medium"], seed_key)
    if water_pct >= 5.0:
        return _seeded_choice_ar(_WATER_LIB["medium"], seed_key)
    if water_pct >= 2.0:
        return _seeded_choice_ar(_WATER_LIB["small"], seed_key)
    return _seeded_choice_ar(_WATER_LIB["tiny"], seed_key)


def _level_ar_bs(value, low, high):
    value = float(value)
    if value < low * 0.5:
        return "منخفض جدًا"
    if value < low:
        return "منخفض"
    if value <= high:
        return "متوسط"
    return "جيد"


# ============================================================
# مكتبة التدخلات الواقعية (Action Library) — الجيل الثاني
# ============================================================
# ملاحظة منهجية جوهرية: نموذج MLP لا يزال يرى فقط خمس نسب مئوية
# (grass/trees/flowers/ground/water). لذلك "الإجراءات الواقعية" أدناه هي
# ترجمة سردية لأي زيادة رقمية يقترحها البحث التكيّفي على سمة بعينها —
# مجموعة أفعال زراعية/تصميمية حقيقية يمكن أن تُنتج فعليًا هذا التغيير في
# التركيبة، بدل جملة وحيدة مبتذلة مثل "تحسين المسطح الأخضر بنسبة 8%".
# هذا يطبّق حرفيًا ما طلبه الباحث: مكتبة تحسينات Landscape بدل رقم واحد.
#
# مفاتيح المستوى هنا مطابقة لبقية الملف: low=اقتصادي، medium=متوازن،
# open=مثالي (وليست economic/balanced/ideal، تفاديًا لأي تعارض تسمية).
# ============================================================

ACTION_LIBRARY = {
    "low": {
        "grass_pct": [
            "تحليل التربة وإضافة المادة العضوية اللازمة لمعالجة ضعف المسطح الأخضر.",
            "تسميد مدروس للمسطح الأخضر لتحسين اللون وزيادة الكثافة.",
            "تهوية التربة ومعالجة المناطق الضعيفة وإعادة البذر موضعيًا في البقع الفارغة.",
            "تحسين نظام الري وضبط توقيته لمعالجة الاصفرار وعدم تجانس اللون.",
            "جزّ منتظم بارتفاع مناسب لتكثيف العشب القائم بدل زراعة مساحات جديدة بالكامل.",
        ],
        "trees_pct": [
            "تقليم الأشجار والشجيرات الموجودة وإزالة الأجزاء الجافة منها.",
            "إضافة عدد محدود جدًا من الشجيرات منخفضة الكلفة في النقاط الأكثر حاجة.",
            "زراعة شتلات صغيرة سريعة التأسيس بدل أشجار كبيرة مكلفة، مع رعاية أولية بسيطة.",
        ],
        "flowers_pct": [
            "زراعة أحواض زهور صغيرة أو نباتات موسمية في أكثر المواقع ظهورًا.",
            "زيادة كثافة الأحواض القائمة بإضافة لون أو لونين لتحسين التباين البصري.",
            "استخدام نباتات معمِّرة رخيصة التكلفة ومتكررة الإزهار بدل الزهور الموسمية المكلفة الاستبدال.",
        ],
        "ground_pct": [
            "إزالة الأعشاب غير المرغوبة وتنظيم حواف المسطحات والممرات.",
            "إضافة طبقة نشارة عضوية أو غطاء أرضي للمساحات المكشوفة لتقليل الفراغ البصري.",
            "زراعة غطاء أرضي زاحف منخفض الكلفة في البقع المكشوفة الصغيرة المتفرقة.",
        ],
    },
    "medium": {
        "grass_pct": [
            "تجديد أجزاء واسعة من المسطح الأخضر مع تحسين التربة قبل إعادة الزراعة.",
            "إعادة تأهيل شاملة للمسطح الأخضر تجمع بين التربة والري والتسميد معًا.",
            "إعادة تسوية المساحات غير المستوية من المسطح الأخضر قبل زراعتها بعشب جديد متجانس.",
        ],
        "trees_pct": [
            "زراعة مجموعات من الشجيرات والأشجار الصغيرة لخلق طبقات نباتية متعددة الارتفاع.",
            "إضافة أشجار وشجيرات جديدة موزَّعة بعناية بدل التوزيع العشوائي السابق.",
            "زراعة صف حدودي من الأشجار متوسطة الحجم لتحديد أطراف الحديقة بصريًا.",
        ],
        "flowers_pct": [
            "إنشاء أحواض زهور منظمة وموزّعة بتوازن بين الممرات ونقاط التقاطع.",
            "تصميم مجموعات زهرية متناسقة الألوان لإضافة تنوع بصري حقيقي.",
            "إضافة أحواض زهور معمِّرة مختلطة بنباتات موسمية لتجديد اللون على مدار السنة.",
        ],
        "water_pct": [
            "إضافة نافورة صغيرة كنقطة جذب بصرية جديدة في الحديقة.",
            "إنشاء شلال جداري صغير أو مجرى مائي قصير كعنصر تركيز.",
            "إضافة بركة زخرفية صغيرة محاطة بنباتات مائية لتكوين نقطة هدوء بصري.",
        ],
        "ground_pct": [
            "تحسين التربة في المساحات المكشوفة تمهيدًا لزراعتها بدل تركها فارغة.",
            "تغطية المساحات المكشوفة الواسعة بغطاء أرضي كثيف بدل تركها ترابًا عاريًا.",
            "استبدال المساحات المكشوفة المتناثرة بممرات محصَّاة أو رصيفة منظمة بدل بقائها فراغًا غير موظَّف.",
        ],
    },
    "open": {
        "grass_pct": [
            "إعادة تصميم المسطح الأخضر بالكامل ليصبح خلفية متجانسة لبقية عناصر التصميم.",
            "تجديد المسطح الأخضر بالكامل بعشب عالي الجودة موحَّد النوع لضمان تجانس اللون والملمس.",
            "إعادة هندسة منحنيات المسطح الأخضر لتوجيه النظر نحو نقاط التركيز الرئيسية في التصميم.",
        ],
        "trees_pct": [
            "زراعة أشجار وشجيرات أكبر حجمًا ومتدرجة الارتفاع لبناء عمق بصري حقيقي.",
            "بناء طبقات نباتية كاملة: أشجار خلفية، شجيرات وسطى، ونباتات منخفضة أمامية.",
            "زراعة أشجار مميزة (Specimen Trees) في نقاط استراتيجية لتكون معالم بصرية قائمة بذاتها.",
        ],
        "flowers_pct": [
            "إنشاء مناطق زهرية متعددة الطبقات ومتنوعة الألوان ضمن تكوين Landscape متكامل.",
            "تصميم جزر زهرية موسمية موزّعة بعناية بين نقاط التركيز الرئيسية.",
            "إنشاء حدود زهرية طويلة (Herbaceous Border) متدرجة الارتفاع والألوان على طول الممرات الرئيسية.",
        ],
        "ground_pct": [
            "إعادة تصميم حواف الحديقة والممرات لتصبح الحديقة متماسكة من المركز إلى الأطراف.",
            "استبدال كل المساحات المكشوفة المتبقية بأرضيات وممرات مصمَّمة بمواد راقية (حجر طبيعي أو خشب مركّب).",
            "دمج المساحات المكشوفة ضمن منصات جلوس أو ساحات صغيرة مبلَّطة بدل بقائها فراغًا غير مستثمَر.",
        ],
    },
}

AR_FEATURE_SHORT = {
    "grass_pct": "المسطح الأخضر",
    "trees_pct": "الغطاء الشجري",
    "flowers_pct": "أحواض الزهور",
    "ground_pct": "الأرض المكشوفة",
    "water_pct": "العنصر المائي",
}

# عناصر إثراء وصفية *غير مُقيَّمة من النموذج* (لا تُغيّر أي سمة رقمية) —
# بالضبط كما اقترحت المذكرة (الإضاءة، الجلوس، الحواف)، ومُوسَّمة بصراحة
# حتى لا تُوحي بتقييم فعلي من الذكاء الاصطناعي لجانب لم يُدرَّب عليه.
NARRATIVE_ENRICHMENTS = {
    "low": [],
    "medium": [
        "إضافة إضاءة خفيفة للممرات والعناصر المهمة لإبراز التصميم ليلًا.",
        "تحسين التوزيع العام للعناصر لتحقيق تنوع وتوازن وعمق بصري أفضل.",
    ],
    "open": [
        "إضاءة جمالية موجّهة للممرات وقواعد الأشجار والعنصر المائي.",
        "التفكير في نقاط جلوس أو عناصر معمارية صغيرة عند توفر مساحة كافية لها.",
        "تحسين حواف الحديقة لتصبح متماسكة بصريًا من المركز إلى الأطراف.",
    ],
}

NARRATIVE_DISCLAIMER_AR = (
    "🔎 ملاحظة منهجية: العناصر السابقة (كالإضاءة أو نقاط الجلوس) اقتراحات "
    "تصميمية تكميلية فقط — لا توجد سمات مثل الإضاءة ضمن بيانات التدريب، "
    "فلم يُقيِّمها النموذج ولا تؤثر في الدرجة المتوقعة أعلاه."
)

# سطر إضاءة مستوى "open" يذكر "العنصر المائي" فقط إن كان التصميم/الخطة
# الناتجة تتضمن فعليًا عنصرًا مائيًا — تفاديًا للإشارة إلى عنصر غير
# موجود عندما لا يتضمّن التصميم ماءً على الإطلاق (خطأ رُصِد فعليًا في
# تنفيذ حقيقي: تصميم بلا أي عنصر مائي كان يظهر له اقتراح إضاءة "لقواعد
# الأشجار والعنصر المائي").
_OPEN_LIGHTING_WITH_WATER = "إضاءة جمالية موجّهة للممرات وقواعد الأشجار والعنصر المائي."
_OPEN_LIGHTING_NO_WATER = "إضاءة جمالية موجّهة للممرات وقواعد الأشجار."


def _enrichment_pool(scenario_key, has_water=False):
    """
    يُعيد قائمة الإثراءات الوصفية المناسبة لهذا المستوى، مع تعديل صياغة
    سطر الإضاءة في مستوى "open" ليذكر العنصر المائي فقط عندما يتضمّنه
    التصميم/الخطة الناتجة فعليًا.
    """
    pool = list(NARRATIVE_ENRICHMENTS.get(scenario_key, []))
    if scenario_key == "open" and pool:
        lighting_line = _OPEN_LIGHTING_WITH_WATER if has_water else _OPEN_LIGHTING_NO_WATER
        pool = [lighting_line] + pool[1:]
    return pool


def diagnose_weakness_ar(base_feats):
    """
    يُشخّص "المحور" الأضعف فعليًا في الحديقة من بين خمسة احتمالات: عشب/
    أشجار/زهور/أرض مكشوفة/ضعف تنوع — ويُستخدَم فقط لاختيار *إطار* افتتاحية
    التقرير (الزاوية السردية)، وليس لتحديد أي سمة يُغيّرها البحث الرياضي
    (تلك مسألة منفصلة تمامًا يحدّدها _find_adaptive_delta في app.py).
    هذا يحقق بالضبط ما طلبته المذكرة: "قد يكون الاقتصادي في صورة كذا... أو
    في صورة أخرى..." حسب مشكلة هذه الحديقة تحديدًا، لا حديقة عشوائية.
    """
    grass = float(base_feats.get("grass_pct", 0.0))
    trees = float(base_feats.get("trees_pct", 0.0))
    flowers = float(base_feats.get("flowers_pct", 0.0))
    ground = float(base_feats.get("ground_pct", 0.0))

    grass_level = _level_ar_bs(grass, 15.0, 35.0)
    trees_level = _level_ar_bs(trees, 15.0, 35.0)
    flowers_level = _level_ar_bs(flowers, 3.0, 8.0)

    level_score = {"منخفض جدًا": 3.0, "منخفض": 2.0, "متوسط": 0.5, "جيد": 0.0}

    scores = {
        "grass": level_score[grass_level],
        "trees": level_score[trees_level],
        "flowers": level_score[flowers_level] * 0.9,
        "ground": 2.5 if ground > 35.0 else (1.0 if ground > 25.0 else 0.0),
    }

    vegetated = grass + trees + flowers
    diversity_score = 0.0
    if vegetated > 1.0:
        dominant_share = max(grass, trees, flowers) / vegetated
        if dominant_share >= 0.75:
            diversity_score = 2.0
        elif dominant_share >= 0.60:
            diversity_score = 1.0
    scores["diversity"] = diversity_score

    axis = max(scores, key=scores.get)
    if scores[axis] <= 0.0:
        axis = "balanced_ok"

    levels = {
        "grass_level": grass_level,
        "trees_level": trees_level,
        "flowers_level": flowers_level,
        "ground_state": "كبيرة" if ground > 35.0 else "معتدلة",
    }
    return axis, scores, levels


def _pick_n(options, seed_key, n=1):
    if not options:
        return []
    rng = random.Random(seed_key)
    pool = list(options)
    rng.shuffle(pool)
    n = max(0, min(n, len(pool)))
    return pool[:n]


def select_real_actions(scenario_key, deltas_used, base_feats, seed_extra=""):
    """
    يبني قائمة إجراءات واقعية (لا تسمية واحدة مبتذلة "زيادة X%") من السمات
    التي حرّكها البحث الرياضي فعليًا فقط (deltas_used) — وليس من أي سمة
    "مُشخَّصة كأضعف محور" بمعزل عن ذلك.

    [إصلاح مهم 2026-09-04] النسخة الأولى كانت تضيف "إجراءً داعمًا" لأضعف
    محور مُشخَّص (diagnose_weakness_ar) حتى لو لم يُحرِّكه البحث الرياضي
    إطلاقًا — فأنتجت تناقضًا فعليًا رصده الباحث بعد التشغيل الحقيقي: خطط
    توصي بـ"زراعة أحواض زهور" بينما جدول "قبل مقابل بعد" يُظهر flowers_pct
    ثابتة عند 0% تمامًا (لأن الخطة لم تُغيّرها أصلًا). أُزيلت هذه الآلية
    كليًا: كل إجراء مذكور في "الإجراءات المقترحة" يجب أن يقابله تغيير رقمي
    فعلي في التركيبة، وإلا فهو ادّعاء غير صحيح — تشخيص المحور الأضعف
    (diagnose_weakness_ar) لا يزال يُستخدَم فقط لاختيار *زاوية* افتتاحية
    التقرير في generate_improvement_report، وليس لاختيار الإجراءات هنا.
    """
    if not deltas_used:
        return []

    tier_lib = ACTION_LIBRARY.get(scenario_key, {})
    used_features = [f for f, d in (deltas_used or {}).items() if f != "ground_pct" and d]

    if not used_features:
        return []

    seed_base = (
        scenario_key + "|" +
        "|".join(f"{f}:{deltas_used[f]:.1f}" for f in sorted(used_features)) +
        "|" + seed_extra
    )

    chosen = []
    for f in used_features:
        options = tier_lib.get(f, [])
        n = 1 if (scenario_key == "low" and len(used_features) > 1) else 2
        for s in _pick_n(options, seed_base + f"|{f}", n=n):
            if s not in chosen:
                chosen.append(s)

    if not chosen and used_features:
        for f in used_features:
            chosen.append(f"تدخّل موجَّه لتحسين {AR_FEATURE_SHORT.get(f, f)}.")

    cap = {"low": 3, "medium": 4, "open": 5}.get(scenario_key, 3)
    return chosen[:cap]


# ============================================================
# إطارات افتتاحية مرتبطة بالتشخيص الفعلي (وليست عشوائية شكلية فقط) —
# نفس السيناريو (مثلاً الاقتصادي) يُنتج صياغة مختلفة فعليًا حسب المشكلة
# الحقيقية لهذه الحديقة تحديدًا، تمامًا كما طلبت المذكرة بمثاليها
# ("ضعف المسطح الأخضر" مقابل "ضعف اكتمال المشهد البصري").
# ============================================================

OPENING_BY_AXIS_AR = {
    "grass": {
        "low": [
            "تم توجيه السيناريو الاقتصادي نحو معالجة ضعف كثافة المسطح الأخضر من خلال تحسين حالة التربة والعناية بالعشب، دون الحاجة إلى إنشاء عناصر جديدة مرتفعة الكلفة.",
            "بما أن المشكلة الرئيسية في هذه الحديقة هي ضعف المسطح الأخضر تحديدًا، ركّز السيناريو الاقتصادي على معالجتها مباشرة عبر التربة والري والتسميد المدروس.",
        ],
        "medium": [
            "انطلق السيناريو المتوازن من معالجة ضعف المسطح الأخضر أولًا كأساس، ثم أضاف عناصر تصميمية جديدة فوقه لتحقيق تطور بصري واضح.",
            "بعد تجديد المسطح الأخضر الذي كان دون المستوى، وسّع السيناريو المتوازن التدخل ليشمل عناصر نباتية إضافية تعزّز التوازن العام.",
        ],
        "open": [
            "أعاد السيناريو المثالي تصميم المسطح الأخضر بالكامل باعتباره كان نقطة الضعف الأصلية، ليصبح خلفية متجانسة تُبرز بقية عناصر التصميم الغنية.",
            "انطلق التحول الكامل للحديقة من معالجة جذرية للمسطح الأخضر الضعيف أصلًا، قبل بناء بقية طبقات التصميم فوقه.",
        ],
    },
    "trees": {
        "low": [
            "بما أن الغطاء الشجري كان محدودًا نسبيًا، ركّز السيناريو الاقتصادي على صيانته وتقليمه وإضافة عدد محدود جدًا من الشجيرات منخفضة الكلفة.",
        ],
        "medium": [
            "بما أن الغطاء الشجري كان الأضعف في هذه الحديقة، أعطاه السيناريو المتوازن أولوية عبر مجموعات جديدة من الأشجار والشجيرات متعددة الارتفاع.",
            "عالج السيناريو المتوازن ضعف الغطاء الشجري تحديدًا بإضافة طبقات نباتية جديدة، مع تحسينات مكمّلة في بقية عناصر الحديقة.",
        ],
        "open": [
            "بما أن الغطاء الشجري كان نقطة الضعف الأساسية، بنى السيناريو المثالي طبقات نباتية كاملة (أشجار خلفية، شجيرات وسطى، نباتات منخفضة) لمعالجتها جذريًا.",
        ],
    },
    "flowers": {
        "low": [
            "بما أن أحواض الزهور كانت شبه غائبة، أضاف السيناريو الاقتصادي زهورًا موسمية محدودة في أكثر النقاط ظهورًا لإدخال لون دون كلفة كبيرة.",
        ],
        "medium": [
            "ركّز السيناريو المتوازن على معالجة ضعف التنوع اللوني عبر أحواض زهور منظمة وموزّعة بعناية بين الممرات ونقاط التقاطع.",
        ],
        "open": [
            "بما أن التنوع اللوني كان الأضعف، بنى السيناريو المثالي مناطق زهرية متعددة الطبقات ضمن تكوين Landscape متكامل بدل الاكتفاء بأحواض متفرقة.",
        ],
    },
    "ground": {
        "low": [
            "تم توجيه السيناريو الاقتصادي نحو معالجة المساحات المكشوفة وتنظيم الحواف، لأن المشكلة الرئيسية ليست نقص عنصر بعينه بل ضعف اكتمال المشهد البصري.",
        ],
        "medium": [
            "بما أن الأرض المكشوفة كانت واسعة نسبيًا، ركّز السيناريو المتوازن على تقليل هذا الفراغ عبر عناصر نباتية وتصميمية جديدة موزَّعة عليه.",
        ],
        "open": [
            "بما أن اتساع الأرض المكشوفة كان نقطة الضعف الأبرز، أعاد السيناريو المثالي تصميم الحواف والممرات بالكامل ليصبح المشهد متماسكًا من المركز إلى الأطراف.",
        ],
    },
    "diversity": {
        "low": [
            "المشكلة هنا ليست نقصًا حادًا في عنصر بعينه بقدر ما هي غلبة عنصر واحد على البقية، فاختار السيناريو الاقتصادي تدخلات محدودة لتخفيف هذا الاختلال دون كلفة كبيرة.",
        ],
        "medium": [
            "بما أن الحديقة تفتقر إلى التنوع الحقيقي بين عناصرها النباتية، ركّز السيناريو المتوازن على تحسين التوزيع وإضافة عناصر جديدة تكسر هيمنة العنصر الواحد.",
        ],
        "open": [
            "عالج السيناريو المثالي ضعف التنوع الأصلي عبر بناء تكوين Landscape كامل يوازن بين العشب والأشجار والزهور والماء، بدل بقاء الحديقة محكومة بعنصر واحد.",
        ],
    },
    "balanced_ok": {
        "low": [
            "الحديقة في وضع مقبول أصلًا من حيث التوازن بين عناصرها، لذلك اكتفى السيناريو الاقتصادي بصيانة خفيفة ترفع الجودة العامة دون تغييرات كبرى.",
        ],
        "medium": [
            "بما أن التركيبة الأساسية للحديقة متوازنة نسبيًا، أضاف السيناريو المتوازن عناصر تصميمية جديدة لرفع مستوى التجربة البصرية دون معالجة خلل حاد.",
        ],
        "open": [
            "انطلاقًا من تركيبة متوازنة أصلًا، استثمر السيناريو المثالي هذا الأساس الجيد لبناء تجربة تصميمية غنية ومتكاملة.",
        ],
    },
}


def _select_opening(scenario_key, axis, seed_key):
    pool = OPENING_BY_AXIS_AR.get(axis, {}).get(scenario_key) or REPORT_VARIANTS_AR[scenario_key]["openings"]
    return random.Random(seed_key + "|opening").choice(pool)


# افتتاحيات محايدة لحالة "لا حاجة لأي تدخل" (الدرجة الأساسية تقع أصلًا ضمن
# أو فوق نطاق هذا المستوى) — [إصلاح 2026-09-04] لا تُستخدَم افتتاحيات
# OPENING_BY_AXIS_AR هنا لأنها مصاغة كـ"تم توجيه السيناريو نحو معالجة..."
# (تصف تدخلًا فعليًا)، وهو ما يُناقض مباشرة جملة "لا حاجة لأي تدخل" التي
# تليها لو استُخدمت في هذه الحالة (كان هذا هو أصل التناقض الذي رصده الباحث:
# افتتاحية "الزهور شبه غائبة، فأضاف السيناريو زهورًا..." تليها جدول قبل/بعد
# لا يُظهر أي تغيير في الزهور).
NO_INTERVENTION_OPENINGS_AR = {
    "low": [
        "درجة الحديقة الحالية تقع أصلًا ضمن نطاق السيناريو الاقتصادي، فلم يقترح النظام أي تدخل إضافي عند هذا المستوى من الكلفة.",
        "بتركيبتها الحالية، تُحقق الحديقة أصلًا مستوى الجمال المطلوب للسيناريو الاقتصادي دون أي تعديل إضافي.",
    ],
    "medium": [
        "درجة الحديقة الحالية تقع أصلًا ضمن نطاق السيناريو المتوازن، فلم يقترح النظام أي تدخل إضافي عند هذا المستوى.",
        "بتركيبتها الحالية، تُحقق الحديقة أصلًا مستوى الجمال المطلوب للسيناريو المتوازن دون الحاجة لتعديلات إضافية.",
    ],
    "open": [
        "درجة الحديقة الحالية تقع أصلًا ضمن نطاق السيناريو المثالي، فلم يقترح النظام أي تدخل إضافي عند هذا المستوى.",
        "بتركيبتها الحالية، تُحقق الحديقة أصلًا أعلى مستويات الجمال المستهدفة دون أي تعديل إضافي.",
    ],
}


def deficiency_rule_sentences_ar(base_feats, scenario_key="low"):
    """
    [قديم — أُبقي للتوافق فقط] جمل مخصَّصة حسب نقص الصورة المرفوعة الفعلي.
    التقرير الفعلي في generate_improvement_report لم يعد يستدعي هذه الدالة
    مباشرة؛ استُبدلت بـ diagnose_weakness_ar + OPENING_BY_AXIS_AR (تنويع
    حقيقي أوسع). أُبقيت هنا فقط تحسبًا لاستدعاء خارجي قديم لها.
    """
    base_feats = base_feats or {}
    grass = float(base_feats.get("grass_pct", 0.0))
    trees = float(base_feats.get("trees_pct", 0.0))
    flowers = float(base_feats.get("flowers_pct", 0.0))
    ground = float(base_feats.get("ground_pct", 0.0))

    grass_level = _level_ar_bs(grass, 15.0, 35.0)
    trees_level = _level_ar_bs(trees, 15.0, 35.0)
    flowers_level = _level_ar_bs(flowers, 3.0, 8.0)

    lines = []

    if grass_level in ("منخفض جدًا", "منخفض"):
        if scenario_key == "low":
            lines.append(
                "بما أن المسطح الأخضر " + grass_level + " في الصورة الأصلية، "
                "يُنصح بمعالجة التربة وتحسين الري وإعادة بذر المناطق الفارغة قبل أي تدخل آخر."
            )
        else:
            lines.append(
                "المسطح الأخضر كان " + grass_level + " في الصورة الأصلية، لذلك أُعطي أولوية في هذه الخطة."
            )

    if trees_level in ("منخفض جدًا", "منخفض") and scenario_key != "low":
        lines.append(
            "الغطاء الشجري كان " + trees_level + " أصلًا، فرُوعي ذلك عند اختيار الإجراءات."
        )

    if flowers_level in ("منخفض جدًا", "منخفض"):
        _flowers_level_fem = "منخفضة جدًا" if flowers_level == "منخفض جدًا" else "منخفضة"
        lines.append(
            "أحواض الزهور كانت " + _flowers_level_fem + " في الصورة الأصلية."
        )

    if ground > 35.0:
        lines.append(
            "الأرض المكشوفة كانت نسبتها كبيرة نسبيًا (" + f"{ground:.0f}%" + ")، "
            "وهذا من العوامل التي تُضعف الإحساس بالاكتمال البصري للحديقة."
        )

    return lines[:2]  # سطران كحد أقصى حتى لا يطغى على بقية التقرير


def generate_improvement_report(scenario_key, plan, base_feats, new_feats):
    """
    إنشاء تقرير عربي تفسيري — الجيل الثاني (Action Library + Rule Engine).

    بخلاف النسخة السابقة (تقرير شبه ثابت لكل مستوى، يتغيّر فقط في صياغة
    الافتتاحية/الخاتمة عشوائيًا)، هذا التقرير يُبنى فعليًا من:
      1) السمات الفعلية المستخرجة من الحديقة (base_feats) عبر diagnose_weakness_ar.
      2) المحور الأضعف فعليًا (عشب/أشجار/زهور/أرض مكشوفة/تنوع) → يحدّد
         "زاوية" الافتتاحية (OPENING_BY_AXIS_AR).
      3) التدخلات التي اختارها النظام فعليًا (plan["deltas"]) → تُترجَم
         لإجراءات واقعية متعددة عبر select_real_actions، بدل جملة "زيادة X%".
      4) مستوى السيناريو (اقتصادي/متوازن/مثالي).
      5) مقدار التحسن المتوقع فعليًا.
    لذلك حديقتان مختلفتان بنفس المستوى تحصلان على تقريرين مختلفين فعليًا
    في المضمون، وليس فقط في الصياغة السطحية.
    """

    if plan is None:
        return (
            "لم يتم العثور على خطة تحسين مناسبة ضمن حدود هذا السيناريو.\n"
            "تدل النتيجة الحالية على أن الإجراءات المتاحة لا تحقق تحسناً كافياً وفق معيار التقييم.\n"
            "لذلك لم يتم اقتراح تدخلات إضافية ضمن هذا المستوى من الكلفة.\n"
            "يمكن الانتقال إلى سيناريو أعلى للسماح بمجموعة أوسع من إجراءات التحسين."
        )

    base_feats = base_feats or {}
    new_feats = new_feats or {}

    diagnosed_axis, _scores, _levels = diagnose_weakness_ar(base_feats)

    # السمات التي زادها البحث الرياضي فعليًا (نتجاهل ground_pct لأنه دائمًا
    # يُموَّل تلقائيًا من الفائض، وليس "تدخلًا" بحد ذاته).
    deltas_used = {
        k: v for k, v in (plan.get("deltas") or {}).items()
        if k != "ground_pct" and v and v > 0
    }

    base_score = float(plan.get("base_score", 0.0))
    new_score = float(plan.get("new_score", 0.0))
    improvement = float(plan.get("improvement", 0.0))

    # [إصلاح مهم 2026-09-04] الافتتاحية يجب أن تروي ما فعلته الخطة *فعليًا*،
    # وليس أضعف محور مُشخَّص بمعزل عنها — وإلا نفس التناقض الذي رصده الباحث
    # (افتتاحية "ركّز السيناريو على الزهور..." بينما الخطة غيّرت العشب فقط).
    # لذلك: "محور الافتتاحية" = السمة صاحبة أكبر دلتا فعلية ضمن deltas_used،
    # وليس diagnosed_axis. diagnosed_axis يُستخدَم فقط لجملة توضيحية إضافية
    # صادقة عند اختلافه فعليًا عمّا فعلته الخطة (بدل حذف المعلومة كليًا).
    FEATURE_TO_AXIS = {
        "grass_pct": "grass", "trees_pct": "trees", "flowers_pct": "flowers",
    }
    action_axis = None
    if deltas_used:
        primary_feature = max(deltas_used, key=lambda f: deltas_used[f])
        action_axis = FEATURE_TO_AXIS.get(primary_feature)  # None لو water_pct فقط

    axis = action_axis or diagnosed_axis  # يُستخدَم فقط لبناء seed_key أدناه

    seed_key = (
        scenario_key + "|" + axis + "|" +
        "|".join(f"{k}:{v:.1f}" for k, v in sorted(deltas_used.items())) +
        f"|{new_score:.4f}"
    )

    real_actions = select_real_actions(scenario_key, deltas_used, base_feats, seed_extra=seed_key)

    # كل تسمية مبنية بصيغة مذكَّر مفرد ("ضعف/اتساع...") عمدًا لتتوافق نحويًا
    # مع فعل "كان" الثابت في جملة الملاحظة أدناه (تفاديًا لخطأ تذكير/تأنيث
    # مثل "أحواض الزهور كان" بدل "كانت").
    AXIS_LABEL_AR = {
        "grass": "ضعف المسطح الأخضر", "trees": "ضعف الغطاء الشجري",
        "flowers": "ضعف أحواض الزهور", "ground": "اتساع الأرض المكشوفة",
        "diversity": "ضعف التنوع بين العناصر",
    }

    if not deltas_used:
        # لا تدخل حقيقي على الإطلاق — افتتاحية محايدة (وليست مصاغة كتدخل
        # فعلي في محور معيّن)، تفاديًا للتناقض مع جدول "قبل مقابل بعد".
        lines = [random.Random(seed_key + "|no_action_opening").choice(NO_INTERVENTION_OPENINGS_AR[scenario_key])]
    elif action_axis:
        # الافتتاحية تروي المحور الذي *عولج فعليًا* (action_axis)، لا أضعف
        # محور مُشخَّص بمعزل عن الخطة.
        lines = [_select_opening(scenario_key, action_axis, seed_key)]
        if diagnosed_axis != action_axis and diagnosed_axis not in ("balanced_ok",):
            # صدق إضافي: لو كان أضعف محور فعلي مختلفًا عمّا عالجته الخطة،
            # نقولها صراحة بدل حذف المعلومة أو التظاهر بأن الخطة عالجتها.
            lines.append(
                f"يُلاحَظ أن {AXIS_LABEL_AR.get(diagnosed_axis, diagnosed_axis)} كان "
                f"أيضًا من نقاط الضعف الواضحة في هذه الحديقة، إلا أن معالجة "
                f"{AXIS_LABEL_AR.get(action_axis, action_axis)} كانت الأكثر تأثيرًا "
                f"في درجة الجمال ضمن هذا المستوى تحديدًا."
            )
    else:
        # تدخل حقيقي لكن على سمة بلا "محور" افتتاحية مخصَّص (الماء فقط) —
        # افتتاحية عامة من REPORT_VARIANTS_AR بدل اختلاق محور غير دقيق.
        lines = [random.Random(seed_key + "|opening").choice(REPORT_VARIANTS_AR[scenario_key]["openings"])]

    if not real_actions:
        lines.append(
            "لم يتطلّب الوصول إلى هذا المستوى أي تدخل إضافي؛ الحديقة تقع أصلًا "
            "ضمن (أو فوق) نطاق هذا السيناريو بتركيبتها الحالية."
        )
    elif len(real_actions) == 1:
        lines.append(f"تتضمن الخطة إجراءً رئيسيًا هو: {real_actions[0]}")
    else:
        lines.append("تعتمد الخطة على مجموعة من الإجراءات الواقعية التالية:")
        for a in real_actions:
            lines.append(f"• {a}")

    # تسمية وصفية للعنصر المائي الناتج (إن وُجد ماء جديد في الخطة)
    new_water = float(new_feats.get("water_pct", 0.0))
    old_water = float(base_feats.get("water_pct", 0.0))
    if new_water > old_water + 0.05:
        water_name = describe_water_feature(new_water, tier=scenario_key)
        if water_name:
            lines.append(f"يتمثل العنصر المائي المقترح في: {water_name}.")

    # إثراءات وصفية غير مُقيَّمة من النموذج (متوازن/مثالي فقط) + إفصاح صريح
    has_water_final = new_water > 0.5
    enrich_pool = _enrichment_pool(scenario_key, has_water=has_water_final)
    if enrich_pool:
        n_enrich = 1 if scenario_key == "medium" else 2
        chosen_enrich = _pick_n(enrich_pool, seed_key + "|enrich", n=n_enrich)
        if chosen_enrich:
            for e in chosen_enrich:
                lines.append(f"✨ {e}")
            lines.append(NARRATIVE_DISCLAIMER_AR)

    lines.append(
        f"ارتفعت درجة الجمال المتوقعة من {base_score:.2f} إلى {new_score:.2f}، "
        f"أي تحسن مقداره {improvement:.2f}."
    )

    lines.append(random.Random(seed_key + "|closing").choice(REPORT_VARIANTS_AR[scenario_key]["closings"]))

    return "\n".join(lines)


# الفئات التي تمثّل أقساماً حصرية متبادلة من نفس مساحة الصورة الثابتة.
# مجموع هذه الفئات تحديدًا لا يجوز أن يرتفع بعد أي إجراء تحسين، لأن أي
# زيادة في عشب/أشجار/زهور يجب أن تأتي من تقليل الأرض المكشوفة أو من
# مساحة غير مصنَّفة (other) ضمنيًا، لا من "توليد" مساحة إضافية من العدم.
CONSERVED_LAND_COVER_COLS = ["grass_pct", "trees_pct", "flowers_pct", "ground_pct"]

# ============================================================
# مستويات الكلفة (نوعية، وليست أرقامًا)
# ============================================================
#
# لا توجد معادلة أكاديمية عالمية لكلفة زراعة كل عنصر (بخلاف معادلات
# الكتلة الحيوية/الكربون)، لأن الكلفة تعتمد كليًا على السوق المحلي.
# لذلك يُصنَّف كل إجراء يدويًا إلى مستوى نوعي واحد من ثلاثة، بدل رقم
# مالي غير موثّق. ترتيب المستويات ثابت لأغراض المقارنة (low < medium < high).
COST_TIER_ORDER = {"low": 0, "medium": 1, "high": 2}
COST_TIER_AR = {"low": "منخفضة", "medium": "متوسطة", "high": "عالية"}


def apply_deltas(feats, deltas, conserve_cols=None):
    """
    يطبّق دلتا الإجراءات على السمات، مع الحفاظ على أن مجموع فئات الغطاء
    الأرضي المتبادلة (conserve_cols) لا يتجاوز مجموعها الأصلي قبل التحسين.

    آلية الحفاظ:
      1) أي زيادة صافية في المجموع تُموَّل أولًا بتقليل ground_pct (إن وُجد
         وكان لا يزال قابلاً للتخفيض).
      2) إن لم يكفِ ذلك لامتصاص الفائض بالكامل (كأن يكون ground_pct وصل
         للصفر بالفعل)، يُوزَّع الفائض المتبقي تناسبيًا على الفئات التي
         زادت فعليًا بفعل الإجراءات، حتى يعود المجموع الكلي لقيمته الأصلية.
    """
    conserve_cols = conserve_cols or []
    out = dict(feats)

    for k, v in deltas.items():
        out[k] = float(out.get(k, 0.0) + v)

    for k in out:
        if k.endswith("_pct"):
            out[k] = float(np.clip(out[k], 0.0, 100.0))

    if conserve_cols:
        total_before = sum(float(feats.get(c, 0.0)) for c in conserve_cols)
        total_after = sum(float(out.get(c, 0.0)) for c in conserve_cols)
        surplus = total_after - total_before

        if surplus > 1e-9:
            if "ground_pct" in out and "ground_pct" in conserve_cols:
                reducible = min(surplus, out["ground_pct"])
                out["ground_pct"] -= reducible
                surplus -= reducible

            if surplus > 1e-9:
                increased = {
                    c: deltas.get(c, 0.0)
                    for c in conserve_cols
                    if c != "ground_pct" and deltas.get(c, 0.0) > 0
                }
                total_increase = sum(increased.values())
                if total_increase > 1e-9:
                    for c, inc in increased.items():
                        out[c] = max(0.0, out[c] - surplus * (inc / total_increase))

    return out


def predict_one(model, feats_dict, feature_cols):
    x = np.array([[float(feats_dict.get(c, 0.0)) for c in feature_cols]], dtype=np.float32)
    return float(np.clip(model.predict(x)[0], 1.0, 5.0))


# ============================================================
# نقطة تقييم قابلة للاستبدال (score_fn) — دون تغيير عن السابق
# ============================================================


def _score(model, feats_dict, feature_cols, score_fn=None, use_knowledge_only=False,
           knowledge_score_fn=None):
    """
    الدرجة المستخدمة فعليًا لتقييم أي مجموعة سمات (قبل/بعد أي إجراء).

    - إن كان use_knowledge_only=True وتوفّرت knowledge_score_fn ونجحت في
      إرجاع قيمة: تُستخدم درجة المعرفة وحدها (حاجز الأمان للسمات النادرة).
    - وإلا، إن مُرِّرت score_fn: تُستخدم (الدمج المرجّح كما كان).
    - وإلا: MLP الخام فقط (سلوك أصلي، للتوافق الكامل مع أي استدعاء قديم).
    """
    if use_knowledge_only and knowledge_score_fn is not None:
        k_s = knowledge_score_fn(feats_dict)
        if k_s is not None:
            return float(np.clip(k_s, 1.0, 5.0))
        # فشل حساب درجة المعرفة (لا حالات كافية) → رجوع آمن للدمج/MLP

    if score_fn is not None:
        return float(np.clip(score_fn(feats_dict), 1.0, 5.0))
    return predict_one(model, feats_dict, feature_cols)


def compute_rare_features(repo, feature_cols, nonzero_ratio_threshold=0.3):
    """
    حاجز الأمان الديناميكي: يحدد أي السمات "نادرة" في بيانات الاستبيان
    (repo['case_memory']) — أي أن نسبة الحالات ذات القيمة غير الصفرية لها
    أقل من nonzero_ratio_threshold. لأي سمة نادرة، تنبؤ MLP المحلي غير
    موثوق إحصائيًا (بيانات تدريب شحيحة جدًا حول تغييرها)، فيجب عدم الاعتماد
    على الدمج المرجّح (الذي لا يزال يعطي MLP وزنًا كبيرًا) عند تقييم أي
    إجراء يُغيّر هذه السمة تحديدًا.

    لا حاجة لتثبيت flowers_pct يدويًا — أي سمة نادرة مستقبلية (مثل water_pct
    في عيّنة أصغر) ستُكتشف تلقائيًا بنفس المنطق.
    """
    if not repo:
        return set()

    cases = repo.get("case_memory", [])
    if not cases:
        return set()

    n = len(cases)
    rare = set()
    for f in feature_cols:
        nonzero = 0
        for c in cases:
            try:
                v = float(c.get(f, 0.0) or 0.0)
            except (TypeError, ValueError):
                v = 0.0
            if v > 1e-9:
                nonzero += 1
        ratio = nonzero / n if n else 0.0
        if ratio < nonzero_ratio_threshold:
            rare.add(f)

    return rare


def _touches_rare_feature(deltas, rare_features):
    if not rare_features:
        return False
    return any(k in rare_features for k in deltas)


def valid_interventions(interventions, feature_cols):
    clean = []
    for a in interventions:
        useful = {k: v for k, v in a.get("deltas", {}).items() if k in feature_cols}
        if useful:
            b = dict(a)
            b["deltas"] = useful
            if "cost_tier" not in b:
                raise ValueError(
                    f"الإجراء '{b.get('name')}' بلا 'cost_tier' — كل إجراء يجب أن "
                    f"يحمل 'low'/'medium'/'high' بدل 'cost' الرقمي القديم."
                )
            clean.append(b)
    return clean


def get_action(interventions, name):
    for a in interventions:
        if a["name"] == name:
            return a
    return None


def _plan_cost_tier(actions):
    """كلفة أي توليفة = أعلى مستوى كلفة بين إجراءاتها (أضعف حلقة)."""
    order = max(COST_TIER_ORDER[a["cost_tier"]] for a in actions)
    return {v: k for k, v in COST_TIER_ORDER.items()}[order]


def evaluate_plan(model, base_feats, feature_cols, actions, score_fn=None,
                   rare_features=None, knowledge_score_fn=None):
    actions = list(actions)
    cost_tier = _plan_cost_tier(actions)

    deltas = {}
    for a in actions:
        for k, v in a["deltas"].items():
            if k in feature_cols:
                deltas[k] = deltas.get(k, 0.0) + float(v)

    conserve_cols = [c for c in CONSERVED_LAND_COVER_COLS if c in feature_cols]
    new_feats = apply_deltas(base_feats, deltas, conserve_cols=conserve_cols)

    use_knowledge_only = _touches_rare_feature(deltas, rare_features)

    base_score = _score(model, base_feats, feature_cols, score_fn,
                         use_knowledge_only, knowledge_score_fn)
    new_score = _score(model, new_feats, feature_cols, score_fn,
                        use_knowledge_only, knowledge_score_fn)

    return {
        "actions": actions,
        "action_names": [a["name"] for a in actions],
        "action_names_ar": [AR_ACTION.get(a["name"], a["name"]) for a in actions],
        "cost_tier": cost_tier,
        "cost_tier_order": COST_TIER_ORDER[cost_tier],
        "cost_tier_ar": COST_TIER_AR[cost_tier],
        "base_score": base_score,
        "new_score": new_score,
        "improvement": new_score - base_score,
        "new_feats": new_feats,
        "deltas": deltas,
        "scored_via_knowledge_only": use_knowledge_only,
    }


def _all_combinations(interventions):
    combos = []
    n = len(interventions)
    for r in range(1, n + 1):
        for idxs in itertools.combinations(range(n), r):
            combos.append([interventions[i] for i in idxs])
    return combos


def _best_within_tier_cap(plans, max_tier_order):
    """أفضل خطة من بين التي لا يتجاوز مستوى كلفتها max_tier_order."""
    pool = [p for p in plans if p["cost_tier_order"] <= max_tier_order]
    if not pool:
        return None
    return sorted(
        pool,
        key=lambda p: (-p["improvement"], p["cost_tier_order"]),
    )[0]


def _best_overall(plans):
    if not plans:
        return None
    return sorted(plans, key=lambda p: (-p["new_score"], p["cost_tier_order"]))[0]


def plan_to_row(key, plan, model, base_feats, feature_cols, score_fn=None,
                rare_features=None, knowledge_score_fn=None):

    info = SCENARIO_INFO[key]

    base_score = _score(
        model,
        base_feats,
        feature_cols,
        score_fn
    )

    if plan is None:
        report = generate_improvement_report(
            key,
            None,
            base_feats,
            base_feats
        )

        return {
            "scenario_key": key,
            "scenario": info["scenario"],
            "scenario_ar": info["scenario_ar"],
            "description_ar": info["description_ar"],
            "actions": "لا توجد خطة مناسبة",
            "actions_ar": "لا توجد خطة مناسبة",
            "cost_tier": None,
            "cost_tier_ar": "—",
            "base_score": base_score,
            "new_score": base_score,
            "improvement": 0.0,
            "actions_count": 0,
            "scored_via_knowledge_only": False,
            "report_ar": report,
        }

    report = generate_improvement_report(
        key,
        plan,
        base_feats,
        plan["new_feats"]
    )

    return {
        "scenario_key": key,
        "scenario": info["scenario"],
        "scenario_ar": info["scenario_ar"],
        "description_ar": info["description_ar"],

        "actions": " | ".join(plan["action_names"]),
        "actions_ar": " | ".join(plan["action_names_ar"]),

        "cost_tier": plan["cost_tier"],
        "cost_tier_ar": plan["cost_tier_ar"],

        "base_score": plan["base_score"],
        "new_score": plan["new_score"],
        "improvement": plan["improvement"],

        "actions_count": len(plan["actions"]),
        "scored_via_knowledge_only":
            plan.get("scored_via_knowledge_only", False),

        # التقرير التفسيري الجديد
        "report_ar": report,

        # جملة سردية اختيارية (فقط للسيناريوهات المبنية بالبحث التكيّفي
        # في app.py) — سلسلة فارغة إن لم تكن موجودة (السيناريوهات العادية
        # المبنية عبر evaluate_plan لا تملك هذا الحقل).
        "adaptive_summary_ar": plan.get("adaptive_summary_ar", ""),
        "target_range_ar": plan.get("target_range_ar", ""),
    }
def generate_three_budget_scenarios(
    model,
    base_feats,
    feature_cols,
    interventions,
    score_fn=None,
    rare_features=None,
    knowledge_score_fn=None,
    open_score_fn=None,
):
    """
    التصنيف بمستويات كلفة نوعية (low/medium/high) بدل أرقام مالية:
    - الاقتصادي: أفضل توليفة تتكوّن حصرًا من إجراءات "منخفضة" الكلفة.
    - المتوازن: أفضل توليفة لا تتجاوز "متوسطة" (منخفضة + متوسطة مسموح، عالية ممنوعة).
    - المثالي: أفضل توليفة مطلقًا، بلا أي قيد على مستوى الكلفة.

    كل إجراء في interventions يجب أن يحمل "cost_tier": "low"|"medium"|"high"
    بدل "cost" الرقمي القديم. كلفة أي توليفة = أعلى مستوى بين إجراءاتها.

    score_fn / rare_features / knowledge_score_fn: كما في التصميم السابق
    (الدمج المرجّح وحاجز الأمان للسمات النادرة) — تُستخدم للاقتصادي والمتوازن.

    open_score_fn (اختياري): دالة تسجيل منفصلة تُستخدم حصرًا لاختيار
    وعرض السيناريو المثالي، بمعزل تام عن score_fn/rare_features/
    knowledge_score_fn (أي بلا حاجز أمان وبلا دمج معرفة). السبب: درجة
    المعرفة (KNN) لا يمكنها رياضيًا تجاوز أعلى درجة بشرية حقيقية شُوهدت
    في الاستبيان (متوسط مرجَّح لأقرب حالات حقيقية)، فتفرض هذا السقف على
    "المثالي" أيضًا مهما كبُر التدخل. إن لم تُمرَّر، يُستخدم score_fn
    نفسها للمثالي أيضًا (السلوك السابق، بلا تغيير).
    """
    interventions = valid_interventions(interventions, feature_cols)
    effective_open_score_fn = open_score_fn if open_score_fn is not None else score_fn

    if not interventions:
        rows = [
            plan_to_row("low", None, model, base_feats, feature_cols, score_fn,
                        rare_features, knowledge_score_fn),
            plan_to_row("medium", None, model, base_feats, feature_cols, score_fn,
                        rare_features, knowledge_score_fn),
            plan_to_row("open", None, model, base_feats, feature_cols, effective_open_score_fn,
                        None, None),
        ]
        return pd.DataFrame(rows), {"low": None, "medium": None, "open": None}

    combos = _all_combinations(interventions)

    # تقييم الاقتصادي/المتوازن: بالدمج المرجّح وحاجز الأمان كما كان
    evaluated = [
        evaluate_plan(model, base_feats, feature_cols, c, score_fn,
                      rare_features, knowledge_score_fn)
        for c in combos
    ]
    improving = [p for p in evaluated if p["improvement"] > 1e-9]

    low = _best_within_tier_cap(improving, COST_TIER_ORDER["low"])
    medium = _best_within_tier_cap(improving, COST_TIER_ORDER["medium"])

    # تقييم المثالي: مسار منفصل تمامًا (MLP خام إن مُرِّرت open_score_fn)،
    # بلا حاجز أمان السمات النادرة وبلا دمج معرفة — نثق باستقراء النموذج
    # عمدًا هنا فقط، وهذا سبب فصله عن مسار الاقتصادي/المتوازن.
    evaluated_open = [
        evaluate_plan(model, base_feats, feature_cols, c, effective_open_score_fn,
                      None, None)
        for c in combos
    ]
    improving_open = [p for p in evaluated_open if p["improvement"] > 1e-9]
    open_plan = _best_overall(improving_open)

    rows = [
        plan_to_row("low", low, model, base_feats, feature_cols, score_fn,
                    rare_features, knowledge_score_fn),
        plan_to_row("medium", medium, model, base_feats, feature_cols, score_fn,
                    rare_features, knowledge_score_fn),
        plan_to_row("open", open_plan, model, base_feats, feature_cols, effective_open_score_fn,
                    None, None),
    ]
    df = pd.DataFrame(rows)
    plans = {"low": low, "medium": medium, "open": open_plan}
    return df, plans