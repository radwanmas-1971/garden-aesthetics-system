# app.py
import os
import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image, ImageChops
import numpy as np
import joblib
import html
from core.case_memory_builder import build_case_memory
from core.knowledge_discovery import build_knowledge_discovery
from core.knowledge_repository_builder import build_knowledge_repository
from core.reasoning_engine import load_knowledge_repository, build_reasoning
from core.knowledge_weight_learner import learn_human_knowledge_weights
from core.beauty_level_analysis import build_beauty_level_analysis
from core.budget_scenarios import (
    generate_three_budget_scenarios,
    evaluate_plan,
    plan_to_row,
    compute_rare_features,
    _all_combinations,
    valid_interventions,
    apply_deltas as bs_apply_deltas,
    CONSERVED_LAND_COVER_COLS,
    select_real_actions,
    diagnose_weakness_ar,
)
from core.garden_integration import (
    create_unified_garden_design,
)
from core.calibration_learner import learn_calibration_weights
from core.score_calibration import calibrate_score, load_calibration_weights, knowledge_score_from_cases


def autocrop_white(img: Image.Image, bg_threshold: int = 250) -> Image.Image:
    """
    Auto-crop white margins from an image (useful for Graphviz PNG).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")

    # create near-white background
    bg = Image.new("RGB", img.size, (255, 255, 255))

    # difference
    diff = ImageChops.difference(img, bg)

    # make small differences invisible -> keep only non-white areas
    # (convert to grayscale and threshold)
    diff_gray = diff.convert("L")
    # invert threshold: anything darker than bg_threshold is content
    mask = diff_gray.point(lambda p: 255 if p < bg_threshold else 0)

    bbox = mask.getbbox()
    if bbox:
        return img.crop(bbox)
    return img

def resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
    """
    Resize keeping aspect ratio (only if wider than target_w).
    """
    w, h = img.size
    if w <= target_w:
        return img
    new_h = int(h * (target_w / w))
    return img.resize((target_w, new_h), Image.LANCZOS)

def png_bytes_to_pil(png_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(png_bytes))



from core.feature_extractor import run_feature_extraction_on_folder, load_model, analyze_one_image
from core.dataset_builder import build_dataset
from core.trainer import train_mlp_kfold

# ---------------- Desktop pickers (Windows/macOS/Linux local) ----------------
@st.cache_resource(show_spinner=False)
def _gui_dialogs_available():
    """
    يتحقّق فعليًا (وقت التشغيل) من إمكانية فتح نوافذ tkinter في هذه
    البيئة، بدل تخمين ذلك من اسم نظام التشغيل. على استضافة سحابية بلا
    واجهة رسومية (مثل Streamlit Community Cloud) يفشل هذا الفحص فورًا،
    فتُخفى أزرار الاستعراض 📂 تلقائيًا ويُكتفى بحقل نصي — بدل إظهار زر
    يفشل بصمت أو برسالة تحذير عند كل ضغطة. محليًا (Windows/عند توفر
    واجهة رسومية) يبقى الزر يعمل كما كان دائمًا. يُنفَّذ مرة واحدة فقط
    لكل عملية تشغيل (مُخبَّأ عبر st.cache_resource) تفاديًا لفتح/إغلاق
    نافذة Tk فعلية في كل إعادة تشغيل لصفحة Streamlit.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


def pick_folder_dialog(title="Select folder"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        return folder if folder else None
    except Exception as e:
        st.warning(f"Folder picker not available: {e}")
        return None


def pick_file_dialog(title="Select file", filetypes=(("All files", "*.*"),)):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path if path else None
    except Exception as e:
        st.warning(f"File picker not available: {e}")
        return None


# ---------------- Helpers ----------------
def safe_load_image(path):
    try:
        return Image.open(path)
    except Exception:
        return None


def path_exists_ok(p, kind="path"):
    if not os.path.exists(p):
        st.error(f"{kind} not found: {p}")
        return False
    return True
# ============================================================
# أضف هذه الكتلة كـ st.markdown منفصلة بعد الكتلة السابقة (rtl_auto_direction_fix.py)
# تُخصَّص لقائمة st.radio (الصفحات) في الشريط الجانبي:
#   - تعكس ترتيب [دائرة الاختيار + النص] بحيث تصبح الدائرة يمينًا
#     والنص يمتد منها نحو اليسار (قراءة RTL طبيعية).
#   - تُحاذي كامل القائمة إلى الحافة اليمنى للشريط الجانبي بدل
#     التصاقها باليسار.
#   - تُحاذي عنوان الودجت نفسه ("📄 الصفحات") لليمين أيضًا.
# ============================================================

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    .stMarkdown, .stButton, .stTextInput, .stSlider, .stCheckbox, .stRadio,
    .stMetric, .stAlert, h1, h2, h3, h4, h5, h6, p, label {
        direction: rtl;
        text-align: right;
        unicode-bidi: embed;
    }
    /* Keep tabular/code data left-to-right for readability of English
       column names (grass_pct, trees_pct, ...) and numeric values. */
    [data-testid="stDataFrame"], [data-testid="stTable"], code, pre,
    .stDataFrame, .stTable {
        direction: ltr;
        text-align: left;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------- Page Config ----------------
st.set_page_config(page_title="Garden Aesthetics System", layout="wide")

# ---------------- Global RTL (Arabic UI) ----------------
# ============================================================
# ============================================================
# استبدل بهذه الكتلة كاملةً (تحلّ محل النسخة السابقة rtl_sidebar_left_fix.py)
#
# الفرق الجوهري عن النسخة السابقة:
#   - بدل فرض "direction: rtl; text-align: right;" على كل شيء،
#     نستخدم "unicode-bidi: plaintext; text-align: start;"
#   - unicode-bidi: plaintext يجعل المتصفح يحدد اتجاه كل عنصر نصي
#     تلقائيًا بناءً على أول حرف قوي فيه:
#       * إذا بدأ النص بحرف عربي  -> يُعامَل كـ RTL ويُحاذى يمينًا
#       * إذا بدأ النص بحرف/رقم إنجليزي -> يُعامَل كـ LTR ويُحاذى يسارًا
#   - "text-align: start" (وليس right/left) يتبع الاتجاه المكتشف
#     تلقائيًا بدل فرض جهة ثابتة.
#   - حاويات التخطيط (stAppViewContainer/stSidebar) تبقى LTR كما
#     في الحل السابق حتى لا ينعكس موضع الشريط الجانبي.
# ============================================================

st.markdown(
    """
    <style>
    /* 1) إبقاء الهيكل العام LTR حتى لا ينعكس موضع الشريط الجانبي */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        direction: ltr;
    }

    /* 2) محاذاة تلقائية حسب اللغة الفعلية لكل عنصر نصي (شريط جانبي) */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] .stTextInput,
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stCheckbox,
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stMetric,
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        unicode-bidi: plaintext;
        text-align: start;
    }

    /* 3) نفس المحاذاة التلقائية للمحتوى الرئيسي */
    .main .stMarkdown, .main .stButton, .main .stTextInput,
    .main .stNumberInput, .main .stSlider, .main .stCheckbox,
    .main .stRadio, .main .stMetric, .main .stAlert,
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    .main p, .main label, .main span {
        unicode-bidi: plaintext;
        text-align: start;
    }

    /* 4) الجداول/الأكواد تبقى LTR دومًا (أرقام وأسماء أعمدة إنجليزية) */
    [data-testid="stDataFrame"], [data-testid="stTable"], code, pre,
    .stDataFrame, .stTable {
        direction: ltr;
        text-align: left;
    }
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        unicode-bidi: plaintext;
        text-align: start;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------- Session defaults ----------------
if "show_about" not in st.session_state:
    st.session_state["show_about"] = False

# مسار الجذر الافتراضي: نسبي لموقع app.py نفسه (لا مسار مطلق على قرص
# مطوّر بعينه مثل D:\Abd9-2-2026\...) — هذا يجعل المشروع قابلاً للتشغيل
# من أي جهاز أو خادم سحابي بلا تعديل يدوي، بشرط رفع مجلد "data" (ويحوي
# الصور وملف الاستبيان ومخرجات training_outputs/experience_outputs) إلى
# GitHub مع الكود. المسار يبقى قابلاً للتغيير يدويًا من الشريط الجانبي
# في أي وقت كما كان الحال دائمًا — هذا فقط تغيير للقيمة الافتراضية عند
# أول تشغيل.
#
# تجاوز محلي اختياري (local_paths.py): إن وُجد ملف باسم local_paths.py
# بجانب app.py (لا يُرفَع إلى GitHub أبدًا — أضِفه إلى .gitignore)
# ويحوي IMAGE_FOLDER و SURVEY_XLSX، تُستخدَم قيمهما كافتراضي بدل مجلد
# "data" النسبي. هذا يسمح لكل جهاز محلي (مثل جهاز الباحث الحالي) بأن
# يكون له مسار افتراضي دائم خاص به (نفس مسارك القديم مثلاً)، بلا حاجة
# لإنشاء مجلد "data" فعليًا ولا لكتابة المسار يدويًا في كل مرة — بينما
# يبقى الإعداد الافتراضي على أي جهاز/خادم آخر (بلا local_paths.py) هو
# مجلد "data" النسبي كما هو مطلوب للنشر السحابي.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_APP_DIR, "data")
_DEFAULT_IMAGE_FOLDER = os.path.join(_DEFAULT_DATA_DIR, "Pictures")
_DEFAULT_SURVEY_XLSX = os.path.join(_DEFAULT_DATA_DIR, "survey.xlsx")
try:
    import local_paths as _local_paths  # ملف محلي اختياري، غير موجود افتراضيًا
    _DEFAULT_IMAGE_FOLDER = getattr(_local_paths, "IMAGE_FOLDER", _DEFAULT_IMAGE_FOLDER)
    _DEFAULT_SURVEY_XLSX = getattr(_local_paths, "SURVEY_XLSX", _DEFAULT_SURVEY_XLSX)
except ImportError:
    pass

if "image_folder" not in st.session_state:
    st.session_state["image_folder"] = _DEFAULT_IMAGE_FOLDER
if "survey_xlsx" not in st.session_state:
    st.session_state["survey_xlsx"] = _DEFAULT_SURVEY_XLSX

# outputs_dir يتبع فولدر الصور تلقائيًا
outputs_dir = os.path.join(st.session_state["image_folder"], "outputs")

# ---------------- Header (مُوسّط) ----------------
st.markdown(
    """
    <div style="text-align:center; margin-top:6px;">
        <div style="font-size:36px; font-weight:900;">
             🌿 نظام تقييم جمال الحدائق (اطروحة دكتوراه)
        </div>
        <div style="font-size:15px; color:rgba(0,0,0,0.65); margin-top:6px;">
            استخراج السمات ← دمج الاستبيان ← استبعاد الجودة ← تدريب النموذج ← معايرة الدرجة ← المراجعة البصرية
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar Settings ----------------
st.sidebar.header("⚙️ الإعدادات")

st.sidebar.divider()
if st.sidebar.button("ℹ️ عن المشروع"):
    st.session_state["show_about"] = True
    st.rerun()

# تم إخفاء زر "مخطط المنهجية" من الشريط الجانبي لأنه متاح أصلاً ضمن الخطوة 10.

st.sidebar.subheader("📁 Paths")

# أزرار الاستعراض 📂 تعتمد على tkinter (نافذة سطح مكتب) — تُخفى تلقائيًا
# إن لم تتوفّر واجهة رسومية فعليًا (كما على الاستضافة السحابية)، ويُكتفى
# بحقل نصي لإدخال/تعديل المسار يدويًا. الفحص فعلي وقت التشغيل (وليس
# تخمينًا)، انظر _gui_dialogs_available أعلاه.
_gui_ok = _gui_dialogs_available()

# Images folder: العنوان وزر الاستعراض في سطر واحد (إن توفّرت واجهة رسومية)، ومسار المجلد في سطر مستقل تحته.
if _gui_ok:
    img_label_col, img_button_col = st.sidebar.columns([2, 1], gap="small")
else:
    img_label_col = st.sidebar
    img_button_col = None
with img_label_col:
    st.markdown("##### Images folder")
if img_button_col is not None:
    with img_button_col:
        if st.button("📂", key="browse_images_folder", use_container_width=True,
                     help="استعراض واختيار مجلد الصور"):
            chosen = pick_folder_dialog("Select Images Folder")
            if chosen:
                st.session_state["image_folder"] = chosen
                # لا بد من تحديث مفتاح صندوق النص نفسه أيضًا (وليس فقط
                # "image_folder") — وإلا فإن Streamlit يتجاهل معامل value=
                # في المرة القادمة (لأن للصندوق مفتاحًا منفصلًا له حالته
                # الخاصة المحفوظة أصلًا)، فيستمر بعرض المسار القديم، ثم
                # السطر اللاحق (st.session_state["image_folder"] = _img_folder)
                # كان سيُعيد الكتابة فوق الاختيار الجديد بالمسار القديم المعروض.
                st.session_state["images_folder_path_input"] = chosen
                st.rerun()

_img_folder = st.sidebar.text_input(
    "Images folder path",
    value=st.session_state["image_folder"],
    key="images_folder_path_input",
    label_visibility="collapsed",
)

# تحديث بالقيمة المكتوبة يدويًا
st.session_state["image_folder"] = _img_folder
outputs_dir = os.path.join(st.session_state["image_folder"], "outputs")

# Survey Excel: العنوان وزر الاستعراض في سطر واحد (إن توفّرت واجهة رسومية)، ومسار الملف في سطر مستقل تحته.
if _gui_ok:
    survey_label_col, survey_button_col = st.sidebar.columns([2, 1], gap="small")
else:
    survey_label_col = st.sidebar
    survey_button_col = None
with survey_label_col:
    st.markdown("##### Survey Excel")
if survey_button_col is not None:
    with survey_button_col:
        if st.button("📂", key="browse_survey_file", use_container_width=True,
                     help="استعراض واختيار ملف Survey Excel"):
            chosen = pick_file_dialog(
                "Select Survey Excel",
                filetypes=(("Excel files", "*.xlsx *.xls"), ("All files", "*.*")),
            )
            if chosen:
                st.session_state["survey_xlsx"] = chosen
                # نفس إصلاح مجلد الصور أعلاه: تحديث مفتاح صندوق النص نفسه.
                st.session_state["survey_excel_path_input"] = chosen
                st.rerun()

_survey = st.sidebar.text_input(
    "Survey Excel path",
    value=st.session_state["survey_xlsx"],
    key="survey_excel_path_input",
    label_visibility="collapsed",
)

st.session_state["survey_xlsx"] = _survey

# ============================================================
# Reusable numeric stepper — no Streamlit sliders anywhere
# الشكل: زر − | مربع قيمة قابل للكتابة | زر +
# ============================================================
def _stepper_is_integer(min_value, max_value, default_value, step):
    values = (min_value, max_value, default_value, step)
    return all(float(v).is_integer() for v in values)


def _format_stepper_value(value, min_value, max_value, default_value, step):
    if _stepper_is_integer(min_value, max_value, default_value, step):
        return str(int(round(float(value))))
    return f"{float(value):.2f}"


def _stepper_adjust(key, delta, min_value, max_value, default_value, step):
    current = float(st.session_state.get(key, default_value))
    new_value = max(min_value, min(max_value, current + delta))
    if _stepper_is_integer(min_value, max_value, default_value, step):
        new_value = int(round(new_value))

    st.session_state[key] = new_value
    st.session_state[f"{key}__input"] = _format_stepper_value(
        new_value, min_value, max_value, default_value, step
    )


def _stepper_text_changed(key, min_value, max_value, default_value, step):
    input_key = f"{key}__input"
    raw = str(st.session_state.get(input_key, "")).strip().replace(",", ".")

    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(st.session_state.get(key, default_value))

    value = max(min_value, min(max_value, value))
    if _stepper_is_integer(min_value, max_value, default_value, step):
        value = int(round(value))

    st.session_state[key] = value
    st.session_state[input_key] = _format_stepper_value(
        value, min_value, max_value, default_value, step
    )


def numeric_stepper(
    container,
    label,
    min_value,
    max_value,
    default_value,
    step,
    key,
    help_text=None,
):
    """
    Numeric control with exactly one minus button and one plus button.
    The middle value is a plain editable text box, so Streamlit's own
    number-input +/- buttons do not appear.
    """
    input_key = f"{key}__input"

    if key not in st.session_state:
        st.session_state[key] = default_value

    if input_key not in st.session_state:
        st.session_state[input_key] = _format_stepper_value(
            st.session_state[key], min_value, max_value, default_value, step
        )

    container.markdown(f"**{label}**")
    c_minus, c_value, c_plus = container.columns([1, 3.4, 1])

    with c_minus:
        st.button(
            "−",
            key=f"{key}__minus",
            use_container_width=True,
            on_click=_stepper_adjust,
            args=(key, -step, min_value, max_value, default_value, step),
        )

    with c_value:
        text_kwargs = dict(
            label=label,
            key=input_key,
            label_visibility="collapsed",
            on_change=_stepper_text_changed,
            args=(key, min_value, max_value, default_value, step),
        )
        if help_text is not None:
            text_kwargs["help"] = help_text
        st.text_input(**text_kwargs)

    with c_plus:
        st.button(
            "+",
            key=f"{key}__plus",
            use_container_width=True,
            on_click=_stepper_adjust,
            args=(key, step, min_value, max_value, default_value, step),
        )

    return st.session_state[key]


# ---------------- Feature Extraction Options ----------------
st.sidebar.divider()
st.sidebar.subheader("🧩 Feature Extraction Options")
crop_top_ratio = numeric_stepper(
    st.sidebar, "Crop top ratio", 0.0, 0.7, 0.00, 0.01, "crop_top_ratio_control"
)
flower_expand_radius = numeric_stepper(
    st.sidebar, "Flower expand radius", 0, 60, 18, 1, "flower_expand_radius_control"
)
disable_water = st.sidebar.checkbox("Disable water (force water=0)", value=False)
is_aerial = st.sidebar.checkbox(
    "🚁 هذه صور جوية (Aerial/Drone/Nadir)",
    value=False,
    help=(
        "فعِّله إن كانت الصور مُلتقطة من الأعلى (طائرة مسيّرة/جوية). "
        "يُشغِّل تصحيحات إضافية للماء/السياج/الممرات مخصَّصة لهذا المنظور. "
        "أبقِه معطَّلًا للصور الجانبية العادية — تفعيله على صورة جانبية "
        "قد يُحوّل مبنى أو جدارًا حقيقيًا خطأً إلى أرض أو ماء أو أشجار."
    ),
)
max_images = st.sidebar.number_input("Max images (0 = all)", min_value=0, value=0, step=1)

# ---------------- Interpretability ----------------
st.sidebar.subheader("🧾 Interpretability (Not for training)")
show_context_metrics = st.sidebar.checkbox(
    "Show sky/building/wall in tables (interpretation only)", value=True
)
include_context_in_training = st.sidebar.checkbox(
    "Include sky/building/wall as training features (NOT recommended)", value=False
)

# ---------------- Quality Exclusion Thresholds ----------------
st.sidebar.divider()
st.sidebar.subheader("🧹 Quality Exclusion Thresholds")
blur_min = numeric_stepper(
    st.sidebar, "blur_min", 0.0, 300.0, 45.0, 1.0, "quality_blur_min"
)
bright_min = numeric_stepper(
    st.sidebar, "brightness_min", 0.0, 255.0, 40.0, 1.0, "quality_bright_min"
)
bright_max = numeric_stepper(
    st.sidebar, "brightness_max", 0.0, 255.0, 220.0, 1.0, "quality_bright_max"
)
other_max = numeric_stepper(
    st.sidebar, "other_max (%)", 0.0, 100.0, 40.0, 1.0, "quality_other_max"
)
apply_exclusion = st.sidebar.checkbox("Actually exclude bad images", value=True)

# ---------------- Training ----------------
st.sidebar.divider()
st.sidebar.subheader("🧠 Training")
use_water_feature = st.sidebar.checkbox("Use water_pct as feature", value=False)
hidden1 = st.sidebar.number_input("Hidden layer 1", min_value=4, value=32, step=4)
hidden2 = st.sidebar.number_input("Hidden layer 2", min_value=0, value=16, step=4)

# ---------------- Pages ----------------
if "page_selected" not in st.session_state:
    st.session_state["page_selected"] = "1) Extract Features"

# لو تم الضغط على زر القفز
if st.session_state.get("page_jump"):
    st.session_state["page_selected"] = st.session_state["page_jump"]
    st.session_state["page_jump"] = None

def load_features_list(features_path: str):
    with open(features_path, "r", encoding="utf-8") as f:
        cols = [x.strip() for x in f.readlines() if x.strip()]
    return cols

def clip01_5(x):
    return float(np.clip(x, 1.0, 5.0))

def predict_one(model, feats_dict: dict, feature_cols: list):
    x = np.array([[float(feats_dict.get(c, 0.0)) for c in feature_cols]], dtype=np.float32)
    y = float(model.predict(x)[0])
    return clip01_5(y)

def sensitivity_report(model, feats_dict: dict, feature_cols: list, step=1.0):
    """
    Local sensitivity: increase each feature by +step (in percentage points) and see score change.
    Assumes features are in 0..100 (%). We clamp to [0,100].
    """
    base_pred = predict_one(model, feats_dict, feature_cols)
    rows = []
    for c in feature_cols:
        d2 = dict(feats_dict)
        d2[c] = float(np.clip(d2.get(c, 0.0) + step, 0.0, 100.0))
        p2 = predict_one(model, d2, feature_cols)
        rows.append({"feature": c, "delta(+step)": step, "score_change": p2 - base_pred})
    df = pd.DataFrame(rows).sort_values("score_change", ascending=False)
    return base_pred, df

def apply_deltas(feats: dict, deltas: dict):
    out = dict(feats)
    for k, v in deltas.items():
        out[k] = float(out.get(k, 0.0) + v)
    # clamp % to [0,100]
    for k in out:
        if k.endswith("_pct"):
            out[k] = float(np.clip(out[k], 0.0, 100.0))
    return out




PAGE_LABELS_AR = {
    "1) Extract Features": "1) استخراج السمات",
    "2) Browse Images + Overlays": "2) استعراض الصور + التراكب",
    "3) Merge + Exclude": "3) الدمج + الاستبعاد",
    "4) Human Knowledge Learning": "4) تعلّم المعرفة البشرية",
    "5) Train Model": "5) تدريب النموذج",
    "6) Score Calibration (Knowledge-Guided)": "6) معايرة الدرجة (موجّهة بالمعرفة)",
    "7) Landscape Assessment & Explainable AI": "7) تقييم المشهد والذكاء الاصطناعي القابل للتفسير",
    "8) Improvement Planner": "8) مخطط التحسين",
    "9) Garden Integration & Smart Design": "9) دمج الحدائق والتصميم الذكي",
    "10) Methodology Diagram": "10) مخطط المنهجية",
}

page = st.sidebar.radio(
    "📄 الصفحات",
    [
        "1) Extract Features",
        "2) Browse Images + Overlays",
        "3) Merge + Exclude",
        "4) Human Knowledge Learning",
        "5) Train Model",
        "6) Score Calibration (Knowledge-Guided)",
        "7) Landscape Assessment & Explainable AI",
        "8) Improvement Planner",
        "9) Garden Integration & Smart Design",
        "10) Methodology Diagram",
        
    ],
    format_func=lambda x: PAGE_LABELS_AR.get(x, x),
    key="page_selected",
)

# ============================================================
# About Page (RTL scoped فقط هنا – لا يلمس Sidebar)
# ============================================================
if st.session_state.get("show_about", False):

    # ---------- Assets ----------
    ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
    uomosul_logo = safe_load_image(os.path.join(ASSETS_DIR, "uomosul_logo.png"))
    college_logo = safe_load_image(os.path.join(ASSETS_DIR, "college_logo.png"))
    supervisor1_img = safe_load_image(os.path.join(ASSETS_DIR, "supervisor1.jpg"))
    profile_img = safe_load_image(os.path.join(ASSETS_DIR, "profile.jpg"))

    # ---------- CSS scoped ----------
    st.markdown(
        """
    <style>
      .about-scope{direction:rtl; text-align:right; font-family:"Cairo","Tahoma","Arial",sans-serif;}
      .about-title{font-size:32px; font-weight:900; text-align:center; margin:8px 0 2px;}
      .about-sub{font-size:16.5px; font-weight:700; text-align:center; color:rgba(0,0,0,0.68); margin-bottom:14px; line-height:1.9;}
      .divider{height:1px; background:rgba(0,0,0,0.12); margin:12px 0 16px;}

      .card{border:1px solid rgba(0,0,0,0.08); border-radius:18px; padding:16px 18px; background:#fff;
            box-shadow:0 10px 24px rgba(0,0,0,0.06); margin-bottom:14px;}
      .card h3{margin:0 0 10px; font-size:20px; font-weight:900; text-align:center;}

      .muted{color:rgba(0,0,0,0.72); line-height:2.0; font-size:16px;}
      .note{background:rgba(0,118,255,0.06); border:1px solid rgba(0,118,255,0.14);
            padding:12px 14px; border-radius:14px; margin-top:10px; font-size:14.8px; line-height:1.9;}
      .accent{border-right:6px solid rgba(0,118,255,0.35);}
      .row-title{font-weight:900; font-size:18px; margin-bottom:6px;}
      .role{color:rgba(0,0,0,0.68); font-size:15.2px; line-height:1.9;}
      .quote{background:rgba(0,0,0,0.03); border:1px solid rgba(0,0,0,0.08); padding:12px 14px;
             border-radius:14px; margin-top:10px; font-size:15.2px; line-height:1.95;}

      .center-rtl{direction:rtl; text-align:center; unicode-bidi:bidi-override; font-family:"Cairo","Tahoma","Arial",sans-serif; line-height:1.9;}
      .center-rtl h2{font-size:24px; font-weight:900; margin-bottom:12px;}
      .center-rtl p{font-size:16px;}
      .center-rtl ol{list-style-position:inside; padding:0; margin:0 auto; display:inline-block; text-align:right;}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------- Back button ----------
    barL, barR = st.columns([1, 5])
    with barL:
        if st.button("⬅️ العودة للنظام"):
            st.session_state["show_about"] = False
            st.rerun()
    with barR:
        st.markdown("")

    st.markdown('<div class="about-scope">', unsafe_allow_html=True)

    # ---------- Header with logos ----------
    L, C, R = st.columns([1.1, 2.8, 1.1])
    with L:
        if uomosul_logo:
            st.image(uomosul_logo, width=130)
        else:
            st.caption("ضع شعار الجامعة: assets/uomosul_logo.png")
    with C:
        st.markdown('<div class="about-title">نظام تقييم جمال الحدائق بالذكاء الاصطناعي</div>', unsafe_allow_html=True)
        st.markdown('<div class="about-sub">جامعة الموصل – كلية الزراعة والغابات</div>', unsafe_allow_html=True)
    with R:
        if college_logo:
            st.image(college_logo, width=130)
        else:
            st.caption("ضع شعار الكلية: assets/college_logo.png")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---------- Goal card ----------
    st.markdown(
        """
    <div class="center-rtl card">
      <h2>الجهة العلمية وهدف النظام</h2>

      <p>يهدف هذا النظام إلى:</p>

      <ol>
        <li>تحليل صور الحدائق لاستخراج نسب العناصر (عشب، أشجار، زهور، أرض، …) باستخدام نماذج التعلم العميق.</li>
        <li>ربط السمات المستخرجة بتقييمات البشر للجمال لبناء نموذج تنبؤي داعم للبحث.</li>
      </ol>

      <p style="margin-top:14px; font-weight:800;">
        SegFormer – Feature Extraction – Quality Filtering – Dataset Builder – MLP Aesthetics
      </p>

      <div class="note">
        يتم حساب السماء والمباني والجدران لأغراض <b>التفسير والتحليل</b> فقط (Interpretability)،
        ولا يتم إدخالها ضمن ميزات تدريب الجمال افتراضيًا.
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---------- Supervision card ----------
    st.markdown('<div class="card accent">', unsafe_allow_html=True)
    st.markdown('<h3>الإشراف العلمي</h3>', unsafe_allow_html=True)

    r1, r2 = st.columns([3.2, 1.1])
    with r1:
        st.markdown(
            """
        <div class="row-title">أ. د. علي فاروق المعاضيدي</div>
        <div class="role">عميد كلية الزراعة والغابات — جامعة الموصل <b>(المشرف الأول / الإشراف العام)</b></div>
        """,
            unsafe_allow_html=True,
        )
    with r2:
        if supervisor1_img:
            st.image(supervisor1_img, width=120)
        else:
            st.caption("ضع صورة المشرف: assets/supervisor1.jpg")

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    r3, r4 = st.columns([3.2, 1.1])
    with r3:
        st.markdown(
            """
        <div class="row-title">أ. م. د. رضوان محمد عبدالله</div>
        <div class="role"><b>المشرف الثاني</b> — الإشراف المباشر على الجوانب التحليلية والبرمجية وتطوير النظام</div>
        """,
            unsafe_allow_html=True,
        )
    with r4:
        if profile_img:
            st.image(profile_img, width=120)
        else:
            st.caption("ضع صورتك: assets/profile.jpg")

    st.markdown(
        """
      <div class="quote">
        تم تنفيذ العمل البحثي وتطوير هذا النظام تحت الإشراف العلمي العام للأستاذ الدكتور علي،
        وبإشراف مباشر في الجوانب البرمجية والتحليلية من قبل الأستاذ المساعد الدكتور رضوان محمد عبدالله.
      </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)  # end supervision card

    # ---------- Developer card ----------
    st.markdown(
        """
    <div class="card accent center-rtl">
      <h2 style="margin:0 0 10px 0;">مصمم ومطور النظام</h2>
      <div class="muted">
        <b>أ. م. د. رضوان محمد عبدالله</b><br>
        تصميم الواجهة، بناء خط المعالجة، تطوير سكربتات استخراج السمات، تجهيز البيانات،
        التدريب، وتوثيق المنهجية بما يتوافق مع متطلبات مشروع الدكتوراه.
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.info("📌 مخطط المنهجية موجود في صفحة: 9) مخطط المنهجية")

    st.markdown("</div>", unsafe_allow_html=True)  # end about-scope
    st.stop()

# ============================================================
# Main System Pages
# ============================================================

image_folder = st.session_state["image_folder"]
survey_xlsx = st.session_state["survey_xlsx"]

# ---------------- Page 1: Feature Extraction ----------------
if page == "1) Extract Features":
    
    st.subheader("1) استخراج السمات (SegFormer)")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("**Inputs**")
        st.code(image_folder)
        st.code(outputs_dir)

        mi = None if int(max_images) == 0 else int(max_images)

        if st.button("🚀 تشغيل استخراج السمات"):
            if not path_exists_ok(image_folder, "Images folder"):
                st.stop()

            with st.spinner("Running SegFormer feature extraction..."):
                df, csv_path, xlsx_path = run_feature_extraction_on_folder(
                    image_folder=image_folder,
                    outputs_dir=outputs_dir,
                    crop_top_ratio=crop_top_ratio,
                    flower_expand_radius=flower_expand_radius,
                    disable_water=disable_water,
                    is_aerial=is_aerial,
                    max_images=mi,
                )
            st.success("تم بنجاح!")
            st.write("Saved:")
            st.code(csv_path)
            st.code(xlsx_path)

    with c2:
        st.write("**معاينة results.csv**")
        csv_path = os.path.join(outputs_dir, "results.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            st.write("Rows:", len(df))

            context_cols = ["sky_pct", "building_pct", "wall_pct"]
            if not show_context_metrics:
                df_view = df.drop(columns=[c for c in context_cols if c in df.columns], errors="ignore")
            else:
                df_view = df

            st.dataframe(df_view, use_container_width=True)
        else:
            st.info("لا يوجد results.csv بعد. شغّل استخراج السمات أولاً.")

# ---------------- Page 2: Browse Images ----------------
elif page == "2) Browse Images + Overlays":

    if not os.path.isdir(image_folder):
        st.error("مجلد الصور غير صالح.")
        st.stop()

    exts = (".jpg", ".jpeg", ".png")
    imgs = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(exts)]
    st.write("Images found:", len(imgs))

    if len(imgs) == 0:
        st.warning("لم يتم العثور على صور.")
        st.stop()

    chosen = st.selectbox("Choose image", imgs, index=0)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Original**")
        st.image(os.path.join(image_folder, chosen), use_container_width=True)

    with c2:
        st.write("**Overlay**")
        overlay_path = os.path.join(outputs_dir, f"{os.path.splitext(chosen)[0]}_overlay.png")
        if os.path.isfile(overlay_path):
            st.image(overlay_path, use_container_width=True)
        else:
            st.info("لم يتم إنشاء صورة overlay لهذه الصورة بعد. شغّل استخراج السمات أولاً.")


# ---------------- Page 3: Merge + Exclude ----------------
elif page == "3) Merge + Exclude":
    st.subheader("3) دمج الاستبيان + استبعاد الجودة")

    results_csv = st.text_input("📄 results.csv path", value=os.path.join(outputs_dir, "results.csv"))
    out_dir = st.text_input("💾 dataset output dir", value=os.path.join(image_folder, "dataset_outputs"))

    thresholds = {
        "blur_min": float(blur_min),
        "bright_min": float(bright_min),
        "bright_max": float(bright_max),
        "other_max": float(other_max),
    }

    if st.button("🔗 بناء قاعدة البيانات (الكل + النظيفة)"):
        if not path_exists_ok(image_folder, "Images folder"):
            st.stop()
        if not path_exists_ok(results_csv, "results.csv"):
            st.stop()
        if not path_exists_ok(survey_xlsx, "Survey Excel"):
            st.stop()

        with st.spinner("Merging + computing mean/std + quality metrics + exclusion..."):
            all_df, clean_df, all_path, clean_path = build_dataset(
                image_folder=image_folder,
                results_csv_path=results_csv,
                survey_xlsx_path=survey_xlsx,
                out_dir=out_dir,
                thresholds=thresholds,
                apply_exclusion=apply_exclusion,
            )

        st.success("تم إنشاء قاعدة البيانات!")
        st.write("Saved:")
        st.code(all_path)
        st.code(clean_path)
        st.write(f"Before: {len(all_df)} | After: {len(clean_df)} | Dropped: {len(all_df) - len(clean_df)}")

    all_path = os.path.join(out_dir, "dataset_all_with_flags.xlsx")
    clean_path = os.path.join(out_dir, "dataset_clean.xlsx")

    c1, c2 = st.columns(2)
    with c1:
        st.write("**dataset_all_with_flags.xlsx**")
        if os.path.isfile(all_path):
            df = pd.read_excel(all_path)
            context_cols = ["sky_pct", "building_pct", "wall_pct"]
            if not show_context_metrics:
                df = df.drop(columns=[c for c in context_cols if c in df.columns], errors="ignore")
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("لم يتم إنشاؤها بعد.")
    with c2:
        st.write("**dataset_clean.xlsx**")
        if os.path.isfile(clean_path):
            df = pd.read_excel(clean_path)
            context_cols = ["sky_pct", "building_pct", "wall_pct"]
            if not show_context_metrics:
                df = df.drop(columns=[c for c in context_cols if c in df.columns], errors="ignore")
            st.dataframe(df, use_container_width=True)
            
        else:
            st.info("لم يتم إنشاؤها بعد.")


# ---------------- Page 3: Human Knowledge Learning ----------------
elif page == "4) Human Knowledge Learning":
    st.subheader("4) تعلّم المعرفة البشرية")

    dataset_clean = st.text_input(
        "📄 dataset_clean.xlsx path",
        value=os.path.join(image_folder, "dataset_outputs", "dataset_clean.xlsx"),
    )

    out_dir = st.text_input(
        "💾 experience output dir",
        value=os.path.join(image_folder, "experience_outputs"),
    )

    top_k = st.number_input("Top similar cases", min_value=1, max_value=10, value=5, step=1)

    st.write("### ① Build Case Memory")
    if st.button("🧠 بناء ذاكرة الحالات"):
        if not path_exists_ok(dataset_clean, "dataset_clean.xlsx"):
            st.stop()

        with st.spinner("Building case memory and experience database..."):
            res = build_case_memory(
                dataset_clean_path=dataset_clean,
                out_dir=out_dir,
                top_k=int(top_k),
            )

        st.success("تم إنشاء ذاكرة الحالات بنجاح!")

        st.write("### Case Memory")
        st.code(res["paths"]["case_memory"])
        st.dataframe(res["case_memory"], use_container_width=True)

        st.write("### Representative Cases")
        st.code(res["paths"]["representatives"])
        st.dataframe(res["representatives"], use_container_width=True)

        st.write("### Beauty Range Summary")
        st.code(res["paths"]["range_summary"])
        st.dataframe(res["range_summary"], use_container_width=True)

        st.write("### Case Stories")
        st.code(res["paths"]["case_stories"])
        st.dataframe(res["case_stories"], use_container_width=True)

        st.write("### Similarity Matrix")
        st.code(res["paths"]["similarity_matrix"])
        st.dataframe(res["similarity_matrix"], use_container_width=True)

        st.write("### Experience Memory JSON")
        st.code(res["paths"]["json"])

    st.divider()

    st.write("### ② Discover Human Knowledge")
    if st.button("🧩 اكتشاف المعرفة البشرية"):
        if not path_exists_ok(dataset_clean, "dataset_clean.xlsx"):
            st.stop()

        with st.spinner("Discovering human knowledge from previous cases..."):
            res2 = build_knowledge_discovery(
                dataset_clean_path=dataset_clean,
                out_dir=out_dir,
            )

        st.success("تم اكتشاف المعرفة بنجاح!")

        st.write("### Feature Trends")
        st.code(res2["paths"]["feature_trends"])
        st.dataframe(res2["feature_trends"], use_container_width=True)

        st.write("### Feature Correlations")
        st.code(res2["paths"]["feature_correlations"])
        st.dataframe(res2["feature_correlations"], use_container_width=True)

        st.write("### Beauty Level Patterns")
        st.code(res2["paths"]["beauty_level_patterns"])
        st.dataframe(res2["beauty_level_patterns"], use_container_width=True)

        st.write("### Knowledge Rules")
        st.code(res2["paths"]["knowledge_rules"])
        st.dataframe(res2["knowledge_rules"], use_container_width=True)

        st.write("### Exceptional Cases")
        st.code(res2["paths"]["exceptional_cases"])
        st.dataframe(res2["exceptional_cases"], use_container_width=True)

        st.write("### Knowledge Discovery JSON")
        st.code(res2["paths"]["json"])

        st.divider()

    st.write("### ③ Learn Human Knowledge Weights")
    if st.button("⚖️ تعلّم أوزان المعرفة البشرية"):
        if not path_exists_ok(dataset_clean, "dataset_clean.xlsx"):
            st.stop()

        with st.spinner("Learning human knowledge weights from survey-based scores..."):
            weights_df, weights_path, weights_json = learn_human_knowledge_weights(
                dataset_clean_path=dataset_clean,
                out_dir=out_dir,
            )

        st.success("تم إنشاء أوزان المعرفة البشرية بنجاح!")
        st.code(weights_path)
        st.code(weights_json)
        st.dataframe(weights_df, use_container_width=True)






    st.divider()

    st.write("### Beauty Level Analysis")
    if st.button("📊 بناء جدول مستويات الجمال"):
        if not path_exists_ok(dataset_clean, "dataset_clean.xlsx"):
            st.stop()

        with st.spinner("Building beauty level master table..."):
            summary_df, images_df, master_path, json_path = build_beauty_level_analysis(
                dataset_clean_path=dataset_clean,
                out_dir=out_dir,
            )

        st.success("تم إنشاء جدول مستويات الجمال بنجاح!")

        st.write("### Summary")
        st.code(master_path)
        st.dataframe(summary_df, use_container_width=True)

        st.write("### All Images")
        st.dataframe(images_df, use_container_width=True)

        st.write("### JSON")
        st.code(json_path)






    st.write("### ④ Build Knowledge Repository")
    if st.button("📦 بناء قاعدة المعرفة"):
        if not os.path.isdir(out_dir):
            st.error(f"experience output dir not found: {out_dir}")
            st.stop()

        repo, repo_path, summary_path, summary = build_knowledge_repository(
            experience_dir=out_dir,
            out_dir=out_dir,
        )

        st.success("تم إنشاء قاعدة المعرفة بنجاح!")
        st.code(repo_path)
        st.code(summary_path)
        st.write(summary)

# ---------------- Page 5: Training ----------------
elif page == "5) Train Model":
    st.subheader("5) تدريب نموذج MLP (تحقق متقاطع 5-fold)")

    dataset_clean = st.text_input(
        "📄 dataset_clean.xlsx path",
        value=os.path.join(image_folder, "dataset_outputs", "dataset_clean.xlsx"),
    )

    train_out_dir = st.text_input("💾 training outputs dir", value=os.path.join(image_folder, "training_outputs"))

    if st.button("🧠 ابدأ التدريب"):
        if not path_exists_ok(dataset_clean, "dataset_clean.xlsx"):
            st.stop()

        with st.spinner("Training (5-fold CV)..."):
            res = train_mlp_kfold(
                dataset_xlsx=dataset_clean,
                out_dir=train_out_dir,
                use_water_feature=use_water_feature,
                include_context_features=include_context_in_training,
                hidden=(int(hidden1), int(hidden2)) if int(hidden2) > 0 else (int(hidden1),),
            )

        st.success("اكتمل التدريب!")
        st.write("**Summary:**")
        st.json(res["summary"])

        st.write("**Per-fold:**")
        st.dataframe(res["folds"], use_container_width=True)

        st.write("Saved files:")
        st.code(res["model_path"])
        st.code(res["pred_path"])
        if res.get("imp_path"):
            st.code(res["imp_path"])

        try:
            pred_df = pd.read_csv(res["pred_path"])
            pred_df["abs_err"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
            st.write("**Top 15 errors:**")
            st.dataframe(pred_df.sort_values("abs_err", ascending=False).head(15), use_container_width=True)
        except Exception:
            pass


# ---------------- Page 6: Score Calibration ----------------
elif page == "6) Score Calibration (Knowledge-Guided)":
    st.subheader("6) محرك معايرة الدرجة الموجّه بالمعرفة")

    st.markdown(
        """
        <div dir="rtl" style="text-align:right; font-size:15.5px; line-height:2;
                    background:rgba(0,118,255,0.06); border:1px solid rgba(0,118,255,0.14);
                    border-radius:12px; padding:12px 16px; margin-bottom:10px;">
        هذه المرحلة تجعل <b>الاستبيان (تقييم البشر الحقيقي)</b> هو المرجع الأساسي لدرجة الجمال،
        وليس فقط نموذج MLP. لأي صورة جديدة، يبحث النظام عن أقرب حالات حقيقية من قاعدة الخبرة
        (case_memory) ويحسب منها <b>درجة مبنية على المعرفة (Knowledge Score)</b>، ثم يدمجها مع
        درجة MLP حسب مدى الثقة بكل تقدير، لتُنتج <b>الدرجة المعايرة النهائية</b> التي تُستخدم
        بعد ذلك في التفسير (Reasoning) وخطة التحسين (Improvement Planner).
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_clean = st.text_input(
        "📄 dataset_clean.xlsx path",
        value=os.path.join(image_folder, "dataset_outputs", "dataset_clean.xlsx"),
        key="calib_dataset_clean",
    )

    train_out_dir = st.text_input(
        "💾 training outputs dir",
        value=os.path.join(image_folder, "training_outputs"),
        key="calib_train_out_dir",
    )

    exp_out_dir = st.text_input(
        "💾 experience output dir",
        value=os.path.join(image_folder, "experience_outputs"),
        key="calib_exp_out_dir",
    )

    if st.button("⚖️ تعلّم أوزان المعايرة"):
        if not path_exists_ok(dataset_clean, "dataset_clean.xlsx"):
            st.stop()

        mlp_cv_path = os.path.join(train_out_dir, "mlp_predictions_cv.csv")

        with st.spinner("Running Leave-One-Out K-sweep and fusing with MLP accuracy..."):
            weights, sweep_df, json_path, sweep_path = learn_calibration_weights(
                dataset_clean_path=dataset_clean,
                out_dir=exp_out_dir,
                mlp_cv_predictions_path=mlp_cv_path,
            )

        st.success("تم تعلّم أوزان المعايرة بنجاح!")

        st.write("### K-sweep (Leave-One-Out on real survey cases)")
        st.dataframe(sweep_df, use_container_width=True)
        st.caption(f"Best K = {weights['k_used']} (lowest LOO-MAE = {weights['mae_knowledge']:.3f})")

        st.write("### Fusion weights")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("MLP CV MAE", f"{weights['mae_mlp']:.3f}" if weights["mae_mlp"] else "N/A")
        with c2:
            st.metric(f"Knowledge LOO-MAE (K={weights['k_used']})", f"{weights['mae_knowledge']:.3f}")
        with c3:
            st.metric("w_knowledge (global)", f"{weights['w_knowledge']*100:.1f}%")

        st.info(weights["fusion_note"])

        st.write("Saved files:")
        st.code(json_path)
        st.code(sweep_path)

    st.divider()
    st.caption(
        "Run this stage once after (re)training the MLP model, so the calibration weights "
        "reflect the latest model accuracy. Stages 7 and 8 will automatically use the saved "
        "calibration_weights.json to produce the final calibrated score."
    )


# ---------------- Page 7: Landscape Assessment ----------------
elif page == "7) Landscape Assessment & Explainable AI":
    st.subheader("7) تقييم المشهد والذكاء الاصطناعي القابل للتفسير")

    train_out_dir = st.text_input(
        "💾 training outputs dir",
        value=os.path.join(image_folder, "training_outputs"),
        key="assessment_train_out_dir",
    )

    model_path = os.path.join(train_out_dir, "mlp_model.joblib")
    features_path = os.path.join(train_out_dir, "training_features.txt")
    repo_path = os.path.join(image_folder, "experience_outputs", "knowledge_repository.json")

    if not os.path.isfile(model_path) or not os.path.isfile(features_path):
        st.error("النموذج أو training_features.txt غير موجود. الرجاء تدريب النموذج أولاً.")
        st.code(model_path)
        st.code(features_path)
        st.stop()

    uploaded = st.file_uploader(
        "اختر صورة حديقة جديدة من جهازك",
        type=["jpg", "jpeg", "png"],
        key="assessment_uploaded_image",
    )

    if uploaded is None:
        st.info("ارفع صورة لبدء التقييم.")
        st.stop()

    temp_dir = os.path.join(outputs_dir, "temp_assessment")
    os.makedirs(temp_dir, exist_ok=True)
    temp_image = os.path.join(temp_dir, uploaded.name)

    with open(temp_image, "wb") as f:
        f.write(uploaded.getbuffer())

    with st.spinner("Extracting SegFormer features from the uploaded image..."):
        processor, seg_model = load_model()
        features, overlay_path, _ = analyze_one_image(
            image_path=temp_image,
            processor=processor,
            model=seg_model,
            out_dir=temp_dir,
            crop_top_ratio=crop_top_ratio,
            flower_expand_radius=flower_expand_radius,
            disable_water=disable_water,
            is_aerial=is_aerial,
        )

    r = {
        "grass_pct": features.get("grass", 0.0),
        "trees_pct": features.get("trees", 0.0),
        "flowers_pct": features.get("flowers", 0.0),
        "ground_pct": features.get("ground", 0.0),
        "water_pct": features.get("water", 0.0),
        "sky_pct": features.get("sky", 0.0),
        "building_pct": features.get("building", 0.0),
        "wall_pct": features.get("wall", 0.0),
    }

    c1, c2 = st.columns(2)
    with c1:
        st.write("**الصورة المرفوعة**")
        st.image(temp_image, use_container_width=True)
    with c2:
        st.write("**صورة التراكب (SegFormer)**")
        if os.path.isfile(overlay_path):
            st.image(overlay_path, use_container_width=True)
        else:
            st.info("لم يتم إنشاء صورة overlay.")

    context = {k: r.get(k, None) for k in ["sky_pct", "building_pct", "wall_pct"] if k in r}
    st.write("### السمات المستخرجة (نسب مئوية)")
    base_cols = [c for c in ["grass_pct", "trees_pct", "flowers_pct", "ground_pct", "water_pct"] if c in r]
    st.dataframe(pd.DataFrame([{c: r.get(c, 0.0) for c in base_cols}]), use_container_width=True)

    if show_context_metrics and context:
        st.write("### مقاييس السياق (للتفسير فقط)")
        st.dataframe(pd.DataFrame([context]), use_container_width=True)

    model = joblib.load(model_path)
    feature_cols = load_features_list(features_path)

    st.write("### التنبؤ")
    mlp_score = predict_one(model, r, feature_cols)

    calib_weights_path = os.path.join(image_folder, "experience_outputs", "calibration_weights.json")

    if os.path.isfile(repo_path):
        repo = load_knowledge_repository(repo_path)
        calib = calibrate_score(
            mlp_score=mlp_score, feats=r, repo=repo, feature_cols=feature_cols,
            weights_path=calib_weights_path,
        )
        score = calib["calibrated_score"]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("MLP score (raw)", f"{calib['mlp_score']:.3f}")
        with m2:
            if calib.get("knowledge_score") is not None:
                st.metric(f"Knowledge score (K={calib['k_used']} survey cases)", f"{calib['knowledge_score']:.3f}")
            else:
                st.metric("Knowledge score", "N/A")
        with m3:
            st.metric("Calibrated final score", f"{score:.3f}")

        st.caption(f"Beauty level (calibrated): {calib['beauty_level']}")
        st.markdown(
            f"""
            <div dir="rtl" style="text-align:right; font-size:15px; line-height:1.9; margin-top:4px;">
            • {html.escape(calib.get("note_ar", ""))}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if calib.get("neighbors"):
            with st.expander("Nearest survey-grounded cases used"):
                st.dataframe(pd.DataFrame(calib["neighbors"]), use_container_width=True)
        if not os.path.isfile(calib_weights_path):
            st.info(
                "Using fallback calibration weights (K=7, from the LOO analysis on the "
                "125-case survey). Run Stage 6 once to learn weights tailored to your latest model."
            )
    else:
        score = mlp_score
        st.metric("Predicted aesthetics score (1..5)", f"{score:.3f}")

    st.write("### 🧠 Knowledge Reasoning Explanation")
    if os.path.isfile(repo_path):
        reasoning = build_reasoning(feats=r, predicted_score=score, repo=repo, feature_cols=feature_cols)

        st.write("#### Similar previous cases")
        if reasoning.get("similar_cases"):
            st.dataframe(pd.DataFrame(reasoning["similar_cases"]), use_container_width=True)
        else:
            st.info("لم يتم العثور على حالات مشابهة في قاعدة المعرفة.")

        st.write("#### Matched knowledge rules")
        if reasoning.get("matched_rules"):
            st.dataframe(pd.DataFrame(reasoning["matched_rules"]), use_container_width=True)
        else:
            st.info("لم يتم العثور على قواعد معرفة مطابقة.")

        st.write("#### Arabic reasoning")
        for line in reasoning.get("reasoning_ar", []):
            st.markdown(
                f"""
                <div dir="rtl" style="text-align:right; font-size:18px; line-height:2; margin-bottom:12px;">
                • {html.escape(str(line))}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("قاعدة المعرفة غير موجودة. الرجاء تشغيل المرحلة 4: بناء قاعدة المعرفة أولاً.")
        st.code(repo_path)

    st.write("### Local sensitivity (what helps most if increased by +1%)")
    _, sens_df = sensitivity_report(model, r, feature_cols, step=1.0)
    st.dataframe(sens_df, use_container_width=True)


# ---------------- Page 8: Improvement Planner v5 ----------------
elif page == "8) Improvement Planner":
    st.subheader("8) مخطط التحسين v5 (البدائل الخبيرة النهائية)")

    # ============================================================
    # Helper functions used only inside Page 7
    # ============================================================

    def _beauty_level_ar(score):
        score = float(score)
        if score < 3:
            return "منخفض"
        if score < 4:
            return "متوسط"
        return "مرتفع"

    def _scenario_color(key):
        if key == "low":
            return "#2e7d32", "#f1f8f4"
        if key == "medium":
            return "#b28704", "#fff9e6"
        return "#b71c1c", "#fff0f0"

    def _feature_ar(name):
        return {
            "grass_pct": "العشب",
            "trees_pct": "الأشجار",
            "flowers_pct": "الزهور",
            "ground_pct": "الأرض المكشوفة",
            "water_pct": "الماء",
        }.get(name, name)

    def _action_reason_ar(action_name):
        reasons = {
            "Add flowers bed": "تم اختيار إضافة أحواض الزهور لأنها تزيد التنوع اللوني، وهو من أكثر العوامل التي ترفع الانطباع الجمالي للحديقة.",
            "Improve lawn/soil quality": "تم اختيار تحسين المسطح الأخضر والتربة لأنه يزيد انتظام الغطاء النباتي ويقلل ظهور الأرض المكشوفة.",
            "Plant trees and shrubs": "تم اختيار زراعة أشجار وشجيرات لأنها تضيف عمقًا بصريًا وتوازنًا رأسيًا للمشهد، وتقترب من نسب الغطاء الشجري في الحدائق عالية الجمال.",
            "Add small water feature": "تم اختيار إضافة عنصر مائي صغير (كنافورة أو حوض صغير) لأن الماء أثبت تأثيرًا قويًا جدًا على الجمال المُتوقَّع وفق النموذج، حتى بكمية محدودة.",
            "Add medium water feature": "تم اختيار إضافة عنصر مائي متوسط الحجم (كبركة صغيرة) لأنه يوازن بين الكلفة والتحسّن الملموس في درجة الجمال.",
            "Reduce bare ground (mulch/cover)": "تم اختيار تقليل الأرض المكشوفة لأن ظهور الأرض العارية يخفض الإحساس بالاكتمال والجاذبية البصرية.",
            "Add/Enhance water element": "تم اختيار عنصر الماء لأنه قد يضيف نقطة جذب بصرية إذا كان الماء مستخدمًا ضمن ميزات التدريب.",
            "Increase green spaces": "تم اختيار زيادة المساحات الخضراء لأنها الأساس الأول لأي تحسين تدريجي، وتقلل ظهور الأرض المكشوفة.",
            "Add shrubs": "تم اختيار إضافة شجيرات لأنها تكمّل المساحات الخضراء بعمق بصري إضافي بكلفة منخفضة نسبيًا.",
            "Extensive lawn/soil renovation": "تم اختيار توسيع تجديد المسطح الأخضر والتربة فوق ما سبق لأنه أثبت تجريبيًا أقوى تأثير فردي على الجمال المُتوقَّع بين كل الإجراءات غير المائية.",
            "Additional canopy growth": "تم اختيار زيادة إضافية في الغطاء الشجري فوق ما سبق للاقتراب أكثر من نطاق الدرجة المطلوبة للسيناريو المتوازن دون اللجوء للماء.",
            "Add water pond/fountain": "تم اختيار إضافة بركة أو نافورة ماء كإضافة نهائية للحزمة الكاملة، لأن الماء أثبت أقوى تأثير منفرد على الجمال المُتوقَّع وفق النموذج.",
            "Improve lawn/soil quality (adaptive)": "دلتا هذا الإجراء محسوبة تكيّفيًا خصيصًا لهذه الصورة (بحث تدريجي)، وليست رقمًا ثابتًا — لضمان وقوع الدرجة الناتجة ضمن نطاق السيناريو الاقتصادي المطلوب فعليًا.",
            "Extensive lawn/soil renovation (adaptive)": "دلتا هذا الإجراء محسوبة تكيّفيًا خصيصًا لهذه الصورة (بحث تدريجي)، وليست رقمًا ثابتًا — لضمان وقوع الدرجة الناتجة ضمن نطاق السيناريو المتوازن المطلوب فعليًا.",
        }
        if action_name in reasons:
            return reasons[action_name]

        # ------------------------------------------------------------
        # الإجراءات الجديدة (مكتبة ACTION_LIBRARY في budget_scenarios.py)
        # جملة عربية كاملة وليست مفتاحًا إنجليزيًا ثابتًا كما أعلاه — لذلك
        # لا تُطابِق القاموس السابق أبدًا وكانت تسقط دائمًا إلى رسالة عامة
        # واحدة متطابقة لكل الإجراءات (خطأ رُصِد فعليًا في تشغيل حقيقي:
        # كل الإجراءات المختلفة كانت تُعرَض بنفس جملة السبب). الحل: تحديد
        # السمة المستهدفة من محتوى الجملة نفسها، لإعطاء سبب مناسب فعليًا
        # لكل فئة سمة بدل رسالة عامة واحدة للجميع.
        text = action_name
        if any(k in text for k in ["مائي", "نافورة", "بركة", "شلال", "جدار مائي", "مجرى مائي"]):
            return ("تم اختيار هذا الإجراء لأن الماء أثبت تأثيرًا قويًا جدًا على الجمال "
                    "المُتوقَّع وفق النموذج، حتى بكمية محدودة نسبيًا.")
        if any(k in text for k in ["زهو", "أزهار"]):
            return ("تم اختيار هذا الإجراء لأنه يزيد التنوع اللوني، وهو من أكثر العوامل "
                    "التي ترفع الانطباع الجمالي للحديقة.")
        if any(k in text for k in ["شجر", "شجير", "طبقات نباتية", "غطاء نباتي مرتفع"]):
            return ("تم اختيار هذا الإجراء لأنه يضيف عمقًا بصريًا وتوازنًا رأسيًا للمشهد، "
                    "ويُقرِّب من نسب الغطاء الشجري في الحدائق عالية الجمال.")
        if any(k in text for k in ["مسطح أخضر", "المسطح الأخضر", "العشب", "التربة", "الري",
                                     "تسميد", "جزّ", "بذر", "تهوية"]):
            return ("تم اختيار هذا الإجراء لأنه يزيد انتظام الغطاء النباتي ويقلل ظهور الأرض "
                    "المكشوفة، وفق ما رصده البحث التكيّفي لهذه الصورة تحديدًا.")
        if any(k in text for k in ["أرض مكشوفة", "الأرض المكشوفة", "نشارة", "غطاء أرضي",
                                     "حواف", "ممرات"]):
            return ("تم اختيار هذا الإجراء لأن تقليل ظهور الأرض العارية يرفع الإحساس "
                    "بالاكتمال والتماسك البصري للمشهد.")
        return ("تم اختيار هذا الإجراء لأنه يحقق تحسنًا متوقعًا في درجة الجمال وفق البحث "
                "التكيّفي لهذه الصورة تحديدًا.")

    def _before_after_df(base_feats, new_feats, feature_cols):
        rows = []
        for f in feature_cols:
            if not f.endswith("_pct"):
                continue
            before = float(base_feats.get(f, 0.0))
            after = float(new_feats.get(f, before))
            rows.append({
                "السمة": _feature_ar(f),
                "قبل (%)": round(before, 2),
                "بعد (%)": round(after, 2),
                "التغير (%)": round(after - before, 2),
            })
        return pd.DataFrame(rows)

    def _build_interventions(feature_cols):
        # ============================================================
        # الدلتا مُعايَرة على متوسطات فعلية للصور "عالية الجمال" في بيانات
        # الاستبيان (مرحلة 4، Feature Trends): grass≈22%, trees≈44% —
        # وليست أرقامًا افتراضية. هذا مدعوم أيضًا بمعيار غطاء الأشجار
        # الحضري العالمي (20-40%، American Forests / NYC Urban Forest
        # Plan 2030: هدف 30%) كتحقق خارجي مستقل.
        #
        # الإجراءات الصغيرة السابقة (+3 إلى +6%) لم تكن كافية لنقل صورة
        # نموذجية (عشب>65%, أشجار<20%) إلى ملامح "عالية الجمال" فعليًا —
        # حتى مجموع كل الإجراءات القديمة معًا لم يتجاوز +15-20% تقريبًا.
        # الدلتا هنا أكبر بكثير لتعكس حجم التغيير الحقيقي الذي يُحدثه
        # مشروع تجديد شامل بميزانية مفتوحة، بدل تصنيع درجة نهائية مباشرة.
        interventions = [
            {
                "name": "Add flowers bed",
                "deltas": {"flowers_pct": 9, "ground_pct": -9},
                "cost_tier": "low",
            },
            {
                "name": "Improve lawn/soil quality",
                "deltas": {"grass_pct": 12, "ground_pct": -12},
                "cost_tier": "low",
            },
            {
                "name": "Plant trees and shrubs",
                "deltas": {"trees_pct": 20, "ground_pct": -20},
                "cost_tier": "medium",
            },
            {
                "name": "Reduce bare ground (mulch/cover)",
                "deltas": {"ground_pct": -10, "grass_pct": 6, "flowers_pct": 4},
                "cost_tier": "low",
            },
        ]

        # الماء "سمة نادرة إحصائيًا" في بيانات التدريب — دليل تجريبي مؤكَّد:
        # حتى أصغر كمية ماء ممكنة تدفع النموذج فورًا لسقف 5.0 (علاقة شبه
        # ثنائية وليست تدريجية واقعية، بسبب ندرة الأمثلة الحقيقية للماء في
        # التدريب). "Add tiny water feature" (+1%) اختبار تشخيصي لمعرفة هل
        # العتبة أدق من 3% المُختبَرة سابقًا، أم أن أي كمية غير صفرية كافية.
        if "water_pct" in feature_cols:
            interventions.append({
                "name": "Add tiny water feature",
                "deltas": {"water_pct": 1, "ground_pct": -1},
                "cost_tier": "medium",
            })
            interventions.append({
                "name": "Add/Enhance water element",
                "deltas": {"water_pct": 8, "ground_pct": -8},
                "cost_tier": "high",
            })

        return interventions

    def _render_metric_cards(current_score, feature_cols):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current aesthetics score", f"{current_score:.3f} / 5")
        with c2:
            st.metric("Beauty level", _beauty_level_ar(current_score))
        with c3:
            st.metric("Training features", len(feature_cols))

    def _render_scenario_card(row, plan, base_feats, feature_cols, current_score_displayed=None):
        border, bg = _scenario_color(row["scenario_key"])

        scored_via_knowledge_only = row.get("scored_via_knowledge_only", False)

        note_html = ""
        if scored_via_knowledge_only and current_score_displayed is not None:
            note_html = f"""
<p style="font-size:14px; color:#7a5c00; background:#fff8e1; border-radius:8px; padding:8px 12px;">
⚠️ <b>ملاحظة:</b> أحد الإجراءات في هذا السيناريو يُغيّر سمة نادرة إحصائيًا (مثل الزهور)،
لذلك قُيِّمت "الدرجة الحالية" و"الدرجة المتوقعة" هنا بدرجة المعرفة البشرية وحدها
(<b>{row['base_score']:.2f}</b>)، وليس بالدرجة المعايَرة المعروضة أعلى الصفحة
(<b>{current_score_displayed:.2f}</b>). هذا مقصود لضمان مقارنة "قبل" و"بعد" بنفس
المقياس تمامًا، وليس خطأ حسابيًا.
</p>
"""

        if row.get("target_reached") is False:
            note_html += f"""
<p style="font-size:14px; color:#7a1f1f; background:#fdecec; border-radius:8px; padding:8px 12px;">
⚠️ الدرجة المعروضة ({row['new_score']:.2f}) هي <b>أقصى ما استطاع النظام
الوصول إليه</b> لهذه الصورة، ولم تصل فعليًا للنطاق المطلوب لهذا
السيناريو ({row.get('target_range_ar', '')}).
</p>
"""

        report_ar = row.get("report_ar", "") or ""
        report_html = ""
        if report_ar:
            report_paragraphs = "".join(
                f"<p style='margin:6px 0;'>{line}</p>"
                for line in report_ar.split("\n") if line.strip()
            )
            report_html = f"""
<div style="
margin-top:14px;
padding:14px 16px;
background:rgba(255,255,255,0.55);
border-radius:12px;
border:1px solid rgba(0,0,0,0.08);
font-size:16px;
line-height:1.9;
">
<p style="margin:0 0 8px 0;"><b>📝 التقرير التفسيري</b></p>
{report_paragraphs}
</div>
"""

        st.markdown(
            f"""
<div dir="rtl" style="
text-align:right;
line-height:2;
font-size:18px;
border:2px solid {border};
background:{bg};
border-radius:18px;
padding:20px;
margin-bottom:22px;
">

<h3>{row['scenario_ar']}</h3>

<p><b>الوصف:</b> {row['description_ar']}</p>
<p><b>مستوى الكلفة:</b> {row['cost_tier_ar']}</p>
<p><b>الدرجة الحالية:</b> {row['base_score']:.2f}</p>
<p><b>الدرجة المتوقعة بعد التحسين:</b> {row['new_score']:.2f}</p>
<p><b>مقدار التحسن:</b> {row['improvement']:.2f}</p>
<p><b>عدد الإجراءات:</b> {int(row['actions_count'])}</p>
<p><b>الإجراءات المقترحة:</b> {row['actions_ar']}</p>
{f'<p style="font-weight:600; color:#1a4d2e;">🎯 {row["adaptive_summary_ar"]}</p>' if row.get('adaptive_summary_ar') else ''}
{note_html}
{report_html}
</div>
""",
            unsafe_allow_html=True,
        )

        if plan is not None:
            st.write("#### قبل مقابل بعد")
            bf_df = _before_after_df(base_feats, plan["new_feats"], feature_cols)
            st.dataframe(bf_df, use_container_width=True)

            st.markdown(
                """
<div dir="rtl" style="text-align:right; line-height:2; font-size:17px;">
<b>سبب اختيار الإجراءات:</b>
</div>
""",
                unsafe_allow_html=True,
            )

            for action in plan.get("actions", []):
                ar_name = {
                    "Add flowers bed": "إضافة أحواض زهور",
                    "Improve lawn/soil quality": "تحسين المسطح الأخضر والتربة",
                    "Plant trees and shrubs": "زراعة أشجار وشجيرات",
                    "Reduce bare ground (mulch/cover)": "تقليل الأرض المكشوفة",
                    "Add small water feature": "إضافة عنصر مائي صغير",
                    "Add medium water feature": "إضافة عنصر مائي متوسط",
                    "Add/Enhance water element": "إضافة/تعزيز عنصر الماء",
                    "Increase green spaces": "زيادة المساحات الخضراء",
                    "Add shrubs": "إضافة شجيرات",
                    "Extensive lawn/soil renovation": "توسيع تجديد المسطح الأخضر والتربة",
                    "Additional canopy growth": "زيادة إضافية في الغطاء الشجري",
                    "Add water pond/fountain": "إضافة بركة/نافورة ماء",
                    "Improve lawn/soil quality (adaptive)": "تحسين المسطح الأخضر والتربة (تكيّفي)",
                    "Extensive lawn/soil renovation (adaptive)": "توسيع تجديد المسطح الأخضر والتربة (تكيّفي)",
                }.get(action["name"], action["name"])

                st.markdown(
                    f"""
<div dir="rtl" style="text-align:right; line-height:2; font-size:16px;">
✓ <b>{ar_name}</b>: {_action_reason_ar(action['name'])}
</div>
""",
                    unsafe_allow_html=True,
                )

    def _render_expert_recommendation(scenarios_df):
        valid_df = scenarios_df[scenarios_df["improvement"] > 0].copy()

        if len(valid_df) == 0:
            st.warning("لم يتم العثور على بدائل تحسين تحقق زيادة في الدرجة المتوقعة.")
            return

        balanced = valid_df[valid_df["scenario_key"] == "medium"]
        if len(balanced):
            recommended = balanced.iloc[0]
        else:
            recommended = valid_df.sort_values(["new_score", "improvement"], ascending=False).iloc[0]

        low = valid_df[valid_df["scenario_key"] == "low"]
        ideal = valid_df[valid_df["scenario_key"] == "open"]

        low_txt = ""
        if len(low):
            low_row = low.iloc[0]
            low_txt = f" أما السيناريو الاقتصادي فيحقق تحسنًا مقداره {low_row['improvement']:.2f} بمستوى كلفة {low_row['cost_tier_ar']}، وهو مناسب عند محدودية الميزانية."

        ideal_txt = ""
        if len(ideal):
            ideal_row = ideal.iloc[0]
            ideal_txt = f" بينما يهدف السيناريو المثالي إلى الوصول لأعلى جودة ممكنة بدرجة متوقعة {ideal_row['new_score']:.2f} بمستوى كلفة {ideal_row['cost_tier_ar']}."

        st.markdown(
            f"""
<div dir="rtl" style="
text-align:right;
line-height:2;
font-size:18px;
border:2px solid #1565c0;
background:#eef6ff;
border-radius:18px;
padding:20px;
margin-top:20px;
">

<h3>🧠 توصية الخبير</h3>

<p>
يوصى باعتماد <b>{recommended['scenario_ar']}</b> كخيار عملي متوازن؛
لأنه يرفع درجة الجمال المتوقعة من <b>{recommended['base_score']:.2f}</b>
إلى <b>{recommended['new_score']:.2f}</b>، مع تحسن مقداره
<b>{recommended['improvement']:.2f}</b> وبمستوى كلفة <b>{recommended['cost_tier_ar']}</b>.
</p>

<p>
{low_txt}
{ideal_txt}
</p>

<p>
بذلك لا يفرض النظام حلًا واحدًا، بل يقدم بدائل يمكن الاختيار بينها وفق مستوى الكلفة والهدف الجمالي المطلوب.
</p>

</div>
""",
            unsafe_allow_html=True,
        )

    # ============================================================
    # 1) Required paths
    # ============================================================
    train_out_dir = st.text_input(
        "💾 training outputs dir",
        value=os.path.join(image_folder, "training_outputs"),
        key="planner_v5_train_out_dir",
    )

    model_path = os.path.join(train_out_dir, "mlp_model.joblib")
    features_path = os.path.join(train_out_dir, "training_features.txt")
    repo_path = os.path.join(image_folder, "experience_outputs", "knowledge_repository.json")
    calib_weights_path = os.path.join(image_folder, "experience_outputs", "calibration_weights.json")

    if not os.path.isfile(model_path) or not os.path.isfile(features_path):
        st.error("النموذج أو training_features.txt غير موجود. الرجاء تدريب النموذج أولاً.")
        st.code(model_path)
        st.code(features_path)
        st.stop()

    # ============================================================
    # 2) Upload image
    # ============================================================
    uploaded = st.file_uploader(
        "اختر صورة حديقة من جهازك",
        type=["jpg", "jpeg", "png"],
        key="planner_v5_uploaded_image",
    )

    if uploaded is None:
        st.info("ارفع صورة لتوليد بدائل التحسين النهائية.")
        st.stop()

    temp_dir = os.path.join(outputs_dir, "temp_planner")
    os.makedirs(temp_dir, exist_ok=True)
    temp_image = os.path.join(temp_dir, uploaded.name)

    with open(temp_image, "wb") as f:
        f.write(uploaded.getbuffer())

    # ============================================================
    # 3) Feature extraction
    # ============================================================
    with st.spinner("Extracting SegFormer features from the uploaded image..."):
        processor, seg_model = load_model()
        features, overlay_path, _ = analyze_one_image(
            image_path=temp_image,
            processor=processor,
            model=seg_model,
            out_dir=temp_dir,
            crop_top_ratio=crop_top_ratio,
            flower_expand_radius=flower_expand_radius,
            disable_water=disable_water,
            is_aerial=is_aerial,
        )

    r = {
        "grass_pct": features.get("grass", 0.0),
        "trees_pct": features.get("trees", 0.0),
        "flowers_pct": features.get("flowers", 0.0),
        "ground_pct": features.get("ground", 0.0),
        "water_pct": features.get("water", 0.0),
        "sky_pct": features.get("sky", 0.0),
        "building_pct": features.get("building", 0.0),
        "wall_pct": features.get("wall", 0.0),
    }

    # ============================================================
    # 4) Load model and predict
    # ============================================================
    model = joblib.load(model_path)
    feature_cols = load_features_list(features_path)
    mlp_score = predict_one(model, r, feature_cols)

    if os.path.isfile(repo_path):
        repo = load_knowledge_repository(repo_path)
        calib = calibrate_score(
            mlp_score=mlp_score, feats=r, repo=repo, feature_cols=feature_cols,
            weights_path=calib_weights_path,
        )
        current_score = calib["calibrated_score"]

        pc1, pc2 = st.columns(2)
        with pc1:
            st.metric("MLP score (raw)", f"{mlp_score:.3f}")
        with pc2:
            st.metric("Calibrated aesthetics score (used for planning)", f"{current_score:.3f}")
        st.caption(
            f"Calibrated using Stage 6 (Knowledge-Guided Score Calibration): "
            f"w_knowledge={calib.get('w_knowledge', 0.0)*100:.1f}%, beauty level = {calib['beauty_level']}"
        )
    else:
        current_score = mlp_score
        st.info("قاعدة المعرفة غير موجودة — تم استخدام درجة MLP الخام بدون معايرة.")

    # ============================================================
    # 5) Executive summary
    # ============================================================
    st.markdown(
        f"""
<div dir="rtl" style="
text-align:right;
line-height:2;
font-size:18px;
border:2px solid #2e7d32;
background:#f1f8f4;
border-radius:18px;
padding:20px;
margin-bottom:20px;
">

<h3>📌 الملخص التنفيذي</h3>

<p>
حصلت الصورة على درجة جمالية متوقعة مقدارها <b>{current_score:.2f} من 5</b>،
وتقع ضمن مستوى <b>{_beauty_level_ar(current_score)}</b>.
تعتمد صفحة التحسين على مقارنة عدة بدائل تصميمية، بحيث لا تقدم حلًا واحدًا فقط،
بل تعرض بدائل اقتصادية ومتوازنة ومثالية.
</p>

</div>
""",
        unsafe_allow_html=True,
    )

    _render_metric_cards(current_score, feature_cols)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**الصورة المرفوعة**")
        st.image(temp_image, use_container_width=True)
    with c2:
        st.write("**صورة التراكب (SegFormer)**")
        if os.path.isfile(overlay_path):
            st.image(overlay_path, use_container_width=True)
        else:
            st.info("لم يتم إنشاء صورة overlay.")

    st.write("### السمات المستخرجة")
    visible_cols = [c for c in ["grass_pct", "trees_pct", "flowers_pct", "ground_pct", "water_pct"] if c in r]
    st.dataframe(pd.DataFrame([{c: r.get(c, 0.0) for c in visible_cols}]), use_container_width=True)

# ============================================================
    # 6) Alternatives generation
    # ============================================================
    st.divider()
    st.markdown(
        """
<div dir="rtl" style="text-align:right; line-height:2; font-size:18px;">
<h3>🌿 بدائل تحسين الحديقة</h3>
<p>
اضغط الزر التالي لتوليد ثلاثة بدائل: اقتصادي، متوازن، ومثالي.
كل بديل يتم تقييمه حسب الكلفة، الدرجة المتوقعة، ومقدار التحسن.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

    interventions = _build_interventions(feature_cols)

    # ------------------------------------------------------------
    # دالة تسجيل التوليفات: MLP الخام فقط، بلا مزج مع المعرفة (KNN).
    #
    # لماذا لكل السيناريوهات الثلاثة (وليس فقط "المثالي")؟ دليل تجريبي
    # قاطع عبر 4 جولات معايرة متتالية (دلتا 4/5/6 ثم 6/8/12 ثم 8/10/17
    # ثم 9/12/20): درجة "الدمج" (calibrate_score) للتوليفات بلا ماء
    # تبقى شبه ثابتة تقريبًا (~0 تحسّن) بغض النظر عن حجم الدلتا — لأن
    # مكوّن المعرفة (KNN) هو متوسط مرجَّح لأقرب حالات استبيان حقيقية،
    # وهذا مصمَّم لتقييم دقيق لصور حقيقية موجودة فعليًا، وليس لمحاكاة
    # سيناريوهات افتراضية (ماذا لو غيّرنا السمات؟) — فهو لا يستطيع
    # الاستقراء خارج الحالات الحقيقية التي رآها، فتبقى الدرجة شبه ثابتة
    # مهما كبر التغيير الافتراضي المُطبَّق.
    #
    # الحل: نموذج MLP وحده (دالة مستمرة قابلة للاستقراء الحقيقي) هو
    # الأنسب لمهمة "المحاكاة الافتراضية" هذه تحديدًا. ملاحظة مهمة: هذا
    # لا يمسّ إطلاقًا "الدرجة الحالية" المعروضة أعلى الصفحة لصورة حقيقية
    # مرفوعة فعليًا — تلك تبقى بالدمج الكامل (calibrate_score) كما كانت،
    # لأنها الأدق لتقييم صورة حقيقية موجودة، بعكس محاكاة الإجراءات.
    # ------------------------------------------------------------
    def open_score_fn(feats_dict, _model=model, _feature_cols=feature_cols):
        return predict_one(_model, feats_dict, _feature_cols)

    score_fn = open_score_fn

    # ------------------------------------------------------------
    # جديد: دالة تسجيل بسقف ليّن، للسيناريو المثالي حصرًا.
    #
    # المشكلة: predict_one يُقيَّد بـ np.clip(x, 1, 5) — أي قيمة خام من
    # النموذج تتجاوز 5 (وقد تكون فعليًا 7 أو 10 داخليًا، غير موثوقة أصلًا
    # لأنها خارج نطاق البيانات الحقيقية) تُسطَّح جميعها إلى 5.0 بالضبط.
    # هذا يُنتج "يقينًا زائفًا" — رقمًا نظيفًا متكررًا يُخفي عدم يقين حقيقي
    # شديد في منطقة الاستقراء البعيدة.
    #
    # الحل: بدل القص الحاد عند 5، نستخدم انضغاطًا تدريجيًا (Soft Ceiling)
    # يسمح للدرجة بالاقتراب من 4.8 دون الوصول إليها أبدًا تمامًا — كلما
    # كان التنبؤ الخام أكثر تطرفًا، اقتربت الدرجة المعروضة أكثر من 4.8
    # (لكن لا تساويها أبدًا رياضيًا)، بدل القفز المباشر لـ5.0. هذا أصدق
    # إحصائيًا: يعكس عدم اليقين المتزايد بدل إخفائه خلف رقم صناعي ثابت.
    # ------------------------------------------------------------
    SOFT_CEILING_START = 4.5   # تحته: لا تصحيح، الرقم موثوق نسبيًا
    SOFT_CEILING_MAX = 4.8     # الحد الأعلى النظري الذي تقترب منه الدرجة

    def open_score_fn_soft_ceiling(feats_dict, _model=model, _feature_cols=feature_cols):
        x = np.array([[float(feats_dict.get(c, 0.0)) for c in _feature_cols]], dtype=np.float32)
        raw = float(_model.predict(x)[0])  # بلا أي قص — القيمة الخام الحقيقية

        if raw <= SOFT_CEILING_START:
            return float(np.clip(raw, 1.0, 5.0))

        # انضغاط أُسّي: يقترب تدريجيًا من SOFT_CEILING_MAX بلا الوصول إليه
        excess = raw - SOFT_CEILING_START
        span = SOFT_CEILING_MAX - SOFT_CEILING_START
        compressed = SOFT_CEILING_START + span * (1.0 - np.exp(-excess / 2.0))
        return float(compressed)

    # ------------------------------------------------------------
    # حاجز الأمان للسمات النادرة (كان يُجبر أي إجراء يمسّ سمة نادرة
    # كالزهور على استخدام درجة المعرفة وحدها) — مُعطَّل الآن عمدًا.
    # بعد التحول لتقييم كل السيناريوهات بـ MLP الخام (انظر التعليق أعلاه)،
    # إبقاء هذا الحاجز فعّالًا سيُعيد فرض نفس مشكلة "التشبّع" (KNN) على
    # أي إجراء يمسّ سمة نادرة تحديدًا، ناقضًا الإصلاح نفسه جزئيًا. نثق
    # الآن باستقراء MLP عمدًا لكل سمة، وليس فقط للسمات غير النادرة.
    # rare_features/knowledge_score_fn أُبقيا موجودتين (فارغة/None) فقط
    # لتوافق التوقيع مع generate_three_budget_scenarios دون تعديل بنيته.
    # ------------------------------------------------------------
    rare_features = set()
    knowledge_score_fn = None

    # ------------------------------------------------------------
    # جديد: بنية "متراكمة" ثابتة (وليست بحثًا آليًا عن الأفضل) — كل
    # مستوى = المستوى الأدنى + إجراء إضافي واحد، بالضبط كما طلب الباحث:
    #   اقتصادي  = زيادة المساحات الخضراء + شجيرات
    #   متوازن   = (كل إجراءات الاقتصادي) + أشجار كبيرة
    #   مثالي    = (كل إجراءات المتوازن) + بركة/نافورة ماء
    # هذا أوضح سرديًا ومنطقيًا من البحث الشامل، حتى لو كان تحسّن بعض
    # المستويات ضعيفًا وفق ما تعلّمه النموذج (سنرى الأرقام الفعلية).
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # جديد: بحث تكيّفي عن الدلتا بدل أرقام ثابتة (12%، 20%).
    #
    # لماذا؟ اختبار 4 صور مختلفة بنفس الدلتا الثابتة أظهر: صورة بمساحة
    # أرض مكشوفة قليلة لا تستطيع فعليًا استيعاب +20% عشب (يُموَّل من
    # ground_pct عبر منطق الحفاظ)، فيتطابق الاقتصادي والمتوازن رياضيًا؛
    # وصورة بدرجة أساس أعلى قد تتجاوز حتى بدلتا صغيرة. الحل: نبحث لكل
    # صورة تحديدًا عن أصغر دلتا "مسطح أخضر" تُدخل الدرجة **فعليًا** ضمن
    # نطاق كل مستوى، بدل افتراض رقم واحد يناسب الجميع.
    # ------------------------------------------------------------
    def _find_adaptive_delta_single_feature(base_feats, feature_cols, feature_name,
                                              target_lo, target_hi, score_fn,
                                              max_delta=60, step=1):
        """
        يبحث تدريجيًا (feature_name = step, 2*step, ...) عن أصغر دلتا تجعل
        الدرجة الناتجة تقع ضمن [target_lo, target_hi]. إن لم تصل أي دلتا
        مُختبَرة للنطاق، يُعاد أقرب دلتا وُجدت مع target_reached=False.
        """
        if feature_name not in feature_cols:
            return None
        conserve_cols = [c for c in CONSERVED_LAND_COVER_COLS if c in feature_cols]
        best = None  # (المسافة عن النطاق, دلتا, السمات الجديدة, الدرجة)

        for d in range(step, max_delta + 1, step):
            new_feats = bs_apply_deltas(
                base_feats, {feature_name: d, "ground_pct": -d}, conserve_cols=conserve_cols
            )
            score = score_fn(new_feats)

            if target_lo <= score <= target_hi:
                return d, new_feats, score, True

            dist = (target_lo - score) if score < target_lo else (score - target_hi)
            if best is None or dist < best[0]:
                best = (dist, d, new_feats, score)

        if best is None:
            return None
        _, d, new_feats, score = best
        return d, new_feats, score, False

    def _find_adaptive_delta(base_feats, feature_cols, target_lo, target_hi, score_fn,
                              candidate_features=("grass_pct", "trees_pct", "flowers_pct"),
                              max_delta=60, step=1, pair_max_delta=40, pair_step=3):
        """
        المرحلة 1: يجرّب كل سمة مرشَّحة بمفردها (المسطح أولًا، ثم الأشجار،
        ثم الزهور). يُعيد فورًا أول نتيجة تصل فعليًا للنطاق المطلوب.

        المرحلة 2 (فقط إن فشلت كل سمة منفردة): يأخذ أفضل سمة منفردة
        (الأقرب للنطاق) كقاعدة، ثم يبحث تدريجيًا عن إضافة دلتا **صغيرة**
        من سمة ثانية مختلفة فوقها (بحث حقيقي بخطوات صغيرة، وليس رقمًا
        كبيرًا ثابتًا كما فعلنا سابقًا وأضرّ بالنتائج) — لعله يُقرِّب أكثر
        من النطاق أو يصل إليه فعليًا.

        يُعيد: (قاموس الدلتا المُستخدَمة {سمة: قيمة, ...}, السمات الجديدة,
        الدرجة, target_reached).
        """
        # فحص مبكر: البحث يزيد السمات فقط (لا يستطيع تقليلها لخفض الدرجة).
        # إن كانت الدرجة الأساسية أصلًا ضمن النطاق المطلوب، أو حتى أعلى
        # منه، فلا حاجة لأي تدخل — بل أي زيادة إضافية ستُبعدها عن النطاق
        # (أو تُبقيها فوقه). في كلتا الحالتين، "بلا تدخل" هو الخيار الأمثل.
        base_score_now = score_fn(base_feats)
        if base_score_now >= target_lo:
            return {}, base_feats, base_score_now, True

        best_overall = None  # (المسافة, {fname: d}, السمات, الدرجة)
        for fname in candidate_features:
            result = _find_adaptive_delta_single_feature(
                base_feats, feature_cols, fname, target_lo, target_hi, score_fn,
                max_delta=max_delta, step=step,
            )
            if result is None:
                continue
            d, new_feats, score, reached = result
            if reached:
                return {fname: d}, new_feats, score, True
            dist = (target_lo - score) if score < target_lo else (score - target_hi)
            if best_overall is None or dist < best_overall[0]:
                best_overall = (dist, {fname: d}, new_feats, score)

        if best_overall is None:
            return {}, base_feats, score_fn(base_feats), False

        # المرحلة 2: إضافة سمة ثانية فوق أفضل سمة منفردة وُجدت
        base_dist, base_deltas, base_new_feats, base_score_val = best_overall
        base_fname = next(iter(base_deltas))
        base_d = base_deltas[base_fname]
        remaining = [f for f in candidate_features if f in feature_cols and f != base_fname]

        for second_fname in remaining:
            conserve_cols = [c for c in CONSERVED_LAND_COVER_COLS if c in feature_cols]
            for d2 in range(pair_step, pair_max_delta + 1, pair_step):
                combo_deltas = {base_fname: base_d, second_fname: d2, "ground_pct": -(base_d + d2)}
                new_feats2 = bs_apply_deltas(base_feats, combo_deltas, conserve_cols=conserve_cols)
                score2 = score_fn(new_feats2)

                if target_lo <= score2 <= target_hi:
                    return {base_fname: base_d, second_fname: d2}, new_feats2, score2, True

                dist2 = (target_lo - score2) if score2 < target_lo else (score2 - target_hi)
                if dist2 < best_overall[0]:
                    best_overall = (dist2, {base_fname: base_d, second_fname: d2}, new_feats2, score2)

        _, deltas_used, new_feats, score = best_overall
        return deltas_used, new_feats, score, False

    _FEATURE_ACTION_LABEL = {
        "grass_pct": "تحسين المسطح الأخضر والتربة",
        "trees_pct": "زراعة أشجار وشجيرات",
        "flowers_pct": "إضافة أحواض زهور",
    }

    def _generate_nested_scenarios(model, base_feats, feature_cols, score_fn):
        # نطاقات الدرجة المطلوبة لكل مستوى (متطلَّب من المشرف تحديدًا).
        ECONOMIC_RANGE = (2.0, 3.0)
        BALANCED_RANGE = (3.0, 4.0)
        IDEAL_RANGE = (4.0, 5.0)

        deltas_low, feats_low, score_low, reached_low = _find_adaptive_delta(
            base_feats, feature_cols, *ECONOMIC_RANGE, score_fn
        )
        deltas_med, feats_med, score_med, reached_med = _find_adaptive_delta(
            base_feats, feature_cols, *BALANCED_RANGE, score_fn
        )

        base_score = score_fn(base_feats)

        def _mk_plan(deltas_used, new_feats, new_score, scenario_key):
            # deltas_used: {"grass_pct": 8} أو {"grass_pct": 8, "trees_pct": 6}
            # أو {} (فارغة) إن كانت الدرجة الأساسية أصلًا كافية — عندها
            # لا حاجة لأي تدخل إطلاقًا، وهذه ليست حالة فشل.
            #
            # جديد: بدل جملة وحيدة مبتذلة "تحسين المسطح الأخضر بنسبة 8%"،
            # نستدعي مكتبة التدخلات الواقعية (select_real_actions) لنحصل
            # على 2-4 إجراءات واقعية فعلية تتّسق مع تشخيص أضعف محور في هذه
            # الحديقة تحديدًا — بالضبط ما طلبه الباحث.
            real_actions = select_real_actions(
                scenario_key, deltas_used, base_feats, seed_extra=f"{new_score:.4f}"
            )
            if not real_actions:
                real_actions = ["لا حاجة لأي تدخل — الحديقة ضمن هذا المستوى أصلًا"]
            ground_total = -sum(deltas_used.values())
            return {
                "actions": [{"name": a, "deltas": {}, "cost_tier": scenario_key} for a in real_actions],
                "action_names": real_actions,
                "action_names_ar": real_actions,
                "cost_tier": "low",
                "cost_tier_order": 0,
                "cost_tier_ar": "",
                "base_score": base_score,
                "new_score": new_score,
                "improvement": new_score - base_score,
                "new_feats": new_feats,
                "deltas": {**deltas_used, "ground_pct": ground_total},
                "scored_via_knowledge_only": False,
            }

        def _mk_adaptive_summary_ar(scenario_label_ar, deltas_used, new_score):
            # جملة سردية واحدة تشرح نفسها بنفسها، بلا أي كلمة مبهمة مثل
            # "تكيّفي" — مثال: "السيناريو الاقتصادي يحتاج زيادة 5% في
            # المسطح الأخضر والتربة فقط للوصول إلى درجة جمال متوقعة 3.05."
            if not deltas_used:
                return (
                    f"{scenario_label_ar}: الحديقة تقع أصلًا ضمن (أو تفوق) "
                    f"مستوى هذا السيناريو بدرجتها الحالية {new_score:.2f} — لا حاجة لأي تدخل إضافي."
                )
            parts_ar = []
            for fname, dval in deltas_used.items():
                label = _FEATURE_ACTION_LABEL.get(fname, fname)
                parts_ar.append(f"زيادة {dval:.0f}% في {label}")
            needs_text = " مع ".join(parts_ar)
            return (
                f"{scenario_label_ar} يحتاج {needs_text} فقط للوصول إلى "
                f"درجة جمال متوقعة {new_score:.2f}."
            )

        low_plan = _mk_plan(deltas_low, feats_low, score_low, "low")
        low_plan["cost_tier"] = "low"
        low_plan["cost_tier_ar"] = "منخفضة"
        low_plan["target_range_ar"] = f"{ECONOMIC_RANGE[0]:.2f}-{ECONOMIC_RANGE[1]:.2f}"
        low_plan["adaptive_summary_ar"] = _mk_adaptive_summary_ar(
            "السيناريو الاقتصادي", deltas_low, score_low
        )

        medium_plan = _mk_plan(deltas_med, feats_med, score_med, "medium")
        medium_plan["cost_tier"] = "medium"
        medium_plan["cost_tier_order"] = 1
        medium_plan["cost_tier_ar"] = "متوسطة"
        medium_plan["target_range_ar"] = f"{BALANCED_RANGE[0]:.2f}-{BALANCED_RANGE[1]:.2f}"
        medium_plan["adaptive_summary_ar"] = _mk_adaptive_summary_ar(
            "السيناريو المتوازن", deltas_med, score_med
        )

        # المثالي: يبقى بالتصميم الثابت (توسيع المسطح + ماء) — الماء له
        # استجابة شبه ثنائية مؤكَّدة تجريبيًا (أي كمية غير صفرية تصل للسقف
        # 5.0 تقريبًا دومًا)، فلا حاجة لبحث تكيّفي هنا؛ أي دلتا ماء تكفي.
        lawn_extensive = {
            "name": "Extensive lawn/soil renovation",
            "deltas": {"grass_pct": 20, "ground_pct": -20},
            "cost_tier": "medium",
        }
        water_pond = None
        if "water_pct" in feature_cols:
            water_pond = {
                "name": "Add water pond/fountain",
                "deltas": {"water_pct": 8, "ground_pct": -8},
                "cost_tier": "high",
            }
        high_actions = valid_interventions(
            [lawn_extensive] + ([water_pond] if water_pond else []), feature_cols
        )
        high_plan = evaluate_plan(model, base_feats, feature_cols, high_actions, open_score_fn_soft_ceiling, None, None) if high_actions else None
        if high_plan is not None:
            high_plan["target_range_ar"] = f"{IDEAL_RANGE[0]:.2f}-{IDEAL_RANGE[1]:.2f}"
            high_reached = IDEAL_RANGE[0] <= high_plan["new_score"] <= IDEAL_RANGE[1]
            # استبدال التسمية الثابتة القديمة ("توسيع تجديد المسطح..." +
            # "إضافة بركة/نافورة ماء" فقط) بمكتبة "المثالي" الغنية — نفس
            # منطق select_real_actions المستخدَم للاقتصادي والمتوازن، حتى
            # يتّسق ما تعرضه "الإجراءات المقترحة" مع ما يرويه التقرير التفسيري
            # أدناه (الذي يستدعي نفس الدالة داخليًا عبر generate_improvement_report).
            open_real_actions = select_real_actions(
                "open", high_plan.get("deltas", {}), base_feats,
                seed_extra=f"{high_plan['new_score']:.4f}",
            )
            if open_real_actions:
                high_plan["action_names_ar"] = open_real_actions
                high_plan["action_names"] = open_real_actions
                high_plan["actions"] = [
                    {"name": a, "deltas": {}, "cost_tier": "open"} for a in open_real_actions
                ]
        else:
            high_reached = False

        rows = [
            plan_to_row("low", low_plan, model, base_feats, feature_cols, score_fn, None, None),
            plan_to_row("medium", medium_plan, model, base_feats, feature_cols, score_fn, None, None),
            plan_to_row("open", high_plan, model, base_feats, feature_cols, open_score_fn_soft_ceiling, None, None),
        ]
        for row, reached in zip(rows[:2], [reached_low, reached_med]):
            row["target_reached"] = reached
        rows[2]["target_reached"] = high_reached

        df = pd.DataFrame(rows)
        plans = {"low": low_plan, "medium": medium_plan, "open": high_plan}
        return df, plans

    # ------------------------------------------------------------
    # الزر يعمل دائمًا (بمعرفة أو بدونها — score_fn=None كخيار احتياطي)
    # ------------------------------------------------------------
    if st.button("📊 توليد بدائل تحسين الحديقة", key="planner_v5_generate_alternatives"):
        scenarios_df, scenarios = _generate_nested_scenarios(
            model=model,
            base_feats=r,
            feature_cols=feature_cols,
            score_fn=open_score_fn,
        )

        # ------------------------------------------------------------
        # تشخيص شامل: كل التوليفات الممكنة (وليس فقط الإجراءات المنفردة)،
        # لمعرفة إن كانت السيناريوهات الثلاثة تتطابق لأن لا بديل حقيقي آخر
        # يحقق تحسنًا موجبًا، أو لأن منطق الاختيار يُخفي بدائل موجودة فعلًا.
        # ------------------------------------------------------------
        with st.expander("🔍 تشخيص شامل: كل التوليفات الممكنة"):
            all_combos = _all_combinations(valid_interventions(interventions, feature_cols))
            combo_rows = []
            for combo in all_combos:
                p = evaluate_plan(
                    model, r, feature_cols, combo,
                    score_fn=score_fn,
                    rare_features=rare_features,
                    knowledge_score_fn=knowledge_score_fn,
                )
                combo_rows.append({
                    "الإجراءات": " + ".join(a["name"] for a in combo),
                    "مستوى الكلفة": p["cost_tier_ar"],
                    "الدرجة بعد": round(p["new_score"], 4),
                    "التحسن": round(p["improvement"], 4),
                    "المعيار": "معرفة فقط" if p.get("scored_via_knowledge_only") else "دمج (MLP+معرفة)",
                })

            combo_df = pd.DataFrame(combo_rows).sort_values("التحسن", ascending=False)
            st.dataframe(combo_df, use_container_width=True)

            n_improving = (combo_df["التحسن"] > 1e-9).sum()
            st.caption(f"عدد التوليفات ذات تحسن موجب فعليًا: {n_improving} من أصل {len(combo_df)}")

        st.success("تم توليد بدائل التحسين بنجاح!")

        # --------------------------------------------------------
        # Explain scenario overlaps (fully or partially identical),
        # covering all cases generically via scenario_relation_note_ar
        # computed inside generate_three_budget_scenarios().
        # --------------------------------------------------------
        relation_note = ""
        if "scenario_relation_note_ar" in scenarios_df.columns and len(scenarios_df):
            relation_note = scenarios_df["scenario_relation_note_ar"].iloc[0]
        if relation_note:
            st.info(f"ℹ️ {relation_note}")

        # --------------------------------------------------------
        # Comparison table
        # --------------------------------------------------------
        comparison_df = scenarios_df[[
            "scenario_ar",
            "cost_tier_ar",
            "base_score",
            "new_score",
            "improvement",
            "actions_count",
            "actions_ar",
        ]].copy()

        comparison_df = comparison_df.rename(columns={
            "scenario_ar": "السيناريو",
            "cost_tier_ar": "مستوى الكلفة",
            "base_score": "الدرجة الحالية",
            "new_score": "الدرجة المتوقعة",
            "improvement": "مقدار التحسن",
            "actions_count": "عدد الإجراءات",
            "actions_ar": "الإجراءات",
        })

        st.write("### جدول المقارنة")
        st.dataframe(comparison_df, use_container_width=True)
        # --------------------------------------------------------
        # Dashboard chart
        # --------------------------------------------------------
        st.write("### لوحة مقارنة البدائل")

        chart_df = scenarios_df[["scenario_ar", "cost_tier_ar", "new_score", "improvement"]].copy()
        chart_df = chart_df.rename(columns={
            "scenario_ar": "السيناريو",
            "cost_tier_ar": "مستوى الكلفة",
            "new_score": "الدرجة المتوقعة",
            "improvement": "مقدار التحسن",
        })

        # ملاحظة: الكلفة أصبحت نوعية (منخفضة/متوسطة/عالية) وليست رقمًا،
        # فلا يصح رسمها كعمود أرقام. تُعرض كنص في الجدول أعلاه فقط،
        # والرسم البياني هنا يقتصر على الدرجة المتوقعة (رقمية فعليًا).
        st.line_chart(chart_df.set_index("السيناريو")[["الدرجة المتوقعة"]])

        # --------------------------------------------------------
        # Scenario cards + Before/After
        # --------------------------------------------------------
        st.write("### تفاصيل البدائل")

        for _, row in scenarios_df.iterrows():
            plan = scenarios.get(row["scenario_key"])
            _render_scenario_card(row, plan, r, feature_cols, current_score_displayed=current_score)

        # --------------------------------------------------------
        # Expert recommendation
        # --------------------------------------------------------
        _render_expert_recommendation(scenarios_df)

        # --------------------------------------------------------
        # Export CSV
        # --------------------------------------------------------
        csv_data = comparison_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️ تحميل جدول بدائل التحسين CSV",
            data=csv_data,
            file_name="improvement_alternatives_v5.csv",
            mime="text/csv",
        )
# ============================================================
# Page 9: Garden Integration & Smart Design
# ============================================================
elif page == "9) Garden Integration & Smart Design":

    st.subheader("9) دمج الحدائق والتصميم الذكي")

    st.markdown(
        """
        <div dir="rtl" style="text-align:right; line-height:2;">
        تقوم هذه المرحلة بدمج عدة حدائق متجاورة لتكوين مساحة موحدة،
        ثم تستخدم نموذج MLP الحالي للبحث عن نسب العناصر التي تحقق
        مستوى الجمال المستهدف بأفضل صورة ممكنة.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Step 1: Number of gardens
    # --------------------------------------------------------
    st.markdown("### 1. إدخال الحدائق وأبعادها")

    num_gardens = st.number_input(
        "عدد الحدائق المراد دمجها",
        min_value=2,
        max_value=4,
        value=4,
        step=1,
        key="integration_num_gardens",
    )

    layout_label = st.selectbox(
        "طريقة ترتيب الحدائق",
        [
            "شبكة 2 × 2",
            "صف أفقي",
            "عمود رأسي",
        ],
        key="integration_layout",
    )

    layout_map = {
        "شبكة 2 × 2": "grid_2x2",
        "صف أفقي": "row",
        "عمود رأسي": "column",
    }

    layout = layout_map[layout_label]

    gardens = []

    for i in range(int(num_gardens)):

        st.markdown(f"#### الحديقة {i + 1}")

        c1, c2 = st.columns(2)

        with c1:
            width = st.number_input(
                f"العرض بالمتر — حديقة {i + 1}",
                min_value=0.1,
                value=10.0,
                step=0.5,
                key=f"garden_width_{i}",
            )

        with c2:
            length = st.number_input(
                f"الطول بالمتر — حديقة {i + 1}",
                min_value=0.1,
                value=10.0,
                step=0.5,
                key=f"garden_length_{i}",
            )

        existing_composition = None
        with st.expander(f"📊 (اختياري) التركيبة الحالية لحديقة {i + 1} — لتفعيل تقرير مخصَّص حسب نقصها الفعلي"):
            st.caption(
                "إن أدخلت هذه النسب (من صفحة 1 أو تقديرًا)، سيبني التقرير النهائي "
                "جملًا مخصَّصة تشرح نقاط الضعف الفعلية لهذه الحديقة تحديدًا، بدل وصف عام."
            )
            has_data = st.checkbox("أعرف التركيبة الحالية لهذه الحديقة", value=False, key=f"garden_has_comp_{i}")
            if has_data:
                gc1, gc2, gc3 = st.columns(3)
                with gc1:
                    g_grass = numeric_stepper(st, "عشب %", 0.0, 100.0, 40.0, 1.0, f"garden_grass_{i}")
                    g_trees = numeric_stepper(st, "أشجار %", 0.0, 100.0, 20.0, 1.0, f"garden_trees_{i}")
                with gc2:
                    g_flowers = numeric_stepper(st, "زهور %", 0.0, 100.0, 5.0, 1.0, f"garden_flowers_{i}")
                    g_ground = numeric_stepper(st, "أرض مكشوفة %", 0.0, 100.0, 30.0, 1.0, f"garden_ground_{i}")
                with gc3:
                    g_water = numeric_stepper(st, "ماء %", 0.0, 100.0, 5.0, 1.0, f"garden_water_{i}")
                existing_composition = {
                    "grass_pct": float(g_grass),
                    "trees_pct": float(g_trees),
                    "flowers_pct": float(g_flowers),
                    "ground_pct": float(g_ground),
                    "water_pct": float(g_water),
                }

        gardens.append({
            "name": f"Garden {i + 1}",
            "width": float(width),
            "length": float(length),
            "existing_composition": existing_composition,
        })

    # --------------------------------------------------------
    # Step 2: Target
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 2. تحديد مستوى الجمال المستهدف")

    target_score = numeric_stepper(
        st,
        "درجة الجمال المستهدفة",
        1.0,
        5.0,
        4.5,
        0.05,
        "integration_target_score",
    )

    water_allowed = st.checkbox(
        "السماح بإدخال عنصر مائي ضمن التصميم",
        value=True,
        key="integration_water_allowed",
    )

    search_step = st.selectbox(
        "دقة البحث عن النسب",
        options=[10.0, 5.0],
        index=1,
        format_func=lambda x: f"{x:.0f}%"
    )

    st.info(
        "يتم اختبار تركيبات مختلفة من نسب المسطحات الخضراء والأشجار "
        "والزهور والأرضيات والعناصر المائية، ثم تقييمها باستخدام نموذج "
        "MLP الحالي."
    )


    # --------------------------------------------------------
    # Step 3: Required model
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 3. النموذج المستخدم")

    train_out_dir = st.text_input(
        "مجلد مخرجات تدريب النموذج",
        value=os.path.join(
            image_folder,
            "training_outputs"
        ),
        key="integration_train_out_dir",
    )

    model_path = os.path.join(
        train_out_dir,
        "mlp_model.joblib"
    )

    features_path = os.path.join(
        train_out_dir,
        "training_features.txt"
    )

    st.caption(f"Model: {model_path}")
    st.caption(f"Features: {features_path}")


    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------
    st.markdown("---")

    if st.button(
        "🌳 دمج الحدائق وإنشاء التصميم الذكي",
        type="primary",
        use_container_width=True,
        key="run_garden_integration",
    ):

        try:

            # التحقق من وجود نموذج MLP
            if not os.path.isfile(model_path):
                st.error(
                    "لم يتم العثور على ملف نموذج MLP. "
                    "تأكد من مسار training_outputs."
                )
                st.code(model_path)
                st.stop()

            # التحقق من وجود ملف السمات
            if not os.path.isfile(features_path):
                st.error(
                    "لم يتم العثور على ملف training_features.txt."
                )
                st.code(features_path)
                st.stop()

            # تحميل النموذج
            model = joblib.load(model_path)

            # تحميل قائمة السمات بالطريقة الصحيحة
            feature_cols = load_features_list(features_path)

            with st.spinner(
                "يتم دمج الحدائق والبحث عن أفضل تصميم..."
            ):

                result = create_unified_garden_design(
                    gardens=gardens,
                    layout=layout,
                    model=model,
                    feature_cols=feature_cols,
                    target_score=target_score,
                    water_allowed=water_allowed,
                    step=float(search_step),
                )

            st.session_state[
                "garden_integration_result"
            ] = result

            st.success(
                "تم دمج الحدائق وإنشاء التصميم الذكي بنجاح."
            )

        except Exception as e:
            st.error(
                f"حدث خطأ أثناء إنشاء التصميم: {e}"
            )
            st.exception(e)


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    result = st.session_state.get(
        "garden_integration_result"
    )

    if result is not None:

        geometry = result["geometry"]
        design = result["design"]

        st.markdown("---")
        st.markdown("## النتيجة")

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------
        st.markdown("### أبعاد الحديقة الموحدة")

        g1, g2, g3, g4 = st.columns(4)

        g1.metric(
            "العرض الكلي",
            f"{geometry['width']:.2f} م"
        )

        g2.metric(
            "الطول الكلي",
            f"{geometry['length']:.2f} م"
        )

        g3.metric(
            "المساحة الفعلية للحدائق",
            f"{geometry['area']:.2f} م²"
        )

        g4.metric(
            "عدد الحدائق",
            len(geometry["gardens"])
        )

        # ----------------------------------------------------
        # Beauty result
        # ----------------------------------------------------
        st.markdown("### تقييم التصميم المقترح")

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "الدرجة المستهدفة",
            f"{design['target_score']:.2f} / 5"
        )

        s2.metric(
            "الدرجة المتوقعة",
            f"{design['predicted_score']:.2f} / 5"
        )

        if design["target_reached"]:
            status_text = "تم الوصول إلى الهدف"
        else:
            status_text = "لم يتم الوصول للهدف بالكامل"

        s3.metric(
            "حالة الهدف",
            status_text
        )

        st.caption(
            f"تم اختبار {design['searched_candidates']} تركيباً مختلفاً."
        )

        # ----------------------------------------------------
        # Required percentages and areas
        # ----------------------------------------------------
        st.markdown("### النسب والمساحات المطلوبة للتصميم")

        area_table = result["area_table"].copy()

        area_table["percentage"] = (
            area_table["percentage"]
            .map(lambda x: round(float(x), 2))
        )

        area_table["area_m2"] = (
            area_table["area_m2"]
            .map(lambda x: round(float(x), 2))
        )

        st.dataframe(
            area_table[
                [
                    "element_ar",
                    "percentage",
                    "area_m2",
                ]
            ].rename(
                columns={
                    "element_ar": "العنصر",
                    "percentage": "النسبة %",
                    "area_m2": "المساحة م²",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Original gardens
        # ----------------------------------------------------
        st.markdown("### تفاصيل الحدائق المدمجة")

        gardens_df = pd.DataFrame(
            geometry["gardens"]
        )

        if not gardens_df.empty:

            st.dataframe(
                gardens_df[
                    [
                        "name",
                        "width",
                        "length",
                        "area",
                        "x",
                        "y",
                    ]
                ].rename(
                    columns={
                        "name": "الحديقة",
                        "width": "العرض",
                        "length": "الطول",
                        "area": "المساحة م²",
                        "x": "الموقع X",
                        "y": "الموقع Y",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Conceptual design plan
        # ----------------------------------------------------
        st.markdown("### المخطط الأولي للتصميم")

        zones_df = pd.DataFrame(
            result["zones"]
        )

        st.dataframe(
            zones_df.rename(
                columns={
                    "label_ar": "المنطقة",
                    "type": "النوع",
                    "x": "X",
                    "y": "Y",
                    "width": "العرض",
                    "length": "الطول",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------
        st.markdown("### التقرير التفسيري للتصميم")

        report_text = "\n\n".join(
            [
                f"{i + 1}. {line}"
                for i, line in enumerate(
                    result["report_lines"]
                )
            ]
        )

        st.markdown(
            report_text
        )

        # ----------------------------------------------------
        # Download CSV
        # ----------------------------------------------------
        st.markdown("---")

        csv_data = (
            area_table
            .to_csv(
                index=False,
                encoding="utf-8-sig"
            )
            .encode("utf-8-sig")
        )

        st.download_button(
            "⬇️ تنزيل نسب ومساحات التصميم CSV",
            data=csv_data,
            file_name="unified_garden_design.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ---------------- Page 10: Methodology Diagram ----------------
elif page == "10) Methodology Diagram":

    st.subheader("10) مخطط المنهجية (تدفق عمل النظام)")
   #st.subheader("مخطط المنهجية (تدفق عمل النظام)")

    st.markdown(
        """
          <div style="text-align:center; margin-top:6px;">
          <div style="font-size:16px; color:rgba(0,0,0,0.7); margin-bottom:8px;">
        هذا المخطط يوضح المنهجية الكاملة للنظام من إدخال الصور إلى استخراج السمات ثم دمج الاستبيان
        وفحص الجودة وبناء البيانات والتدريب والمراجعة
        </div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    diagram_path = os.path.join(os.path.dirname(__file__), "assets", "workflow_diagram.png")
    if os.path.isfile(diagram_path):
        st.image(diagram_path, use_container_width=True)
        with open(diagram_path, "rb") as f:
            st.download_button(
                "⬇️ تنزيل المخطط PNG",
                data=f,
                file_name="workflow_diagram.png",
                mime="image/png",
            )
    else:
        st.error(f"Diagram not found: {diagram_path}")