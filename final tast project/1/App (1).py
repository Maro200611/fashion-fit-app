"""
Fashion Fit Predictor — Enhanced Streamlit interface
-----------------------------------------------------
Works with the new pipeline:
  - PyTorch ResNet101 (torchvision) for visual features
  - joblib-loaded XGBoost model
  - New artifact names: numeric_scaler.pkl, categorical_encoder.pkl,
    best_xgboost_fit_model.pkl, category_image_features.csv
  - cm / kg numeric fallbacks
  - spaCy-based text cleaning (optional review text)

New features vs. the old app:
  * Multi-image batch upload
  * Optional user overrides (height, weight, age, body type, review text)
  * Cropped garment thumbnails per detection
  * Dominant color extraction per garment
  * Interactive Plotly probability charts
  * Fit-confidence gauge
  * Downloadable JSON report
  * Session history / comparison
  * Dark-mode-friendly modern UI
  * Robust error handling & progress feedback

Run with:
    streamlit run app.py
"""
import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import io
import json
import base64
import colorsys
import tempfile
import traceback
from io import BytesIO
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ======================================================================
# CONFIG — edit paths to match your machine
# ======================================================================
DATA_DIR = r"D:\Downloads\final tast project" # <— change if needed

MODEL_PATH             = os.path.join(DATA_DIR, "best_xgboost_fit_model.pkl")
SCALER_PATH            = os.path.join(DATA_DIR, "numeric_scaler.pkl")
ENCODER_PATH           = os.path.join(DATA_DIR, "categorical_encoder.pkl")
VECTORIZER_PATH        = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH     = os.path.join(DATA_DIR, "fit_label_encoder.pkl")
CATEGORY_FEATURES_PATH = os.path.join(DATA_DIR, "category_image_features.csv")
FALLBACKS_PATH         = os.path.join(DATA_DIR, "training_fallbacks.json")
FINAL_DF_PATH          = os.path.join(DATA_DIR, "final_df_with_clean_text.parquet")
POSE_MODEL_PATH        = os.path.join(DATA_DIR, "pose_landmarker.task")
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

# Brand logo, shown in the top-right corner of the app (falls back gracefully if missing)
LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logo_icon.png"
)
TARGET_CATEGORICAL_FEATURES = 30

GARMENT_LABELS = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket",
    "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
}

FASHIONPEDIA_TO_INFO = {
    "shirt, blouse": ("shirt", "Shirts"),
    "top, t-shirt, sweatshirt": ("top", "Tops"),
    "sweater": ("sweater", "Sweaters"),
    "cardigan": ("cardigan", "Sweaters"),
    "jacket": ("jacket", "Jackets"),
    "vest": ("vest", "Waistcoat"),
    "pants": ("jeans", "Jeans"),
    "shorts": ("shorts", "Shorts"),
    "skirt": ("skirt", "Skirts"),
    "coat": ("jacket", "Jackets"),
    "dress": ("dress", "Dresses"),
    "jumpsuit": ("jumpsuit", "Jumpsuit"),
    "cape": ("tunic", "Tunics"),
}

GARMENT_EMOJI = {
    "dress": "👗", "jumpsuit": "👗", "skirt": "👗",
    "shirt, blouse": "👔", "top, t-shirt, sweatshirt": "👕",
    "sweater": "🧶", "cardigan": "🧥", "jacket": "🧥", "coat": "🧥", "vest": "🦺",
    "pants": "👖", "shorts": "🩳", "cape": "🦸",
}

FIT_EMOJI = {"fit": "✅", "small": "⚠️", "large": "⚠️", "true to size": "✅"}

st.set_page_config(
    page_title="Fashion Fit Predictor",
    layout="wide",
    page_icon="👗",
    initial_sidebar_state="expanded",
)


# ======================================================================
# CACHED LOADERS
# ======================================================================
@st.cache_data(show_spinner=False)
def load_logo_b64(path=LOGO_PATH):
    """Load the brand logo as base64 so it can be embedded inline via HTML/CSS."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading fit-prediction artifacts...")
def load_inference_artifacts():
    import joblib

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    category_features = pd.read_csv(CATEGORY_FEATURES_PATH, index_col=0)
    model = joblib.load(MODEL_PATH)
    try:
      model.set_params(device="cpu")   # sklearn-API XGBoost (XGBClassifier/XGBRegressor)
    except Exception:
      try:
        model.set_param({"device": "cpu"})   # raw Booster fallback
      except Exception:
        pass

    categorical_columns = list(encoder.feature_names_in_)

    if os.path.exists(FALLBACKS_PATH):
        with open(FALLBACKS_PATH) as f:
            fallbacks = json.load(f)
    elif os.path.exists(FINAL_DF_PATH):
        fallbacks = _generate_training_fallbacks(categorical_columns)
    else:
        # Emergency safe defaults
        fallbacks = {
            "numeric": {"height_cm_median": 165.0, "weight_kg_median": 63.0, "age_median": 32.0},
            "categorical": {c: "Unknown" for c in categorical_columns},
        }

    return {
        "scaler": scaler, "encoder": encoder, "vectorizer": vectorizer,
        "label_encoder": label_encoder, "category_features": category_features,
        "model": model, "fallbacks": fallbacks, "categorical_columns": categorical_columns,
    }


def _generate_training_fallbacks(categorical_columns):
    df = pd.read_parquet(FINAL_DF_PATH)
    numeric_defaults = {
        "height_cm_median": float(df["height_cm"].median()),
        "weight_kg_median": float(df["weight_kg"].median()),
        "age_median": float(df["age"].median()),
    }
    categorical_defaults = {}
    for col in categorical_columns:
        if col in df.columns:
            series = df[col].fillna("Unknown").astype(str).str.strip()
            categorical_defaults[col] = series.mode().iloc[0]
        else:
            categorical_defaults[col] = "Unknown"
    fallbacks = {"numeric": numeric_defaults, "categorical": categorical_defaults}
    with open(FALLBACKS_PATH, "w") as f:
        json.dump(fallbacks, f, indent=2)
    return fallbacks


@st.cache_resource(show_spinner="Loading garment detector (YOLOS)...")
def load_garment_detector():
    from transformers import YolosImageProcessor, YolosForObjectDetection
    processor = YolosImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    detection_model = YolosForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
    detection_model.eval()
    return processor, detection_model


@st.cache_resource(show_spinner="Loading pose landmarker...")
def load_pose_landmarker():
    import urllib.request
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not os.path.exists(POSE_MODEL_PATH):
        os.makedirs(os.path.dirname(POSE_MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(options), mp


@st.cache_resource(show_spinner="Loading spaCy NLP model...")
def load_spacy_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        return None


# ======================================================================
# PIPELINE FUNCTIONS
# ======================================================================
def clean_review_text(text, nlp):
    pre = re.sub(r"[^a-zA-Z\s]", "", str(text).lower())
    if nlp is None:
        return pre
    doc = nlp(pre)
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.lemma_.strip()]
    return " ".join(tokens)


def estimate_body_type_from_pil(pil_image, pose_landmarker, mp_module):
    tmp_path = os.path.join(tempfile.gettempdir(), f"_pose_{os.getpid()}.jpg")
    pil_image.save(tmp_path)
    mp_image = mp_module.Image.create_from_file(tmp_path)
    result = pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print("No person detected — defaulting body_type to 'unknown'")
        return "unknown", None

    lm = result.pose_landmarks[0]
    shoulder_width = abs(lm[11].x - lm[12].x)
    hip_width = abs(lm[23].x - lm[24].x)
    ratio = shoulder_width / hip_width if hip_width > 0 else 1.0

    if ratio > 1.8:
        body_type = "athletic"
    elif ratio <= 1.8:
        body_type = "pear"
    else:
        body_type = "hourglass"

    print(f"Estimated body_type: {body_type} (shoulder/hip ratio: {ratio:.2f}, thresholds unvalidated)")
    return body_type, ratio

def detect_garments(pil_image, processor, detection_model, category_features, confidence_threshold=0.5):
    import torch

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = detection_model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=confidence_threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        fashionpedia_label = detection_model.config.id2label[label.item()]
        if fashionpedia_label not in GARMENT_LABELS:
            continue

        info = FASHIONPEDIA_TO_INFO.get(fashionpedia_label)
        if info is None:
            continue

        raw_category, image_category = info
        if image_category not in category_features.index:
            continue

        box = [max(0, round(i)) for i in box.tolist()]
        detections.append({
            "fashionpedia_label": fashionpedia_label,
            "confidence": float(score.item()),
            "box": box,
            "raw_category": raw_category,
            "image_category": image_category,
        })

    return detections


def get_dominant_color(pil_crop, n_colors=3):
    try:
        from sklearn.cluster import KMeans
        img_small = pil_crop.resize((50, 50))
        pixels = np.array(img_small).reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, n_init=5, random_state=42)
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_)
        dominant_rgb = kmeans.cluster_centers_[np.argmax(counts)].astype(int)
        rgb = tuple(int(v) for v in dominant_rgb)
        return color_name_from_rgb(rgb), rgb
    except Exception:
        return "unknown", (128, 128, 128)


def color_name_from_rgb(rgb):
    """Simple color naming without needing webcolors."""
    r, g, b = rgb
    # Named color palette
    palette = {
        "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
        "red": (220, 20, 30), "maroon": (128, 0, 0), "pink": (255, 182, 193),
        "orange": (255, 140, 0), "yellow": (255, 220, 0), "beige": (222, 200, 170),
        "brown": (139, 69, 19), "tan": (210, 180, 140),
        "green": (34, 139, 34), "olive": (128, 128, 0), "teal": (0, 128, 128),
        "blue": (30, 100, 220), "navy": (0, 0, 128), "cyan": (0, 200, 220),
        "purple": (128, 0, 128), "lavender": (200, 180, 230),
    }
    best, best_d = "unknown", float("inf")
    for name, (nr, ng, nb) in palette.items():
        d = (r - nr) ** 2 + (g - ng) ** 2 + (b - nb) ** 2
        if d < best_d:
            best_d, best = d, name
    return best


# ----------------------------------------------------------------
# Outfit color harmony scoring
# ----------------------------------------------------------------
def rgb_to_hsv_deg(rgb):
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, s, v


def is_neutral(rgb, sat_thresh=0.15, val_low=0.12, val_high=0.92):
    h, s, v = rgb_to_hsv_deg(rgb)
    return s < sat_thresh or v < val_low or v > val_high


def hue_distance(h1, h2):
    d = abs(h1 - h2) % 360
    return min(d, 360 - d)


def pair_harmony_score(rgb1, rgb2):
    if is_neutral(rgb1) or is_neutral(rgb2):
        return 0.95
    h1, _, _ = rgb_to_hsv_deg(rgb1)
    h2, _, _ = rgb_to_hsv_deg(rgb2)
    d = hue_distance(h1, h2)
    # complementary (180), split-complementary (150), triadic (120), analogous (30), match (0)
    best = max(1 - abs(d - target) / 180 for target in [0, 30, 120, 150, 180])
    return best


def outfit_color_score(colors):
    """colors: list of RGB tuples, one per detected garment in the photo."""
    if len(colors) < 2:
        return 100.0
    pairs = [pair_harmony_score(c1, c2) for c1, c2 in combinations(colors, 2)]
    return round(float(np.mean(pairs)) * 100, 1)


def suggest_colors(rgb):
    """Given one garment's RGB, propose complementary/analogous/neutral alternatives."""
    h, _, _ = rgb_to_hsv_deg(rgb)
    options = {
        "complementary": (h + 180) % 360,
        "analogous": (h + 30) % 360,
    }
    suggestions = []
    for name, hue in options.items():
        r, g, b = colorsys.hsv_to_rgb(hue / 360, 0.55, 0.75)
        suggestions.append((name, tuple(int(c * 255) for c in (r, g, b))))
    suggestions += [("neutral", (255, 255, 255)), ("neutral", (30, 30, 30)), ("neutral", (128, 128, 128))]
    return suggestions


def build_categorical_row(categorical_columns, defaults, overrides):
    return [overrides.get(col, defaults.get(col, "Unknown")) for col in categorical_columns]


def encode_categorical(encoder, categorical_columns, row_values):
    df_row = pd.DataFrame([row_values], columns=categorical_columns)
    encoded = encoder.transform(df_row)
    encoded = np.asarray(encoded.todense()) if hasattr(encoded, "todense") else np.asarray(encoded)

    if encoded.shape[1] >= TARGET_CATEGORICAL_FEATURES:
        return encoded[:, :TARGET_CATEGORICAL_FEATURES]

    padding = np.zeros((encoded.shape[0], TARGET_CATEGORICAL_FEATURES - encoded.shape[1]), dtype=np.float32)
    return np.hstack([encoded, padding])


def build_feature_vector(image_category, raw_category, body_type, review_text,
                         nlp, artifacts, user_numeric=None, user_categorical=None):
    vis = artifacts["category_features"].loc[image_category].to_numpy(dtype=np.float32).reshape(1, -1)

    if review_text:
        text_vec = artifacts["vectorizer"].transform([clean_review_text(review_text, nlp)]).toarray()
    else:
        text_vec = artifacts["vectorizer"].transform([""]).toarray()

    n = artifacts["fallbacks"]["numeric"]
    height = user_numeric.get("height_cm", n["height_cm_median"]) if user_numeric else n["height_cm_median"]
    weight = user_numeric.get("weight_kg", n["weight_kg_median"]) if user_numeric else n["weight_kg_median"]
    age    = user_numeric.get("age",       n["age_median"])       if user_numeric else n["age_median"]
    numeric_vec = artifacts["scaler"].transform([[height, weight, age]])

    overrides = {"body type": body_type, "category": raw_category}
    if user_categorical:
        overrides.update(user_categorical)
    row_values = build_categorical_row(
        artifacts["categorical_columns"], artifacts["fallbacks"]["categorical"], overrides
    )
    cat_vec = encode_categorical(artifacts["encoder"], artifacts["categorical_columns"], row_values)

    return np.hstack([vis, text_vec, numeric_vec, cat_vec])


def predict_one(image_category, raw_category, body_type, review_text, nlp, artifacts,
                user_numeric=None, user_categorical=None):
    X = build_feature_vector(image_category, raw_category, body_type, review_text,
                             nlp, artifacts, user_numeric, user_categorical)
    pred_encoded = artifacts["model"].predict(X)
    pred_label = artifacts["label_encoder"].inverse_transform(pred_encoded)[0]
    pred_proba = artifacts["model"].predict_proba(X)[0]
    return pred_label, dict(zip(artifacts["label_encoder"].classes_, pred_proba))


# ======================================================================
# UI HELPERS
# ======================================================================
def bodytype_emoji(bt):
    return {"athletic": "💪", "pear": "🍐", "hourglass": "⏳", "unknown": "🧍"}.get(bt, "🧍")


def draw_boxes(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x0, y0, x1, y1 = det["box"]
        color = det.get("color_rgb", (145, 28, 47))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        tag = f"{det['image_category']} · {det['confidence']*100:.0f}%"
        # background box for text
        bbox = draw.textbbox((x0, max(0, y0 - 22)), tag, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, max(0, y0 - 22)), tag, fill="white", font=font)
    return annotated


def pil_to_b64(image, fmt="PNG"):
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def render_image_boxed(image, max_height_vh=65):
    b64 = pil_to_b64(image)
    st.markdown(
        f'<div class="fp-img-frame" style="max-height:{max_height_vh}vh;">'
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-height:{max_height_vh}vh;max-width:100%;object-fit:contain;border-radius:10px;" />'
        f'</div>', unsafe_allow_html=True,
    )


def probability_bar_chart(prob_dict, highlight_key=None):
    import plotly.graph_objects as go
    items = sorted(prob_dict.items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in items]
    vals = [v * 100 for _, v in items]
    colors = [NAVY if k == highlight_key else ACCENT if kv == max(vals) else "#D9D3C8"
              for k, kv in zip(labels, vals)]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        marker=dict(color=colors, line=dict(width=0)),
    ))
    fig.update_layout(
        height=180, margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(range=[0, 105], showgrid=False, showticklabels=False),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def fit_gauge(confidence, label):
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 26}},
        title={"text": f"<b>{label.upper()}</b>", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": ACCENT if confidence > 0.6 else TEAL},
            "steps": [
                {"range": [0, 40], "color": "#F6E9D6"},
                {"range": [40, 70], "color": "#F0D9DE"},
                {"range": [70, 100], "color": "#E4E1E8"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ======================================================================
# STYLING
# ======================================================================
# Palette matched to the brand logo (maroon / navy / tan)
ACCENT      = "#911C2F"  # maroon — from the logo's icon block
ACCENT_DARK = "#3D0A14"  # deep maroon for text on light accent backgrounds
TEAL        = "#B8863F"  # warm gold/tan — from the logo's hand illustration
NAVY        = "#232733"  # navy — from the logo's wordmark
CARD_BG     = "#FAF9F6"
BORDER      = "#E8E1D8"
MUTED       = "#8A8780"

st.markdown(f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}
    .stApp {{
        background: radial-gradient(ellipse 80% 60% at 15% 0%, #F7E3E7 0%, #FFFDFB 55%),
                    radial-gradient(ellipse 70% 50% at 100% 20%, #F3E9D8 0%, #FFFDFB 55%),
                    #FFFDFB !important;
        background-attachment: fixed;
    }}

    .fp-logo-corner {{
        position: fixed; top: 14px; right: 22px; z-index: 999;
        width: 56px; height: auto;
        filter: drop-shadow(0 3px 10px rgba(0,0,0,0.18));
        transition: transform 0.25s ease;
    }}
    .fp-logo-corner:hover {{ transform: scale(1.08) rotate(-3deg); }}
    .stApp, .stApp p, .stApp label, .stApp span, .stApp div {{ color: {NAVY}; }}
    .block-container {{padding-top: 2rem; max-width: 1200px;}}

    @keyframes fp-fadein {{from {{opacity:0; transform:translateY(10px);}} to {{opacity:1; transform:translateY(0);}}}}
    @keyframes fp-shine  {{0% {{background-position:0% 50%;}} 100% {{background-position:200% 50%;}}}}
    @keyframes fp-pulse  {{0%,100% {{box-shadow:0 0 0 0 {TEAL}55;}} 50% {{box-shadow:0 0 0 8px {TEAL}00;}}}}

    .fp-header h1 {{
        font-size: 36px; font-weight: 700; margin: 0; letter-spacing: -0.02em;
        background: linear-gradient(90deg, {ACCENT}, {TEAL}, {NAVY}, {ACCENT});
        background-size: 300% auto;
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: fp-shine 6s linear infinite;
    }}
    .fp-header p {{font-size: 15px; color: {MUTED}; margin: 4px 0 1.5rem;}}

    div[data-testid="stFileUploaderDropzone"] {{
        background: {CARD_BG}; border: 1.5px dashed {BORDER}; border-radius: 14px;
        transition: all 0.2s;
    }}
    div[data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {ACCENT}; background: {ACCENT}0d;
    }}

    .fp-card {{
        background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 16px 18px; margin-bottom: 12px;
        animation: fp-fadein 0.5s ease both;
        transition: all 0.2s;
    }}
    .fp-card:hover {{
        transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        border-color: {ACCENT}55;
    }}
    .fp-card h4 {{margin: 0 0 6px; font-size: 16px; font-weight: 600;}}
    .fp-card .fp-meta {{font-size: 13px; color: {MUTED}; margin: 0 0 10px;}}

    .fp-swatch {{
        display: inline-block; width: 12px; height: 12px; border-radius: 3px;
        margin-right: 6px; vertical-align: -1px; border: 1px solid rgba(0,0,0,0.08);
    }}
    .fp-badge {{
        display: inline-block; font-size: 12px; font-weight: 600; padding: 4px 12px;
        border-radius: 20px; background: {ACCENT}22; color: {ACCENT_DARK};
        animation: fp-pulse 2.5s ease-in-out infinite;
    }}
    .fp-badge-teal {{
        background: {TEAL}22; color: {TEAL}; animation: fp-pulse 2.5s ease-in-out infinite;
    }}

    .fp-section-label {{
        font-size: 13px; font-weight: 700; color: {NAVY};
        margin: 8px 0 10px; letter-spacing: 0.05em; text-transform: uppercase;
    }}
    .fp-bodytype {{
        background: linear-gradient(135deg, {CARD_BG}, #F6E9D6);
        border: 1px solid {BORDER}; border-radius: 14px; padding: 20px;
        margin-bottom: 16px; animation: fp-fadein 0.5s ease both;
    }}
    .fp-bt-label {{font-size: 11px; color: {MUTED}; margin: 0; text-transform: uppercase; letter-spacing: 0.08em;}}
    .fp-bt-value {{font-size: 24px; font-weight: 700; color: {ACCENT_DARK}; margin: 4px 0 0;}}
    .fp-bt-ratio {{font-size: 12px; color: {MUTED}; margin: 4px 0 0;}}

    .fp-img-frame {{
        background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 14px;
        overflow: hidden; padding: 8px;
        display: flex; align-items: center; justify-content: center;
        transition: transform 0.3s;
    }}
    .fp-img-frame:hover {{ transform: scale(1.01); }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {CARD_BG}; border-radius: 8px 8px 0 0; padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{ background: {ACCENT}22 !important; color: {ACCENT_DARK} !important; }}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# APP
# ======================================================================
_logo_b64 = load_logo_b64()
if _logo_b64:
    st.markdown(
        f'<img src="data:image/png;base64,{_logo_b64}" class="fp-logo-corner" alt="brand logo" />',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="fp-header"><h1>👗 Fashion Fit Predictor</h1>'
    '<p>Upload a photo — detect garments, estimate body type, predict fit per item.</p></div>',
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Detection confidence", 0.1, 0.9, 0.5, 0.05)
    show_crops = st.checkbox("Show garment thumbnails", value=True)
    show_gauge = st.checkbox("Show fit-confidence gauge", value=True)

    st.markdown("---")
    st.markdown("### 👤 Personal Info (optional)")
    st.caption("Override the training-set medians for a more personalized fit.")

    use_personal = st.checkbox("Use my measurements", value=False)
    if use_personal:
        user_height = st.number_input("Height (cm)", 120, 220, 165)
        user_weight = st.number_input("Weight (kg)", 30, 200, 63)
        user_age = st.number_input("Age", 10, 100, 30)
        user_body_override = st.selectbox(
            "Body type override",
            ["auto-detect", "athletic", "pear", "hourglass", "unknown"],
        )
    else:
        user_height = user_weight = user_age = None
        user_body_override = "auto-detect"

    st.markdown("---")
    st.markdown("### 📝 Review Text (optional)")
    review_text = st.text_area(
        "Add review text for a stronger signal",
        placeholder="e.g., 'Ran slightly small in the waist but looked great!'",
        height=80,
    )

    st.markdown("---")
    if "history" in st.session_state and st.session_state.history:
        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []


# ---------- MAIN ----------
uploaded_files = st.file_uploader(
    "📸 Upload one or more photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👆 Upload one or more photos to run the pipeline.")
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📚 Session History")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"{h['timestamp']} · {len(h['results'])} garment(s) · body: {h['body_type']}"):
                for r in h["results"]:
                    st.write(f"- **{r['category']}** → {r['predicted_fit']} "
                             f"({max(r['probabilities'].values())*100:.0f}%)")
    st.stop()

# ---------- LOAD MODELS ----------
try:
    artifacts = load_inference_artifacts()
    processor, detection_model = load_garment_detector()
    pose_landmarker, mp_module = load_pose_landmarker()
    nlp = load_spacy_nlp() if review_text.strip() else None
except Exception:
    st.error("💥 Failed to load models/artifacts.")
    st.code(traceback.format_exc())
    st.stop()

# ---------- PROCESS EACH IMAGE ----------
tabs = st.tabs([f"📷 Image {i+1}" for i in range(len(uploaded_files))])

for tab, uploaded_file in zip(tabs, uploaded_files):
    with tab:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("🔍 Running full pipeline..."):
            # Body type
            try:
                if user_body_override != "auto-detect":
                    body_type, ratio = user_body_override, None
                else:
                    body_type, ratio = estimate_body_type_from_pil(image, pose_landmarker, mp_module)
            except Exception:
                body_type, ratio = "unknown", None
                st.warning("Body type estimation failed. Continuing with 'unknown'.")

            # Detect garments
            try:
                detections = detect_garments(
                    image, processor, detection_model, artifacts["category_features"],
                    confidence_threshold,
                )
            except Exception:
                st.error("Garment detection failed.")
                st.code(traceback.format_exc())
                detections = []

            # Add color info + fit prediction per garment
            results = []
            user_numeric = None
            if use_personal:
                user_numeric = {
                    "height_cm": float(user_height),
                    "weight_kg": float(user_weight),
                    "age": float(user_age),
                }

            for det in detections:
                try:
                    crop = image.crop(det["box"])
                    color_name, color_rgb = get_dominant_color(crop)
                    det["color_name"] = color_name
                    det["color_rgb"] = color_rgb
                    det["crop_b64"] = pil_to_b64(crop, fmt="PNG")

                    pred_label, pred_proba = predict_one(
                        det["image_category"], det["raw_category"], body_type,
                        review_text.strip() or None, nlp, artifacts,
                        user_numeric=user_numeric,
                    )
                    det["predicted_fit"] = pred_label
                    det["fit_probabilities"] = pred_proba
                    det["category"] = det["image_category"]
                    results.append(det)
                except Exception:
                    st.warning(f"Fit prediction failed for '{det.get('fashionpedia_label', '?')}'.")
                    st.code(traceback.format_exc())

            # Outfit color harmony
            color_score, color_verdict, color_suggestions = None, None, None
            if len(results) >= 2:
                detected_colors = [tuple(r["color_rgb"]) for r in results]
                color_score = outfit_color_score(detected_colors)
                if color_score >= 85:
                    color_verdict = "✅ Recommended — these colors work well together!"
                else:
                    color_verdict = "⚠️ Not the best pairing — here's what would work better:"
                    # base suggestions on the first non-neutral garment (neutrals pair with anything)
                    base_rgb = next((c for c in detected_colors if not is_neutral(c)), detected_colors[0])
                    color_suggestions = suggest_colors(base_rgb)

            # Fallback if no garments
            if not results and detections == []:
                st.info("No garments detected — running fallback (Dresses).")
                try:
                    pred_label, pred_proba = predict_one(
                        "Dresses", "dress", body_type,
                        review_text.strip() or None, nlp, artifacts,
                        user_numeric=user_numeric,
                    )
                    results.append({
                        "fashionpedia_label": "fallback",
                        "category": "Dresses",
                        "image_category": "Dresses",
                        "confidence": 0.0,
                        "box": None,
                        "predicted_fit": pred_label,
                        "fit_probabilities": pred_proba,
                        "color_name": "n/a",
                        "color_rgb": (128, 128, 128),
                    })
                except Exception:
                    st.error("Fallback prediction failed.")
                    st.code(traceback.format_exc())

        # ---------- DISPLAY ----------
        col1, col2 = st.columns([1.05, 1])

        with col1:
            st.markdown('<p class="fp-section-label">📸 Annotated Photo</p>', unsafe_allow_html=True)
            boxed_dets = [d for d in results if d.get("box")]
            render_image_boxed(draw_boxes(image, boxed_dets) if boxed_dets else image)

            # Body type card
            st.markdown('<p class="fp-section-label" style="margin-top:20px;">🧍 Body Type</p>',
                        unsafe_allow_html=True)
            if body_type != "unknown":
                ratio_txt = f"shoulder/hip ratio {ratio:.2f}" if ratio else "manually set"
                st.markdown(
                    f'<div class="fp-bodytype"><p class="fp-bt-label">Estimated</p>'
                    f'<p class="fp-bt-value">{bodytype_emoji(body_type)} {body_type.capitalize()}</p>'
                    f'<p class="fp-bt-ratio">{ratio_txt}</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="fp-bodytype"><p class="fp-bt-value" style="font-size:16px;">'
                    '🧍 No person detected — using \'unknown\'</p></div>',
                    unsafe_allow_html=True,
                )

            # Color harmony card
            if color_score is not None:
                st.markdown('<p class="fp-section-label" style="margin-top:20px;">🎨 Color Harmony</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="fp-bodytype"><p class="fp-bt-label">Outfit score: {color_score}/100</p>'
                    f'<p class="fp-bt-value" style="font-size:16px;">{color_verdict}</p></div>',
                    unsafe_allow_html=True,
                )
                if color_suggestions:
                    swatch_html = "".join(
                        f'<div style="display:inline-block;text-align:center;margin:6px 8px 0 0;">'
                        f'<div style="width:32px;height:32px;border-radius:6px;'
                        f'background:#{r:02x}{g:02x}{b:02x};border:1px solid {BORDER};"></div>'
                        f'<div style="font-size:11px;margin-top:2px;">{name}</div></div>'
                        for name, (r, g, b) in color_suggestions
                    )
                    st.markdown(f'<div>{swatch_html}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(
                f'<p class="fp-section-label">✨ Results ({len(results)} garment{"s" if len(results)!=1 else ""})</p>',
                unsafe_allow_html=True,
            )
            if not results:
                st.info("No garments detected above threshold. Try lowering it.")

            for i, det in enumerate(results):
                emoji = GARMENT_EMOJI.get(det["fashionpedia_label"], "🏷️")
                fit_em = FIT_EMOJI.get(det["predicted_fit"], "📏")
                swatch = "#%02x%02x%02x" % det["color_rgb"]
                top_prob = max(det["fit_probabilities"].values())

                with st.container():
                    st.markdown(
                        f'<div class="fp-card" style="animation-delay:{i*0.08:.2f}s;">'
                        f'<h4>{emoji} {det["fashionpedia_label"].title()}</h4>'
                        f'<p class="fp-meta">'
                        f'Detected at <b>{det["confidence"]*100:.0f}%</b> · '
                        f'Category <b>{det["category"]}</b> · '
                        f'<span class="fp-swatch" style="background:{swatch};"></span>{det["color_name"]}'
                        f'</p>'
                        f'<span class="fp-badge">{fit_em} Predicted fit: {det["predicted_fit"]}</span> '
                        f'<span class="fp-badge fp-badge-teal">Confidence: {top_prob*100:.0f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    sub_cols = st.columns([1, 2] if show_crops and det.get("crop_b64") else [1])
                    if show_crops and det.get("crop_b64"):
                        with sub_cols[0]:
                            st.markdown(
                                f'<img src="data:image/png;base64,{det["crop_b64"]}" '
                                f'style="width:100%;border-radius:10px;border:1px solid {BORDER};" />',
                                unsafe_allow_html=True,
                            )
                        chart_col = sub_cols[1]
                    else:
                        chart_col = sub_cols[0]

                    with chart_col:
                        st.plotly_chart(
                            probability_bar_chart(det["fit_probabilities"], det["predicted_fit"]),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key=f"bar_{tab}_{i}_{id(det)}",
                        )
                        if show_gauge:
                            st.plotly_chart(
                                fit_gauge(top_prob, det["predicted_fit"]),
                                use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"gauge_{tab}_{i}_{id(det)}",
                            )

        # ---------- REPORT & HISTORY ----------
        if results:
            report = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "body_type": body_type,
                "shoulder_hip_ratio": ratio,
                "user_measurements": user_numeric,
                "review_text": review_text.strip() or None,
                "color_harmony_score": color_score,
                "results": [
                    {
                        "garment": r["fashionpedia_label"],
                        "category": r["category"],
                        "detection_confidence": r["confidence"],
                        "color": r["color_name"],
                        "predicted_fit": r["predicted_fit"],
                        "probabilities": {k: float(v) for k, v in r["fit_probabilities"].items()},
                    }
                    for r in results
                ],
            }
            st.session_state.history.append({
                "timestamp": report["timestamp"],
                "body_type": body_type,
                "results": [{"category": r["category"], "predicted_fit": r["predicted_fit"],
                             "probabilities": r["fit_probabilities"]} for r in results],
            })

            st.download_button(
                "📥 Download JSON report",
                data=json.dumps(report, indent=2),
                file_name=f"fit_report_{report['timestamp'].replace(':', '-')}.json",
                mime="application/json",
                key=f"dl_{tab}",
            )

        st.caption(
            "ℹ️ Numeric fields (height/weight/age) use training-set medians unless you override them "
            "in the sidebar. Body type comes from MediaPipe pose landmarks."
        )