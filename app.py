
import json
import re
from html import escape
from pathlib import Path
from textwrap import dedent

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from utils.llm_response import generate_disease_explanation


# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_plant_disease.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "dataset" / "class_names.json"
KNOWLEDGE_PATH = (
    PROJECT_DIR / "knowledge" / "class_knowledge_cards.json"
)

IMAGE_SIZE = (224, 224)
DISEASE_SECTION_TITLES = (
    "Disease Overview",
    "Cause and Pathogen",
    "Symptoms",
    "Favorable Conditions",
    "Spread",
    "Management and Prevention",
)


def parse_disease_sections(explanation):
    title_pattern = "|".join(
        re.escape(title) for title in DISEASE_SECTION_TITLES
    )
    heading_pattern = re.compile(
        rf"^[ \t]*(?:#{{1,6}}[ \t]*)?(?:\*\*)?"
        rf"(?:\d+[.)][ \t]*)?(?:\*\*)?({title_pattern})"
        rf"(?:\*\*)?[ \t]*:?[ \t]*",
        re.IGNORECASE | re.MULTILINE,
    )
    headings = list(heading_pattern.finditer(explanation))
    sections = []

    for index, heading in enumerate(headings):
        body_end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(explanation)
        )
        title = next(
            title
            for title in DISEASE_SECTION_TITLES
            if title.casefold() == heading.group(1).casefold()
        )
        sections.append((title, explanation[heading.end():body_end].strip()))

    if tuple(title for title, _ in sections) != DISEASE_SECTION_TITLES:
        raise ValueError(
            "The AI explanation did not contain all six sections in order."
        )

    return sections


# --------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(71, 126, 67, 0.22), transparent 38%),
            linear-gradient(145deg, #071b12 0%, #0b2418 52%, #102d1d 100%);
        color: #edf5e9;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 27, 18, 0.72);
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 860px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        position: relative;
        z-index: 1;
    }

    .falling-leaf-layer {
        position: fixed;
        inset: 0;
        z-index: 0;
        overflow: hidden;
        pointer-events: none;
    }

    .falling-leaf {
        position: absolute;
        top: -14vh;
        height: auto;
        opacity: 0;
        filter: drop-shadow(0 5px 10px rgba(0, 0, 0, 0.18));
        animation: leaf-drift-fall var(--fall-duration) linear var(--fall-delay) infinite;
        will-change: transform, opacity;
    }

    h1, h2, h3, p, label, [data-testid="stCaptionContainer"] {
        color: #edf5e9;
    }

    [data-testid="stAppViewContainer"] .main .stMarkdown p:not(.result-eyebrow):not(.result-name):not(.top-three-eyebrow),
    [data-testid="stAppViewContainer"] .main .stMarkdown li,
    [data-testid="stAlert"] p {
        color: #e4efe0;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    [data-testid="stAlert"] {
        border: 1px solid rgba(185, 239, 138, 0.25);
        border-radius: 16px;
        background: rgba(185, 239, 138, 0.08);
        color: #e8f4df;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed rgba(185, 239, 138, 0.55);
        border-radius: 18px;
        background: rgba(16, 41, 28, 0.78);
        transition: border-color 180ms ease, background 180ms ease, transform 180ms ease;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #b9ef8a;
        background: rgba(35, 75, 43, 0.8);
        transform: translateY(-2px);
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stButton"] button {
        border: 1px solid rgba(185, 239, 138, 0.58);
        border-radius: 999px;
        background: #b9ef8a;
        color: #102719;
        font-weight: 700;
        transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease;
    }

    [data-testid="stFileUploader"] button:hover,
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px) scale(1.02);
        background: #d0f7ae;
        box-shadow: 0 8px 24px rgba(129, 190, 94, 0.25);
    }

    [data-testid="stButton"] button:active {
        transform: translateY(0) scale(0.98);
    }

    [data-testid="stImage"] {
        animation: image-arrive 650ms cubic-bezier(0.2, 0.75, 0.25, 1) both;
        overflow: hidden;
        border: 1px solid rgba(185, 239, 138, 0.22);
        border-radius: 20px;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.24);
    }

    .result-card {
        margin: 1.1rem 0 1.5rem;
        padding: 1.5rem 1.65rem;
        border: 1px solid rgba(185, 239, 138, 0.32);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(28, 67, 39, 0.96), rgba(15, 42, 27, 0.98));
        box-shadow: 0 18px 46px rgba(0, 0, 0, 0.24);
        animation: result-card-enter 1s cubic-bezier(0.2, 0.75, 0.25, 1) both;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        margin: 0.8rem 0;
        border: 1px solid rgba(185, 239, 138, 0.32);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(28, 67, 39, 0.82), rgba(15, 42, 27, 0.9));
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] h4 {
        margin: 0.15rem 0 0.4rem;
        color: #b9ef8a;
        font-size: 1.15rem;
        font-weight: 750;
    }

    .result-eyebrow, .top-three-eyebrow {
        margin: 0 0 0.45rem;
        color: #b9ef8a;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .result-name {
        margin: 0;
        color: #f3f8ee;
        font-size: clamp(1.3rem, 3vw, 1.85rem);
        font-weight: 750;
        line-height: 1.25;
    }

    .confidence-line {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin-top: 1.3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(237, 245, 233, 0.16);
        color: #d7e8d0;
    }

    .confidence-value {
        color: #c9f49f;
        font-size: 1.55rem;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
    }

    .top-three-card {
        margin: 1.1rem 0;
        padding: 1.45rem 1.55rem;
        border: 1px solid rgba(185, 239, 138, 0.2);
        border-radius: 22px;
        background: rgba(12, 35, 23, 0.88);
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.18);
        animation: top-three-enter 950ms cubic-bezier(0.2, 0.75, 0.25, 1) 280ms both;
    }

    .prediction-row {
        padding: 1rem 0 0.9rem;
        border-top: 1px solid rgba(237, 245, 233, 0.1);
        animation: prediction-fade 600ms ease-out both;
    }

    .prediction-heading {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.65rem;
        color: #edf5e9;
        line-height: 1.4;
    }

    .prediction-rank {
        color: #b9ef8a;
        font-weight: 800;
    }

    .prediction-score {
        color: #c9f49f;
        font-weight: 750;
        white-space: nowrap;
    }

    .prediction-track {
        height: 7px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(237, 245, 233, 0.12);
    }

    .prediction-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #83c76c, #c9f49f);
        transform-origin: left;
        animation: fill-in 800ms ease-out both;
    }

    @keyframes result-card-enter {
        0% { opacity: 0; filter: blur(7px); transform: translateY(28px) scale(0.96); }
        70% { opacity: 1; filter: blur(0); transform: translateY(-3px) scale(1.01); }
        100% { opacity: 1; filter: blur(0); transform: translateY(0) scale(1); }
    }

    @keyframes top-three-enter {
        0% { opacity: 0; clip-path: inset(0 0 100% 0 round 22px); filter: blur(6px); transform: translateY(22px) scale(0.97); }
        65% { opacity: 1; clip-path: inset(0 0 0 0 round 22px); filter: blur(0); transform: translateY(-2px) scale(1.005); }
        100% { opacity: 1; clip-path: inset(0 0 0 0 round 22px); filter: blur(0); transform: translateY(0) scale(1); }
    }

    @keyframes prediction-fade {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fill-in {
        from { transform: scaleX(0); }
        to { transform: scaleX(1); }
    }

    @keyframes image-arrive {
        from { opacity: 0; transform: translateY(14px) scale(0.985); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes leaf-drift-fall {
        0% { opacity: 0; transform: translate3d(0, -5vh, 0) rotate(-35deg); }
        8% { opacity: var(--leaf-opacity); }
        50% { transform: translate3d(var(--leaf-drift), 48vh, 0) rotate(190deg); }
        92% { opacity: var(--leaf-opacity); }
        100% { opacity: 0; transform: translate3d(0, 110vh, 0) rotate(385deg); }
    }

    @media (min-width: 1450px) {
        .falling-leaf-layer {
            display: block;
        }
    }

    @media (max-width: 1449px) {
        .falling-leaf-layer {
            display: none;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

leaf_positions = [
    ("left: 4vw;", "38px", "15s", "-12s", "44px", "0.66"),
    ("left: 10vw;", "52px", "19s", "-7s", "-38px", "0.56"),
    ("left: 16vw;", "30px", "17s", "-15s", "30px", "0.5"),
    ("right: 4vw;", "46px", "18s", "-4s", "-42px", "0.62"),
    ("right: 10vw;", "34px", "14s", "-10s", "36px", "0.54"),
    ("right: 16vw;", "54px", "20s", "-17s", "-32px", "0.48"),
]
leaf_shape = """
<svg viewBox="0 0 100 150" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <path d="M50 4 C86 24 99 58 86 91 C76 117 58 138 50 146 C40 136 21 117 12 91 C0 57 14 23 50 4Z" fill="#83c96d"/>
    <path d="M50 140 C51 99 50 55 50 12 M50 91 C38 78 28 66 19 51 M50 112 C64 97 74 82 82 66 M50 69 C61 58 68 47 74 36" fill="none" stroke="#d8f2b5" stroke-linecap="round" stroke-width="2.4" opacity=".8"/>
</svg>
"""
falling_leaves = "".join(
    (
        f'<span class="falling-leaf" style="{side} width: {size}; '
        f'--fall-duration: {duration}; --fall-delay: {delay}; '
        f'--leaf-drift: {drift}; --leaf-opacity: {opacity};">'
        f'{leaf_shape}</span>'
    )
    for side, size, duration, delay, drift, opacity in leaf_positions
)
st.markdown(
    f'<div class="falling-leaf-layer" aria-hidden="true">{falling_leaves}</div>',
    unsafe_allow_html=True,
)

st.title("🌿 Plant Disease Detection")
st.write(
    "Upload a plant leaf image to see the model's predicted "
    "disease category and its top three predictions."
)


with st.expander(
    "⚠️ Supported Plants & Disclaimer",
    expanded=True
):
    st.markdown("""
    **Supported plants (14):**

    Apple, Blueberry, Cherry, Corn (Maize), Grape,
    Orange, Peach, Bell Pepper, Potato, Raspberry,
    Soybean, Squash, Strawberry, and Tomato.

    **Important disclaimer:**

    This model was trained on the PlantVillage dataset
    and recognizes only its 38 trained categories.
    Predictions for unsupported plants, diseases, or
    unfamiliar real-world images may be incorrect.

    The confidence score does not guarantee correctness.
    This application is intended for educational purposes
    and is not a substitute for professional agricultural advice.
    """)


st.info(
    "This application recognizes the 38 categories included in "
    "the training dataset. Predictions are estimates, not a "
    "guaranteed diagnosis."
)

with st.expander(
    "📥 AI Assistant Setup — Required Model Files",
    expanded=False
):
    st.markdown("""
    **The AI Plant Disease Assistant requires two additional model files.**

    These files are not included in this GitHub repository because
    of their large file sizes.

    **Setup instructions:**

    1. Open the provided Google Colab notebook.
    2. Run the notebook and download both model files.
    3. Place both downloaded files inside the project's `models` folder.

    **Required filenames:**

    - `Qwen3-4B-Q4_K_M.gguf`
    - `plant_disease_adapter.gguf`

    **Important:** Both files are required for the fine-tuned AI
    Plant Disease Assistant to work.

    The CNN-based plant disease prediction feature uses a separate
    model and can work independently.
    """)

# --------------------------------------------------
# 3. LOAD THE TRAINED MODEL AND CLASS NAMES
# --------------------------------------------------

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
    )


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)

    if not isinstance(names, list) or not names:
        raise ValueError(
            "class_names.json must contain a non-empty JSON list."
        )

    return names


try:
    model = load_trained_model()
    class_names = load_class_names()

    if model.output_shape[-1] != len(class_names):
        st.error(
            "The model output count does not match the number "
            "of class names. Please check your model and JSON file."
        )
        st.stop()

except Exception as error:
    st.error(f"Could not load the model or class names: {error}")
    st.stop()


# --------------------------------------------------
# 4. PREPROCESS THE UPLOADED IMAGE
# --------------------------------------------------

def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    # Must match the preprocessing used during training.
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return np.expand_dims(image_array, axis=0)


def format_class_name(class_name):
    return class_name.replace("___", " — ").replace("_", " ")


# --------------------------------------------------
# LOAD AND MATCH PLANT DISEASE KNOWLEDGE CARDS
# --------------------------------------------------

@st.cache_data
def load_knowledge_cards():
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge cards not found: {KNOWLEDGE_PATH}"
        )

    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_class_name(name):
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def find_knowledge_card(data, target_class):
    target = normalize_class_name(target_class)

    if isinstance(data, dict):

        # Direct dictionary lookup
        for key, value in data.items():
            if normalize_class_name(key) == target:
                return value

        # Search records containing a class/disease identifier
        identifier_fields = (
            "class_name",
            "class",
            "label",
            "disease",
            "disease_name",
            "name",
            "category",
        )

        for field in identifier_fields:
            for key, value in data.items():
                if key.lower() == field and normalize_class_name(value) == target:
                    return data

        # Search nested dictionaries
        for value in data.values():
            result = find_knowledge_card(value, target_class)
            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:
            result = find_knowledge_card(item, target_class)
            if result is not None:
                return result

    return None


# --------------------------------------------------
# 5. IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a single plant leaf.",
)


# --------------------------------------------------
# 6. PREDICT THE PLANT DISEASE
# --------------------------------------------------

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)
        original_image = ImageOps.exif_transpose(
            original_image
        ).convert("RGB")

        st.subheader("Uploaded image")
        st.image(
            original_image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

        if st.button("🔍 Analyze Leaf", type="primary"):
            with st.spinner("Analyzing the leaf image..."):
                input_array = prepare_image(original_image)
                probabilities = model.predict(
                    input_array,
                    verbose=0,
                )[0]

            top_indices = np.argsort(probabilities)[::-1][:3]
            best_index = int(top_indices[0])
            best_class = class_names[best_index]
            confidence = float(probabilities[best_index]) * 100

            st.subheader("Prediction result")

            st.markdown(
                f"""
                <section class="result-card">
                    <p class="result-eyebrow">Most likely match</p>
                    <p class="result-name">{escape(format_class_name(best_class))}</p>
                    <div class="confidence-line">
                        <span>Model confidence</span>
                        <span class="confidence-value">{confidence:.2f}%</span>
                    </div>
                </section>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "This percentage is the model's predicted probability, "
                "not a measure of guaranteed correctness."
            )

            st.subheader("Top 3 predictions")

            prediction_rows = []
            for rank, index in enumerate(top_indices, start=1):
                label = format_class_name(class_names[int(index)])
                score = float(probabilities[int(index)]) * 100
                percentage = min(max(score, 0.0), 100.0)
                prediction_rows.append(
                    dedent(
                        f"""
                        <div class="prediction-row" style="animation-delay: {0.35 + rank * 0.16:.2f}s">
                            <div class="prediction-heading">
                                <span><span class="prediction-rank">{rank}.</span> {escape(label)}</span>
                                <span class="prediction-score">{score:.2f}%</span>
                            </div>
                            <div class="prediction-track">
                                <div class="prediction-fill" style="width: {percentage:.2f}%; animation-delay: {0.55 + rank * 0.16:.2f}s"></div>
                            </div>
                        </div>
                        """
                    ).strip()
                )

            st.markdown(
                '<section class="top-three-card">'
                '<p class="top-three-eyebrow">Other likely matches</p>'
                + "".join(prediction_rows)
                + "</section>",
                unsafe_allow_html=True,
            )
                        # --------------------------------------------------
            # 7. FINE-TUNED LLM DISEASE EXPLANATION
            # --------------------------------------------------

            st.subheader("🌱 AI Plant Disease Assistant")

            try:
                knowledge_data = load_knowledge_cards()

                knowledge_card = find_knowledge_card(
                    knowledge_data,
                    best_class
                )

                if knowledge_card is None:
                    st.warning(
                        "No matching knowledge card was found "
                        f"for {format_class_name(best_class)}."
                    )

                else:
                    knowledge_text = json.dumps(
                        knowledge_card,
                        ensure_ascii=False,
                        indent=2
                    )

                    with st.spinner(
                        "Generating disease explanation with your fine-tuned AI model..."
                    ):
                        explanation = generate_disease_explanation(
                            best_class,
                            knowledge_text
                        )

                    for title, section_text in parse_disease_sections(explanation):
                        with st.container(border=True):
                            st.markdown(f"#### {title}:")
                            st.markdown(section_text)

            except Exception as error:
                st.error(
                    f"Could not generate the AI explanation: {error}"
                )

            st.divider()

            st.caption(
                "The model was trained on the PlantVillage dataset. "
                "Results may be less reliable for blurry photos, "
                "unfamiliar plants, unusual lighting, or leaf images "
                "outside its 38 trained categories. For important "
                "crop decisions, confirm the result with a qualified "
                "agricultural expert."
            )

    except Exception as error:
        st.error(f"Could not analyze this image: {error}")

else:
    st.markdown(
        "### How to use\n"
        "1. Upload a JPG, JPEG, or PNG leaf image.\n"
        "2. Click **Analyze Leaf**.\n"
        "3. Review the predicted category and top three results."
    )
