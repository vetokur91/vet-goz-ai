from __future__ import annotations

import base64
import html
import logging
import os
import time
from pathlib import Path
from typing import Mapping

# Model çıkarımı için kullanılacak CPU iş parçacığı sayısı.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import streamlit as st
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "v2_best.pt"
VIDEO_PATH = REPO_ROOT / "site_background.mp4"
BACKGROUND_PATH = REPO_ROOT / "site_background.png"

LLM_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
EXPECTED_CLASS_COUNT = 24
APP_VERSION = "Prototip v5.1 · Streamlit"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("vet-goz-ai-streamlit")


TURKCE_ISIMLER = {
    "anterior_uveitis": "Anterior Üveit",
    "blepharitis": "Blefarit",
    "cataract": "Katarakt",
    "cherry_eye": "Cherry Eye (Kiraz Göz)",
    "ciliar_disorders": "Kirpik Anomalileri (Distichiasis/Trichiasis)",
    "conjunctivitis": "Konjonktivit",
    "corneal_edema": "Korneal Ödem",
    "corneal_pigmentation": "Korneal Pigmentasyon",
    "corneal_sequestrum": "Korneal Sekestrum",
    "corneal_ulcer": "Korneal Ülser",
    "dermoid": "Dermoid",
    "ectropion": "Ektropion",
    "entropion": "Entropion",
    "eosinophilic_keratitis": "Eozinofilik Keratit",
    "glaucoma": "Glokom",
    "healthy": "Sağlıklı",
    "hyphema": "Hifema",
    "hypopyon": "Hipopiyon",
    "iris_atrophy": "İris Atrofisi",
    "iris_hyperpigmentation": "İris Hiperpigmentasyonu",
    "lens_luxation": "Lens Luksasyonu",
    "neoplasia": "Neoplazi / Kitle",
    "pannus": "Pannus (Kronik Yüzeysel Keratit)",
    "persistent_pupillary_membran": "Persistan Pupiller Membran (PPM)",
}


SYSTEM_PROMPT = """Sen veteriner hekimlere yönelik bir veteriner oftalmoloji karar-destek asistanısın.
Sana bir görüntü sınıflandırma modelinin kedi veya köpek ön segment göz fotoğrafı için ürettiği
ilk beş sınıf ve model güven skorları verilecek. Bunlar kalibre edilmiş hastalık olasılıkları değildir.
Görüntüyü doğrudan görmediğin için kesin tanı koyma ve hastaya özgü reçete oluşturma.

Türkçe, kısa ve klinik olarak temkinli bir yanıt üret. Aşağıdaki başlıkları kullan:

### Klinik Yorum
En yüksek skorlu sınıfı yalnızca "öncelikli model çıktısı" olarak belirt. Skorlar birbirine yakınsa
veya teknik güven düşükse bunu açıkça vurgula ve ayırıcı tanı yaklaşımı kullan.

### Tanıyı Kesinleştirmek İçin Muayene ve Tetkikler
Hastalığa uygun somut muayene ve tetkikleri öncelik sırasıyla yaz. Alakasız testleri sıralama.

### Tanı Doğrulanırsa Genel Tedavi Yaklaşımı
İlaç grubu ve cerrahi yaklaşım düzeyinde bilgi ver. Marka, doz, uygulama sıklığı veya reçete yazma.
Korneal ülser dışlanmadan topikal kortikosteroid önermeme gibi kritik kontrendikasyonları belirt.

Yanıtın başında tek cümlelik prototip uyarısı bulunmalı. 350 kelimeyi geçme.
"""


st.set_page_config(
    page_title="Veteriner Göz Hastalıkları AI Asistanı",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def file_to_data_uri(path_string: str, mime_type: str) -> str:
    path = Path(path_string)
    if not path.is_file():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def install_background() -> None:
    video_uri = file_to_data_uri(str(VIDEO_PATH), "video/mp4")
    poster_uri = file_to_data_uri(str(BACKGROUND_PATH), "image/png")

    if video_uri:
        media_html = f"""
        <video id="site-background-video" autoplay muted loop playsinline preload="auto"
               poster="{poster_uri}">
            <source src="{video_uri}" type="video/mp4">
        </video>
        """
    elif poster_uri:
        media_html = f'<div id="site-background-image" style="background-image:url(\'{poster_uri}\')"></div>'
    else:
        media_html = '<div id="site-background-image"></div>'

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: #050505;
            --panel: rgba(12, 12, 12, 0.90);
            --panel-soft: rgba(19, 19, 19, 0.88);
            --orange: #ff9d2e;
            --orange-light: #ffc16b;
            --cream: #f6efe7;
            --muted: #b9afa5;
            --line: rgba(255, 157, 46, 0.28);
            --line-strong: rgba(255, 157, 46, 0.68);
        }}

        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
        }}

        body {{
            background: var(--bg);
        }}

        #site-background-video,
        #site-background-image {{
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            z-index: -4;
            pointer-events: none;
        }}

        #site-background-video {{
            object-fit: cover;
            object-position: center right;
            opacity: 0.96;
            filter: brightness(0.90) contrast(1.10) saturate(1.08);
        }}

        #site-background-image {{
            background-position: center right;
            background-repeat: no-repeat;
            background-size: cover;
            opacity: 0.94;
            filter: brightness(0.88) contrast(1.10) saturate(1.08);
        }}

        .site-background-overlay {{
            position: fixed;
            inset: 0;
            z-index: -3;
            pointer-events: none;
            background:
                linear-gradient(
                    90deg,
                    rgba(5,5,5,0.94) 0%,
                    rgba(5,5,5,0.82) 27%,
                    rgba(5,5,5,0.56) 48%,
                    rgba(5,5,5,0.18) 72%,
                    rgba(5,5,5,0.06) 100%
                ),
                linear-gradient(
                    180deg,
                    rgba(5,5,5,0.16) 0%,
                    rgba(5,5,5,0.03) 45%,
                    rgba(5,5,5,0.72) 100%
                );
        }}

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .stApp {{
            background: transparent !important;
        }}

        [data-testid="stHeader"] {{
            background: rgba(5,5,5,0.42) !important;
            backdrop-filter: blur(12px);
        }}

        [data-testid="stToolbar"],
        #MainMenu,
        footer {{
            visibility: hidden;
        }}

        .block-container {{
            max-width: 1320px;
            padding-top: 2.1rem;
            padding-bottom: 3rem;
        }}

        .topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            padding: 0.8rem 0 1.1rem;
            border-bottom: 1px solid var(--line);
            margin-bottom: 2.2rem;
        }}

        .brand {{
            color: var(--orange-light);
            font-weight: 760;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            font-size: 0.77rem;
        }}

        .version-chip {{
            color: var(--orange-light);
            background: rgba(255,157,46,0.08);
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.42rem 0.78rem;
            font-size: 0.74rem;
            font-weight: 700;
        }}

        .hero {{
            max-width: 780px;
            padding: 1.4rem 0 2rem;
        }}

        .hero-kicker,
        .panel-number,
        .results-heading,
        .developer-role {{
            color: var(--orange);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }}

        .hero h1 {{
            margin: 0.7rem 0 0.9rem;
            color: var(--cream);
            font-size: clamp(2.35rem, 5vw, 4.8rem);
            line-height: 0.98;
            letter-spacing: -0.055em;
            text-shadow: 0 8px 40px rgba(0,0,0,0.60);
        }}

        .hero h1 span {{
            display: block;
            color: var(--orange);
        }}

        .hero p {{
            margin: 0;
            max-width: 700px;
            color: #d0c6bc;
            font-size: 1rem;
            line-height: 1.75;
            text-shadow: 0 3px 18px rgba(0,0,0,0.72);
        }}

        .hero-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.25rem;
        }}

        .hero-badge {{
            border: 1px solid var(--line);
            background: rgba(13,13,13,0.65);
            color: var(--orange-light);
            border-radius: 999px;
            padding: 0.44rem 0.72rem;
            font-size: 0.75rem;
            font-weight: 680;
            backdrop-filter: blur(10px);
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: linear-gradient(145deg, rgba(12,12,12,0.94), rgba(19,19,19,0.86));
            border: 1px solid var(--line) !important;
            border-radius: 22px !important;
            box-shadow: 0 26px 80px rgba(0,0,0,0.42);
            backdrop-filter: blur(18px) saturate(1.05);
        }}

        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding: 0.4rem;
        }}

        .panel-title {{
            padding-bottom: 1rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(255,157,46,0.15);
        }}

        .panel-title h2 {{
            color: var(--cream);
            margin: 0.34rem 0 0.32rem;
            font-size: 1.55rem;
            letter-spacing: -0.025em;
        }}

        .panel-title p {{
            color: var(--muted);
            margin: 0;
            line-height: 1.58;
            font-size: 0.88rem;
        }}

        .field-label {{
            color: var(--orange-light);
            font-size: 0.78rem;
            font-weight: 780;
            letter-spacing: 0.055em;
            text-transform: uppercase;
            margin: 0.3rem 0 0.55rem;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            min-height: 225px;
            background: rgba(5,5,5,0.72) !important;
            border: 1px dashed var(--line-strong) !important;
            border-radius: 16px !important;
        }}

        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] div {{
            color: #d9cfc5 !important;
        }}

        [data-testid="stCameraInput"] {{
            background: rgba(5,5,5,0.72);
            border: 1px dashed var(--line);
            border-radius: 16px;
            padding: 0.65rem;
        }}

        [data-testid="stRadio"] label p {{
            color: #ddd2c7 !important;
        }}

        .stButton > button[kind="primary"] {{
            width: 100%;
            min-height: 3.25rem;
            border: 1px solid #ffae4d !important;
            border-radius: 13px !important;
            color: #120b04 !important;
            background: linear-gradient(110deg, #ff8a00, #ffb34e) !important;
            font-weight: 850 !important;
            letter-spacing: 0.02em;
            box-shadow: 0 14px 34px rgba(255,138,0,0.24);
        }}

        .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.08);
            transform: translateY(-1px);
        }}

        .stButton > button[kind="secondary"] {{
            width: 100%;
            min-height: 2.9rem;
            border-radius: 12px !important;
            border: 1px solid var(--line-strong) !important;
            background: rgba(255,157,46,0.08) !important;
            color: var(--orange-light) !important;
            font-weight: 760 !important;
        }}

        .helper-text {{
            color: #9f968e;
            font-size: 0.79rem;
            line-height: 1.55;
            margin-top: 0.65rem;
        }}

        .confidence-card {{
            border: 1px solid var(--line);
            background: rgba(5,5,5,0.76);
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
        }}

        .confidence-high {{
            border-color: rgba(91, 215, 152, 0.55);
        }}

        .confidence-medium {{
            border-color: rgba(255, 183, 77, 0.64);
        }}

        .confidence-low {{
            border-color: rgba(255, 112, 112, 0.60);
        }}

        .eyebrow {{
            color: var(--orange);
            font-size: 0.68rem;
            letter-spacing: 0.13em;
            font-weight: 850;
        }}

        .diagnosis-name {{
            color: var(--cream);
            font-size: 1.65rem;
            line-height: 1.15;
            font-weight: 820;
            margin-top: 0.42rem;
        }}

        .confidence-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem 1rem;
            margin-top: 0.8rem;
            color: #d0c5bb;
            font-size: 0.76rem;
        }}

        .confidence-disclaimer {{
            color: #8f877f;
            font-size: 0.7rem;
            line-height: 1.5;
            margin: 0.8rem 0 0;
        }}

        .result-list {{
            display: flex;
            flex-direction: column;
            gap: 0.84rem;
            margin-top: 0.8rem;
        }}

        .result-row {{
            padding: 0.75rem 0.8rem;
            border: 1px solid rgba(255,157,46,0.12);
            background: rgba(5,5,5,0.62);
            border-radius: 12px;
        }}

        .result-head {{
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            margin-bottom: 0.52rem;
        }}

        .result-name-wrap {{
            display: flex;
            align-items: center;
            min-width: 0;
            gap: 0.68rem;
        }}

        .result-rank {{
            color: var(--orange);
            font-size: 0.66rem;
            font-weight: 860;
            letter-spacing: 0.07em;
        }}

        .result-name {{
            color: #eee5dc;
            font-size: 0.85rem;
            font-weight: 690;
        }}

        .result-score {{
            color: var(--orange-light);
            font-size: 0.8rem;
            font-weight: 840;
            white-space: nowrap;
        }}

        .result-track {{
            height: 6px;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
        }}

        .result-fill {{
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #ff7900, #ffc16b);
            box-shadow: 0 0 14px rgba(255,157,46,0.42);
        }}

        .empty-results {{
            display: grid;
            place-items: center;
            min-height: 240px;
            text-align: center;
            color: #8f877f;
            border: 1px dashed rgba(255,157,46,0.20);
            border-radius: 14px;
            background: rgba(5,5,5,0.44);
            padding: 1.5rem;
            line-height: 1.65;
        }}

        .metric-line {{
            color: #9f968e;
            font-size: 0.74rem;
            margin-top: 0.75rem;
        }}

        [data-testid="stMarkdownContainer"] h3 {{
            color: var(--orange-light);
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {{
            color: #d8cec4;
            line-height: 1.68;
        }}

        .developer-section {{
            margin-top: 2.5rem;
            border-top: 1px solid var(--line);
            padding-top: 1.7rem;
        }}

        .developer-section h2 {{
            color: var(--cream);
            margin: 0;
            font-size: 1.35rem;
        }}

        .developer-section > p {{
            color: var(--muted);
            margin: 0.35rem 0 1.1rem;
        }}

        .developer-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
        }}

        .developer-card {{
            background: rgba(12,12,12,0.80);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem;
            backdrop-filter: blur(12px);
        }}

        .developer-name {{
            color: #eee4da;
            margin-top: 0.33rem;
            font-weight: 730;
        }}

        .site-footer {{
            color: #79716a;
            text-align: center;
            font-size: 0.72rem;
            line-height: 1.6;
            margin-top: 2.2rem;
            padding: 1.2rem 0 0.3rem;
        }}

        @media (max-width: 800px) {{
            #site-background-video {{
                object-position: 68% center;
                opacity: 0.76;
            }}

            #site-background-image {{
                background-position: 68% center;
                opacity: 0.76;
            }}

            .site-background-overlay {{
                background:
                    linear-gradient(
                        90deg,
                        rgba(5,5,5,0.92) 0%,
                        rgba(5,5,5,0.72) 65%,
                        rgba(5,5,5,0.38) 100%
                    ),
                    linear-gradient(
                        180deg,
                        rgba(5,5,5,0.18) 0%,
                        rgba(5,5,5,0.78) 100%
                    );
            }}

            .block-container {{
                padding-left: 0.8rem;
                padding-right: 0.8rem;
                padding-top: 1rem;
            }}

            .hero h1 {{
                font-size: 2.55rem;
            }}

            .developer-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>

        {media_html}
        <div class="site-background-overlay"></div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH.name}. "
            "Dosyanın GitHub deposunun ana dizininde olduğundan emin olun."
        )

    from ultralytics import YOLO

    try:
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        LOGGER.debug("Torch iş parçacığı ayarlanamadı", exc_info=True)

    LOGGER.info("Model yükleniyor: %s", MODEL_PATH)
    return YOLO(str(MODEL_PATH))


def prepare_image(image: Image.Image) -> Image.Image:
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    prepared.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    return prepared


def predict_scores(image: Image.Image) -> dict[str, float]:
    model = load_model()
    prepared = prepare_image(image)

    result = model.predict(
        source=prepared,
        verbose=False,
        imgsz=224,
        device="cpu",
    )[0]

    if result.probs is None:
        raise RuntimeError("Model sınıflandırma skoru üretmedi.")

    probabilities = [float(value) for value in result.probs.data.cpu().tolist()]
    names = result.names

    if len(probabilities) != EXPECTED_CLASS_COUNT:
        raise RuntimeError(
            f"Beklenen {EXPECTED_CLASS_COUNT} sınıf yerine "
            f"{len(probabilities)} sınıf üretildi."
        )

    scores: dict[str, float] = {}
    for index, probability in enumerate(probabilities):
        english_name = names[index] if isinstance(names, dict) else names[index]
        turkish_name = TURKCE_ISIMLER.get(str(english_name), str(english_name))
        scores[turkish_name] = probability

    return scores


def confidence_metrics(
    scores: Mapping[str, float],
) -> tuple[str, float, float, str]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score

    if top_score >= 0.75 and margin >= 0.25:
        level = "Yüksek teknik güven"
    elif top_score >= 0.50 and margin >= 0.10:
        level = "Orta teknik güven"
    else:
        level = "Düşük / belirsiz teknik güven"

    return top_name, top_score, margin, level


def confidence_card(scores: Mapping[str, float]) -> str:
    top_name, top_score, margin, level = confidence_metrics(scores)

    if level.startswith("Yüksek"):
        level_class = "confidence-high"
    elif level.startswith("Orta"):
        level_class = "confidence-medium"
    else:
        level_class = "confidence-low"

    return f"""
    <section class="confidence-card {level_class}">
        <div class="eyebrow">ÖNCELİKLİ MODEL ÇIKTISI</div>
        <div class="diagnosis-name">{html.escape(top_name)}</div>
        <div class="confidence-meta">
            <span>{html.escape(level)}</span>
            <span>En yüksek skor: %{top_score * 100:.1f}</span>
            <span>İlk iki sınıf farkı: %{margin * 100:.1f}</span>
        </div>
        <p class="confidence-disclaimer">
            Bu skor klinik doğruluk veya kalibre edilmiş hastalık olasılığı değildir.
        </p>
    </section>
    """


def top_five_results(scores: Mapping[str, float]) -> str:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    rows: list[str] = []

    for rank, (name, score) in enumerate(ranked, start=1):
        width = max(2.0, min(score * 100, 100.0))
        rows.append(
            f"""
            <div class="result-row">
                <div class="result-head">
                    <div class="result-name-wrap">
                        <span class="result-rank">{rank:02d}</span>
                        <span class="result-name">{html.escape(name)}</span>
                    </div>
                    <span class="result-score">%{score * 100:.1f}</span>
                </div>
                <div class="result-track">
                    <div class="result-fill" style="width:{width:.1f}%"></div>
                </div>
            </div>
            """
        )

    return '<div class="result-list">' + "".join(rows) + "</div>"


def format_top_scores(
    scores: Mapping[str, float],
    top_n: int = 5,
) -> str:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return "\n".join(f"- {name}: %{score * 100:.1f}" for name, score in ranked)


def get_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key

    try:
        secret_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        secret_key = ""

    return secret_key or None


def generate_ai_comment(
    scores: Mapping[str, float],
    species: str,
) -> str:
    api_key = get_api_key()
    if not api_key:
        return (
            "### AI destekli değerlendirme şu anda pasif\n\n"
            "Streamlit uygulamasının **Secrets** bölümüne `GEMINI_API_KEY` "
            "eklenmediği için metin tabanlı klinik yorum üretilmemektedir."
        )

    from google import genai
    from google.genai import types

    top_name, top_score, margin, level = confidence_metrics(scores)
    prompt = f"""Hayvan türü: {species}
Teknik güven sınıfı: {level}
Öncelikli model çıktısı: {top_name}
En yüksek skor: %{top_score * 100:.1f}
İlk iki sınıf skor farkı: %{margin * 100:.1f}

İlk beş model çıktısı:
{format_top_scores(scores)}
"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=LLM_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1200,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )

    return response.text or "AI değerlendirmesi boş yanıt döndürdü."


def initialize_state() -> None:
    defaults = {
        "scores": None,
        "analysis_seconds": None,
        "ai_comment": None,
        "analysed_species": "Belirtilmedi",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


install_background()
initialize_state()


st.markdown(
    f"""
    <div class="topbar">
        <div class="brand">Veteriner Oftalmoloji × Yapay Zekâ</div>
        <div class="version-chip">{APP_VERSION}</div>
    </div>

    <header class="hero">
        <div class="hero-kicker">24 sınıflı ön segment görüntü modeli</div>
        <h1>Veteriner göz hastalıklarında <span>akıllı karar desteği.</span></h1>
        <p>
            Kedi ve köpeklerin ön segment göz fotoğraflarını analiz eder,
            ilk beş model skorunu şeffaf biçimde sunar ve isteğe bağlı
            klinik değerlendirme oluşturur.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">Kedi ve köpek</span>
            <span class="hero-badge">24 sınıf</span>
            <span class="hero-badge">Araştırma prototipi</span>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)


input_column, result_column = st.columns([1, 1], gap="large")

with input_column:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-title">
                <span class="panel-number">01 / GÖRÜNTÜ GİRİŞİ</span>
                <h2>Olgu ve fotoğraf bilgisi</h2>
                <p>Net, yakın plan ve iyi aydınlatılmış bir göz fotoğrafı yükleyin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="field-label">Görüntü kaynağı</div>', unsafe_allow_html=True)
        source_mode = st.radio(
            "Görüntü kaynağı",
            ["Dosya yükle", "Kameradan çek"],
            horizontal=True,
            label_visibility="collapsed",
        )

        uploaded_image = None
        if source_mode == "Dosya yükle":
            uploaded_image = st.file_uploader(
                "Göz fotoğrafı",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
        else:
            uploaded_image = st.camera_input(
                "Göz fotoğrafı çek",
                label_visibility="collapsed",
            )

        st.markdown('<div class="field-label">Hayvan türü</div>', unsafe_allow_html=True)
        species = st.radio(
            "Hayvan türü",
            ["Kedi", "Köpek", "Belirtilmedi"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(
                image,
                caption="Analiz edilecek görüntü",
                use_container_width=True,
            )
        else:
            image = None

        st.markdown(
            """
            <div class="helper-text">
                Gözün kadrajı mümkün olduğunca doldurması önerilir. El, cerrahi alet,
                geniş yüz kadrajı ve yoğun arka plan model skorlarını etkileyebilir.
            </div>
            """,
            unsafe_allow_html=True,
        )

        analyse_clicked = st.button(
            "Görüntüyü analiz et",
            type="primary",
            use_container_width=True,
        )

        if analyse_clicked:
            if image is None:
                st.warning("Önce bir göz fotoğrafı yükleyin veya çekin.")
            else:
                st.session_state["ai_comment"] = None
                try:
                    with st.spinner("Model hazırlanıyor ve görüntü analiz ediliyor..."):
                        started = time.perf_counter()
                        scores = predict_scores(image)
                        elapsed = time.perf_counter() - started

                    st.session_state["scores"] = scores
                    st.session_state["analysis_seconds"] = elapsed
                    st.session_state["analysed_species"] = species
                except Exception as exc:
                    LOGGER.exception("Görüntü analizi başarısız")
                    st.session_state["scores"] = None
                    st.session_state["analysis_seconds"] = None
                    st.error(f"Analiz tamamlanamadı: {exc}")


with result_column:
    with st.container(border=True):
        st.markdown(
            """
            <div class="panel-title">
                <span class="panel-number">02 / MODEL ÇIKTISI</span>
                <h2>Analiz sonuçları</h2>
                <p>Öncelikli model çıktısı ve ilk beş güven skoru burada gösterilir.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        scores = st.session_state.get("scores")
        if scores:
            st.markdown(confidence_card(scores), unsafe_allow_html=True)
            st.markdown(
                '<div class="results-heading">İlk beş model güven skoru</div>',
                unsafe_allow_html=True,
            )
            st.markdown(top_five_results(scores), unsafe_allow_html=True)

            elapsed = st.session_state.get("analysis_seconds")
            if elapsed is not None:
                st.markdown(
                    f'<div class="metric-line">Model analiz süresi: {elapsed:.2f} saniye</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <section class="confidence-card">
                    <div class="eyebrow">ANALİZ BEKLENİYOR</div>
                    <div class="diagnosis-name">Henüz görüntü analiz edilmedi</div>
                    <p class="confidence-disclaimer">
                        Bir göz fotoğrafı yükleyip analiz düğmesine basın.
                    </p>
                </section>
                <div class="empty-results">
                    Sonuçlar hastalık adı, yüzde skor ve turuncu ilerleme çubuğuyla
                    burada sıralanacaktır.
                </div>
                """,
                unsafe_allow_html=True,
            )


with st.container(border=True):
    st.markdown(
        """
        <div class="panel-title">
            <span class="panel-number">03 / KLİNİK YORUM</span>
            <h2>AI destekli klinik değerlendirme</h2>
            <p>
                Model sonucu bekletilmez. Klinik yorum, sonuç alındıktan sonra
                ayrı olarak oluşturulur.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_scores = st.session_state.get("scores")
    api_key_available = get_api_key() is not None

    if not current_scores:
        st.info("Klinik değerlendirme oluşturmak için önce bir görüntüyü analiz edin.")
    elif not api_key_available:
        st.markdown(
            """
            ### AI destekli değerlendirme şu anda pasif

            Streamlit Community Cloud'da uygulamanın **Settings → Secrets**
            alanına `GEMINI_API_KEY` eklendiğinde bu bölüm çalışacaktır.
            """
        )
    else:
        generate_clicked = st.button(
            "Klinik AI yorumunu oluştur",
            type="secondary",
            use_container_width=True,
        )

        if generate_clicked:
            try:
                with st.spinner("Klinik değerlendirme hazırlanıyor..."):
                    st.session_state["ai_comment"] = generate_ai_comment(
                        current_scores,
                        st.session_state.get("analysed_species", "Belirtilmedi"),
                    )
            except Exception as exc:
                LOGGER.exception("AI klinik değerlendirme alınamadı")
                st.error(f"Klinik değerlendirme alınamadı: {exc}")

        if st.session_state.get("ai_comment"):
            st.markdown(st.session_state["ai_comment"])
        else:
            st.markdown(
                "Model sonuçları hazır. Klinik yorum için yukarıdaki düğmeye basın."
            )


st.markdown(
    """
    <section class="developer-section">
        <h2>Geliştiriciler</h2>
        <p>Veteriner oftalmoloji ve yapay zekâ araştırma ekibi</p>
        <div class="developer-grid">
            <article class="developer-card">
                <div class="developer-role">Geliştirici</div>
                <div class="developer-name">Doç. Dr. Sıtkıcan Okur</div>
            </article>
            <article class="developer-card">
                <div class="developer-role">Geliştirici</div>
                <div class="developer-name">Vet. Hek. Büşra Baykal</div>
            </article>
            <article class="developer-card">
                <div class="developer-role">Geliştirici</div>
                <div class="developer-name">Vet. Hek. Tuğçe Kartal</div>
            </article>
        </div>
    </section>

    <footer class="site-footer">
        Bu araç klinik teşhis veya tedavi aracının yerine geçmez.
        Nihai karar hastayı muayene eden veteriner hekime aittir.
    </footer>
    """,
    unsafe_allow_html=True,
)
