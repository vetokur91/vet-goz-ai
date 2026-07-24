"""Veteriner Göz Hastalıkları AI Karar-Destek Asistanı — siyah/turuncu profesyonel arayüz."""

from __future__ import annotations

import base64
import html
import logging
import os
import threading
from pathlib import Path
from typing import Mapping

# Render Free gibi düşük kaynaklı ortamlarda CPU kullanımını sınırlar.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gradio as gr


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "v2_best.pt"
BACKGROUND_PATH = BASE_DIR / "site_background.png"
VIDEO_PATH = BASE_DIR / "site_background.mp4"
LLM_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
EXPECTED_CLASS_COUNT = 24

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("vet-goz-ai")


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


_model = None
_model_lock = threading.Lock()


def get_model():
    """YOLO sınıflandırma modelini ilk analiz isteğinde güvenli biçimde yükler."""
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {MODEL_PATH.name}. "
                "GitHub ana dizininde v2_best.pt bulunmalıdır."
            )

        # İş parçacığı sınırları model yüklenmeden önce uygulanır.
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Daha önce bir Torch işlemi başladıysa bu ayar ikinci kez yapılamayabilir.
            pass

        from ultralytics import YOLO

        LOGGER.info("Model yükleniyor: %s", MODEL_PATH)
        model = YOLO(str(MODEL_PATH), task="classify")

        try:
            model.model.eval()
        except Exception:
            LOGGER.debug("Model eval moduna alınamadı", exc_info=True)

        _model = model
        LOGGER.info("Model başarıyla yüklendi")
        return _model


def get_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def predict_scores(image) -> dict[str, float]:
    if image is None:
        raise ValueError("Analiz edilecek görüntü bulunamadı.")

    # Saydam, gri tonlu veya farklı moddaki görselleri modelin beklediği RGB biçimine çevirir.
    if hasattr(image, "convert"):
        image = image.convert("RGB")

    model = get_model()

    import torch

    with torch.inference_mode():
        result = model.predict(
            source=image,
            verbose=False,
            imgsz=224,
            device="cpu",
            save=False,
        )[0]

    if result.probs is None:
        raise RuntimeError("Model sınıflandırma olasılığı üretmedi.")

    probabilities = [
        float(value)
        for value in result.probs.data.detach().cpu().tolist()
    ]
    names = result.names

    if len(probabilities) != EXPECTED_CLASS_COUNT:
        raise RuntimeError(
            f"Beklenen {EXPECTED_CLASS_COUNT} sınıf yerine "
            f"{len(probabilities)} sınıf üretildi."
        )

    scores: dict[str, float] = {}
    for index, probability in enumerate(probabilities):
        english_name = names[index]
        turkish_name = TURKCE_ISIMLER.get(english_name, english_name)
        scores[turkish_name] = probability

    return scores

def confidence_metrics(scores: Mapping[str, float]) -> tuple[str, float, float, str]:
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

    rows = []
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
                    <div class="result-fill" style="width: {width:.1f}%"></div>
                </div>
            </div>
            """
        )

    return '<div class="result-list">' + "".join(rows) + "</div>"


def format_top_scores(scores: Mapping[str, float], top_n: int = 5) -> str:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return "\n".join(f"- {name}: %{score * 100:.1f}" for name, score in ranked)


def generate_ai_comment(scores: Mapping[str, float], species: str) -> str:
    api_key = get_api_key()
    if not api_key:
        return (
            "### AI destekli değerlendirme şu anda pasif\n\n"
            "Gemini API anahtarı tanımlanmadığı için bu bölümde metin tabanlı klinik yorum "
            "üretilmemektedir. Görüntü sınıflandırma sonuçları yukarıda gösterilmektedir."
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

    try:
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
    except Exception:
        LOGGER.exception("Gemini değerlendirmesi alınamadı")
        return (
            "### AI değerlendirmesi alınamadı\n\n"
            "Görüntü sınıflandırma sonuçları geçerlidir. Metin tabanlı değerlendirme için daha sonra tekrar deneyin."
        )


def analyse_image(image, species: str):
    """Önce yükleme durumunu, ardından model sonucunu akış halinde gösterir."""
    if image is None:
        yield (
            """
            <section class="confidence-card confidence-low">
                <div class="eyebrow">GÖRÜNTÜ GEREKLİ</div>
                <div class="diagnosis-name">Önce bir göz fotoğrafı yükleyin</div>
                <p class="confidence-disclaimer">
                    Fotoğraf yüklendikten sonra analiz düğmesine yeniden basın.
                </p>
            </section>
            """,
            """
            <div class="empty-results">
                Analiz için geçerli bir JPG veya PNG görüntüsü yükleyin.
            </div>
            """,
            {},
            "### Görüntü bekleniyor\n\nÖnce bir göz fotoğrafı yükleyin.",
        )
        return

    # Kullanıcı model yüklenirken boş bir ekran görmesin.
    yield (
        """
        <section class="confidence-card confidence-medium">
            <div class="eyebrow">MODEL ÇALIŞIYOR</div>
            <div class="diagnosis-name">Görüntü analiz ediliyor</div>
            <p class="confidence-disclaimer">
                Ücretsiz sunucuda ilk analiz model yüklenirken daha uzun sürebilir.
                Sayfayı yenilemeden bekleyin.
            </p>
        </section>
        """,
        """
        <div class="empty-results">
            Model hazırlanıyor ve görüntü işleniyor…
        </div>
        """,
        {},
        "### Analiz sürüyor\n\nÖnce görüntü modeli çalıştırılıyor.",
    )

    try:
        scores = predict_scores(image)
        ai_waiting = (
            "### Görüntü analizi tamamlandı\n\n"
            "Model sonuçları hazır. AI destekli klinik değerlendirme ayrı olarak hazırlanıyor."
            if get_api_key()
            else
            "### Görüntü analizi tamamlandı\n\n"
            "Model sonuçları hazır. Gemini API anahtarı tanımlanmadığı için "
            "metin tabanlı klinik değerlendirme pasiftir."
        )
        yield (
            confidence_card(scores),
            top_five_results(scores),
            scores,
            ai_waiting,
        )
    except Exception as exc:
        LOGGER.exception("Görüntü analizi başarısız")
        safe_error = html.escape(f"{type(exc).__name__}: {exc}")[:500]
        yield (
            f"""
            <section class="confidence-card confidence-low">
                <div class="eyebrow">TEKNİK HATA</div>
                <div class="diagnosis-name">Analiz tamamlanamadı</div>
                <p class="confidence-disclaimer">
                    Sunucu mesajı: {safe_error}
                </p>
            </section>
            """,
            """
            <div class="empty-results">
                Model sonuç üretmedi. Aşağıdaki hata mesajı Render günlüklerinde de görülebilir.
            </div>
            """,
            {},
            f"### Analiz hatası\n\n`{type(exc).__name__}: {str(exc)[:400]}`",
        )


def generate_ai_after_model(scores: Mapping[str, float], species: str) -> str:
    """Gemini çağrısını model sonucundan ayırır; böylece LLM gecikse bile skorlar görünür."""
    if not scores:
        return "### Klinik değerlendirme üretilemedi\n\nÖnce görüntü modeli başarılı biçimde çalışmalıdır."
    return generate_ai_comment(scores, species)

def background_data_uri() -> str:
    if not BACKGROUND_PATH.is_file():
        LOGGER.warning("Arka plan görseli bulunamadı: %s", BACKGROUND_PATH)
        return ""

    encoded = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


BACKGROUND_DATA_URI = background_data_uri()
VIDEO_URL = f"/gradio_api/file={VIDEO_PATH}"
ALLOWED_PATHS = [str(VIDEO_PATH)] if VIDEO_PATH.is_file() else []


CUSTOM_CSS = f"""
:root {{
    --bg: #050505;
    --panel: rgba(13, 13, 13, 0.95);
    --panel-soft: rgba(20, 20, 20, 0.93);
    --line: rgba(255, 157, 46, 0.24);
    --line-strong: rgba(255, 157, 46, 0.60);
    --orange: #ff9d2e;
    --orange-light: #ffc16b;
    --cream: #f6efe7;
    --muted: #b7aea5;
    --muted-dark: #7f7871;
}}

html,
body {{
    min-height: 100%;
    margin: 0;
    background: var(--bg) !important;
    color: var(--cream) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}}

body {{
    overflow-x: hidden;
}}

.background-layer {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background: #050505;
}}

.background-eye {{
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
    background-image: url("{BACKGROUND_DATA_URI}");
    background-size: cover;
    background-position: center right;
    background-repeat: no-repeat;
    overflow: hidden;
    opacity: 0.96;
    filter: brightness(.94) contrast(1.08) saturate(1.12);
    -webkit-mask-image: linear-gradient(90deg, rgba(0,0,0,.18) 0%, rgba(0,0,0,.42) 26%, rgba(0,0,0,.82) 48%, black 66%);
    mask-image: linear-gradient(90deg, rgba(0,0,0,.18) 0%, rgba(0,0,0,.42) 26%, rgba(0,0,0,.82) 48%, black 66%);
}}

.background-eye video {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center right;
}}

.background-vignette {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at 80% 38%, rgba(255,157,46,.07), transparent 38%),
        linear-gradient(90deg, rgba(5,5,5,.98) 0%, rgba(5,5,5,.93) 27%, rgba(5,5,5,.67) 47%, rgba(5,5,5,.20) 69%, rgba(5,5,5,.04) 100%),
        linear-gradient(180deg, rgba(5,5,5,.08) 0%, rgba(5,5,5,.18) 62%, rgba(5,5,5,.38) 100%);
}}

.gradio-container {{
    position: relative;
    z-index: 1;
    max-width: none !important;
    min-height: 100vh;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    color: var(--cream) !important;
}}

#app-shell {{
    width: min(1320px, calc(100% - 48px));
    margin: 0 auto;
    padding: 56px 0 48px;
    gap: 0 !important;
}}

#hero {{
    max-width: 790px;
    padding: 8px 0 42px;
}}

.hero-kicker {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
    color: var(--orange);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
}}

.hero-kicker::before {{
    content: "";
    width: 34px;
    height: 2px;
    background: var(--orange);
}}

#hero h1 {{
    margin: 0;
    color: var(--cream) !important;
    font-size: clamp(42px, 5vw, 74px);
    line-height: 1.02;
    font-weight: 760;
    letter-spacing: -0.045em;
    text-shadow: 0 12px 36px rgba(0,0,0,.68);
}}

#hero h1 span {{
    color: var(--orange) !important;
}}

#hero p {{
    max-width: 700px;
    margin: 24px 0 0;
    color: var(--muted) !important;
    font-size: 17px;
    line-height: 1.72;
}}

.hero-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 26px;
}}

.hero-badge {{
    display: inline-flex;
    align-items: center;
    min-height: 34px;
    padding: 0 13px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(12,12,12,.78);
    color: var(--orange-light) !important;
    font-size: 12px;
    font-weight: 700;
}}

.main-grid {{
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 24px !important;
    align-items: stretch !important;
    margin: 0 !important;
}}

.panel {{
    height: 100%;
    min-width: 0;
    padding: 30px !important;
    border: 1px solid var(--line) !important;
    border-radius: 22px !important;
    background: var(--panel) !important;
    box-shadow: 0 24px 70px rgba(0,0,0,.42) !important;
    backdrop-filter: blur(14px);
}}

.panel-title {{
    margin-bottom: 22px;
}}

.panel-title .panel-number {{
    display: block;
    margin-bottom: 7px;
    color: var(--orange) !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .16em;
}}

.panel-title h2 {{
    margin: 0 !important;
    color: var(--orange) !important;
    font-size: 25px !important;
    line-height: 1.2 !important;
    font-weight: 750 !important;
}}

.panel-title p {{
    margin: 8px 0 0 !important;
    color: var(--muted) !important;
    font-size: 14px !important;
    line-height: 1.55 !important;
}}

.field-label {{
    margin: 19px 0 9px;
    color: var(--orange) !important;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: .02em;
}}

#eye-upload {{
    overflow: hidden;
    border: 1px dashed rgba(255,157,46,.42) !important;
    border-radius: 16px !important;
    background: #0a0a0a !important;
}}

#eye-upload > div,
#eye-upload .block,
#eye-upload .wrap,
#eye-upload .image-container {{
    background: #0a0a0a !important;
    color: var(--cream) !important;
}}

#eye-upload button,
#eye-upload span,
#eye-upload p {{
    color: var(--muted) !important;
}}

#species-radio {{
    padding: 4px 0 0;
}}

#species-radio > div {{
    gap: 10px !important;
}}

#species-radio label {{
    min-height: 42px;
    padding: 10px 14px !important;
    border: 1px solid rgba(255,157,46,.22) !important;
    border-radius: 12px !important;
    background: #0b0b0b !important;
    color: var(--orange-light) !important;
}}

#species-radio label span {{
    color: var(--orange-light) !important;
    font-weight: 700 !important;
}}

#species-radio label:has(input:checked) {{
    border-color: var(--orange) !important;
    background: rgba(255,157,46,.12) !important;
    box-shadow: inset 0 0 0 1px rgba(255,157,46,.16);
}}

#species-radio input {{
    accent-color: var(--orange) !important;
}}

.helper-text,
.helper-text * {{
    margin-top: 14px !important;
    color: var(--muted-dark) !important;
    font-size: 13px !important;
    line-height: 1.58 !important;
}}

#analyse-button {{
    margin-top: 20px !important;
}}

#analyse-button button {{
    width: 100%;
    min-height: 52px;
    border: 1px solid var(--orange) !important;
    border-radius: 13px !important;
    background: var(--orange) !important;
    color: #0a0a0a !important;
    font-size: 15px !important;
    font-weight: 850 !important;
    box-shadow: 0 12px 32px rgba(255,157,46,.16);
    transition: transform .16s ease, filter .16s ease;
}}

#analyse-button button:hover {{
    transform: translateY(-1px);
    filter: brightness(1.06);
}}

.confidence-card {{
    padding: 23px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: #0a0a0a;
}}

.confidence-card.confidence-high {{ border-left: 4px solid #78c58f; }}
.confidence-card.confidence-medium {{ border-left: 4px solid var(--orange); }}
.confidence-card.confidence-low {{ border-left: 4px solid #c76d58; }}

.eyebrow {{
    color: var(--orange) !important;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: .16em;
}}

.diagnosis-name {{
    margin-top: 9px;
    color: var(--cream) !important;
    font-size: 28px;
    font-weight: 760;
    line-height: 1.18;
}}

.confidence-meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 9px 16px;
    margin-top: 15px;
    color: var(--orange-light) !important;
    font-size: 12px;
    font-weight: 700;
}}

.confidence-disclaimer {{
    margin: 13px 0 0;
    color: var(--muted-dark) !important;
    font-size: 12px;
    line-height: 1.55;
}}

.results-heading {{
    margin: 24px 0 13px;
    color: var(--orange) !important;
    font-size: 13px;
    font-weight: 850;
    letter-spacing: .04em;
}}

.result-list {{
    display: grid;
    gap: 14px;
}}

.result-row {{
    padding: 14px 15px;
    border: 1px solid rgba(255,157,46,.14);
    border-radius: 13px;
    background: #0a0a0a;
}}

.result-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
}}

.result-name-wrap {{
    display: flex;
    align-items: center;
    min-width: 0;
    gap: 11px;
}}

.result-rank {{
    flex: 0 0 auto;
    color: var(--orange) !important;
    font-size: 11px;
    font-weight: 850;
}}

.result-name {{
    overflow: hidden;
    color: var(--cream) !important;
    font-size: 14px;
    font-weight: 700;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

.result-score {{
    flex: 0 0 auto;
    color: var(--orange-light) !important;
    font-size: 13px;
    font-weight: 850;
}}

.result-track {{
    height: 6px;
    margin-top: 10px;
    overflow: hidden;
    border-radius: 999px;
    background: #24201c;
}}

.result-fill {{
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #d96f16, var(--orange), var(--orange-light));
}}

.empty-results {{
    min-height: 212px;
    display: grid;
    place-items: center;
    padding: 24px;
    border: 1px dashed rgba(255,157,46,.20);
    border-radius: 16px;
    background: #0a0a0a;
    color: var(--muted-dark) !important;
    font-size: 14px;
    text-align: center;
}}

.full-panel {{
    margin-top: 24px !important;
}}

#ai-output {{
    min-height: 130px;
    padding: 22px 24px !important;
    border: 1px solid rgba(255,157,46,.14) !important;
    border-radius: 16px !important;
    background: #0a0a0a !important;
}}

#ai-output h1,
#ai-output h2,
#ai-output h3 {{
    color: var(--orange) !important;
}}

#ai-output p,
#ai-output li,
#ai-output strong {{
    color: var(--cream) !important;
    line-height: 1.68 !important;
}}

.developer-section {{
    margin-top: 24px;
    padding: 28px 30px;
    border: 1px solid var(--line);
    border-radius: 22px;
    background: rgba(10,10,10,.93);
}}

.developer-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    padding-bottom: 18px;
    border-bottom: 1px solid rgba(255,157,46,.14);
}}

.developer-header h2 {{
    margin: 0;
    color: var(--orange) !important;
    font-size: 22px;
}}

.developer-header p {{
    margin: 0;
    color: var(--muted-dark) !important;
    font-size: 13px;
}}

.developer-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 18px;
}}

.developer-card {{
    padding: 18px;
    border: 1px solid rgba(255,157,46,.16);
    border-radius: 14px;
    background: #0d0d0d;
}}

.developer-role {{
    color: var(--orange) !important;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: .14em;
}}

.developer-name {{
    margin-top: 8px;
    color: var(--cream) !important;
    font-size: 16px;
    font-weight: 760;
}}

.site-footer {{
    padding: 26px 4px 0;
    color: var(--muted-dark) !important;
    font-size: 12px;
    line-height: 1.65;
    text-align: center;
}}

@media (max-width: 980px) {{
    #app-shell {{
        width: min(100% - 28px, 760px);
        padding-top: 34px;
    }}

    .main-grid {{
        grid-template-columns: 1fr;
    }}

    .background-eye {{
        width: 100vw;
        opacity: .72;
        background-position: 72% center;
        filter: brightness(.88) contrast(1.06) saturate(1.08);
        -webkit-mask-image: linear-gradient(90deg, rgba(0,0,0,.24) 0%, rgba(0,0,0,.55) 42%, black 72%);
        mask-image: linear-gradient(90deg, rgba(0,0,0,.24) 0%, rgba(0,0,0,.55) 42%, black 72%);
    }}

    .background-eye video {{
        object-position: 72% center;
    }}

    .developer-grid {{
        grid-template-columns: 1fr;
    }}
}}

@media (max-width: 640px) {{
    #app-shell {{
        width: calc(100% - 20px);
        padding: 24px 0 30px;
    }}

    #hero {{
        padding-bottom: 28px;
    }}

    #hero h1 {{
        font-size: 38px;
    }}

    #hero p {{
        font-size: 15px;
    }}

    .panel {{
        padding: 20px !important;
        border-radius: 17px !important;
    }}

    .developer-section {{
        padding: 22px 20px;
    }}

    .developer-header {{
        align-items: flex-start;
        flex-direction: column;
        gap: 7px;
    }}

    .confidence-meta {{
        flex-direction: column;
        gap: 5px;
    }}
}}
"""


with gr.Blocks(title="Veteriner Göz Hastalıkları — AI Asistanı") as demo:
    gr.HTML(
        f"""
        <div class="background-layer"></div>
        <div class="background-eye" aria-hidden="true">
            <video autoplay muted loop playsinline preload="metadata" poster="{BACKGROUND_DATA_URI}">
                <source src="{VIDEO_URL}" type="video/mp4">
            </video>
        </div>
        <div class="background-vignette"></div>
        """
    )

    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            """
            <header id="hero">
                <div class="hero-kicker">Veteriner oftalmoloji ve yapay zekâ</div>
                <h1>
                    Veteriner Göz Hastalıkları<br>
                    <span>AI Karar-Destek Asistanı</span>
                </h1>
                <p>
                    Kedi ve köpeklerin ön segment göz fotoğraflarını 24 sınıfta analiz eden,
                    model güven skorlarını şeffaf biçimde sunan araştırma prototipi.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge">24 sınıflı model</span>
                    <span class="hero-badge">Kedi ve köpek</span>
                    <span class="hero-badge">Prototip v5.1</span>
                </div>
            </header>
            """
        )

        with gr.Row(elem_classes="main-grid"):
            with gr.Column(elem_classes="panel"):
                gr.HTML(
                    """
                    <div class="panel-title">
                        <span class="panel-number">01 / GÖRÜNTÜ GİRİŞİ</span>
                        <h2>Olgu ve fotoğraf bilgisi</h2>
                        <p>Analiz için net, yakın plan ve iyi aydınlatılmış bir göz fotoğrafı yükleyin.</p>
                    </div>
                    <div class="field-label">Göz fotoğrafı</div>
                    """
                )

                image_input = gr.Image(
                    type="pil",
                    label=None,
                    show_label=False,
                    height=390,
                    sources=["upload", "webcam", "clipboard"],
                    elem_id="eye-upload",
                )

                gr.HTML('<div class="field-label">Hayvan türü</div>')

                species_input = gr.Radio(
                    choices=["Kedi", "Köpek", "Belirtilmedi"],
                    value="Belirtilmedi",
                    label=None,
                    show_label=False,
                    elem_id="species-radio",
                )

                gr.Markdown(
                    "Gözü kadrajı mümkün olduğunca dolduracak şekilde yükleyin. El, alet, geniş yüz "
                    "kadrajı ve yoğun arka plan model skorlarını etkileyebilir.",
                    elem_classes="helper-text",
                )

                analyse_button = gr.Button(
                    "Görüntüyü analiz et",
                    variant="primary",
                    size="lg",
                    elem_id="analyse-button",
                )

            with gr.Column(elem_classes="panel"):
                gr.HTML(
                    """
                    <div class="panel-title">
                        <span class="panel-number">02 / MODEL ÇIKTISI</span>
                        <h2>Analiz sonuçları</h2>
                        <p>Öncelikli model çıktısı ve ilk beş sınıf güven skoru burada gösterilir.</p>
                    </div>
                    """
                )

                confidence_output = gr.HTML(
                    """
                    <section class="confidence-card">
                        <div class="eyebrow">ANALİZ BEKLENİYOR</div>
                        <div class="diagnosis-name">Henüz görüntü analiz edilmedi</div>
                        <p class="confidence-disclaimer">
                            Bir göz fotoğrafı yükleyip analiz düğmesine basın.
                        </p>
                    </section>
                    """
                )

                gr.HTML('<div class="results-heading">İLK BEŞ MODEL GÜVEN SKORU</div>')

                results_output = gr.HTML(
                    """
                    <div class="empty-results">
                        Sonuçlar; hastalık adı, yüzde skor ve görsel ilerleme çubuğuyla burada sıralanacaktır.
                    </div>
                    """
                )

        with gr.Column(elem_classes="panel full-panel"):
            gr.HTML(
                """
                <div class="panel-title">
                    <span class="panel-number">03 / KLİNİK YORUM</span>
                    <h2>AI destekli klinik değerlendirme</h2>
                    <p>Muayene ve tetkik yaklaşımı ile genel tedavi yönlendirmesi bu bölümde sunulur.</p>
                </div>
                """
            )

            scores_state = gr.State({})

            ai_output = gr.Markdown(
                "Analiz tamamlandığında değerlendirme bu alanda gösterilecektir.",
                elem_id="ai-output",
            )

        gr.HTML(
            """
            <section class="developer-section">
                <div class="developer-header">
                    <h2>Geliştiriciler</h2>
                    <p>Veteriner oftalmoloji ve yapay zekâ araştırma ekibi</p>
                </div>
                <div class="developer-grid">
                    <article class="developer-card">
                        <div class="developer-role">GELİŞTİRİCİ</div>
                        <div class="developer-name">Doç. Dr. Sıtkıcan Okur</div>
                    </article>
                    <article class="developer-card">
                        <div class="developer-role">GELİŞTİRİCİ</div>
                        <div class="developer-name">Vet. Hek. Büşra Baykal</div>
                    </article>
                    <article class="developer-card">
                        <div class="developer-role">GELİŞTİRİCİ</div>
                        <div class="developer-name">Vet. Hek. Tuğçe Kartal</div>
                    </article>
                </div>
            </section>

            <footer class="site-footer">
                Bu araç klinik teşhis veya tedavi aracının yerine geçmez.
                Nihai karar hastayı muayene eden veteriner hekime aittir.
            </footer>
            """
        )

    analysis_event = analyse_button.click(
        fn=analyse_image,
        inputs=[image_input, species_input],
        outputs=[confidence_output, results_output, scores_state, ai_output],
        api_name="analiz_et",
        show_progress="full",
    )

    analysis_event.then(
        fn=generate_ai_after_model,
        inputs=[scores_state, species_input],
        outputs=ai_output,
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=10).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=False,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
        allowed_paths=ALLOWED_PATHS,
    )
