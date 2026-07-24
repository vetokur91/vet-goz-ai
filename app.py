"""Veteriner Göz Hastalıkları AI Asistanı — Render/Gradio production app."""

from __future__ import annotations

import base64
import html
import logging
import os
from pathlib import Path
from typing import Mapping

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "v2_best.pt"
BACKGROUND_PATH = BASE_DIR / "site_background.png"
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


SYSTEM_PROMPT = """
Sen veteriner hekimlere yönelik bir veteriner oftalmoloji
karar-destek asistanısın.

Sana bir görüntü sınıflandırma modelinin kedi veya köpek ön segment
göz fotoğrafı için ürettiği ilk beş sınıf ve model güven skorları
verilecek.

Bu skorlar kalibre edilmiş hastalık olasılıkları değildir.
Görüntüyü doğrudan görmediğin için kesin tanı koyma ve hastaya özgü
reçete oluşturma.

Türkçe, kısa ve klinik olarak temkinli bir yanıt üret.

Aşağıdaki başlıkları kullan:

### Klinik Yorum

En yüksek skorlu sınıfı yalnızca "öncelikli model çıktısı" olarak belirt.
Skorlar birbirine yakınsa veya teknik güven düşükse bunu açıkça vurgula
ve ayırıcı tanı yaklaşımı kullan.

### Tanıyı Kesinleştirmek İçin Muayene ve Tetkikler

Hastalığa uygun somut muayene ve tetkikleri öncelik sırasıyla yaz.
Alakasız testleri sıralama.

### Tanı Doğrulanırsa Genel Tedavi Yaklaşımı

İlaç grubu ve cerrahi yaklaşım düzeyinde bilgi ver.
Marka, doz, uygulama sıklığı veya hastaya özgü reçete yazma.

Korneal ülser dışlanmadan topikal kortikosteroid önerilmemesi gibi
kritik kontrendikasyonları belirt.

Yanıtın başında tek cümlelik prototip uyarısı bulunmalı.
Yanıt 350 kelimeyi geçmemelidir.
"""


_model = None


def get_model():
    """YOLO modelini ilk analiz isteğinde yükler."""

    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {MODEL_PATH}"
            )

        from ultralytics import YOLO

        LOGGER.info("Model yükleniyor: %s", MODEL_PATH)
        _model = YOLO(str(MODEL_PATH))

    return _model


def get_api_key() -> str | None:
    """Render ortamındaki Gemini API anahtarını döndürür."""

    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def predict_scores(image) -> dict[str, float]:
    """Yüklenen görüntü için 24 sınıflı model skorlarını üretir."""

    model = get_model()

    result = model.predict(
        source=image,
        verbose=False,
        imgsz=224,
        device="cpu",
    )[0]

    probabilities = [
        float(value)
        for value in result.probs.data.tolist()
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
        turkish_name = TURKCE_ISIMLER.get(
            english_name,
            english_name,
        )
        scores[turkish_name] = probability

    return scores


def confidence_metrics(
    scores: Mapping[str, float],
) -> tuple[str, float, float, str]:
    """En yüksek skor, ilk iki sınıf farkı ve güven düzeyini hesaplar."""

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_name, top_score = ranked[0]

    second_score = (
        ranked[1][1]
        if len(ranked) > 1
        else 0.0
    )

    margin = top_score - second_score

    if top_score >= 0.75 and margin >= 0.25:
        level = "Yüksek teknik güven"

    elif top_score >= 0.50 and margin >= 0.10:
        level = "Orta teknik güven"

    else:
        level = "Düşük / belirsiz teknik güven"

    return top_name, top_score, margin, level


def confidence_card(
    scores: Mapping[str, float],
) -> str:
    """Öncelikli model çıktısını HTML kartı olarak hazırlar."""

    top_name, top_score, margin, level = confidence_metrics(scores)

    if level.startswith("Yüksek"):
        css_class = "status-high"

    elif level.startswith("Orta"):
        css_class = "status-medium"

    else:
        css_class = "status-low"

    return f"""
    <div class="status-card {css_class}">
        <div class="status-kicker">
            ÖNCELİKLİ MODEL ÇIKTISI
        </div>

        <div class="status-title">
            {html.escape(top_name)}
        </div>

        <div class="status-row">
            <span>{html.escape(level)}</span>
            <span>En yüksek skor: %{top_score * 100:.1f}</span>
            <span>İlk iki sınıf farkı: %{margin * 100:.1f}</span>
        </div>

        <div class="status-note">
            Bu skor klinik doğruluk veya kalibre edilmiş
            hastalık olasılığı değildir.
        </div>
    </div>
    """


def format_top_scores(
    scores: Mapping[str, float],
    top_n: int = 5,
) -> str:
    """İlk beş model çıktısını Gemini için metne dönüştürür."""

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_n]

    return "\n".join(
        f"- {name}: %{score * 100:.1f}"
        for name, score in ranked
    )


def generate_ai_comment(
    scores: Mapping[str, float],
    species: str,
) -> str:
    """Gemini API aktifse klinik değerlendirme oluşturur."""

    api_key = get_api_key()

    if not api_key:
        return (
            "⚠️ **AI destekli klinik değerlendirme şu anda aktif değildir.** "
            "Görüntü sınıflandırma sonuçları yukarıda gösterilmektedir. "
            "Bu bölüm, Render ortamına güvenli şekilde Gemini API anahtarı "
            "eklendiğinde çalışacaktır."
        )

    from google import genai
    from google.genai import types

    top_name, top_score, margin, level = confidence_metrics(scores)

    prompt = f"""
Hayvan türü: {species}

Teknik güven sınıfı:
{level}

Öncelikli model çıktısı:
{top_name}

En yüksek skor:
%{top_score * 100:.1f}

İlk iki sınıf skor farkı:
%{margin * 100:.1f}

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
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0
                ),
            ),
        )

        return (
            response.text
            or "⚠️ AI değerlendirmesi boş yanıt döndürdü."
        )

    except Exception:
        LOGGER.exception(
            "Gemini değerlendirmesi alınamadı"
        )

        return (
            "⚠️ **AI değerlendirmesi şu anda alınamadı.** "
            "Görüntü sınıflandırma sonuçları geçerlidir. "
            "Metin tabanlı değerlendirme için daha sonra "
            "yeniden deneyebilirsiniz."
        )


def analyse_image(image, species: str):
    """Görüntü analizini gerçekleştirir."""

    if image is None:
        return (
            {},
            """
            <div class="status-card status-low">
                <div class="status-title">
                    Önce bir göz fotoğrafı yükleyin.
                </div>
            </div>
            """,
            "",
        )

    try:
        scores = predict_scores(image)

        return (
            scores,
            confidence_card(scores),
            generate_ai_comment(scores, species),
        )

    except Exception:
        LOGGER.exception(
            "Görüntü analizi başarısız"
        )

        return (
            {},
            """
            <div class="status-card status-low">
                <div class="status-title">
                    Analiz tamamlanamadı.
                </div>

                <div class="status-note">
                    Dosyanın geçerli bir JPG veya PNG görüntüsü
                    olduğundan emin olun.
                </div>
            </div>
            """,
            (
                "⚠️ Teknik bir hata oluştu. "
                "Lütfen başka bir görüntüyle yeniden deneyin."
            ),
        )


def background_data_uri() -> str:
    """Arka plan görselini CSS içinde kullanılabilir hale getirir."""

    if not BACKGROUND_PATH.is_file():
        LOGGER.warning(
            "Arka plan görseli bulunamadı: %s",
            BACKGROUND_PATH,
        )
        return ""

    encoded = base64.b64encode(
        BACKGROUND_PATH.read_bytes()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


BACKGROUND_DATA_URI = background_data_uri()


CUSTOM_CSS = f"""
:root {{
    color-scheme: dark;
}}

html,
body {{
    min-height: 100%;
    background: #030711 !important;
}}

body {{
    overflow-x: hidden;
}}

.background-image {{
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    pointer-events: none;

    background-image: url("{BACKGROUND_DATA_URI}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;

    filter:
        saturate(.78)
        brightness(.58)
        contrast(1.08);
}}

.background-overlay {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;

    background:
        linear-gradient(
            90deg,
            rgba(2, 6, 15, .99) 0%,
            rgba(3, 8, 18, .96) 32%,
            rgba(4, 11, 23, .80) 55%,
            rgba(4, 10, 20, .53) 76%,
            rgba(3, 8, 18, .64) 100%
        ),
        linear-gradient(
            180deg,
            rgba(3, 8, 18, .52) 0%,
            rgba(3, 8, 18, .24) 45%,
            rgba(3, 8, 18, .90) 100%
        );
}}

.gradio-container {{
    background: transparent !important;
    min-height: 100vh;
    position: relative;
    z-index: 1;
}}

#app-shell {{
    max-width: 1120px;
    margin: 0 auto;
    padding: 28px 18px 38px;
}}

#hero {{
    padding: 34px 4px 24px;
    max-width: 720px;
}}

#hero h1 {{
    color: #ffffff !important;
    font-size: clamp(30px, 4.2vw, 49px);
    line-height: 1.07;
    margin: 0;
    letter-spacing: -1.2px;
    text-shadow:
        0 3px 8px rgba(0, 0, 0, .85),
        0 8px 34px rgba(0, 0, 0, .70);
}}

#hero p {{
    color: #e1eef1 !important;
    max-width: 650px;
    font-size: 16px;
    line-height: 1.68;
    margin: 15px 0 0;
    text-shadow:
        0 2px 7px rgba(0, 0, 0, .95);
}}

.hero-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
}}

.hero-badge {{
    border:
        1px solid rgba(141, 244, 229, .48);

    background:
        rgba(4, 24, 37, .88);

    backdrop-filter: blur(12px);

    color: #dcfffa !important;

    border-radius: 999px;
    padding: 7px 12px;
    font-size: 12px;
    font-weight: 700;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, .28);
}}

.glass-card {{
    background:
        rgba(4, 13, 26, .94) !important;

    backdrop-filter:
        blur(20px)
        saturate(1.05);

    border:
        1px solid rgba(119, 226, 217, .32) !important;

    border-radius: 18px !important;
    padding: 20px !important;

    box-shadow:
        0 24px 80px rgba(0, 0, 0, .55) !important;
}}

.glass-card *,
.glass-card label span {{
    color: #f4fcfd !important;
}}

.glass-card .block,
.glass-card .wrap,
.glass-card .styler {{
    background:
        rgba(3, 10, 21, .78) !important;
}}

.section-heading h3 {{
    color: #92f5e5 !important;
    margin-top: 0 !important;
    font-weight: 760 !important;
}}

.helper,
.helper * {{
    color: #d2e2e6 !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
}}

.primary-action button {{
    background:
        linear-gradient(
            115deg,
            #0f9f92,
            #087fb6
        ) !important;

    border: 0 !important;
    color: #ffffff !important;
    font-weight: 760 !important;
    min-height: 48px;

    box-shadow:
        0 10px 30px rgba(8, 145, 178, .32);
}}

.primary-action button:hover {{
    filter: brightness(1.10);
}}

.status-card {{
    border-radius: 15px;
    padding: 17px 18px;
    border: 1px solid;
    margin: 4px 0 14px;

    background:
        rgba(2, 9, 20, .90);
}}

.status-high {{
    border-color:
        rgba(52, 211, 153, .58);
}}

.status-medium {{
    border-color:
        rgba(250, 204, 21, .58);
}}

.status-low {{
    border-color:
        rgba(251, 113, 133, .62);
}}

.status-kicker {{
    color: #92ddff !important;
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 800;
}}

.status-title {{
    color: #ffffff !important;
    font-size: 23px;
    font-weight: 780;
    margin-top: 5px;
}}

.status-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;

    color: #dfedf0 !important;
    font-size: 12px;
    margin-top: 9px;
}}

.status-note {{
    color: #c7d7db !important;
    font-size: 11px;
    margin-top: 9px;
}}

#footer {{
    color: #d9e8eb !important;
    text-align: center;
    font-size: 12px;
    padding: 24px 0 7px;
    line-height: 1.75;

    text-shadow:
        0 2px 8px rgba(0, 0, 0, .95);
}}

#footer strong {{
    color: #a8fff0 !important;
}}

.developer-box {{
    display: inline-block;

    margin:
        14px auto 13px;

    padding:
        14px 24px;

    border:
        1px solid rgba(141, 244, 229, .32);

    border-radius: 14px;

    background:
        rgba(3, 13, 25, .82);

    box-shadow:
        0 12px 34px rgba(0, 0, 0, .36);
}}

.developer-title {{
    color: #91f5e4 !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.2px;
    margin-bottom: 7px;
}}

.developer-name {{
    color: #ffffff !important;
    font-size: 13px;
    font-weight: 650;
    line-height: 1.8;
}}

@media (max-width: 760px) {{

    .background-image {{
        background-position: 66% center;

        filter:
            saturate(.74)
            brightness(.50)
            contrast(1.08);
    }}

    .background-overlay {{
        background:
            linear-gradient(
                90deg,
                rgba(3, 8, 18, .98) 0%,
                rgba(3, 8, 18, .92) 60%,
                rgba(3, 8, 18, .72) 100%
            ),
            linear-gradient(
                180deg,
                rgba(3, 8, 18, .32) 0%,
                rgba(3, 8, 18, .62) 54%,
                rgba(3, 8, 18, .96) 100%
            );
    }}

    #app-shell {{
        padding: 12px 10px 26px;
    }}

    #hero {{
        padding: 24px 4px 20px;
    }}

    .glass-card {{
        padding: 14px !important;
        border-radius: 14px !important;
    }}

    .developer-box {{
        padding: 12px 17px;
    }}
}}
"""


with gr.Blocks(
    title="Veteriner Göz Hastalıkları — AI Asistanı"
) as demo:

    gr.HTML(
        """
        <div class="background-image"></div>
        <div class="background-overlay"></div>
        """
    )

    with gr.Column(elem_id="app-shell"):

        gr.HTML(
            """
            <section id="hero">

                <div class="hero-badges">
                    <span class="hero-badge">
                        24 sınıflı görüntü modeli
                    </span>

                    <span class="hero-badge">
                        Kedi & köpek
                    </span>

                    <span class="hero-badge">
                        Araştırma prototipi v2.1
                    </span>
                </div>

                <h1>
                    Veteriner Göz Hastalıkları
                    <br>
                    AI Karar-Destek Asistanı
                </h1>

                <p>
                    Ön segment göz fotoğrafını sınıflandırır,
                    ilk beş model çıktısını gösterir ve
                    veteriner hekim için tanıyı kesinleştirme
                    yaklaşımı sunar.
                </p>

            </section>
            """
        )

        with gr.Group(elem_classes="glass-card"):

            with gr.Row(equal_height=True):

                with gr.Column(scale=6):

                    gr.Markdown(
                        "### Görüntü ve olgu bilgisi",
                        elem_classes="section-heading",
                    )

                    image_input = gr.Image(
                        type="pil",
                        label="Göz fotoğrafı",
                        height=330,
                        sources=[
                            "upload",
                            "webcam",
                            "clipboard",
                        ],
                    )

                    species_input = gr.Radio(
                        choices=[
                            "Kedi",
                            "Köpek",
                            "Belirtilmedi",
                        ],
                        value="Belirtilmedi",
                        label="Hayvan türü",
                    )

                    gr.Markdown(
                        (
                            "Gözü yakın planda, net, iyi aydınlatılmış "
                            "ve mümkün olduğunca arka plansız yükleyin. "
                            "Görüntüdeki el, cerrahi alet veya geniş yüz "
                            "kadrajı model skorlarını olumsuz etkileyebilir."
                        ),
                        elem_classes="helper",
                    )

                    analyse_button = gr.Button(
                        "Görüntüyü analiz et",
                        variant="primary",
                        size="lg",
                        elem_classes="primary-action",
                    )

                with gr.Column(scale=5):

                    gr.Markdown(
                        "### Model sonuçları",
                        elem_classes="section-heading",
                    )

                    confidence_output = gr.HTML(
                        """
                        <div class="status-card">
                            <div class="status-title">
                                Analiz bekleniyor
                            </div>

                            <div class="status-note">
                                Bir göz fotoğrafı yükleyerek başlayın.
                            </div>
                        </div>
                        """
                    )

                    labels_output = gr.Label(
                        num_top_classes=5,
                        label="İlk beş model güven skoru",
                    )

                    gr.Markdown(
                        (
                            "Model skoru tek başına tanı değildir. "
                            "Düşük skorlu veya birbirine yakın sonuçlar, "
                            "görüntünün yeniden çekilmesini ve ayrıntılı "
                            "oftalmik muayeneyi gerektirir."
                        ),
                        elem_classes="helper",
                    )

        with gr.Group(elem_classes="glass-card"):

            gr.Markdown(
                "### AI destekli klinik değerlendirme",
                elem_classes="section-heading",
            )

            ai_output = gr.Markdown(
                (
                    "Analiz sonrasında muayene ve tetkik önerileri "
                    "ile genel tedavi yaklaşımı burada gösterilecektir."
                )
            )

        gr.HTML(
            """
            <footer id="footer">

                <strong>
                    Veteriner Göz Hastalıkları AI Asistanı
                    — Prototip v2.1
                </strong>

                <br>

                <div class="developer-box">

                    <div class="developer-title">
                        GELİŞTİRİCİLER
                    </div>

                    <div class="developer-name">
                        Doç. Dr. Sıtkıcan Okur
                    </div>

                    <div class="developer-name">
                        Vet. Hek. Büşra Baykal
                    </div>

                    <div class="developer-name">
                        Vet. Hek. Tuğçe Kartal
                    </div>

                </div>

                <br>

                Bu araç klinik teşhis veya tedavi aracının
                yerine geçmez.

                <br>

                Nihai karar, hastayı muayene eden
                veteriner hekime aittir.

            </footer>
            """
        )

    analyse_button.click(
        fn=analyse_image,
        inputs=[
            image_input,
            species_input,
        ],
        outputs=[
            labels_output,
            confidence_output,
            ai_output,
        ],
        api_name="analiz_et",
    )


if __name__ == "__main__":

    demo.queue(
        default_concurrency_limit=1
    ).launch(
        server_name="0.0.0.0",
        server_port=int(
            os.getenv("PORT", "7860")
        ),
        show_error=False,
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="cyan",
        ),
        css=CUSTOM_CSS,
    )
