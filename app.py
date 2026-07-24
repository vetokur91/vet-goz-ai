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

SYSTEM_PROMPT = """Sen veteriner hekimlere yönelik bir veteriner oftalmoloji karar-destek asistanısın.
Sana bir görüntü sınıflandırma modelinin kedi veya köpek ön segment göz fotoğrafı için ürettiği
ilk beş sınıf ve model güven skorları verilecek. Bunlar kalibre edilmiş hastalık olasılıkları değildir.
Görüntüyü doğrudan görmediğin için kesin tanı koyma ve hastaya özgü reçete oluşturma.

Türkçe, kısa ve klinik olarak temkinli bir yanıt üret. Aşağıdaki başlıkları kullan:

### Klinik Yorum
En yüksek skorlu sınıfı yalnızca "öncelikli model çıktısı" olarak belirt. Skorlar birbirine yakınsa
veya teknik güven düşükse bunu açıkça vurgula ve ayırıcı tanı yaklaşımı kullan.

### Tanıyı Kesinleştirmek İçin Muayene ve Tetkikler
Hastalığa uygun somut muayene/tetkikleri öncelik sırasıyla yaz. Alakasız testleri sıralama.

### Tanı Doğrulanırsa Genel Tedavi Yaklaşımı
İlaç grubu ve cerrahi yaklaşım düzeyinde bilgi ver. Marka, doz, uygulama sıklığı veya reçete yazma.
Korneal ülser dışlanmadan topikal kortikosteroid önermeme gibi kritik kontrendikasyonları belirt.

Yanıtın başında tek cümlelik prototip uyarısı bulunmalı. 350 kelimeyi geçme.
"""

_model = None


def get_model():
    """Load the YOLO classifier lazily so Render can bind to its port quickly."""
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
        from ultralytics import YOLO

        LOGGER.info("Model yükleniyor: %s", MODEL_PATH)
        _model = YOLO(str(MODEL_PATH))
    return _model


def get_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def predict_scores(image) -> dict[str, float]:
    model = get_model()
    result = model.predict(image, verbose=False)[0]
    probabilities = [float(value) for value in result.probs.data.tolist()]
    names = result.names

    if len(probabilities) != EXPECTED_CLASS_COUNT:
        raise RuntimeError(
            f"Beklenen {EXPECTED_CLASS_COUNT} sınıf yerine {len(probabilities)} sınıf üretildi."
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

    # Heuristic communication bands; these are not calibration thresholds.
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
        css_class = "status-high"
    elif level.startswith("Orta"):
        css_class = "status-medium"
    else:
        css_class = "status-low"

    return f"""
    <div class="status-card {css_class}">
      <div class="status-kicker">ÖNCELİKLİ MODEL ÇIKTISI</div>
      <div class="status-title">{html.escape(top_name)}</div>
      <div class="status-row">
        <span>{html.escape(level)}</span>
        <span>En yüksek skor: %{top_score * 100:.1f}</span>
        <span>İlk iki sınıf farkı: %{margin * 100:.1f}</span>
      </div>
      <div class="status-note">Bu skor klinik doğruluk veya kalibre edilmiş hastalık olasılığı değildir.</div>
    </div>
    """


def format_top_scores(scores: Mapping[str, float], top_n: int = 5) -> str:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return "\n".join(f"- {name}: %{score * 100:.1f}" for name, score in ranked)


def generate_ai_comment(scores: Mapping[str, float], species: str) -> str:
    api_key = get_api_key()
    if not api_key:
        return (
            "⚠️ **Gemini değerlendirmesi etkin değil.** Render ortamında `GEMINI_API_KEY` "
            "tanımlandığında bu bölüm otomatik olarak çalışacaktır. Modelin ilk beş skoru yukarıda gösterilmektedir."
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
        return response.text or "⚠️ AI değerlendirmesi boş yanıt döndürdü."
    except Exception:
        LOGGER.exception("Gemini değerlendirmesi alınamadı")
        return (
            "⚠️ **AI değerlendirmesi şu anda alınamadı.** Görüntü sınıflandırma sonuçları geçerlidir; "
            "metin tabanlı klinik yorum için daha sonra yeniden deneyin."
        )


def analyse_image(image, species: str):
    if image is None:
        return (
            {},
            '<div class="status-card status-low"><div class="status-title">Önce bir göz fotoğrafı yükleyin.</div></div>',
            "",
        )

    try:
        scores = predict_scores(image)
        return scores, confidence_card(scores), generate_ai_comment(scores, species)
    except Exception:
        LOGGER.exception("Görüntü analizi başarısız")
        return (
            {},
            '<div class="status-card status-low"><div class="status-title">Analiz tamamlanamadı.</div>'
            '<div class="status-note">Dosyanın geçerli bir JPG/PNG görüntüsü olduğundan emin olun.</div></div>',
            "⚠️ Teknik bir hata oluştu. Lütfen başka bir görüntüyle yeniden deneyin.",
        )


def background_data_uri() -> str:
    """Return the congress-style background as an embedded data URI."""
    if not BACKGROUND_PATH.is_file():
        LOGGER.warning("Arka plan görseli bulunamadı: %s", BACKGROUND_PATH)
        return ""
    encoded = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


BACKGROUND_DATA_URI = background_data_uri()

CUSTOM_CSS = f"""
:root {{ color-scheme: dark; }}
html, body {{ min-height: 100%; background: #050914 !important; }}
body {{ overflow-x: hidden; }}
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
    filter: saturate(.88) contrast(1.03);
}}
.background-overlay {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    background:
      linear-gradient(90deg, rgba(3,8,18,.96) 0%, rgba(3,8,18,.88) 29%, rgba(4,12,25,.58) 54%, rgba(4,10,20,.24) 76%, rgba(3,8,18,.42) 100%),
      linear-gradient(180deg, rgba(3,8,18,.34) 0%, rgba(3,8,18,.08) 45%, rgba(3,8,18,.78) 100%);
}}
.gradio-container {{ background: transparent !important; min-height: 100vh; position: relative; z-index: 1; }}
#app-shell {{ max-width: 1120px; margin: 0 auto; padding: 28px 18px 38px; }}
#hero {{ padding: 34px 4px 24px; max-width: 720px; }}
#hero h1 {{ color: #f1feff; font-size: clamp(30px, 4.2vw, 49px); line-height: 1.07; margin: 0; letter-spacing: -1.2px; text-shadow: 0 4px 28px rgba(0,0,0,.46); }}
#hero p {{ color: #b8d4da; max-width: 650px; font-size: 15px; line-height: 1.68; margin: 15px 0 0; text-shadow: 0 2px 14px rgba(0,0,0,.38); }}
.hero-badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
.hero-badge {{ border: 1px solid rgba(94,234,212,.26); background: rgba(7,45,56,.48); backdrop-filter: blur(10px); color: #a7f3e8; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 650; }}
.glass-card {{ background: rgba(6, 16, 30, .79) !important; backdrop-filter: blur(18px) saturate(1.08); border: 1px solid rgba(119,226,217,.19) !important; border-radius: 18px !important; padding: 20px !important; box-shadow: 0 24px 80px rgba(0,0,0,.38) !important; }}
.glass-card *, .glass-card label span {{ color: #dff7f5 !important; }}
.glass-card .block, .glass-card .wrap, .glass-card .styler {{ background: rgba(3,11,22,.55) !important; }}
.section-heading h3 {{ color: #67e8d3 !important; margin-top: 0 !important; }}
.helper, .helper * {{ color: #91b5bc !important; font-size: 13px !important; line-height: 1.55 !important; }}
.primary-action button {{ background: linear-gradient(115deg,#0d9488,#0284c7) !important; border: 0 !important; color: white !important; font-weight: 750 !important; min-height: 48px; box-shadow: 0 10px 30px rgba(8,145,178,.25); }}
.status-card {{ border-radius: 15px; padding: 17px 18px; border: 1px solid; margin: 4px 0 14px; background: rgba(3,11,22,.68); }}
.status-high {{ border-color: rgba(52,211,153,.46); }}
.status-medium {{ border-color: rgba(250,204,21,.46); }}
.status-low {{ border-color: rgba(251,113,133,.50); }}
.status-kicker {{ color: #7dd3fc; font-size: 10px; letter-spacing: 1.5px; font-weight: 800; }}
.status-title {{ color: #ecfeff; font-size: 23px; font-weight: 780; margin-top: 5px; }}
.status-row {{ display: flex; flex-wrap: wrap; gap: 8px 16px; color: #b6d9dc; font-size: 12px; margin-top: 9px; }}
.status-note {{ color: #789ba2; font-size: 11px; margin-top: 9px; }}
#footer {{ color: #789aa2; text-align: center; font-size: 11px; padding: 20px 0 4px; line-height: 1.6; text-shadow: 0 2px 10px rgba(0,0,0,.52); }}
#footer strong {{ color: #67e8d3; }}
@media (max-width: 760px) {{
  .background-image {{ background-position: 66% center; }}
  .background-overlay {{
    background:
      linear-gradient(90deg, rgba(3,8,18,.95) 0%, rgba(3,8,18,.78) 60%, rgba(3,8,18,.50) 100%),
      linear-gradient(180deg, rgba(3,8,18,.18) 0%, rgba(3,8,18,.45) 54%, rgba(3,8,18,.92) 100%);
  }}
  #app-shell {{ padding: 12px 10px 26px; }}
  #hero {{ padding: 24px 4px 20px; }}
  .glass-card {{ padding: 14px !important; border-radius: 14px !important; }}
}}
"""

with gr.Blocks(title="Veteriner Göz Hastalıkları — AI Asistanı") as demo:
    gr.HTML('<div class="background-image"></div><div class="background-overlay"></div>')
    with gr.Column(elem_id="app-shell"):
        gr.HTML(
            """
            <section id="hero">
              <div class="hero-badges">
                <span class="hero-badge">24 sınıflı görüntü modeli</span>
                <span class="hero-badge">Kedi & köpek</span>
                <span class="hero-badge">Araştırma prototipi v2.1</span>
              </div>
              <h1>Veteriner Göz Hastalıkları<br>AI Karar-Destek Asistanı</h1>
              <p>Ön segment göz fotoğrafını sınıflandırır, ilk beş model çıktısını gösterir ve veteriner hekim için tanıyı kesinleştirme yaklaşımı üretir.</p>
            </section>
            """
        )

        with gr.Group(elem_classes="glass-card"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=6):
                    gr.Markdown("### Görüntü ve olgu bilgisi", elem_classes="section-heading")
                    image_input = gr.Image(
                        type="pil",
                        label="Göz fotoğrafı",
                        height=330,
                        sources=["upload", "webcam", "clipboard"],
                    )
                    species_input = gr.Radio(
                        choices=["Kedi", "Köpek", "Belirtilmedi"],
                        value="Belirtilmedi",
                        label="Hayvan türü",
                    )
                    gr.Markdown(
                        "Gözü yakın planda, net, iyi aydınlatılmış ve mümkün olduğunca arka plansız yükleyin. "
                        "Görüntüdeki el, cerrahi alet veya geniş yüz kadrajı model skorlarını bozabilir.",
                        elem_classes="helper",
                    )
                    analyse_button = gr.Button(
                        "Görüntüyü analiz et",
                        variant="primary",
                        size="lg",
                        elem_classes="primary-action",
                    )
                with gr.Column(scale=5):
                    gr.Markdown("### Model sonuçları", elem_classes="section-heading")
                    confidence_output = gr.HTML(
                        '<div class="status-card"><div class="status-title">Analiz bekleniyor</div>'
                        '<div class="status-note">Bir göz fotoğrafı yükleyerek başlayın.</div></div>'
                    )
                    labels_output = gr.Label(
                        num_top_classes=5,
                        label="İlk beş model güven skoru",
                    )
                    gr.Markdown(
                        "Model skoru tek başına tanı değildir. Düşük skorlu veya birbirine yakın sonuçlar, "
                        "görüntünün yeniden çekilmesini ve ayrıntılı oftalmik muayeneyi gerektirir.",
                        elem_classes="helper",
                    )

        with gr.Group(elem_classes="glass-card"):
            gr.Markdown("### AI destekli klinik değerlendirme", elem_classes="section-heading")
            ai_output = gr.Markdown(
                "Analiz sonrasında muayene/tetkik önerileri ve genel tedavi yaklaşımı burada gösterilecektir."
            )

        gr.HTML(
            """
            <footer id="footer">
              <strong>Veteriner Göz Hastalıkları AI Asistanı — Prototip v2.1</strong><br>
              Klinik teşhis veya tedavi aracının yerine geçmez. Nihai karar, hastayı muayene eden veteriner hekime aittir.
            </footer>
            """
        )

    analyse_button.click(
        fn=analyse_image,
        inputs=[image_input, species_input],
        outputs=[labels_output, confidence_output, ai_output],
        api_name="analiz_et",
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=False,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"),
        css=CUSTOM_CSS,
    )
