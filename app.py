"""
Veteriner Goz Hastaliklari - AI Teshis Asistani (Hugging Face Spaces surumu)

Bu dosya scripts/demo_arayuz.py ile ayni mantigi kullanir, sadece dosya yollari
Hugging Face Spaces ortamina (goreceli yollar, ortam degiskeni tabanli API
anahtari) gore ayarlanmistir.
"""
import os
from pathlib import Path

import gradio as gr

MODEL_PATH = Path(__file__).parent / "v2_best.pt"
LLM_MODEL_NAME = "gemini-flash-latest"

_model = None


def get_model():
    """Modeli ilk kullanimda yukler (lazy loading) - boylece program aciliste
    hemen porta baglanir, Render/Hugging Face gibi platformlarin baslangic
    zaman asimina takilmaz. Ilk fotograf biraz yavas olabilir, sonrakiler hizli olur."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(MODEL_PATH))
    return _model

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

SYSTEM_PROMPT = """Sen bir veteriner göz hastalıkları yapay zeka asistanısın. Bir görüntü
sınıflandırma modeli, kedi/köpek ön segment göz fotoğrafına bakarak olası hastalıkları
yüzde olarak sıraladı; bu yüzdeler sana verilecek.

En olası tanı üzerinden (ÜST KATEGORİ düzeyinde - alt tip/evre/nükleer skleroz gibi
fotoğraftan çıkarılamayacak ayrımlara girme), Türkçe ve veteriner hekime hitaben
AŞAĞIDAKİ İKİ BAŞLIĞI yaz:

### 🔬 Tanıyı Kesinleştirmek İçin Muayene/Tetkik Yöntemleri
Bu tanının doğru olup olmadığını netleştirmek için hekimin yapması gereken somut
muayene bulgularını ve basit tetkikleri madde madde listele (örn. Schirmer testi,
fluorescein boyama, tonometri, yarık lamba muayenesi, göz dibi muayenesi vb. -
hastalığa uygun olanları seç, alakasız olanları listeleme).

### 💊 Tanı Doğrulanırsa Uygulanacak Tedavi Yöntemleri
Hem medikal (ilaç GRUBU bazında - örn. "topikal antibiyotik", "topikal steroid
olmayan antiinflamatuar" - spesifik marka/doz YAZMA) hem cerrahi (varsa, hangi
durumda gerekli olduğunu belirterek) tedavi seçeneklerini madde madde listele.

Cevabının EN BAŞINA, tek cümlelik kısa bir uyarı ekle: bu bir geliştirme aşamasındaki
prototip yapay zeka aracıdır, klinik teşhis/tedavi yerine geçmez, nihai karar her
zaman muayene eden hekime aittir.

Kısa ve öz ol, madde işaretleri kullan, 300 kelimeyi geçme.
"""

def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    return None


def get_client():
    key = get_api_key()
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)


def tahmin_et(img):
    model = get_model()
    results = model.predict(img, verbose=False)
    r = results[0]
    probs = r.probs.data.tolist()
    names = r.names
    sonuc = {}
    for idx, p in enumerate(probs):
        eng_name = names[idx]
        tr_name = TURKCE_ISIMLER.get(eng_name, eng_name)
        sonuc[tr_name] = float(p)
    return sonuc


def format_top_for_prompt(sonuc, top_n=5):
    top = sorted(sonuc.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return "\n".join(f"- {name}: %{prob * 100:.1f}" for name, prob in top)


API_KEY_UYARISI = (
    "⚠️ AI değerlendirmesi için henüz bir API anahtarı ayarlanmadı. "
    "Yüzde teşhis sonuçları yukarıda görünüyor, ama muayene/tedavi önerisi kısmı şu an aktif değil."
)


def ai_yorumu(sonuc):
    client = get_client()
    if client is None:
        return API_KEY_UYARISI
    from google.genai import types
    top_text = format_top_for_prompt(sonuc)
    try:
        response = client.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=f"Model tahminleri:\n{top_text}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1000,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI değerlendirmesi alınamadı: {e}"


def analiz_et(img):
    if img is None:
        return {}, ""
    sonuc = tahmin_et(img)
    yorum = ai_yorumu(sonuc)
    return sonuc, yorum


CUSTOM_CSS = """
.gradio-container {
    background: #f8fafc !important;
    min-height: 100vh;
}

/* ---- Baslik ---- */
#baslik-kutusu {
    text-align: center;
    padding: 32px 20px;
    border-radius: 16px;
    background: linear-gradient(120deg, #0f766e 0%, #0891b2 100%);
    color: white;
    margin-bottom: 20px;
}
#baslik-kutusu h1 {
    margin: 0;
    font-size: 26px;
    font-weight: 700;
}
#baslik-kutusu p {
    margin: 8px 0 0 0;
    opacity: 0.92;
    font-size: 14px;
}

/* ---- Kartlar ---- */
.ana-kart, .degerlendirme-kart {
    background: white !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06) !important;
    border: 1px solid #e2e8f0 !important;
    margin-bottom: 16px !important;
}

/* ---- Buton ---- */
.teshis-butonu button {
    background: linear-gradient(120deg, #0d9488, #0891b2) !important;
    border: none !important;
    font-weight: 600 !important;
}

/* ---- Basliklar ---- */
.bolum-basligi h3 {
    color: #0f766e !important;
    font-weight: 700 !important;
}

/* ---- Ipucu metni ---- */
.ipucu-metni {
    font-size: 13px !important;
    color: #475569 !important;
}

/* ---- AI degerlendirme metni ---- */
.degerlendirme-metni {
    font-size: 15px;
    line-height: 1.6;
}

/* ---- Alt bilgi ---- */
#altbilgi {
    text-align: center;
    padding: 16px;
    color: #94a3b8;
    font-size: 12px;
}
"""

with gr.Blocks(title="Veteriner Göz Hastalıkları — AI Teşhis Asistanı") as demo:
    gr.HTML(
        """
        <div id="baslik-kutusu">
            <h1>🐾 Veteriner Göz Hastalıkları — AI Teşhis Asistanı</h1>
            <p>Ön segment göz fotoğrafından olası hastalığı ve tedavi yönlendirmesini değerlendirir · Prototip v2</p>
        </div>
        """
    )

    with gr.Group(elem_classes="ana-kart"):
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(type="pil", label="📷 Göz Fotoğrafı Yükleyin")
                gr.Markdown(
                    "💡 *En iyi sonuç için fotoğrafı gözü kapsayacak şekilde YAKINDAN çekin/kırpın; "
                    "el, makas, geniş yüz veya arka plan mümkün olduğunca kadraj dışında kalsın.*",
                    elem_classes="ipucu-metni",
                )
                analiz_btn = gr.Button("🔍 Teşhis Et", variant="primary", size="lg", elem_classes="teshis-butonu")
            with gr.Column(scale=1):
                label_output = gr.Label(num_top_classes=5, label="📊 Olası Tanılar (%)")

    with gr.Group(elem_classes="degerlendirme-kart"):
        gr.Markdown("### 🧠 AI Değerlendirmesi: Kesinleştirme ve Tedavi Önerisi", elem_classes="bolum-basligi")
        yorum_output = gr.Markdown(elem_classes="degerlendirme-metni")

    gr.HTML(
        """
        <div id="altbilgi">
            <strong>Veteriner Göz Hastalıkları AI Asistanı</strong> — Geliştirme Prototipi v2<br>
            Bu araç klinik teşhis/tedavi yerine geçmez. Nihai karar her zaman muayene eden veteriner hekime aittir.
        </div>
        """
    )

    analiz_btn.click(fn=analiz_et, inputs=img_input, outputs=[label_output, yorum_output])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="cyan",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
    )
