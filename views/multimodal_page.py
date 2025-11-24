import streamlit as st
import os
from models.image_handler import ImageHandler
from models.text_handler import TextHandler
try:
    from models.audio_handler import AudioHandler
except ImportError:
    AudioHandler = None

@st.cache_resource
def load_handlers():
    img_h = ImageHandler()
    txt_h = TextHandler()
    aud_h = AudioHandler() if AudioHandler else None
    return img_h, txt_h, aud_h

def show():
    st.subheader("📚 Analisis Investigasi Lengkap (All-in-One)")
    st.caption("Kombinasikan teks, gambar, dan audio untuk analisis komprehensif")

    try:
        img_handler, txt_handler, aud_handler = load_handlers()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return

    # --- INPUT SECTION (3 Kolom) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("1. Bukti Teks")
        input_text = st.text_area("Judul Berita", height=150, placeholder="Masukkan judul...")

    with col2:
        st.info("2. Bukti Gambar")
        input_image = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

    with col3:
        st.info("3. Bukti Audio")
        input_audio = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    # --- TOMBOL EKSEKUSI ---
    st.markdown("---")
    analyze_btn = st.button("🔍 ANALISIS SEMUA BUKTI", type="primary", use_container_width=True)

    if analyze_btn:
        if not input_text and not input_image and not input_audio:
            st.warning("⚠️ Masukkan minimal satu bukti (Teks, Gambar, atau Audio)!")
            return

        with st.spinner('Sedang melakukan investigasi menyeluruh...'):
            results = []
            
            # 1. Analisis Teks
            if input_text:
                label, score = txt_handler.predict(input_text)
                results.append({"type": "Teks", "label": label, "score": score})
            
            # 2. Analisis Gambar
            if input_image:
                # Kita pakai MobileNet sbg default biar cepat
                label, score = img_handler.predict(input_image, "MobileNet")
                results.append({"type": "Gambar", "label": label, "score": score})

            # 3. Analisis Audio
            if input_audio and aud_handler:
                temp_path = "temp_multi_audio.wav"
                with open(temp_path, "wb") as f:
                    f.write(input_audio.getbuffer())
                
                label, score = aud_handler.predict(temp_path)
                results.append({"type": "Audio", "label": label, "score": score})
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # --- TAMPILKAN HASIL ---
            st.subheader("📊 Laporan Hasil Investigasi")
            
            any_hoax = any(r['label'].lower() in ['hoax', 'fake', 'fake/generated'] for r in results)
            
            if any_hoax:
                st.error("### 🚨 KESIMPULAN: INDIKASI HOAX / PALSU")
                st.write("Ditemukan anomali pada salah satu atau lebih bukti yang Anda lampirkan.")
            else:
                st.success("### ✅ KESIMPULAN: INDIKASI VALID")
                st.write("Semua bukti yang dilampirkan terlihat asli dan valid.")

            st.divider()
            
            # Tampilkan Rincian Card
            r_col1, r_col2, r_col3 = st.columns(3)
            
            for i, res in enumerate(results):
                # Pilih kolom secara dinamis
                target_col = [r_col1, r_col2, r_col3][i % 3]
                
                with target_col:
                    status_color = "red" if res['label'].lower() in ['hoax', 'fake', 'fake/generated'] else "green"
                    target_col.markdown(f"""
                    <div style='padding: 10px; border: 1px solid #ddd; border-radius: 10px; border-top: 5px solid {status_color};'>
                        <h4 style='margin:0;'>Analisis {res['type']}</h4>
                        <h2 style='color: {status_color}; margin:0;'>{res['label']}</h2>
                        <p>Confidence: <strong>{res['score']:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)