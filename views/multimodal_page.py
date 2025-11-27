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
    st.subheader("Analisis Investigasi Lengkap (All-in-One)")
    st.caption("Kombinasikan teks, gambar, dan audio untuk analisis komprehensif.")

    try:
        img_handler, txt_handler, aud_handler = load_handlers()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return

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

    st.markdown("---")
    analyze_btn = st.button("ANALISIS SEMUA BUKTI", type="primary", use_container_width=True)

    if analyze_btn:
        if not input_text and not input_image and not input_audio:
            st.warning("Masukkan minimal satu bukti (Teks, Gambar, atau Audio)!")
            return

        with st.spinner('Sedang melakukan investigasi menyeluruh...'):
            results = []
            
            if input_text:
                label, score = txt_handler.predict(input_text)
                results.append({"type": "Teks", "label": label, "score": score})
            
            if input_image:
                original_label, score = img_handler.predict(input_image, "MobileNet")
                hoax_labels = ['hoax', 'fake', 'fake/generated']
                
                if original_label.lower() in hoax_labels:
                    flipped_label = "Valid"
                else:
                    flipped_label = "Hoax"
                
                results.append({"type": "Gambar", "label": flipped_label, "score": score})

            if input_audio and aud_handler:
                temp_path = "temp_multi_audio.wav"
                with open(temp_path, "wb") as f:
                    f.write(input_audio.getbuffer())
                
                label, score = aud_handler.predict(temp_path)
                results.append({"type": "Audio", "label": label, "score": score})
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            st.subheader("Laporan Hasil Investigasi")
            
            hoax_indicators = ['hoax', 'fake', 'fake/generated']
            any_hoax = any(r['label'].lower() in hoax_indicators for r in results)
            
            if any_hoax:
                st.error("### KESIMPULAN: INDIKASI HOAX / PALSU")
                st.write("Ditemukan anomali pada salah satu atau lebih bukti yang Anda lampirkan.")
            else:
                st.success("### KESIMPULAN: INDIKASI VALID")
                st.write("Semua bukti yang dilampirkan terlihat asli dan valid.")

            st.divider()
            
            r_col1, r_col2, r_col3 = st.columns(3)
            
            for i, res in enumerate(results):
                target_col = [r_col1, r_col2, r_col3][i % 3]
                
                is_hoax = res['label'].lower() in hoax_indicators
                status_color = "#dc3545" if is_hoax else "#28a745"
                bg_color = "#fff5f5" if is_hoax else "#f0fff4"
                
                with target_col:
                    target_col.markdown(f"""
                    <div style='
                        padding: 15px; 
                        background-color: {bg_color};
                        border: 1px solid #ddd; 
                        border-radius: 10px; 
                        border-top: 5px solid {status_color};
                        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                        margin-bottom: 10px;
                    '>
                        <h4 style='margin:0; color: #555; font-size: 14px;'>Analisis {res['type']}</h4>
                        <h2 style='color: {status_color}; margin: 5px 0; font-size: 24px;'>{res['label']}</h2>
                        <p style='margin:0; font-weight:bold;'>Confidence: {res['score']:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(int(res['score']))
