import torch
import os
import numpy as np
import librosa
import traceback # <-- Tambahan penting buat melacak error
from transformers import Wav2Vec2ForSequenceClassification

class AudioHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "facebook/wav2vec2-base-960h"
        self.model = None
        self.class_names = ["Real", "Fake"] 
        self.load_model()

    def load_model(self):
        if self.model is not None:
            return

        print(f"🔄 Sedang memuat model audio dari {self.model_name}...")
        try:
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model Audio Siap.")
            
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")
            traceback.print_exc() # Cetak error lengkap ke terminal
            self.model = None

    def preprocess_audio(self, audio_file):
        print(f"📂 Mencoba memproses file: {audio_file}")
        
        # 1. Load Audio pakai Librosa
        # PENTING: Librosa butuh FFmpeg untuk MP3 di Windows
        try:
            speech_array, sr = librosa.load(audio_file, sr=16000, mono=True)
            print(f"✅ Audio berhasil dibaca. Sample Rate: {sr}, Panjang: {len(speech_array)}")
        except Exception as e:
            print("❌ Gagal membaca file audio dengan Librosa.")
            raise e # Lempar error biar ditangkap fungsi predict

        # 2. Ubah ke Tensor
        waveform = torch.tensor(speech_array).float()

        # 3. Potong durasi maks 10 detik (biar RAM aman)
        max_len = 16000 * 10
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]

        # 4. Pastikan dimensi benar [1, length]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        return waveform.to(self.device)

    def predict(self, audio_file):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return "Error: Model Gagal Diload (Cek Internet)", 0.0

        try:
            input_tensor = self.preprocess_audio(audio_file)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits

                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence, predicted_idx = torch.max(probs, dim=-1)
                
                idx = predicted_idx.item()
                score = confidence.item() * 100
                
                label = self.class_names[idx] if idx < len(self.class_names) else "Unknown"
                
            return label, score

        except Exception as e:
            # INI BAGIAN PENTING:
            print("\n!!! ERROR TERJADI SAAT PREDIKSI !!!")
            print(f"Pesan: {str(e)}")
            traceback.print_exc() # Cetak detail error ke terminal
            return f"Error Proses: {str(e)}", 0.0