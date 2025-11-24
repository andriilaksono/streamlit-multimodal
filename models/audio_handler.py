import torch
import os
import numpy as np
import librosa # <--- Kita pakai ini, jauh lebih stabil daripada Torchaudio di Windows
from transformers import Wav2Vec2ForSequenceClassification

class AudioHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join("assets", "audio_models", "wav2vec2.pt")
        self.model = None
        self.class_names = ["Real", "Fake/Generated"] 
        
        self.load_model()

    def load_model(self):
        if self.model is not None:
            return

        print(f"Memuat model audio dari {self.model_path}...")
        try:
            # 1. Bangun Arsitektur
            try:
                model_name = "facebook/wav2vec2-base"
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=2
                )
            except Exception as e:
                print(f"Gagal download config base: {e}")
                return

            # 2. Load Weights
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                
                # Load state dict
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                print(f"⚠️ File {self.model_path} tidak ditemukan. Menggunakan model base kosong.")

            self.model.to(self.device)
            self.model.eval()
            print("✅ Model Audio berhasil dimuat.")
            
        except Exception as e:
            print(f"❌ Error Fatal saat load model audio: {e}")

    def preprocess_audio(self, audio_file):
        # --- PERUBAHAN TOTAL DI SINI ---
        # Kita pakai LIBROSA. 
        # sr=16000 artinya otomatis diubah ke 16kHz (resample otomatis).
        # mono=True artinya otomatis diubah jadi 1 channel.
        speech_array, _ = librosa.load(audio_file, sr=16000, mono=True)

        # Librosa outputnya Numpy, kita ubah ke Tensor PyTorch
        waveform = torch.tensor(speech_array).float()

        # Squeeze/Unsqueeze logic
        # Kita butuh input bersih (1 dimensi array panjang)
        return waveform

    def predict(self, audio_file):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return "Error: Model Gagal Diload", 0.0

        try:
            # Preprocess pakai Librosa
            waveform = self.preprocess_audio(audio_file)
            
            # Wav2Vec2 butuh input [Batch, Length]. Kita tambah dimensi batch di depan.
            input_tensor = waveform.unsqueeze(0).to(self.device)

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
            print(f"Error saat prediksi audio: {e}")
            return f"Error Proses: {str(e)}", 0.0