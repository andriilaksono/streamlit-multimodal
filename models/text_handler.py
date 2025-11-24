import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import os

class TextHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(os.getcwd(), "assets", "text_models")
        self.label_map = {'valid': 1, 'hoax': 0}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is not None:
            return True
        print(f"Mencoba memuat model teks dari: {self.model_path}")
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Folder model tidak ditemukan: {self.model_path}")
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model teks berhasil dimuat.")
            return True
        except Exception as e:
            print(f"FATAL ERROR Text Model: {e}")
            return False

    def predict(self, text):
        if self.model is None:
            success = self.load_model()
            if not success:
                return "Error Model Missing", 0.0
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                confidence, predicted_idx = torch.max(probs, dim=-1)
                prediksi_angka = predicted_idx.item()
                confidence_score = confidence.item() * 100
                prediksi_label = self.reverse_label_map.get(prediksi_angka, "Unknown")
            return prediksi_label, confidence_score
        except Exception as e:
            print(f"Error processing text: {e}")
            return "Error Processing", 0.0