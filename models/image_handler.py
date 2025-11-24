import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

class ImageHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Valid', 'Hoax'] 
        self.model = None
        
        # Langsung load model saat inisialisasi agar cepat
        self.load_model()

    def load_model(self):
        if self.model is not None:
            return

        print("Sedang memuat model MobileNet...")    
        try:
            # --- HANYA MOBILENET (ResNet dihapus) ---
            self.model = models.mobilenet_v3_small(pretrained=False)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, 2)
            
            # Path khusus MobileNet
            path = os.path.join("assets", "images_models", "mobilenet.pth")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"File model tidak ditemukan di: {path}")

            # Load Weights
            checkpoint = torch.load(path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                self.model = checkpoint
                
            self.model.to(self.device)
            self.model.eval()
            print("✅ Sukses memuat MobileNet")

        except Exception as e:
            print(f"ERROR di load_model: {e}")
            self.model = None

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    # Parameter 'model_name' dihapus karena cuma ada satu model
    def predict(self, image_file, model_name=None): 
        if self.model is None:
            self.load_model()
            if self.model is None:
                return "Error Load Model", 0.0

        try:
            image = Image.open(image_file).convert('RGB')
            input_tensor = self.preprocess_image(image)

            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)
            
            label = self.class_names[predicted_class.item()]
            score = confidence.item() * 100
            
            return label, score
        except Exception as e:
            print(f"Error saat prediksi: {e}")
            return "Error Prediksi", 0.0