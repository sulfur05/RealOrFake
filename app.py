from flask import Flask, request, render_template
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

app = Flask(__name__)

# Load pretrained ViT deepfake detector
model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            image = Image.open(img_file).convert('RGB')
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
            
            # Hugging‑Face labels: 0 = Fake, 1 = Real
            prediction = {
                "Fake": round(probs[0], 3),
                "Real": round(probs[1], 3)
            }
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
