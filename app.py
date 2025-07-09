# app.py
from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os

app = Flask(__name__)

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model/fake_real_cnn.pth', map_location='cpu', weights_only=False))

model.eval()

# Image preprocessor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            image = Image.open(img_file).convert('RGB')
            img_tensor = transform(image).unsqueeze(0)
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()

            label = 'Fake' if pred == 0 else 'Real'
            return render_template('index.html', prediction=label)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
