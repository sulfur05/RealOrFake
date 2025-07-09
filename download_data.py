import os
import requests
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Create directories
os.makedirs('data/train/fake', exist_ok=True)
os.makedirs('data/train/real', exist_ok=True)

# 2. Download 100 fake faces from thispersondoesnotexist.com
print("📥 Downloading fake faces...")
for i in tqdm(range(1, 101)):
    try:
        r = requests.get("https://thispersondoesnotexist.com/", headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        with open(f"data/train/fake/fake_{i:03d}.jpg", "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"❌ Failed to download fake image {i}: {e}")

# 3. Download real face images from LFW using scikit-learn
print("\n📸 Downloading real faces from LFW dataset...")
lfw_people = fetch_lfw_people(min_faces_per_person=1, resize=1.0, color=True)
images = lfw_people.images
for i in range(100):
    img_array = (images[i] * 255).astype(np.uint8)  # Convert float32 to uint8
    img = Image.fromarray(img_array)
    img.save(f"data/train/real/real_{i:03d}.jpg")

print("\n✅ Done! 100 fake and 100 real images are ready.")
