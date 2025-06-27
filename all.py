import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("pneumonia_classifier.h5")

# Path ke folder gambar X-ray
folder_path = r"C:\computer vision\try"  # <- ubah sesuai tempat gambar kamu

# Ambil semua nama file gambar
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Prediksi & tampilkan semua gambar
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)

    # Baca dan preprocess gambar
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Prediksi
    pred = model.predict(img_input)[0][0]
    label = "PNEUMONIA" if pred >= 0.5 else "NORMAL"
    confidence = f"{pred:.2f}"

    # Tampilkan gambar + hasil
    plt.imshow(img_rgb)
    plt.title(f"{label} ({confidence})")
    plt.axis('off')
    plt.show()