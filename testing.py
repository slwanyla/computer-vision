import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model
model = load_model("pneumonia_classifier.h5")

# Load dan preprocess gambar
img_path = r"C:\computer vision\n2.jpg"  # ← Ubah sesuai lokasi gambar
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB agar warna benar di matplotlib
img_resized = cv2.resize(img_rgb, (150, 150))
img_input = img_resized / 255.0
img_input = np.expand_dims(img_input, axis=0)

# Prediksi
pred = model.predict(img_input)[0][0]
label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

# Tampilkan hasil prediksi dan gambar
plt.imshow(img_rgb)
plt.title(f"Hasil: {label} ({pred:.2f})")
plt.axis('off')
plt.show()
