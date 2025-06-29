import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from gradcam import make_gradcam_heatmap

dataset_path = "chest_xray"

# Load data
datagen = ImageDataGenerator(rescale=1./255)

train = datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

val = datagen.flow_from_directory(
    os.path.join(dataset_path, "val"),
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

# Model CNN
inputs = Input(shape=(150, 150, 3))
x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D()(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(128, (3,3), activation='relu', name='last_conv')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=5, validation_data=val)

# Fungsi untuk baca dan siapkan gambar
def load_custom_image(dataset_path, folder_type, class_type, filename):
    img_path = os.path.join(dataset_path, folder_type, class_type, filename)
    print(f"📷 Membaca gambar: {class_type} | {filename}")
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Gambar tidak ditemukan.")
        exit()
    img = cv2.resize(img, (150, 150))
    return img

# Input gambar pneumonia
folder_type_p = input("Masukkan folder (train/val) untuk gambar PNEUMONIA: ")
file_p = input("Masukkan nama file gambar PNEUMONIA: ")
img_p = load_custom_image(dataset_path, folder_type_p, "PNEUMONIA", file_p)
input_p = np.expand_dims(img_p / 255.0, axis=0)
heatmap_p = make_gradcam_heatmap(input_p, model, 'last_conv')

# Input gambar normal
folder_type_n = input("Masukkan folder (train/val) untuk gambar NORMAL: ")
file_n = input("Masukkan nama file gambar NORMAL: ")
img_n = load_custom_image(dataset_path, folder_type_n, "NORMAL", file_n)
input_n = np.expand_dims(img_n / 255.0, axis=0)
heatmap_n = make_gradcam_heatmap(input_n, model, 'last_conv')

# Tampilkan hasil
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.title("X-ray Pneumonia")
plt.imshow(img_p)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Grad-CAM Pneumonia")
plt.imshow(img_p)
plt.imshow(heatmap_p, cmap='jet', alpha=0.5)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("X-ray Normal")
plt.imshow(img_n)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Grad-CAM Normal")
plt.imshow(img_n)
plt.imshow(heatmap_n, cmap='jet', alpha=0.5)
plt.axis('off')

plt.tight_layout()
plt.show()
