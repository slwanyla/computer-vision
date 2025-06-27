import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 1. Path dataset (struktur folder: train/PNEUMONIA, train/NORMAL, val/PNEUMONIA, val/NORMAL)
base_dir = r"C:\computer vision\chest_xray"

# 2. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=(150, 150),
    class_mode='binary',
    batch_size=32
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(base_dir, "val"),
    target_size=(150, 150),
    class_mode='binary',
    batch_size=32
)

# 3. CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 4. Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train Model
history = model.fit(train_gen, epochs=10, validation_data=val_gen)

# 6. Save Model
model.save("pneumonia_classifier.h5")

# Optional: Plot akurasi
plt.plot(history.history['accuracy'], label='Akurasi Train')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
