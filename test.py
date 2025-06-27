import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Tambahkan ini:
base_dir = r"C:\computer vision\chest_xray"

# Path ke direktori test
test_dir = os.path.join(base_dir, "test")
print("Base dir:", base_dir)

# Load model yang sudah dilatih
model = load_model("pneumonia_classifier.h5")

# Buat ImageDataGenerator untuk test
test_datagen = ImageDataGenerator(rescale=1./255)

# Buat generator test
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

# Evaluasi model
loss, accuracy = model.evaluate(test_gen)
print(f"Akurasi Test: {accuracy:.2f}")
