import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

# Load Dataset
IMG_SIZE = 128
DATASET_DIR = 'dataset'
CSV_PATH = os.path.join(DATASET_DIR, 'labels.csv')

df = pd.read_csv(CSV_PATH)
df['image_path'] = df['image_name'].apply(lambda x: os.path.join(DATASET_DIR, 'images', x))

# Encode Labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['category'])
num_classes = len(le.classes_)

# Train/Test Split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_encoded'], random_state=42)

# Load Images 
def load_images(paths):
    images = []
    for path in tqdm(paths):
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img / 255.0)
    return np.array(images)

X_train = load_images(train_df['image_path'])
X_test = load_images(test_df['image_path'])
y_train = train_df['label_encoded'].values[:len(X_train)]
y_test = test_df['label_encoded'].values[:len(X_test)]

y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Build CNN 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train 
model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=20, batch_size=32)

# Save Model + Encoder
model.save('model/cnn_model.h5')

with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Model training complete and saved!")