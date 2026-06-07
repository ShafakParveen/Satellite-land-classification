import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from config import DATASET_PATH, CLASSES, IMAGE_SIZE, TEST_SIZE, RANDOM_STATE


def load_images():
    X, y = [], []

    print("Loading dataset images...")

    for label, class_name in enumerate(CLASSES):
        class_folder = os.path.join(DATASET_PATH, class_name)

        if not os.path.exists(class_folder):
            raise FileNotFoundError(f"Folder not found: {class_folder}")

        for image_name in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_name)
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                X.append(np.array(img))
                y.append(label)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")

    X = np.array(X) / 255.0
    y = np.array(y)

    print(f"Dataset loaded. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test