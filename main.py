import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


DATASET_PATH = "data/EuroSAT_data"
CLASSES = ["Forest", "AnnualCrop", "Residential"]
IMAGE_SIZE = 64
EPOCHS = 9
BATCH_SIZE = 32
X = []  # images
y = [] # labels

print ("loading dataset images...")

for label, class_name in enumerate(CLASSES):
    class_folder = os.path.join(DATASET_PATH, class_name)

    if not os.path.exists(class_folder):
        raise FileNotFoundError(f"Folder not found: {class_folder}")
    
    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)

        try:
            img = Image.open(image_path).convert("RGB")
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

            img_array = np.array(img)
            X.append(img_array)
            y.append(label)

        except Exception as e:
            print("Error loading:", image_path, e)
        
X = np.array(X)
y = np.array(y)
print("Dataset loaded successfullly.")
print("X shape:", X.shape)
print("y shape:", y.shape)

X = X/255.0
print("Images normalized successfully.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

#CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128,(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("CNN model created and compiled successfully.")
model.summary()

#model training
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

#Accuracy Graph

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(" CNN Training & Validation Accuracy")
plt.legend()
plt.savefig("outputs/accuracy_graph.png")
plt.show()

#loss graph

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(" CNN Training & Validation Loss")
plt.legend()
plt.savefig("outputs/loss_graph.png")
plt.show()

#model evalution

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Final test accuracy:", test_accuracy)
print("Final test loss:", test_loss)

#confusion matrix

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

disp.plot()
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()

model.save("cnn_lulc_eurosat_model.keras")

print("Process completed successfully.")
print("Results are in output folder.")

#sample prediction
plt.figure(figsize=(10, 8))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[i])

    actual_class = CLASSES[y_test[i]]
    pred_label = CLASSES[y_pred[i]]

    plt.title(f"Actual: {actual_class}\nPredicted: {pred_label}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/sample_predictions.png")
plt.show()

print("Sample predictions saveed successfully")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

#classification report

report = classification_report(y_test, y_pred, target_names=CLASSES)

print("Classification Report:")
print(report)

with open("outputs/classification_report.txt", "w") as file:
    file.write(report)

print("Classification report saved successfully")
