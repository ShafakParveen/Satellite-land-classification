import os
from data_loader import load_images, split_data
from model import build_model
from train import train_model, plot_training
from evaluate import (
    evaluate_model,
    get_predictions,
    plot_confusion_matrix,
    plot_sample_predictions,
    save_classification_report,
)
from config import MODEL_SAVE_PATH
 
os.makedirs("outputs", exist_ok=True)
 
X, y = load_images()
X_train, X_test, y_train, y_test = split_data(X, y)
 
model = build_model()
 
history = train_model(model, X_train, y_train, X_test, y_test)
plot_training(history)
 
evaluate_model(model, X_test, y_test)
 
y_pred = get_predictions(model, X_test)
 
plot_confusion_matrix(y_test, y_pred)
plot_sample_predictions(X_test, y_test, y_pred)
save_classification_report(y_test, y_pred)
 
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
print("All done. Results are in the outputs/ folder.")
