import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from config import CLASSES

#evaluate_model
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Final test loss:     {test_loss:.4f}")
    return test_loss, test_accuracy

#predictions
def get_predictions(model, X_test):
    y_pred_probs = model.predict(X_test)
    return np.argmax(y_pred_probs, axis=1)

#plot_confusion_matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved.")

#sample_predictions
def plot_sample_predictions(X_test, y_test, y_pred):
    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[i])
        plt.title(f"Actual: {CLASSES[y_test[i]]}\nPredicted: {CLASSES[y_pred[i]]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/sample_predictions.png")
    plt.show()
    print("Sample predictions saved.")

#save_classification_report(_)
def save_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, target_names=CLASSES)
    print("Classification Report:")
    print(report)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)
    print("Classification report saved.")