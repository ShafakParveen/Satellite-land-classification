import matplotlib.pyplot as plt

from config import EPOCHS, BATCH_SIZE


def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test)
    )
    return history


def plot_training(history):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("CNN Training & Validation Accuracy")
    plt.legend()
    plt.savefig("outputs/accuracy_graph.png")
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("CNN Training & Validation Loss")
    plt.legend()
    plt.savefig("outputs/loss_graph.png")
    plt.show()

    print("Training graphs saved.")