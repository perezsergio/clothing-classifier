from keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.saving import load_model


def fit_and_save_model(
    model,
    model_name,
    x_train,
    y_train,
    x_validate,
    y_validate,
    batch_sze=4096,
    num_epochs=50,
):
    """
    Given a model, a training and a validation set,
    train the model, save it an return the training history
    """
    # Define loss function, optimizer, and performance metrics
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(lr=0.001),
        metrics=["accuracy"],
    )
    # Train it
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_sze,
        epochs=num_epochs,
        verbose=0,
        validation_data=(x_validate, y_validate),
    )
    # Save it so that we don't have to train it again
    model.save(f"../models/{model_name}.model")
    # Return history
    return history


def visualize_performance(history, model_name, x_validate, y_validate):
    """
    Given a model, its history and the validation set,
    visualize its performance with a confusion matrix and a line plot of the accuracy vs num epochs.
    """
    model = load_model(f"../models/{model_name}.model")
    _, ax = plt.subplots(1, 2, figsize=(12, 5))

    ## Accuracy per epochs trained
    ax[0].plot(history.history["accuracy"], label="Accuracy")
    ax[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax[0].legend()
    ax[0].set_title("Accuracy per epochs trained")

    ## Confusion matrix
    # Compute
    y_predicted = np.argmax(model.predict(x_validate, verbose=0), axis=1)
    y_true = y_validate
    cm = confusion_matrix(y_true, y_predicted)
    # Plot
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="crest",
        ax=ax[1],
    )
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")
    ax[1].set_title("Confusion Matrix")
    # Annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[1].text(
                j + 0.5, i + 0.5, str(cm[i, j]), ha="center", va="center", color="black"
            )

    # Print accuracy for the last epoch
    print("Final accuracy: ", round(history.history["accuracy"][-1], 4))
    print("Final validation accuracy: ", round(history.history["val_accuracy"][-1], 4))
