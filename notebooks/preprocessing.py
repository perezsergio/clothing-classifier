import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def normalize_and_extract_features_labels(data):
    features = data[:, 1:] / 255.0
    labels = data[:, 0]
    return features, labels


def get_train_test_sets():
    # Read
    train_df = pd.read_csv("../data/fashion-mnist_train.csv", sep=",")
    test_df = pd.read_csv("../data/fashion-mnist_test.csv", sep=",")
    # Transform
    train_data = np.array(train_df, dtype="float32")
    test_data = np.array(test_df, dtype="float32")
    x_train, y_train = normalize_and_extract_features_labels(train_data)
    x_test, y_test = normalize_and_extract_features_labels(test_data)
    # Train test split
    x_train, x_validate, y_train, y_validate = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    # image shape is 28 x 28 x 1. (Squared 28px x 28px image in black and white)
    image_shape = (28, 28, 1)
    # Reshape features to fit image size
    x_train = x_train.reshape(x_train.shape[0], *image_shape)
    x_test = x_test.reshape(x_test.shape[0], *image_shape)
    x_validate = x_validate.reshape(x_validate.shape[0], *image_shape)
    # Return features and labels for training, validation and test sets.
    return x_train, x_validate, x_test, y_train, y_validate, y_test
