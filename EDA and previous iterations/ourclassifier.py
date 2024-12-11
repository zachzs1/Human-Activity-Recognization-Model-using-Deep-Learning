# -*- coding: utf-8 -*-
"""
CNN for Human Activity Recognition
"""

import numpy as np
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_end_index = 3511  # Last row of training data for train/test split


def build_cnn_model(input_shape, num_classes):
    """
    Build a simple CNN model for activity recognition.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def predict_test(train_data, train_labels, test_data, num_classes):
    """
    Train the CNN model and predict on test data.
    """
    # Adjust labels to zero-based indexing
    train_labels_zero_based = train_labels - train_labels.min()

    # One-hot encode labels
    train_labels_cat = to_categorical(train_labels_zero_based, num_classes=num_classes)

    # Build and train CNN model
    input_shape = (train_data.shape[1], train_data.shape[2])
    model = build_cnn_model(input_shape, num_classes)
    model.fit(train_data, train_labels_cat, epochs=10, batch_size=128, verbose=1)

    # Predict on test data
    test_outputs = model.predict(test_data)
    test_predictions = np.argmax(test_outputs, axis=1)
    return test_predictions


# Run this code only if being used as a script
if __name__ == "__main__":
    # Load labels and training sensor data into a 3-D array
    labels = np.loadtxt('../Data/labels_train_1.csv', dtype='int')
    data_slice_0 = np.loadtxt(sensor_names[0] + '_train_1.csv', delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1], len(sensor_names)))
    data[:, :, 0] = data_slice_0
    del data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(sensor_names[sensor_index] + '_train_1.csv', delimiter=',')

    # Split into training and test by row index
    train_data = data[:train_end_index + 1, :, :]
    train_labels = labels[:train_end_index + 1]
    test_data = data[train_end_index + 1:, :, :]
    test_labels = labels[train_end_index + 1:]

    # Get number of classes
    num_classes = len(np.unique(labels))

    # Predict test labels using CNN
    test_outputs = predict_test(train_data, train_labels, test_data, num_classes)

    # Compute micro and macro-averaged F1 scores
    micro_f1 = f1_score(test_labels, test_outputs, average='micro')
    macro_f1 = f1_score(test_labels, test_outputs, average='macro')
    print(f'Micro-averaged F1 score: {micro_f1}')
    print(f'Macro-averaged F1 score: {macro_f1}')
