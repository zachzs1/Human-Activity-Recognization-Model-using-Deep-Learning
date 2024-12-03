import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import LSTM, Dense, Dropout
from tf.keras.utils import to_categorical

# Load your IMU data (example file names)
acc_x_train = pd.read_csv('Acc_x_train_1.csv', header=None).values
acc_y_train = pd.read_csv('Acc_y_train_1.csv', header=None).values
acc_z_train = pd.read_csv('Acc_z_train_1.csv', header=None).values
gyro_x_train = pd.read_csv('Gyro_x_train_1.csv', header=None).values
gyro_y_train = pd.read_csv('Gyro_y_train_1.csv', header=None).values
gyro_z_train = pd.read_csv('Gyro_z_train_1.csv', header=None).values

# Load the activity labels
labels = pd.read_csv('labels_train_1.csv', header=None).values

# Stack the accelerometer and gyroscope data along the second axis (features axis)
data = np.stack([acc_x_train, acc_y_train, acc_z_train, gyro_x_train, gyro_y_train, gyro_z_train], axis=-1)

# Normalize the data (standard scaling)
scaler = StandardScaler()
data_reshaped = data.reshape(-1, data.shape[-1])  # Flatten the time windows
data_normalized = scaler.fit_transform(data_reshaped)
data_normalized = data_normalized.reshape(data.shape)  # Reshape back to (samples, time_steps, features)

# One-hot encode the labels
labels_one_hot = to_categorical(labels)

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels_one_hot, test_size=0.2, random_state=42)

# Function to build and evaluate the model
def predict_test(train_data, train_labels, test_data):
    # Build the LSTM model
    model = Sequential()

    # Add LSTM layer(s)
    model.add(LSTM(64, input_shape=(train_data.shape[1], train_data.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))  # Dropout to prevent overfitting

    # Add the output layer (4 activity classes)
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=10, batch_size=64)

    # Predict on the test set
    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert one-hot to class labels
    return y_pred_classes

# This block runs when the script is executed directly
if __name__ == "__main__":
    # Make predictions on the test data
    y_pred_classes = predict_test(X_train, y_train, X_test)

    # Convert the true labels to class labels
    y_true_classes = np.argmax(y_test, axis=1)

    # Calculate the Micro and Macro F1 scores
    f1_micro = f1_score(y_true_classes, y_pred_classes, average='micro')
    f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro')

    # Print the F1 scores
    print(f"Micro-averaged F1 score: {f1_micro}")
    print(f"Macro-averaged F1 score: {f1_macro}")
