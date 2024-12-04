import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical

# Load your IMU data (example file names)
acc_x_train = pd.read_csv('Acc_x_train_1.csv', header=None).values
acc_y_train = pd.read_csv('Acc_y_train_1.csv', header=None).values
acc_z_train = pd.read_csv('Acc_z_train_1.csv', header=None).values
gyro_x_train = pd.read_csv('Gyr_x_train_1.csv', header=None).values
gyro_y_train = pd.read_csv('Gyr_y_train_1.csv', header=None).values
gyro_z_train = pd.read_csv('Gyr_z_train_1.csv', header=None).values

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

# Debugging: Check the shapes of your data
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Check if X_train has the correct shape for LSTM input (samples, time_steps, features)
# Example: X_train.shape should be (samples, time_steps, features)
if len(X_train.shape) != 3:
    raise ValueError(f"X_train has incorrect shape: {X_train.shape}. Expected shape: (samples, time_steps, features).")

# Check if y_train has the correct shape (samples, num_classes)
if len(y_train.shape) != 2:
    raise ValueError(f"y_train has incorrect shape: {y_train.shape}. Expected shape: (samples, num_classes).")

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Number of classes in the output layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Predict on the test data
y_pred = model.predict(X_test)

# Convert predictions to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate F1 score
f1_micro = f1_score(y_test_classes, y_pred_classes, average='micro')
f1_macro = f1_score(y_test_classes, y_pred_classes, average='macro')

# Print the F1 scores
print(f"Micro-averaged F1 score: {f1_micro}")
print(f"Macro-averaged F1 score: {f1_macro}")
