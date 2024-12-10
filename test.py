import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np

# Function to smooth the predictions over a time window
def smooth_predictions(y_pred_classes, window_size=5):
    smoothed_predictions = []
    for i in range(len(y_pred_classes)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(y_pred_classes), i + window_size // 2 + 1)
        smoothed_predictions.append(np.argmax(np.bincount(y_pred_classes[start_idx:end_idx])))
    return np.array(smoothed_predictions)

def scale_data(train_data, test_data):
    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler on the training data and transform it
    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)
    
    # Transform the test data using the same scaler
    test_data_scaled = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

    return train_data_scaled, test_data_scaled

# Main prediction function
def predict_test(train_data, train_labels, test_data):

    train_data_scaled, test_data_scaled = scale_data(train_data, test_data)
    
    # Adjust train_labels to be 0-indexed for one-hot encoding
    train_labels = train_labels - 1  # Shift labels from [1, 4] to [0, 3]
    y_train = to_categorical(train_labels, num_classes=4)

    # Define the CNN-LSTM model
    model = Sequential()
    
    # Convolutional layers for feature extraction
    
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=(60, 6)))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    # Fit the model with callbacks
    model.fit(train_data_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

    # Predict on test data
    y_pred = model.predict(test_data_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert softmax output to class labels

    # Shift back to original labels (1-4 instead of 0-3)
    y_pred_classes = y_pred_classes + 1

    return y_pred_classes



if __name__ == "__main__":
    
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

    # Make predictions on the test data
    y_pred_classes = predict_test(X_train, y_train, X_test)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate F1 score
    f1_micro = f1_score(y_test_classes, y_pred_classes, average='micro')
    f1_macro = f1_score(y_test_classes, y_pred_classes, average='macro')

    # Print the F1 scores
    print(f"Micro-averaged F1 score: {f1_micro}")
    print(f"Macro-averaged F1 score: {f1_macro}")
    '''
    # Examine outputs compared to labels
    n_test = test_labels.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n_test), test_labels, 'b.')
    plt.xlabel('Time window')
    plt.ylabel('Target')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(n_test), test_outputs, 'r.')
    plt.xlabel('Time window')
    plt.ylabel('Output (predicted target)')
    plt.show()
    '''