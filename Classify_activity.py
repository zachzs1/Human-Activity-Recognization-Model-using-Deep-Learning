import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def predict_test(train_data, train_labels, test_data):
    # Preprocessing: Scale the data
    scaler = StandardScaler()

    # Reshape to 2D for scaling
    X_train = train_data.reshape(train_data.shape[0], -1)  # Flatten
    X_test = test_data.reshape(test_data.shape[0], -1)  # Flatten

    # Apply StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape back to 3D for LSTM
    X_train = X_train.reshape((X_train.shape[0], 60, 6))
    X_test = X_test.reshape((X_test.shape[0], 60, 6))

    # Adjust train_labels to be 0-indexed for one-hot encoding
    train_labels = train_labels - 1  # Shift labels from [1, 4] to [0, 3]

    # One-hot encode train_labels
    y_train = to_categorical(train_labels, num_classes=4)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))  # 64 neurons, ReLU activation
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))  # 4 output classes
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2, verbose=1)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_pred_classes = y_pred_classes + 1

    return y_pred_classes



if __name__ == "__main__":
    
    acc_x_train = pd.read_csv('Acc_x_train_1.csv', header=None).values
    acc_y_train = pd.read_csv('Acc_y_train_1.csv', header=None).values
    acc_z_train = pd.read_csv('Acc_z_train_1.csv', header=None).values
    gyro_x_train = pd.read_csv('Gyr_x_train_1.csv', header=None).values
    gyro_y_train = pd.read_csv('Gyr_y_train_1.csv', header=None).values
    gyro_z_train = pd.read_csv('Gyr_z_train_1.csv', header=None).values

    labels = pd.read_csv('labels_train_1.csv', header=None).values

    data = np.stack([acc_x_train, acc_y_train, acc_z_train, gyro_x_train, gyro_y_train, gyro_z_train], axis=-1)

    scaler = StandardScaler()
    data_reshaped = data.reshape(-1, data.shape[-1])  
    data_normalized = scaler.fit_transform(data_reshaped)
    data_normalized = data_normalized.reshape(data.shape)  

    labels_one_hot = to_categorical(labels)

    X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels_one_hot, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    if len(X_train.shape) != 3:
        raise ValueError(f"X_train has incorrect shape: {X_train.shape}. Expected shape: (samples, time_steps, features).")

    if len(y_train.shape) != 2:
        raise ValueError(f"y_train has incorrect shape: {y_train.shape}. Expected shape: (samples, num_classes).")

    y_pred_classes = predict_test(X_train, y_train, X_test)
    y_test_classes = np.argmax(y_test, axis=1)

    f1_micro = f1_score(y_test_classes, y_pred_classes, average='micro')
    f1_macro = f1_score(y_test_classes, y_pred_classes, average='macro')

    print(f"Micro-averaged F1 score: {f1_micro}")
    print(f"Macro-averaged F1 score: {f1_macro}")
