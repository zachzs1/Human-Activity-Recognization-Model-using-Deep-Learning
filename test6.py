import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, MaxPooling1D, LSTM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def predict_test(train_data, train_labels, test_data):
    # Standardize the data: Normalize each axis (column-wise) to zero mean and unit variance
    scaler = StandardScaler()
    train_data = train_data.reshape(-1, 6)  # Flatten the last two dimensions to apply scaling
    train_data = scaler.fit_transform(train_data)
    train_data = train_data.reshape(-1, 60, 6)  # Reshape back to (num_samples, 60, 6)

    test_data = test_data.reshape(-1, 6)
    test_data = scaler.transform(test_data)
    test_data = test_data.reshape(-1, 60, 6)

    # One-hot encode the labels (4 classes)
    train_labels = train_labels - 1
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=4)

    # Define the CNN model
    model = models.Sequential()
    '''    
    # 1st Convolutional Layer
    model.add(Conv1D(64, 3, activation='relu', input_shape=(60, 6)))
    model.add(MaxPooling1D(2))

    # 2nd Convolutional Layer
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(2))

    # 3rd Convolutional Layer (optional, can be fine-tuned)
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(MaxPooling1D(2))

    # Flatten the output from Conv1D layers
    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout for regularization

    # Output Layer (4 classes)
    model.add(layers.Dense(4, activation='softmax'))  # Softmax for multi-class classification
    ''' 
    model.add(LSTM(128, input_shape=(60, 6), activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels, epochs=50, batch_size=64, validation_split=0.4)

    # Predict the labels for the test data
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1) + 1  # Convert one-hot to integer labels

    # Return the predicted labels for the test data
    return predicted_labels
