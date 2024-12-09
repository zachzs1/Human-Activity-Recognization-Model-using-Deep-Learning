from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

# Function to smooth the predictions over a time window
def smooth_predictions(y_pred_classes, window_size=5):
    smoothed_predictions = []
    for i in range(len(y_pred_classes)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(y_pred_classes), i + window_size // 2 + 1)
        smoothed_predictions.append(np.argmax(np.bincount(y_pred_classes[start_idx:end_idx])))
    return np.array(smoothed_predictions)

# Main prediction function
def predict_test(train_data, train_labels, test_data):
    # Preprocessing: Scale the data
    scaler = StandardScaler()

    # Reshape to 2D for scaling (flatten)
    X_train = train_data.reshape(train_data.shape[0], -1)
    X_test = test_data.reshape(test_data.shape[0], -1)

    # Apply StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape back to 3D for CNN (time steps, features)
    X_train = X_train.reshape((X_train.shape[0], 60, 6))  # 60 time steps, 6 features
    X_test = X_test.reshape((X_test.shape[0], 60, 6))  # Same reshaping for test set

    # Adjust train_labels to be 0-indexed for one-hot encoding
    train_labels = train_labels - 1  # Shift labels from [1, 4] to [0, 3]
    y_train = to_categorical(train_labels, num_classes=4)

    # Define the CNN model for HAR
    model = Sequential()

    # Convolutional layers for feature extraction
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(BatchNormalization())  # Added Batch Normalization
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))  # Dropout for regularization
    
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())  # Added Batch Normalization
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))  # Dropout for regularization
    
    # Flatten the output to feed it into the Dense layers
    model.add(Flatten())
    
    # Dense layers for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Dense(4, activation='softmax'))  # 4 output classes (adjust as needed)
    
    # Compile the model with a fixed learning rate
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Fit the model with more epochs and without early stopping
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, class_weight=class_weight_dict, verbose=1)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert softmax output to class labels

    # Smooth predictions to reduce abrupt class changes
    smoothed_y_pred_classes = smooth_predictions(y_pred_classes, window_size=5)

    # Shift back to original labels (1-4 instead of 0-3)
    smoothed_y_pred_classes = smoothed_y_pred_classes + 1

    return smoothed_y_pred_classes
