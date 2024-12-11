import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

def predict_test(train_data, train_labels, test_data, n_splits=5):
    # Standardize the data: Normalize each axis (column-wise) to zero mean and unit variance
    scaler = StandardScaler()
    train_data = train_data.reshape(-1, 6)  # Flatten the last two dimensions to apply scaling
    train_data = scaler.fit_transform(train_data)
    train_data = train_data.reshape(-1, 60, 6)  # Reshape back to (num_samples, 60, 6)
    
    # Prepare test data
    test_data = test_data.reshape(-1, 6)
    test_data = scaler.transform(test_data)  # Only transform the test data
    test_data = test_data.reshape(-1, 60, 6)

    # One-hot encode the labels (4 classes)
    train_labels = train_labels - 1  # Assuming labels are 1-4, convert to 0-3 for one-hot encoding
    train_labels = to_categorical(train_labels, num_classes=4)

    # Define the LSTM model creation function
    def create_lstm_model():
        model = models.Sequential()
        model.add(layers.LSTM(128, input_shape=(60, 6), activation='relu', return_sequences=True))
        model.add(layers.Dropout(0.5))
        model.add(layers.LSTM(64, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))  # 4 output classes
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)  # Use 5-fold cross-validation
    f1_scores = []

    # Perform cross-validation
    for train_idx, val_idx in tscv.split(train_data):
        X_train, X_val = train_data[train_idx], train_data[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]

        # Create a new LSTM model for each fold
        model = create_lstm_model()

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), verbose=1)

        # Evaluate the model on the validation set
        val_preds = model.predict(X_val)
        val_preds_labels = np.argmax(val_preds, axis=1)
        val_true_labels = np.argmax(y_val, axis=1)

        # Compute classification report and F1 score for the fold
        report = classification_report(val_true_labels, val_preds_labels, output_dict=True)
        f1_scores.append(report['accuracy'])  # You can also append other metrics like f1-score, precision, recall


    # Final model training on all training data for predictions on test data
    final_model = create_lstm_model()
    final_model.fit(train_data, train_labels, epochs=50, batch_size=64, verbose=1)

    # Predict the labels for the test data
    predictions = final_model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1) + 1  # Convert one-hot to integer labels (1-4)

    # Return the predicted labels for the test data and the mean F1 score from cross-validation
    return predicted_labels

# Example of calling the function
# train_data, train_labels, test_data are your data arrays
# predicted_labels, mean_f1 = predict_test(train_data, train_labels, test_data)
