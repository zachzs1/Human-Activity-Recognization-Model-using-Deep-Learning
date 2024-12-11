import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import mode
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)


def extract_features(data):
    """
    Enhanced feature extraction with spectral entropy and low-frequency power.
    """

    def compute_spectral_entropy(data):
        from scipy.signal import welch
        entropy_features = []
        for row in data:
            _, power_spectrum = welch(row, nperseg=min(len(row), 60))
            power_spectrum = power_spectrum / np.sum(power_spectrum)
            entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-6))
            entropy_features.append(entropy)
        return np.array(entropy_features).reshape(-1, 1)

    def compute_low_frequency_power(data):
        low_freq_power = []
        for row in data:
            fft_coeffs = np.abs(np.fft.rfft(row))
            low_power = np.sum(fft_coeffs[:10])
            total_power = np.sum(fft_coeffs)
            low_freq_power.append(low_power / (total_power + 1e-6))
        return np.array(low_freq_power).reshape(-1, 1)

    def compute_mid_frequency_power(data):
        mid_freq_power = []
        for row in data:
            fft_coeffs = np.abs(np.fft.rfft(row))
            mid_power = np.sum(fft_coeffs[10:30])
            total_power = np.sum(fft_coeffs)
            mid_freq_power.append(mid_power / (total_power + 1e-6))
        return np.array(mid_freq_power).reshape(-1, 1)

    def compute_signal_variance(data):
        return np.var(data, axis=1).reshape(-1, 1)

    # Existing feature computation
    mag = np.sqrt(np.sum(data ** 2, axis=1))
    high_fft = np.abs(np.fft.rfft(data))[:, 15:30]

    # New feature computation
    entropy = compute_spectral_entropy(data)
    low_freq_power = compute_low_frequency_power(data)
    mid_freq_power = compute_mid_frequency_power(data)
    signal_variance = compute_signal_variance(data)

    # Combine all features
    return np.hstack([mag.reshape(-1, 1), high_fft, entropy, low_freq_power, mid_freq_power, signal_variance])


def augment_data(raw_data, features, labels, target_class=4):
    """
    Augment data for the target class using SMOTE, ensuring alignment between raw data and features.
    """
    raw_flat = raw_data.reshape(raw_data.shape[0], -1)
    smote = SMOTE(sampling_strategy={target_class - 1: 2 * np.sum(labels == target_class - 1)}, random_state=42)
    raw_resampled, labels_resampled = smote.fit_resample(raw_flat, labels)
    features_resampled, _ = smote.fit_resample(features, labels)

    # Reshape raw data back to 3D
    raw_resampled = raw_resampled.reshape(-1, raw_data.shape[1], raw_data.shape[2])
    return raw_resampled, features_resampled, labels_resampled


def create_multitask_model(input_shape_raw, input_shape_features, num_classes=4):
    # Raw Input
    input_raw = Input(shape=input_shape_raw, name="raw_input")
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(input_raw)
    x = layers.Conv1D(filters=128, kernel_size=5, activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    attention = layers.Attention()([x, x])
    x = layers.GlobalAveragePooling1D()(attention)

    # Feature Input
    input_features = Input(shape=input_shape_features, name="feature_input")
    y = layers.Dense(256, activation="relu")(input_features)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(128, activation="relu")(y)

    # Fusion
    combined = layers.Concatenate()([x, y])
    z = layers.Dense(128, activation="relu")(combined)
    z = layers.Dropout(0.3)(z)

    # Main Output
    main_output = layers.Dense(num_classes, activation="softmax", name="main_output")(z)

    # Auxiliary Outputs
    aux_output_1 = layers.Dense(1, activation="sigmoid", name="class_1_aux")(z)
    aux_output_4 = layers.Dense(1, activation="sigmoid", name="class_4_aux")(z)

    # Model
    model = models.Model(inputs=[input_raw, input_features], outputs=[main_output, aux_output_1, aux_output_4])
    model.compile(
        optimizer="adam",
        loss={
            "main_output": "sparse_categorical_crossentropy",
            "class_1_aux": "binary_crossentropy",
            "class_4_aux": "binary_crossentropy",
        },
        loss_weights={"main_output": 1.0, "class_1_aux": 0.5, "class_4_aux": 0.5},
        metrics=["accuracy"]
    )
    return model


def apply_confidence_threshold(predictions, class_idx, threshold):
    """
    Adjust predictions based on a confidence threshold for a specific class.

    Parameters:
    - predictions: np.ndarray of shape (num_samples, num_classes), containing probability scores.
    - class_idx: Index of the class to apply the threshold.
    - threshold: Confidence threshold for the class.

    Returns:
    - Adjusted predictions with the same shape as input.
    """
    adjusted_predictions = predictions.copy()
    confident_indices = predictions[:, class_idx] >= threshold

    # Set all probabilities to zero for confident predictions
    adjusted_predictions[confident_indices, :] = 0
    # Assign 100% confidence to the class index for confident predictions
    adjusted_predictions[confident_indices, class_idx] = 1
    return adjusted_predictions


def smooth_predictions(predictions, window_size=5):
    """
    Apply mode filtering to smooth predictions over a sliding window.

    Parameters:
    - predictions: 1D array of predicted class labels (e.g., [1, 2, 2, 4, ...]).
    - window_size: Size of the sliding window (must be odd for symmetric smoothing).

    Returns:
    - Smoothed predictions.
    """
    smoothed = predictions.copy()
    half_window = window_size // 2

    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        smoothed[i] = mode(predictions[start:end])[0][0]

    return smoothed


def plot_training_history(history):
    """
    Plots the training and validation accuracy over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['main_output_accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_main_output_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_test(train_data, train_labels, test_data, confidence_threshold=0.92, smoothing_window=5):
    """
    Train a multi-task model with enhanced features and refine predictions for Class 4.
    """
    train_labels = train_labels - 1

    # Feature extraction
    train_features = np.hstack([
        extract_features(train_data[:, :, i]) for i in range(train_data.shape[2])
    ])
    test_features = np.hstack([
        extract_features(test_data[:, :, i]) for i in range(test_data.shape[2])
    ])

    # Rebalance data for Class 4
    raw_train, train_features, train_labels = augment_data(train_data, train_features, train_labels, target_class=4)

    # Train-validation split
    raw_train, raw_val, features_train, features_val, labels_train, labels_val = train_test_split(
        raw_train, train_features, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    # Create multi-task model
    model = create_multitask_model(
        input_shape_raw=(train_data.shape[1], train_data.shape[2]),
        input_shape_features=(train_features.shape[1],),
        num_classes=4
    )

    # Train multi-task model
    history = model.fit(
        [raw_train, features_train],
        {"main_output": labels_train, "class_1_aux": (labels_train == 0).astype(int),
         "class_4_aux": (labels_train == 3).astype(int)},
        validation_data=(
            [raw_val, features_val],
            {"main_output": labels_val, "class_1_aux": (labels_val == 0).astype(int),
             "class_4_aux": (labels_val == 3).astype(int)},
        ),
        epochs=50,
        batch_size=32,
        verbose=1
    )

    plot_training_history(history)

    # Predict test data
    predictions = model.predict([test_data, test_features])[0]  # Main output
    # predictions[:, 3] *= 1.3  # Boost Class 4 probabilities
    adjusted_predictions = apply_confidence_threshold(predictions, class_idx=3, threshold=confidence_threshold)
    final_labels = np.argmax(adjusted_predictions, axis=1) + 1  # Convert to 1-4
    smoothed_labels = smooth_predictions(final_labels, window_size=smoothing_window)

    return smoothed_labels