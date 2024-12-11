import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import mode
import random

def predict_test(train_data, train_labels, test_data):
    #seed setting for consistent results
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    confidence_threshold=0.92
    smoothing_window=5
    def extract_features(data):
        #features extraction
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


        mag = np.sqrt(np.sum(data**2, axis=1))
        high_fft = np.abs(np.fft.rfft(data))[:, 15:30]

        entropy = compute_spectral_entropy(data)
        low_freq_power = compute_low_frequency_power(data)
        mid_freq_power = compute_mid_frequency_power(data)

        #combining existing features and the newly calculated ones
        return np.hstack([mag.reshape(-1, 1), high_fft, entropy, low_freq_power, mid_freq_power])

    #class imbalance handling
    def augment_data(raw_data, features, labels, target_class=4):
        raw_flat = raw_data.reshape(raw_data.shape[0], -1)
        smote = SMOTE(sampling_strategy={target_class - 1: 2 * np.sum(labels == target_class - 1)}, random_state=42)
        raw_resampled, labels_resampled = smote.fit_resample(raw_flat, labels)
        features_resampled, _ = smote.fit_resample(features, labels)

        raw_resampled = raw_resampled.reshape(-1, raw_data.shape[1], raw_data.shape[2])
        return raw_resampled, features_resampled, labels_resampled

    def create_multitask_model(input_shape_raw, input_shape_features, num_classes=4):
        #raw input
        input_raw = Input(shape=input_shape_raw, name="raw_input")
        x = layers.Conv1D(filters=64, kernel_size=3, activation="relu")(input_raw)
        x = layers.Conv1D(filters=128, kernel_size=5, activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        attention = layers.Attention()([x, x])
        x = layers.GlobalAveragePooling1D()(attention)

        #feautre input
        input_features = Input(shape=input_shape_features, name="feature_input")
        y = layers.Dense(256, activation="relu")(input_features)
        y = layers.Dropout(0.3, seed=42)(y)
        y = layers.Dense(128, activation="relu")(y)

        #combining
        combined = layers.Concatenate()([x, y])
        z = layers.Dense(128, activation="relu")(combined)
        z = layers.Dropout(0.3, seed=42)(z)

        #outputs
        main_output = layers.Dense(num_classes, activation="softmax", name="main_output")(z)
        aux_output_1 = layers.Dense(1, activation="sigmoid", name="class_1_aux")(z)
        aux_output_4 = layers.Dense(1, activation="sigmoid", name="class_4_aux")(z)

        #compiling the model
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

    #Post Processing:
    #confidence threshold
    def apply_confidence_threshold(predictions, class_idx, threshold):
        adjusted_predictions = predictions.copy()
        confident_indices = predictions[:, class_idx] >= threshold
        adjusted_predictions[confident_indices, :] = 0
        adjusted_predictions[confident_indices, class_idx] = 1
        return adjusted_predictions

    #smoothening for classes 1 and 4
    def smooth_predictions(predictions, window_size=5):
        smoothed = predictions.copy()
        half_window = window_size // 2
        for i in range(len(predictions)):
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            smoothed[i] = mode(predictions[start:end])[0][0]
        return smoothed

    train_labels = train_labels - 1

    #feature extraction
    train_features = np.hstack([
        extract_features(train_data[:, :, i]) for i in range(train_data.shape[2])
    ])
    test_features = np.hstack([
        extract_features(test_data[:, :, i]) for i in range(test_data.shape[2])
    ])

    #SMOTE
    raw_train, train_features, train_labels = augment_data(train_data, train_features, train_labels, target_class=4)

    #stratified training
    raw_train, raw_val, features_train, features_val, labels_train, labels_val = train_test_split(
        raw_train, train_features, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    #making the model multitask
    model = create_multitask_model(
        input_shape_raw=(train_data.shape[1], train_data.shape[2]),
        input_shape_features=(train_features.shape[1],),
        num_classes=4
    )

    #training the model
    model.fit(
        [raw_train, features_train],
        {"main_output": labels_train, "class_1_aux": (labels_train == 0).astype(int), "class_4_aux": (labels_train == 3).astype(int)},
        validation_data=(
            [raw_val, features_val],
            {"main_output": labels_val, "class_1_aux": (labels_val == 0).astype(int), "class_4_aux": (labels_val == 3).astype(int)},
        ),
        epochs= 50,
        batch_size=32,
        #comment this out to reduce messages in the terminal
        verbose=1
    )

    #final prediction
    predictions = model.predict([test_data, test_features])[0]
    adjusted_predictions = apply_confidence_threshold(predictions, class_idx=3, threshold=confidence_threshold)
    final_labels = np.argmax(adjusted_predictions, axis=1) + 1
    smoothed_labels = smooth_predictions(final_labels, window_size=smoothing_window)

    return smoothed_labels