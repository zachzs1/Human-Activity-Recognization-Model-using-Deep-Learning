from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for tackling class imbalance.
    Args:
        gamma (float): Focusing parameter for hard-to-classify examples.
        alpha (float): Weighting factor for balancing class contributions.
    Returns:
        Loss function.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        cross_entropy = -y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    return loss_fn


def augment_data(X, y):
    """
    Augment data for underrepresented classes with jittering and scaling.
    Args:
        X: Feature data.
        y: Corresponding labels.
    Returns:
        Augmented data and labels.
    """
    noise = np.random.normal(0, 0.02, X.shape)
    scaling = np.random.uniform(0.9, 1.1, size=(X.shape[0], 1, 1))
    X_noisy = X + noise
    X_scaled = X * scaling
    X_augmented = np.concatenate([X, X_noisy, X_scaled], axis=0)
    y_augmented = np.concatenate([y, y, y], axis=0)
    return X_augmented, y_augmented


def predict_test(train_data, train_labels, test_data):
    """
    Train and test a CNN model with class balancing and data augmentation.
    Args:
        train_data: Training feature data.
        train_labels: Corresponding labels for training data.
        test_data: Test feature data to classify.
    Returns:
        Predicted classes for the test data.
    """
    # Preprocessing: Scale the data
    scaler = StandardScaler()
    X_train = train_data.reshape(train_data.shape[0], -1)
    X_test = test_data.reshape(test_data.shape[0], -1)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 60, 6))
    X_test = X_test.reshape((X_test.shape[0], 60, 6))

    # Adjust labels to one-hot encoding
    train_labels = train_labels - 1  # Shift labels to zero-indexed
    y_train = to_categorical(train_labels, num_classes=4)

    # Apply SMOTE for oversampling
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    y_train_labels = np.argmax(y_train, axis=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train_labels)

    # Reshape data back to original format and one-hot encode labels
    X_train = X_train_resampled.reshape((-1, 60, 6))
    y_train = to_categorical(y_train_resampled, num_classes=4)

    # Augment data
    X_train, y_train = augment_data(X_train, y_train)

    # Define the CNN model
    model = Sequential()

    # Convolutional layers with L2 regularization and Batch Normalization
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu', kernel_regularizer=l2(0.01), input_shape=(60, 6)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    '''
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    '''
    # Global pooling and dense layers
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(loss=focal_loss(gamma=2., alpha=0.25), optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    # Training callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1, callbacks=[reduce_lr])

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) + 1

    return y_pred_classes
