from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def predict_test(train_data, train_labels, test_data):
    # Preprocessing: Scale the data
    scaler = StandardScaler()
    
    # Reshape to 2D for scaling (flatten)
    X_train = train_data.reshape(train_data.shape[0], -1)
    X_test = test_data.reshape(test_data.shape[0], -1)

    # Apply StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape back to 3D for CNN (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], 60, 6))  # 60 time steps, 6 features
    X_test = X_test.reshape((X_test.shape[0], 60, 6))  # Same reshaping for test set

    # Adjust labels for one-hot encoding
    train_labels = train_labels - 1  # Shift labels from [1, 4] to [0, 3]
    y_train = to_categorical(train_labels, num_classes=4)

    # Augment training data by adding Gaussian noise
    def augment_data(X, y):
        noise = np.random.normal(0, 0.01, X.shape)
        scale = np.random.uniform(0.9, 1.1, size=(X.shape[0], 1, 1))
        X_noisy = X + noise
        X_scaled = X * scale
        X_augmented = np.concatenate([X, X_noisy, X_scaled], axis=0)
        y_augmented = np.concatenate([y, y, y], axis=0)
        return X_augmented, y_augmented

    X_train, y_train = augment_data(X_train, y_train)

    # Compute class weights for handling class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Define the CNN model
    model = Sequential()

    # Convolutional layers with Batch Normalization and Dropout
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=(60, 6)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Reduced dropout rate

    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Reduced dropout rate

    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))  # Reduced dropout rate

    # Global pooling
    model.add(GlobalAveragePooling1D())

    # Dense layers with Dropout
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))  # Moderate dropout rate
    model.add(Dense(4, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])

    # Training callbacks
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

    # Fit the model
    model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, verbose=1,
               class_weight=class_weight_dict)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert softmax output to class labels

    # Shift back to original labels (1-4 instead of 0-3)
    y_pred_classes = y_pred_classes + 1

    return y_pred_classes
