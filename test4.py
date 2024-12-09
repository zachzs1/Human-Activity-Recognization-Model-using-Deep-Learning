from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

def predict_test(train_data, train_labels, test_data):
    
    # Preprocessing: Scale the data
    scaler = StandardScaler()
    X_train = train_data.reshape(train_data.shape[0], -1)  # Flatten for scaling
    X_test = test_data.reshape(test_data.shape[0], -1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train.reshape((X_train.shape[0], 60, 6))  # Reshape back for CNN
    X_test = X_test.reshape((X_test.shape[0], 60, 6))
    
    # Adjust labels for one-hot encoding
    train_labels = train_labels - 1
    y_train = to_categorical(train_labels, num_classes=4)
    
    # Oversample Class 3 using SMOTE
    smote = SMOTE(sampling_strategy={2: max(np.bincount(train_labels))}, random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Flatten for SMOTE
    y_train_numeric = np.argmax(y_train, axis=1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train_numeric)
    X_train = X_train_resampled.reshape(-1, 60, 6)  # Reshape back to 3D
    y_train = to_categorical(y_train_resampled, num_classes=4)
    
    # Compute class weights for handling class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Define the CNN model
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=(60, 6)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])
    
    # Training callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
    
    # Fit the model
    model.fit(
        X_train, y_train,
        epochs=50, batch_size=64, validation_split=0.2, verbose=1,
        class_weight=class_weight_dict, callbacks=[reduce_lr]
    )
    
    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) + 1  # Convert back to 1-indexed labels
    
    return y_pred_classes

