# model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def predict_test(train_data, train_labels, test_data):
    """
    Function to train a Random Forest classifier and return predicted classes for the test data.
    
    Args:
    - train_data (numpy.ndarray or pandas.DataFrame): The training data features.
    - train_labels (numpy.ndarray or pandas.Series): The labels for the training data.
    - test_data (numpy.ndarray or pandas.DataFrame): The test data features for prediction.
    
    Returns:
    - predictions (numpy.ndarray): The predicted class labels for the test data.
    """
    
    # Ensure train_data and test_data are 2D (samples, features)
    # If the data is 3D (e.g., (samples, timesteps, features)), flatten it into 2D
    if len(train_data.shape) == 3:
        n_samples, n_timesteps, n_features = train_data.shape
        train_data = train_data.reshape(n_samples, n_timesteps * n_features)
    if len(test_data.shape) == 3:
        n_samples, n_timesteps, n_features = test_data.shape
        test_data = test_data.reshape(n_samples, n_timesteps * n_features)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data)
    X_test_scaled = scaler.transform(test_data)
    
    rf_model = RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=42)

    # Fit the grid search to the data
    rf_model.fit(X_train_scaled, train_labels)


    # Evaluate the best model
    y_pred = rf_model.predict(X_test_scaled)
        
    # Return the predicted classes
    return y_pred
