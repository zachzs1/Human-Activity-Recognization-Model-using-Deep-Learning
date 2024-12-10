from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier


def predict_test(train_data, train_labels, test_data):
    """
    XGBoost classifier for multi-class classification.
    """
    try:
        # Reshape and flatten data for compatibility with XGBoost
        train_data_flat = train_data.reshape(train_data.shape[0], -1)  # Flatten to 2D
        test_data_flat = test_data.reshape(test_data.shape[0], -1)    # Flatten to 2D

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_data_flat)
        X_test = scaler.transform(test_data_flat)

        # Adjust labels to be 0-indexed
        train_labels_adjusted = train_labels.flatten() - 1  # Shift labels from [1, 4] to [0, 3]

        # Define XGBoost model
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

        # Fit the model
        model.fit(X_train, train_labels_adjusted)

        # Predict on test data
        test_predictions = model.predict(X_test)

        # Convert back to original label range [1, 4]
        test_predictions += 1

        return test_predictions

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example usage with synthetic data (replace with actual data)
    np.random.seed(42)

    # Generate synthetic train and test data
    train_data = np.random.rand(1000, 60, 6)  # Replace with actual train data
    train_labels = np.random.randint(1, 5, size=(1000, 1))  # Replace with actual train labels
    test_data = np.random.rand(200, 60, 6)  # Replace with actual test data

    # Predict using logistic regression
    predictions = predict_test(train_data, train_labels, test_data)

    # Display predictions
    print(f"Predictions on test data: {predictions}")
