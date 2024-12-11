import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def predict_test(train_data, train_labels, test_data):
    """
    Clusters the input 3D data into 4 clusters using k-means.

    Args:
    - data: np.ndarray of shape (n_samples, n_timesteps, n_features), where:
        * n_samples: number of samples (time windows).
        * n_timesteps: number of timesteps per sample.
        * n_features: number of features per timestep.
    
    Returns:
    - cluster_labels: np.ndarray of shape (n_samples,), cluster labels for each sample.
    """
    # Reshape 3D array to 2D: (n_samples, features)
    n_samples, n_timesteps, n_features = train_data.shape
    reshaped_data = train_data.reshape(n_samples, n_timesteps * n_features)
    
    # Normalize the features
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(reshaped_data)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_data)
    
    return cluster_labels

# Example usage:
# Assuming `test_data` is your 3D numpy array of shape (n_samples, n_timesteps, n_features)
# cluster_labels = predict_test(test_data)
