import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.signal import welch
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

# Load data
acc_x_train = pd.read_csv('../Data/Acc_x_train_1.csv', header=None).values
acc_y_train = pd.read_csv('../Data/Acc_y_train_1.csv', header=None).values
acc_z_train = pd.read_csv('../Data/Acc_z_train_1.csv', header=None).values
gyro_x_train = pd.read_csv('../Data/Gyr_x_train_1.csv', header=None).values
gyro_y_train = pd.read_csv('../Data/Gyr_y_train_1.csv', header=None).values
gyro_z_train = pd.read_csv('../Data/Gyr_z_train_1.csv', header=None).values
labels_1 = pd.read_csv('../Data/labels_train_1.csv', header=None).values

acc_x_train_2 = pd.read_csv('../Data/Acc_x_train_2.csv', header=None).values
acc_y_train_2 = pd.read_csv('../Data/Acc_y_train_2.csv', header=None).values
acc_z_train_2 = pd.read_csv('../Data/Acc_z_train_2.csv', header=None).values
gyro_x_train_2 = pd.read_csv('../Data/Gyr_x_train_2.csv', header=None).values
gyro_y_train_2 = pd.read_csv('../Data/Gyr_y_train_2.csv', header=None).values
gyro_z_train_2 = pd.read_csv('../Data/Gyr_z_train_2.csv', header=None).values
labels_2 = pd.read_csv('../Data/labels_train_2.csv', header=None).values

# Combine sensor data for both datasets
acc_x_train_combined = np.vstack((acc_x_train, acc_x_train_2))
acc_y_train_combined = np.vstack((acc_y_train, acc_y_train_2))
acc_z_train_combined = np.vstack((acc_z_train, acc_z_train_2))
gyro_x_train_combined = np.vstack((gyro_x_train, gyro_x_train_2))
gyro_y_train_combined = np.vstack((gyro_y_train, gyro_y_train_2))
gyro_z_train_combined = np.vstack((gyro_z_train, gyro_z_train_2))

# Combine labels
labels = np.vstack((labels_1, labels_2))

# Combine sensor data into a single array (samples × time steps × channels)
data = np.stack(
    [acc_x_train_combined, acc_y_train_combined, acc_z_train_combined,
     gyro_x_train_combined, gyro_y_train_combined, gyro_z_train_combined], axis=2
)


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


# Visualization of SMOTE Process
def visualize_smote(raw_data, features, labels, target_class=4):
    """
    Visualize the effects of SMOTE augmentation on the dataset.
    """
    # Original class distribution
    labels_flat = labels.flatten()
    class_counts_original = pd.Series(labels_flat).value_counts().sort_index()

    print("Original Class Distribution:")
    print(class_counts_original)

    # Perform augmentation
    raw_resampled, features_resampled, labels_resampled = augment_data(raw_data, features, labels_flat, target_class)

    # Augmented class distribution
    labels_resampled_series = pd.Series(labels_resampled).value_counts().sort_index()

    print("\nAugmented Class Distribution:")
    print(labels_resampled_series)

    # Plot original vs augmented class distributions
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_counts_original.index, y=class_counts_original.values, color="blue", alpha=0.7,
                label="Original")
    sns.barplot(x=labels_resampled_series.index, y=labels_resampled_series.values, color="orange", alpha=0.7,
                label="Augmented")
    plt.title("Class Distribution Before and After SMOTE")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Visualize feature distributions before and after augmentation for a specific class
    target_class_idx = target_class - 1  # Convert class label to zero-indexed
    original_features = features[labels_flat == target_class_idx]
    augmented_features = features_resampled[labels_resampled == target_class_idx]

    feature_names = [f"Feature_{i}" for i in range(features.shape[1])]

    print("\nVisualizing Feature Distributions for Target Class:")
    for i, feature_name in enumerate(feature_names[:5]):  # Limit to first 5 features for clarity
        plt.figure(figsize=(8, 4))
        sns.kdeplot(original_features[:, i], color="blue", label="Original", fill=True, alpha=0.3)
        sns.kdeplot(augmented_features[:, i], color="orange", label="Augmented", fill=True, alpha=0.3)
        plt.title(f"Feature {i + 1} Distribution (Target Class {target_class})")
        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.legend()
        plt.show()


# Example Usage
# Assuming `data`, `features`, and `labels` are preloaded
# For this example, let's assume features are just a flattened version of the data
features = data.reshape(data.shape[0], -1)  # Flatten data into 2D features for SMOTE
visualize_smote(data, features, labels, target_class=4)


'''
# Define feature extraction functions
def compute_spectral_entropy(data):
    entropy_features = []
    for row in data:
        _, power_spectrum = welch(row, nperseg=min(len(row), 30))
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

# Apply feature engineering
entropy = compute_spectral_entropy(data_flat)
low_freq_power = compute_low_frequency_power(data_flat)
mid_freq_power = compute_mid_frequency_power(data_flat)
variance = compute_signal_variance(data_flat)

# Combine features into a single dataframe
features = np.hstack([entropy, low_freq_power, mid_freq_power, variance])
feature_names = ['Entropy', 'Low_Freq_Power', 'Mid_Freq_Power', 'Variance']
features_df = pd.DataFrame(features, columns=feature_names)

# Visualize each feature
print("\nVisualizing Each Feature Distribution:")
for feature_name in feature_names:
    sns.histplot(features_df[feature_name], kde=True)
    plt.title(f"Distribution of {feature_name}")
    plt.show()

# Pairplot of all extracted features
print("\nPairplot of Extracted Features:")
sns.pairplot(features_df)
plt.title("Pairplot of Extracted Features")
plt.show()

# Correlation heatmap
print("\nHeatmap of Feature Correlations:")
correlation_matrix = features_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap of Extracted Features")
plt.show()

# Z-Score for outlier detection
print("\nOutlier Detection with Z-Score:")
z_scores = np.abs(zscore(features_df))
outliers = np.any(z_scores > 3, axis=1)
print(f"Number of Outliers Detected: {np.sum(outliers)}")
sns.boxplot(data=features_df, orient="h")
plt.title("Boxplot of Extracted Features (Outliers Highlighted)")
plt.show()

# EDA Step 1: Initial Dataset Overview
print("1. Initial Dataset Overview")
print(f"Data shape: {data.shape}")  # Data shape
print(f"First 5 samples: {data[:5]}")  # First 5 samples

# Summary statistics (mean, std, min, max, etc.) for each feature across all time steps
print("\nSummary Statistics (mean, std, min, max, etc.):")
print(f"Mean: {data.mean(axis=0)}")
print(f"Standard Deviation: {data.std(axis=0)}")
print(f"Min: {data.min(axis=0)}")
print(f"Max: {data.max(axis=0)}")

# EDA Step 3: Visualizing Distributions of Features
print("\n3. Visualizing Distributions of Features")
# Visualize the distribution of a few random features (e.g., first 3 features)
for i in range(3):  # Change range to visualize more features if needed
    sns.displot(data[:, :, i].flatten(), kde=True)
    plt.title(f"Feature Distribution - {i+1}")
    plt.show()

# EDA Step 4: Correlation Matrix
print("\n4. Correlation Matrix")
# Flatten data into 2D (samples x features) for correlation calculation
data_flat = data.reshape(-1, data.shape[-1])
correlation_matrix = np.corrcoef(data_flat, rowvar=False)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

# EDA Step 5: Pair Plot (Scatterplot Matrix) - Limited to first 3 features
data_df = pd.DataFrame(data_flat[:, :3], columns=['Acc_X', 'Acc_Y', 'Acc_Z'])

# Visualize pairplot of the first 3 features (Accelerometer X, Y, Z)
sns.pairplot(data_df)
plt.title("Pairplot of Accelerometer Data")
plt.show()

# EDA Step 6: Box Plots to Check for Outliers (e.g., Acc_x_1)
print("\n6. Box Plots to Check for Outliers")
sns.boxplot(x=data[:, 0, 0])  # Visualize outliers for Acc_x_1
plt.title("Boxplot of Acc_x_1")
plt.show()

# EDA Step 7: Feature Distribution by Target Variable
print("\n7. Feature Distribution by Target Variable")
sns.boxplot(x=labels.flatten(), y=data[:, 0, 0].flatten())  # Distribution of Acc_x_1 by labels
plt.title("Boxplot of Acc_x_1 by Labels")
plt.show()

# EDA Step 8: Handling Categorical Variables (If applicable)
print("\n8. Handling Categorical Variables")
# Here, labels are categorical, so check their distribution
unique_labels, label_counts = np.unique(labels, return_counts=True)
print(f"Label distribution: {dict(zip(unique_labels, label_counts))}")
sns.countplot(x=labels.flatten())
plt.title("Label Distribution")
plt.show()

# EDA Step 9: Pairwise Relationships Between Features
data_df = pd.DataFrame(data_flat[:, :6], columns=['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])

# Visualize pairplot of the first 6 features (Accelerometer X, Y, Z, Gyroscope X,Y,Z)
sns.pairplot(data_df)
plt.title("Pairplot of First 6 Features")
plt.show()

# EDA Step 10: Feature Engineering (Create New Feature)
print("\n10. Feature Engineering (Create New Feature)")
new_feature = data[:, 0, 0] * data[:, 0, 1]  # Create a new feature by multiplying Acc_x_1 and Acc_y_1
sns.displot(new_feature.flatten(), kde=True)
plt.title("Distribution of New Feature (Acc_x_1 * Acc_y_1)")
plt.show()

# EDA Step 11: Checking for Class Imbalance
print("\n11. Checking for Class Imbalance")
print(f"Class distribution: {dict(zip(unique_labels, label_counts))}")
sns.countplot(x=labels.flatten())
plt.title("Class Imbalance Check")
plt.show()

# EDA Step 12: Outlier Detection Using Z-Score
print("\n12. Outlier Detection Using Z-Score")
z_scores = np.abs(zscore(data_flat, axis=0))
outliers = np.any(z_scores > 3, axis=1)  # Detect rows with outliers
print(f"Outliers detected: {np.sum(outliers)}")

# EDA Step 14: Normalization / Standardization
print("\n14. Normalization / Standardization")
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_flat)
sns.histplot(data_scaled[:, 0], kde=True)
plt.title("Normalized Distribution of First Feature")
plt.show()

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
data_minmax_scaled = scaler_minmax.fit_transform(data_flat)
sns.histplot(data_minmax_scaled[:, 0], kde=True)
plt.title("Min-Max Scaled Distribution of First Feature")
plt.show()


# EDA Step 15: Heatmap of Pairwise Correlations Between Features
print("\n15. Heatmap of Pairwise Correlations Between Features")
plt.figure(figsize=(12, 8))
sns.heatmap(np.corrcoef(data_flat, rowvar=False), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Pairwise Correlations Between Features")
plt.show()
'''