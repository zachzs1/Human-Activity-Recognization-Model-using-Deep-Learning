import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load data
acc_x_train = pd.read_csv('Acc_x_train_1.csv', header=None).values
acc_y_train = pd.read_csv('Acc_y_train_1.csv', header=None).values
acc_z_train = pd.read_csv('Acc_z_train_1.csv', header=None).values
gyro_x_train = pd.read_csv('Gyr_x_train_1.csv', header=None).values
gyro_y_train = pd.read_csv('Gyr_y_train_1.csv', header=None).values
gyro_z_train = pd.read_csv('Gyr_z_train_1.csv', header=None).values
labels_1 = pd.read_csv('labels_train_1.csv', header=None).values

acc_x_train_2 = pd.read_csv('Acc_x_train_2.csv', header=None).values
acc_y_train_2 = pd.read_csv('Acc_y_train_2.csv', header=None).values
acc_z_train_2 = pd.read_csv('Acc_z_train_2.csv', header=None).values
gyro_x_train_2 = pd.read_csv('Gyr_x_train_2.csv', header=None).values
gyro_y_train_2 = pd.read_csv('Gyr_y_train_2.csv', header=None).values
gyro_z_train_2 = pd.read_csv('Gyr_z_train_2.csv', header=None).values
labels_2 = pd.read_csv('labels_train_2.csv', header=None).values

# Combine sensor data for both datasets
acc_x_train_combined = np.vstack((acc_x_train, acc_x_train_2))
acc_y_train_combined = np.vstack((acc_y_train, acc_y_train_2))
acc_z_train_combined = np.vstack((acc_z_train, acc_z_train_2))
gyro_x_train_combined = np.vstack((gyro_x_train, gyro_x_train_2))
gyro_y_train_combined = np.vstack((gyro_y_train, gyro_y_train_2))
gyro_z_train_combined = np.vstack((gyro_z_train, gyro_z_train_2))

# Combine labels
labels = np.vstack((labels_1, labels_2))


# Combine sensor data into a single array
data = np.stack([acc_x_train_combined, acc_y_train_combined, acc_z_train_combined, gyro_x_train_combined, gyro_y_train_combined, gyro_z_train_combined], axis=-1)

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
