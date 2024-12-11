# -*- coding: utf-8 -*-
"""
Script used for final evaluation of classifier accuracy using a train-test split

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from Classify_activity import predict_test
from sklearn.metrics import classification_report, confusion_matrix

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_1_suffix = '_train_1.csv'
train_2_suffix = '_train_2.csv'

def load_sensor_data(sensor_names, suffix):
    data_slice_0 = np.loadtxt(sensor_names[0] + suffix, delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                     len(sensor_names)))
    data[:, :, 0] = data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
            sensor_names[sensor_index] + suffix, delimiter=',')
    
    return data
    
# Load labels and sensor data into 3-D array
train_1_labels = np.loadtxt('labels' + train_1_suffix, dtype='int')
train_1_data = load_sensor_data(sensor_names, train_1_suffix)
train_2_labels = np.loadtxt('labels' + train_2_suffix, dtype='int')
train_2_data = load_sensor_data(sensor_names, train_2_suffix)

# Combine training data
train_labels = np.hstack((train_1_labels, train_2_labels))
train_data = np.vstack((train_1_data, train_2_data))

# Create train-test split
train_data_split, test_data_split, train_labels_split, test_labels_split = train_test_split(
    train_data, train_labels, test_size=0.2, random_state= 42, shuffle=False)

# Predict activities on test data
test_outputs = predict_test(train_data_split, train_labels_split, test_data_split)

# Compute micro and macro-averaged F1 scores
micro_f1 = f1_score(test_labels_split, test_outputs, average='micro')
macro_f1 = f1_score(test_labels_split, test_outputs, average='macro')
print(f'Micro-averaged F1 score: {micro_f1}')
print(f'Macro-averaged F1 score: {macro_f1}')

# Examine outputs compared to labels
n_test = test_labels_split.size
plt.subplot(2, 1, 1)
plt.plot(np.arange(n_test), test_labels_split, 'b.')
plt.xlabel('Time window')
plt.ylabel('Target')
plt.subplot(2, 1, 2)
plt.plot(np.arange(n_test), test_outputs, 'r.')
plt.xlabel('Time window')
plt.ylabel('Output (predicted target)')
plt.show()

# Compute per-class metrics
report = classification_report(test_labels_split, test_outputs, output_dict=True)
f1_scores = [report[str(label)]['f1-score'] for label in range(1, 5)]  # Adjust range for your labels

    # Visualize per-class F1 scores
plt.figure(figsize=(8, 6))
plt.bar(range(1, 5), f1_scores, tick_label=[f'Class {i}' for i in range(1, 5)])
plt.title("Per-Class F1 Scores")
plt.xlabel("Class")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Confusion matrix and classification report
print("Classification Report:")
print(classification_report(test_labels_split, test_outputs))
cm = confusion_matrix(test_labels_split, test_outputs)
print("Confusion Matrix:")
print(cm)