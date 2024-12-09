# -*- coding: utf-8 -*-
"""
Script used for pre-submission evaluation of classifier accuracy on training
set 2

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from test4 import predict_test
from sklearn.metrics import classification_report, confusion_matrix

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_suffix = '_train_1.csv'
test_suffix = '_train_2.csv'

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
train_labels = np.loadtxt('labels' + train_suffix, dtype='int')
train_data = load_sensor_data(sensor_names, train_suffix)
test_labels = np.loadtxt('labels' + test_suffix, dtype='int')
test_data = load_sensor_data(sensor_names, test_suffix)

# Predict activities on test data
test_outputs = predict_test(train_data, train_labels, test_data)

# Compute micro and macro-averaged F1 scores
micro_f1 = f1_score(test_labels, test_outputs, average='micro')
macro_f1 = f1_score(test_labels, test_outputs, average='macro')
print(f'Micro-averaged F1 score: {micro_f1}')
print(f'Macro-averaged F1 score: {macro_f1}')

# Examine outputs compared to labels
n_test = test_labels.size
plt.subplot(2, 1, 1)
plt.plot(np.arange(n_test), test_labels, 'b.')
plt.xlabel('Time window')
plt.ylabel('Target')
plt.subplot(2, 1, 2)
plt.plot(np.arange(n_test), test_outputs, 'r.')
plt.xlabel('Time window')
plt.ylabel('Output (predicted target)')
plt.show()

 # Compute per-class metrics
report = classification_report(test_labels, test_outputs, output_dict=True)
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
print(classification_report(test_labels, test_outputs))
cm = confusion_matrix(test_labels, test_outputs)
print("Confusion Matrix:")
print(cm)