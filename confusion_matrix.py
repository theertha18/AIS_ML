import numpy as np
import pandas as pd
import time
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from CNN_Model import *
from XG_Boost import *
import pickle


if __name__ == "__main__":
    start_time = time.time()
    # Load the dataset
    df = pd.read_csv(r'D:\AIS_ML\Dataset\adc_1m_hard_surface_2.csv')
    df = df.iloc[:, 16:]
    distances_highest_peak = np.zeros((len(df.index),), dtype=float)
    for i, row in enumerate(df.values):
        distances_highest_peak[i] = calculate_distance(row)
    print("Labels for the first 1000 rows:", distances_highest_peak[:len(df.index)])

    # Assuming dMAN is a predefined value
    dMAN = 1  # Manually measured distance
    group_labels = label_groups(distances_highest_peak, dMAN)
    print("Labels for the first 1000 rows:", group_labels[:len(df.index)])

    # Count the number of hits and fails
    num_hits = group_labels.count("hit")
    num_fails = group_labels.count("fail")

    # Print the counts
    print("Number of hits:", num_hits)
    print("Number of fails:", num_fails)

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(df.values, group_labels, test_size=0.2, random_state=42)

    # Convert DataFrames to NumPy arrays with integer dtype
    x_test = np.array(x_test, dtype=int)

    # Convert labels to one-hot encoded vectors
    y_test = pd.get_dummies(y_test).values

    # Load the saved model
    # model = tf.keras.models.load_model(r'D:\AIS_ML\Output\cnn_model_1.h5')
    with open(r'D:\AIS_ML\Output\xg_boost_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Evaluate the model on test data
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(np.argmax(y_test, axis=1), predicted_classes)
    f1 = f1_score(np.argmax(y_test, axis=1), predicted_classes, average='weighted')
    recall = recall_score(np.argmax(y_test, axis=1), predicted_classes, average='weighted')
    tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis=1), predicted_classes).ravel()

    # Print evaluation metrics
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Recall:", recall)
    print("True Positives:", tp)
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)

    # Plot confusion matrix
    plot_confusion_matrix(np.argmax(y_test, axis=1), predicted_classes)

    # Generate classification report
    report = classification_report(np.argmax(y_test, axis=1), predicted_classes)
    print("Classification Report:")
    print(report)

    # Count how many samples were classified into different labels
    unique_labels, counts = np.unique(predictions, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    hits_count = label_counts.get(1.0, 0)
    fails_count = sum(label_counts.values()) - hits_count

    print("Hits count:", hits_count)
    print("Fails count:", fails_count)

    end_time = time.time()
    print("Time taken to calculate distances and evaluate the model:", end_time - start_time, "seconds")
