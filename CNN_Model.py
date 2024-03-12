import numpy as np
import pandas as pd
from scipy.signal import hilbert
import time
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Calculate distances
def calculate_distance(row, dt=0.35, threshold=1.5, velocity_of_sound=343):
    fhat = np.fft.fft(row)
    PSD = fhat * np.conj(fhat) / len(row)
    indices = PSD > threshold
    fhat = indices * fhat
    ffilt = np.fft.ifft(fhat)
    analytical_signal = hilbert(ffilt.real)
    env = np.abs(analytical_signal)
    highest_peak_index = np.argmax(env)
    pos_highest_peak = highest_peak_index * dt
    distance = 0.5 * pos_highest_peak * 2 * 1e-6 * velocity_of_sound
    return distance

# Define a function to label the groups
def label_groups(distances, dMAN):
    labels = []
    for distance in distances:
        if abs(distance - dMAN) < 0.01:  # Assuming dMAN is manually measured distance
            labels.append("hit")
        else:
            labels.append("fail")
    return labels


def train_model(xtrain, xtest, ytrain, ytest):
    verbose, epochs, batch_size = 1, 10, 32
    model = Sequential()
    # Model architecture
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(xtrain.shape[1], 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(1052, activation='relu'))
    model.add(Dense(len(ytrain[0]), activation='softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # Fit the model
    model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # Evaluate model
    _, accuracy = model.evaluate(xtest, ytest, batch_size=batch_size, verbose=verbose)
    accuracy = accuracy * 100.0
    print('Accuracy of Model: ', accuracy)
    # Save the model
    model.save(r'D:\AIS_ML\Output\cnn_model_2.h5')
    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    # Load the dataset
    df = pd.read_csv(r'D:\AIS_ML\AIS_ML\Dataset\adc_1m_hard_surface.csv')
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
    x_train, x_test, y_train, y_test = train_test_split(df, group_labels, test_size=0.2, random_state=42)
    x_train = np.array(x_train, dtype=int)
    x_test = np.array(x_test, dtype=int)
    # Convert labels to one-hot encoded vectors
    y_train = pd.get_dummies(y_train).values
    y_test = pd.get_dummies(y_test).values
    model = train_model(x_train, x_test, y_train, y_test)

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

    end_time = time.time()
    print("Time taken to calculate distances and train the model:", end_time - start_time, "seconds")
