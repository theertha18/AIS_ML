import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
from keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import tensorflow as tf

def apply_window(signal, window_type='hanning'):
    if window_type == 'hanning':
        window = np.hanning(len(signal))
    else:
        raise ValueError("Unsupported window type")
    return signal * window

def read_and_prepare_data(dataset_path):
    dataframe = pd.read_csv(dataset_path)
    df = dataframe.iloc[:, 16:]
    return df

def reduce_noise_and_label(df, dt):
    distances = np.zeros((len(df.index),), dtype=float)
    peaks_list = np.zeros((len(df.index),), dtype=int)

    for i in range(len(df.index)):
        f = df.iloc[i, :]

        # Apply window function to the signal
        windowed_signal = apply_window(f)

        n = len(windowed_signal)
        time = np.arange(n) * dt  # Time array

        fhat = np.fft.fft(windowed_signal, n)
        PSD = fhat * np.conj(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)
        L = np.arange(1, n // 2, dtype='int')

        indices = PSD > 1.5  # Thresholding the Power Spectral Density
        PSDclean = PSD * indices
        fhat = indices * fhat
        ffilt = np.fft.ifft(fhat)

        analytical_signal = hilbert(ffilt.real)
        env = np.abs(analytical_signal)
        peaks, _ = find_peaks(env, distance=n)

        if len(peaks) > 0:
            pos_highest_peak = peaks[0]  # Position of the highest peak, assuming it's the first one
            distance = 0.5 * pos_highest_peak * dt * 2 * 343  # Calculate distance
            distances[i] = distance  # Store the distance for this signal
            peaks_list[i] = pos_highest_peak  # Store the peak position for this signal
            print(f"Row {i}: Distance = {distance} units, Peak Position = {pos_highest_peak}")
        else:
            distances[i] = np.nan  # Use NaN to indicate no peak/distance was detected
            peaks_list[i] = -1
            print(f"Row {i}: No peak detected.")
    return peaks_list, distances


def group_labeled_data(peaks_list, signal_length, window_width):
    """
    Groups the labeled data based on peak locations and window width.

    Parameters:
    - peaks_list: List of peak positions for each signal.
    - signal_length: Length of the signals.
    - window_width: Width of the time window to group peaks.

    Returns:
    - y_label: A binary matrix indicating the presence of a peak in each time window.
    """
    # Calculate the number of windows in each signal
    n_windows = signal_length // window_width
    print("No of windows", n_windows)

    # Initialize the label matrix
    y_label = np.zeros((len(peaks_list), n_windows), dtype=int)

    # Loop through each signal and label the windows based on peak presence
    for i, peak in enumerate(peaks_list):
        if peak >= 0:  # Check if a peak was detected
            # Determine which window the peak falls into
            window_index = peak // window_width
            print("window number where peaks fall into", window_index)
            # Set the corresponding label to 1
            if window_index < n_windows:
                y_label[i, window_index] = 1

    return y_label

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
    model.save(r'D:\AIS_ML\Output\cnn_model_1.h5')
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
    dataset_path = r'D:\AIS_ML\AIS_ML\Dataset\adc_1m_hard_surface.csv'
    window_width = 124
    Fs = 1953125  # Sampling frequency in Hz
    dt = 1 / Fs
    df = read_and_prepare_data(dataset_path)
    peaks, distances = 6
    signal_length = len(df.columns)
    y_label = group_labeled_data(peaks, signal_length, window_width)
    print(y_label)
    # x_data = df.values.reshape((df.shape[0], df.shape[1], 1))
    # print(x_data)
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.2, random_state=42)
    # model = train_model(x_train, x_test, y_train, y_test)
    # y_pred = model.predict(x_test)
    # plot_confusion_matrix(y_test, y_pred)

    end_time = time.time()
    print("Time taken to calculate distances and train the model:", end_time - start_time, "seconds")