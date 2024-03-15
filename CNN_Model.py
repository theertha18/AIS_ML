import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert, get_window
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from sklearn.metrics import classification_report

def read_and_prepare_data(dataset_path):
    """
        Reads a dataset from a CSV file and selects a specific portion of its columns.

        Parameters:
        - dataset_path: str, the file path to the CSV dataset.

        Returns:
        - DataFrame containing the selected columns of the original dataset.
        """
    # Load the dataset with no header, as we're assuming it doesn't have one
    dataframe = pd.read_csv(dataset_path, header=None)
    # Select columns starting from the 17th (index 16) onwards, assuming these contain the relevant data
    df = dataframe.iloc[:, 16:]
    return df

def apply_window(signal, window_type='hann'):
    """
    Applies a windowing function to a signal to reduce spectral leakage.

    Parameters:
    - signal: ndarray, the original signal to be windowed.
    - window_type: str or tuple, specifying the type of window to apply.

    Returns:
    - The windowed signal as an array.
    """
    # Validate window type input
    if not isinstance(window_type, (str, tuple)):
        raise ValueError("Window type must be a string or a tuple")
    # Generate the window based on the specified type and signal length
    window = get_window(window_type, len(signal))
    # Apply the window to the signal by element-wise multiplication
    return signal * window

def reduce_noise_and_label(df, dt):
    """
       Applies signal processing techniques to reduce noise, detect peaks, and calculate distances from a DataFrame of signals.

       Parameters:
       - df: DataFrame, containing signals in its rows.
       - dt: float, the time interval between signal samples.

       Returns:
       - peaks_list: ndarray, the positions of the highest peak in each signal.
       - distances: ndarray, the calculated distance for each signal based on the peak position.
       - filtered_signals: ndarray, the signals after noise reduction and filtering.
    """
    distances = np.zeros((len(df.index),), dtype=float)
    peaks_list = np.zeros((len(df.index),), dtype=int)
    filtered_signals = np.zeros(df.shape)  # Initialize array to store filtered signals

    for i in range(len(df.index)):
        f = df.iloc[i, :]

        # Apply window function to the signal to reduce edge effects
        windowed_signal = apply_window(f)

        # Fourier Transform for frequency analysis
        n = len(windowed_signal)
        fhat = np.fft.fft(windowed_signal, n)
        PSD = fhat * np.conj(fhat) / n

        indices = PSD > 1.5  # Thresholding the Power Spectral Density
        fhat = indices * fhat
        ffilt = np.fft.ifft(fhat) # Inverse FFT for filtered signal

        filtered_signals[i, :] = ffilt.real  # Store the filtered signal

        # Analyze the signal envelope to find peaks
        analytical_signal = hilbert(ffilt.real)
        env = np.abs(analytical_signal)
        peaks, _ = find_peaks(env, distance=n)

        # Calculate distance based on peak position
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
    return peaks_list, distances, filtered_signals

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

    # Initialize the label matrix
    y_label = np.zeros((len(peaks_list), n_windows), dtype=int)

    # Loop through each signal and label the windows based on peak presence
    for i, peak in enumerate(peaks_list):
        if peak >= 0:  # Check if a peak was detected
            # Determine which window the peak falls into
            window_index = peak // window_width
            # Set the corresponding label to 1
            if window_index < n_windows:
                y_label[i, window_index] = 1
    return y_label

def train_model(xtrain, xtest, ytrain, ytest):
    """
      Trains a Convolutional Neural Network model on the provided training data and evaluates its performance on the test data.

      Parameters:
      - xtrain: ndarray, training data features.
      - xtest: ndarray, test data features.
      - ytrain: ndarray, training data labels.
      - ytest: ndarray, test data labels.

      Returns:
      - The trained model.
      """
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
    model.save(r'D:\AIS_ML\AIS_ML\Output\cnn_model_test.h5')
    return model

if __name__ == "__main__":
    start_time = time.time()
    # Load the dataset
    dataset_path = r'D:\AIS_ML\AIS_ML\Dataset\adc_1m_hard_surface.csv'
    window_width = 64
    Fs = 1953125  # Sampling frequency in Hz
    dt = 1 / Fs
    df = read_and_prepare_data(dataset_path)
    peaks, distances, filtered_signals = reduce_noise_and_label(df, dt)
    signal_length = len(df.columns)
    y_label = group_labeled_data(peaks, signal_length, window_width)
    x_data = filtered_signals

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_label, test_size=0.2, random_state=42)
    model = train_model(x_train, x_test, y_train, y_test)
    y_pred = model.predict(x_test)
    predicted_classes = np.argmax(y_pred, axis=1)

    # Calculate and print classification report for detailed metrics
    report = classification_report(np.argmax(y_test, axis=1), predicted_classes, output_dict=True)
    print("Classification Report:")
    print(classification_report(np.argmax(y_test, axis=1), predicted_classes))

    # Extract and print specific metrics from the report
    print("Accuracy:", report['accuracy'])
    print("Weighted F1 Score:", report['weighted avg']['f1-score'])
    print("Weighted Recall:", report['weighted avg']['recall'])

    end_time = time.time()
    print("Time taken to calculate distances and train the model:", end_time - start_time, "seconds")
