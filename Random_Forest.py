import numpy as np
import pandas as pd
from scipy.signal import hilbert, get_window, find_peaks
import time, pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report

def read_and_prepare_data(dataset_path):
    dataframe = pd.read_csv(dataset_path)
    df = dataframe.iloc[:, 16:]
    return df

def apply_window(signal, window_type='hann'):
    # Ensure the window type is a valid string or tuple as required by get_window
    if not isinstance(window_type, (str, tuple)):
        raise ValueError("Window type must be a string or a tuple")
    # Generate the window using scipy's get_window function
    window = get_window(window_type, len(signal))
    return signal * window

def reduce_noise_and_label(df, dt):
    distances = np.zeros((len(df.index),), dtype=float)
    peaks_list = np.zeros((len(df.index),), dtype=int)
    filtered_signals = np.zeros(df.shape)  # Initialize array to store filtered signals

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

        filtered_signals[i, :] = ffilt.real  # Store the real part of the filtered signal

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

# Train Random Forest model
def train_model(xtrain, xtest, ytrain, ytest):
    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    rf_random.fit(xtrain, ytrain)
    best_params = rf_random.best_params_
    print("Best parameters:", best_params)
    rf_best = RandomForestClassifier(**best_params)
    rf_best.fit(xtrain, ytrain)

    # Save the model to a pickle file
    with open(r'D:\AIS_ML\AIS_ML\Output\random_forest.pkl', 'wb') as f:
        pickle.dump(rf_best, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Predict on the training and test sets
    RF_pred_test = rf_best.predict(xtest)
    RF_pred_train = rf_best.predict(xtrain)

    # Calculate F1 scores
    f1_train_rf = f1_score(y_true=ytrain, y_pred=RF_pred_train, average="weighted")
    f1_test_rf = f1_score(y_true=ytest, y_pred=RF_pred_test, average="weighted")
    print("Random Forest - Train Data F1 score:", f1_train_rf)
    print("Random Forest - Test Data F1 score:", f1_test_rf)
    return rf_best


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

