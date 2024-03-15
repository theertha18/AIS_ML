from CNN_Model import *
import tensorflow as tf
import time
import numpy as np
import pandas as pd

# Assuming the following functions are defined as in your previous code:
# read_and_prepare_data(dataset_path)
# reduce_noise_and_label(df, dt)
# apply_window(signal, window_type='hanning')

def predict_peaks(csv_path, model_path, window_width, Fs):
    """
    Predicts peaks for signals in a given CSV file using a pre-trained model.

    Parameters:
    - csv_path: Path to the CSV file containing new signals.
    - model_path: Path to the saved Keras model.
    - window_width: Width of the time window used in training.
    - Fs: Sampling frequency used in training.

    Returns:
    - peak_predictions: Predicted peak positions for each signal.
    """
    dt = 1 / Fs
    df = read_and_prepare_data(csv_path)
    _, _, filtered_signals = reduce_noise_and_label(df, dt)

    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Predict using the model
    predictions = model.predict(filtered_signals)

    # Convert predictions to peak positions or other relevant metric
    peak_predictions = np.argmax(predictions, axis=1) * window_width + 50

    return peak_predictions

if __name__ == "__main__":
    start_time = time.time()
    saved_model_path = r'D:\AIS_ML\AIS_ML\Output\cnn_model_test.h5'
    dataset_path = r'D:\AIS_ML\AIS_ML\Dataset\adc_1m_hard_surface_2.csv'
    window_width = 64
    Fs = 1953125  # Sampling frequency in Hz
    dt = 1 / Fs
    df = read_and_prepare_data(dataset_path)
    print(df)
    peaks, distances, filtered_signals = reduce_noise_and_label(df, dt)
    signal_length = len(df.columns)
    y_label = group_labeled_data(peaks, signal_length, window_width)
    x_data = filtered_signals

    peak_predictions = predict_peaks(dataset_path, saved_model_path, window_width, Fs)
    print("Predicted peak positions:", peak_predictions)

    peak_differences = np.abs(peak_predictions - peaks)
    print("Differences between actual and predicted peak positions:", peak_differences)

    end_time = time.time()
    print("Time taken to predict:", end_time - start_time, "seconds")

