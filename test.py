import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import time


def divide_into_sample_sets(data, sample_size):
    num_samples = len(data) // sample_size
    sample_sets = np.array_split(data, num_samples)
    return sample_sets


def convert_to_fft(sample_set):
    fft_data = np.fft.fft(sample_set)
    return fft_data


def label_data(sample_sets, threshold):
    labels = []
    for sample_set in sample_sets:
        # Convert ADC data to FFT data
        fft_data = convert_to_fft(sample_set)

        # Find peaks in the FFT data
        peaks, _ = find_peaks(np.abs(fft_data), height=threshold)

        if len(peaks) > 0:
            # Significant peak found, label as 1
            label = 1
        else:
            # No significant peak found, label as 0
            label = 0

        labels.append(label)
    return labels


def find_first_echo_peak(fft_data, threshold):
    peaks, _ = find_peaks(np.abs(fft_data), height=threshold)
    if len(peaks) > 0:
        first_peak_index = peaks[0]

        # Calculate distance to the first echo
        time_to_echo = first_peak_index / sample_rate
        distance = 0.5 * time_to_echo * velocity_of_sound
        return distance
    else:
        return None


# Example usage
start_time = time.time()  # Record the start time
df = pd.read_csv('D:\\AIS_ML\\Dataset\\adc_1m_hard_surface.csv')
df = df.iloc[:, 17:]
end_time = time.time()  # Record the end time

# Assuming you have your ADC data stored in a numpy array called 'adc_data'
sample_size = 128  # Adjust this as needed
threshold = 1000  # Adjust this threshold value based on your data
velocity_of_sound = 343  # Speed of sound in air in m/s
sample_rate = 1 / sample_size  # Assuming uniform sampling

# Divide ADC data into sample sets
adc_data = df.values.flatten()  # Assuming your data is stored in a DataFrame, this flattens it into a 1D array
sample_sets = divide_into_sample_sets(adc_data, sample_size)

# Label the sample sets
labels = label_data(sample_sets, threshold)

distances_to_echo = []

for sample_set in sample_sets:
    # Convert ADC data to FFT data
    fft_data = convert_to_fft(sample_set)

    # Find the first echo peak
    distance = find_first_echo_peak(fft_data, threshold)

    distances_to_echo.append(distance)

# Now distances_to_echo contains the calculated distances to the first echo for each sample set
# Print out the distances
print("Distances to first echo for each sample set:")
for i, distance in enumerate(distances_to_echo):
    print(f"Sample Set {i + 1}: {distance} meters")

# Print the time taken to read the CSV file
print("Time taken to read CSV file:", end_time - start_time, "seconds")
