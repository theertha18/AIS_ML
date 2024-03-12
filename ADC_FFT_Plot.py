# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert

# Load dataset from CSV into a DataFrame, selecting specific columns
df = pd.read_csv(r'D:\AIS_ML\AIS_ML\Dataset\adc_1m_hard_surface.csv')
df = df.iloc[:, 16:]

# Initialize an array to store peak data (not used in this snippet)
peak = np.zeros((len(df.index),), dtype=int)

# Iterate over each row in the DataFrame (each signal)
for i in range(len(df.index)):
    f = df.iloc[i, :]  # Extract the signal from the current row

    # Signal processing parameters
    n = len(f)  # Length of the signal
    Fs = 1953125  # Sampling frequency in Hz
    dt = 1 / Fs
    time = np.arange(n)  # Time array for plotting

    # Fourier Transform and Power Spectral Density (PSD) calculation
    fhat = np.fft.fft(f, n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n)
    L = np.arange(1, n // 2, dtype='int')

    # Filter the signal by setting a threshold on the PSD
    indices = PSD > 1.5  # Threshold for filtering
    PSDclean = PSD * indices  # Filtered PSD
    fhat = indices * fhat  # Filtered Fourier Transform
    ffilt = np.fft.ifft(fhat)  # Inverse FFT to get the filtered signal

    # Hilbert Transform to compute the analytical signal and its envelope
    analytical_signal = hilbert(ffilt.real)
    env = np.abs(analytical_signal)  # Envelope of the analytical signal

    # Peak detection in the envelope
    x, _ = find_peaks(env, distance=n)

    # Windowing the signal with a Hanning window
    window_size = 128  # Window size
    window = np.hanning(window_size)  # Hanning window
    segment = f.iloc[:window_size]  # First segment of the signal
    windowed_signal = segment * window  # Apply window to the segment

    # FFT of the windowed signal and its PSD
    fft_windowed = np.fft.fft(windowed_signal, n=window_size)
    fft_windowed_PSD = fft_windowed * np.conj(fft_windowed) / window_size
    freq_windowed = (1 / (dt * window_size)) * np.arange(window_size)
    L_windowed = np.arange(1, window_size // 2, dtype='int')

    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    # Plot 1: Original Noisy Signal
    axs[0].plot(time, f, label='Noisy')
    axs[0].set_xlim(time[0], time[-1])
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].set_title('Original Noisy Signal')

    # Plot 2: FFT of Noisy and Filtered Signal
    axs[1].plot(freq[L], PSD[L], color='c', linewidth=1, label='Noisy')
    axs[1].plot(freq[L], PSDclean[L], color='k', linewidth=1.5, label='Filtered')
    axs[1].set_xlim(freq[L[0]], freq[L[-1]])
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Power')
    axs[1].legend()
    axs[1].set_title('FFT of Noisy and Filtered Signal')

    # Plot 3: Filtered Signal with Envelope and Peaks
    axs[2].plot(time, ffilt, label='Filtered Signal')
    axs[2].plot(time, env, label='Envelope')
    axs[2].plot(x, env[x], "x", label='Peaks', color='red')
    axs[2].set_xlim(time[0], time[-1])
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Amplitude')
    axs[2].legend()
    axs[2].set_title('Filtered Signal with Envelope and Peaks')

    # Plot 4: Fourier Sub Scan Spectrum
    axs[3].plot(freq_windowed[L_windowed], np.abs(fft_windowed_PSD[L_windowed]))
    axs[3].set_xlim(freq_windowed[L_windowed[0]], freq_windowed[L_windowed[-1]])
    axs[3].set_xlabel('Frequency')
    axs[3].set_ylabel('Power')
    axs[3].set_title('Fourier Sub Scan Spectrum')

    plt.tight_layout()
    plt.show()

    # Stop after the first signal due to the break statement
    break
