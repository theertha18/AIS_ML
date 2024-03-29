# Reliability Test and Improvement of a Sensor System for Object Detection

## Overview
This project focuses on enhancing the reliability of a sensor system used for object detection. It involves various machine learning models and data processing scripts to analyze and improve the sensor's detection capabilities.

## Installation

 **Clone the repository:**
```bash
   git clone <repository-url>
```
Ensure Python 3.9 or above is installed on your system. You can download it from the official Python website.
Install the required packages:
```bash
   pip install -r requirements.txt
```

## Scripts Description

- **XG_Boost.py**: Implements an XGBoost model for sensor data analysis. It includes data preprocessing, model training, and evaluation functionalities.

- **ADC_FFT_Plot.py**: Processes ADC data and applies FFT for frequency analysis. It includes functionalities for data visualization.

- **CNN_Model.py**: Defines and trains a CNN model for object detection using sensor data, including data preparation and model evaluation steps.

- **Evaluation_Metrics.py**: Provides evaluation metrics for assessing machine learning model performance.

- **Merge_CSV.py**: Merges multiple CSV files into one, facilitating data consolidation.

- **Random_Forest.py**: Employs a Random Forest classifier for sensor data analysis, featuring model training and evaluation.

- **test_model.py**: Tests a pre-trained model's performance on a new dataset, including data preprocessing and evaluation.

- **TXT_to_CSV_converter.py**: Converts text files to CSV format, aiding in data preprocessing.

- **ADC_To_FFT_Plot_Enveloped.py**: This code takes a CSV file containing signals, filters each signal using Fourier Transform and Power Spectral Density, detects peaks in the envelope of the filtered signal,                                         plots the original and zoomed-in versions of the filtered signal with its envelope and peaks, and saves the plots as image files.
- **app.py**: This Flask web application serves as an interface for uploading CSV files containing signals, predicts peaks in the signals using a pre-trained CNN model, and displays the predicted peak positions 
              and distances from peaks on the web page while also calling an external Python script for further processing.
- **index.html**: This HTML template provides a user interface for uploading files, displaying predicted peak positions, windows, and distances from peaks, and rendering a plot image if available.  

## Usage
Each script can be run individually, depending on the specific needs of the analysis or model training process. For example, to run the CNN model training script, use:
```bash
   python CNN_Model.py
```
For GUI, you can run:
```bash
   python app.py
```
Ensure the necessary data files are in the correct directories as expected by each script.
