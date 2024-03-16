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

## Usage
Each script can be run individually, depending on the specific needs of the analysis or model training process. For example, to run the CNN model training script, use:
```bash
python CNN_Model.py
```
Ensure the necessary data files are in the correct directories as expected by each script.
