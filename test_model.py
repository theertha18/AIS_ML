import numpy as np
import pandas as pd
import time
import pickle
from collections import Counter
from keras.models import load_model

if __name__ == "__main__":
    # Load the new data from another CSV file
    new_data = pd.read_csv(r"D:\AIS_ML\Dataset\adc_1m_hard_surface.csv")

    # Preprocess the new data (assuming similar preprocessing steps as the training data)
    new_data = new_data.iloc[:, 16:]  # Assuming similar data selection

    # Load the saved model



    model = load_model(r'D:\AIS_ML\Output\cnn_model_1.h5')

    # Make predictions using the loaded model
    new_data_values = np.array(new_data.values, dtype=int)  # Convert to NumPy array
    predictions = model.predict(new_data_values)

    # Post-process predictions if necessary
    predicted_classes = np.argmax(predictions, axis=1)

    # Print or save the predicted classes
    print("Predicted classes for new data:")
    print(predicted_classes)
    predicted_labels = ["hit" if prediction == 1 else "fail" for prediction in predicted_classes]
    print("Predicted labels for new data:")
    print(predicted_labels)

    # Count the occurrences of each predicted label
    label_counts = Counter(predicted_labels)

    # Print the count of each predicted label
    for label, count in label_counts.items():
        print(f"Predicted label '{label}': {count} instances")

