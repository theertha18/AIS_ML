import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model
from CNN_Model import *

# Load and prepare your data
dataset_path = r'D:\AIS_ML\AIS_ML\Dataset\adc_1m_hard_surface.csv'
window_width = 64
Fs = 1953125  # Sampling frequency in Hz
dt = 1 / Fs
df = read_and_prepare_data(dataset_path)

# Data preprocessing and label extraction
peaks, distances, filtered_signals = reduce_noise_and_label(df, dt)
y_label = group_labeled_data(peaks, len(df.columns), window_width)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(filtered_signals, y_label, test_size=0.2, random_state=42)

# Load the saved model
saved_model_path = r'D:\AIS_ML\AIS_ML\Output\cnn_model_test.h5'
model = tf.keras.models.load_model(saved_model_path)

# Make predictions on the test set
predictions = model.predict(x_test)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1) * window_width

# If your y_test is in one-hot encoded form, convert it to class labels
true_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

# Generate and plot the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Generate and print the classification report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes))

# Function to plot training and validation loss and accuracy - Requires training history
# def plot_history(history):
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Training vs. Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Training vs. Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
# Uncomment and use the following line after model training to visualize metrics
# plot_history(history)
