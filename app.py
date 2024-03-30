from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import numpy as np
import time
import tensorflow as tf
import subprocess
from CNN_Model import read_and_prepare_data, reduce_noise_and_label

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'

# Use relative paths if possible
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Dataset')
MODEL_PATH = os.path.join(os.getcwd(), 'Output', 'cnn_model.h5')

# Or use environment variables
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER')
MODEL_PATH = os.environ.get('MODEL_PATH')

# If environment variables are not set, fallback to default paths
if not UPLOAD_FOLDER:
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Dataset')
if not MODEL_PATH:
    MODEL_PATH = os.path.join(os.getcwd(), 'Output', 'cnn_model.h5')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_PATH'] = MODEL_PATH

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

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
    windows = np.argmax(predictions, axis=1)

    # Convert predictions to peak positions
    peak_positions = windows * window_width

    # Calculate distances from predicted peaks (assuming speed of sound is 343 m/s)
    distances_from_peaks = peak_positions * dt * 343

    return peak_positions, windows, distances_from_peaks

def call_adc_to_fft():
    script_path = os.path.join(os.getcwd(), 'ADC_To_FFT_Plot_Enveloped.py')
    subprocess.run(['python', script_path])

@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    peak_positions = None
    windows = None
    distances_from_peaks = None
    if request.method == 'POST' and form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        model_path = app.config['MODEL_PATH']
        window_width = 64
        Fs = 1953125
        peak_positions, windows, distances_from_peaks = predict_peaks(csv_path, model_path, window_width, Fs)

        # Call adc_to_fft.py script
        call_adc_to_fft()

    return render_template('index.html', form=form, peak_positions=peak_positions, windows=windows, distances_from_peaks=distances_from_peaks)

if __name__ == '__main__':
    app.run(debug=True)
