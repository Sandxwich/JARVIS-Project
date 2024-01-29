import tensorflow as tf
import numpy as np
import sounddevice as sd

# Function to convert audio waveform to spectrogram
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# Function to record audio from the microphone
def record_audio(duration=1, sample_rate=16000):
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

# Function to make a prediction from live audio
def predict_live_audio(model, label_names, duration=1, sample_rate=16000):
    live_audio = record_audio(duration, sample_rate)
    spectrogram = get_spectrogram(live_audio)
    spectrogram = spectrogram[tf.newaxis, ...]
    prediction = model(spectrogram)
    predicted_index = tf.argmax(prediction[0]).numpy()
    return label_names[predicted_index]

# Load the trained model (replace 'path/to/your/model' with the actual path)
model = tf.keras.models.load_model('path/to/your/model')

# Define your label names based on your training
label_names = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']

# Run the live prediction
print("Say something...")
predicted_command = predict_live_audio(model, label_names)
print("Predicted Command:", predicted_command)
