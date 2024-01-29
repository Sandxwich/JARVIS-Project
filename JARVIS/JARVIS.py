import numpy as np

import tensorflow as tf
from tensorflow.keras import models

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

commands = ['left', 'down', 'stop', 'up', 'right', 'no', 'go', 'yes']

loaded_model = tf.saved_model.load("saved")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    while True:
        command = predict_mic()
        if command == "stop":
            terminate()
            break
