import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "best_model.h5")
label_path = os.path.join(BASE_DIR, "..", "models", "labels.npy")

model = load_model(model_path)
labels = np.load(label_path)


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,
        axis=0
    )
    return mfcc


def predict(file):
    try:
        # 🔹 Case 1: If file is a file-like object (Streamlit upload/record)
        if not isinstance(file, str):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file.read())
                file_path = tmp.name
        else:
            # 🔹 Case 2: If already a file path
            file_path = file

        features = extract_features(file_path)
        features = np.expand_dims(features, axis=0)

        prediction = model.predict(features)
        predicted_label = labels[np.argmax(prediction)]

        return predicted_label

    except Exception as e:
        return f"Error: {str(e)}"