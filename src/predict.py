import numpy as np
import librosa
import os
import tempfile
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "rf_model.pkl")
label_path = os.path.join(BASE_DIR, "..", "models", "labels.npy")

labels = np.load(label_path)

with open(model_path, "rb") as f:
    model = pickle.load(f)

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    
    # 🔹 Force live Streamlit volume scale to exactly 1.0 peak
    audio = librosa.util.normalize(audio)
    # Trim background silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,
        axis=0
    )
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    return np.hstack([mfcc, chroma, mel])


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
        
        # Scikit-Learn RandomForest mapping logic
        prediction = model.predict([features])
        predicted_label = labels[prediction[0]]

        return predicted_label

    except Exception as e:
        return f"Error: {str(e)}"