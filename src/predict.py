import numpy as np
import librosa
import os
import tempfile
import pickle
import imageio_ffmpeg

# Automatically add bundled FFmpeg to the system PATH so Librosa can decode WebM browser recordings natively
os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ["PATH"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "svm_model.pkl")
scaler_path = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
label_path = os.path.join(BASE_DIR, "..", "models", "labels.npy")

labels = np.load(label_path)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the fitted scaler
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)
    
    # 🔹 Force live Streamlit volume scale to exactly 1.0 peak
    audio = librosa.util.normalize(audio)
    # Trim background silence
    audio, _ = librosa.effects.trim(audio, top_db=20)
    # 🔹 Aggressive Pre-Emphasis filter to crush low-frequency web-microphone static and hum
    audio = librosa.effects.preemphasis(audio)
    
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
                # Use Streamlit's native getvalue() to guarantee the full byte stream is ripped from RAM regardless of previous reads/seeks.
                if hasattr(file, 'getvalue'):
                    data = file.getvalue()
                else:
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    data = file.read()
                
                tmp.write(data)
                file_path = tmp.name
                
            print(f"DEBUG - Generated Temp File: {file_path} - Bytes Written: {os.path.getsize(file_path)}")
        else:
            # 🔹 Case 2: If already a file path
            file_path = file

        features = extract_features(file_path)
        
        # Scale features using the StandardScaler to match the training environment numerically
        features_scaled = scaler.transform([features])
        
        # Scikit-Learn SVM mapping logic
        prediction = model.predict(features_scaled)
        predicted_label = labels[prediction[0]]

        return predicted_label

    except Exception as e:
        return f"Error: {str(e)}"