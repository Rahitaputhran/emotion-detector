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
    audio, sample_rate = librosa.load(file_path, sr=22050)
    
    # 🔹 Force live Streamlit volume scale to exactly 1.0 peak
    audio = librosa.util.normalize(audio)
    
    # Simple stable trim
    audio, _ = librosa.effects.trim(audio, top_db=30)
    
    # 1. Pitch tracking (Fundamental Frequency)
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sample_rate)
    pitch_vals = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_vals.append(pitch)
    pitch_mean = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0
    
    # 2. MFCCs dropping MFCC0 (which tracks absolute volume / recording level!)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)[1:] 
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # 3. ZCR and RMS (Zero crossing rate tracks friction/anger, RMS tracks energy)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    
    # 4. Chroma and Mel
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    
    return np.hstack([[pitch_mean], mfcc_mean, zcr, rms, chroma, mel])


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