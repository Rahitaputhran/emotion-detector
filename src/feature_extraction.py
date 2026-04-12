import librosa
import numpy as np
import os

def extract_features(file_path, audio_data=None, sr=22050):
    if audio_data is None:
        audio, sr = librosa.load(file_path, sr=sr)
    else:
        audio = audio_data
    
    # 🔹 Force volume scale to exactly 1.0 peak globally
    audio = librosa.util.normalize(audio)
    
    # Simple stable trim
    audio, _ = librosa.effects.trim(audio, top_db=30)
    
    # 1. Pitch tracking (Fundamental Frequency)
    pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
    pitch_vals = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_vals.append(pitch)
    pitch_mean = np.mean(pitch_vals) if len(pitch_vals) > 0 else 0
    
    # 2. MFCCs dropping MFCC0 (which tracks overall volume / recording level!)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)[1:] 
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # 3. ZCR and RMS (Zero crossing rate tracks friction/anger, RMS tracks energy)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    
    # 4. Chroma and Mel
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    
    return np.hstack([[pitch_mean], mfcc_mean, zcr, rms, chroma, mel])


def inject_noise(data, noise_factor=0.015):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data


def build_dataset(data_path):
    X, y = [], []

    for emotion in os.listdir(data_path):
        path = os.path.join(data_path, emotion)
        
        if not os.path.isdir(path): continue

        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            audio, sr = librosa.load(file_path, sr=22050)
            
            # 1. Clean robust features
            try:
                clean_features = extract_features(file_path, audio_data=audio, sr=sr)
                X.append(clean_features)
                y.append(emotion)
            except Exception:
                pass
            
            # 2. Noisy robust features
            try:
                noisy_audio = inject_noise(audio)
                noisy_features = extract_features(file_path, audio_data=noisy_audio, sr=sr)
                X.append(noisy_features)
                y.append(emotion)
            except Exception:
                pass

    return np.array(X), np.array(y)
