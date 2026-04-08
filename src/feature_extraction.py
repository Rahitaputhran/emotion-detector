import librosa
import numpy as np
import os

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, mel])

def build_dataset(data_path):
    X, y = [], []

    for emotion in os.listdir(data_path):
        path = os.path.join(data_path, emotion)

        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(emotion)

    return np.array(X), np.array(y)
