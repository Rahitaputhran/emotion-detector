import librosa

def load_audio(file_path):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    return audio, sr
