import streamlit as st
from src.predict import predict

st.title("Emotion Recognition")

# 🎤 Allow users to bypass aggressive Browser Microphone Compression logic
option = st.radio("Choose Audio Input Method:", ("Record Live Voice", "Upload Uncompressed File"))

audio_file = None
if option == "Record Live Voice":
    audio_file = st.audio_input("Record your voice")
else:
    audio_file = st.file_uploader("Upload a raw .wav audio file", type=["wav", "ogg", "mp3"])

# 🎯 Predict emotion
if audio_file is not None:
    # Playback the audio inside UI
    st.audio(audio_file)
    
    if st.button("Predict Emotion"):
        emotion = predict(audio_file)
        st.write(f"### Emotion Detected: {emotion.upper()}")