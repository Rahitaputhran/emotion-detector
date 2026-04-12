import streamlit as st
from src.predict import predict

st.title("Emotion Recognition")

audio_files = []
# 🔹 Set accept_multiple_files=True to gracefully handle batch inputs without crashing 
uploaded = st.file_uploader("Upload raw .wav audio files", type=["wav", "ogg", "mp3"], accept_multiple_files=True)
if uploaded:
    audio_files = uploaded

# 🎯 Predict emotion
if audio_files:
    # Playback the audio inside UI sequentially
    for file in audio_files:
        if len(audio_files) > 1:
            st.write(f"**{file.name}**")
        st.audio(file)
    
    if st.button("Predict Emotion" + ("s" if len(audio_files) > 1 else "")):
        emotions = []
        # Batch process behind the scenes sequentially so memory doesn't explode
        for file in audio_files:
            emotion = predict(file)
            emotions.append(emotion.upper())
            
        # Compile final comma separated output string identically to teacher's requirements
        st.write(f"### Emotion{'s' if len(audio_files) > 1 else ''} Detected: {', '.join(emotions)}")