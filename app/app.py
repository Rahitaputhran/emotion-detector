import streamlit as st
from src.predict import predict

st.title("Emotion Recognition")

# 🎤 Record audio
audio_file = st.audio_input("Record your voice")

# 🎯 Predict emotion
if audio_file is not None:
    if st.button("Predict Emotion"):
        emotion = predict(audio_file)
        st.write("Emotion:", emotion)