import os
import shutil

ravdess_dir = r"C:\Users\Poorvi\Downloads\Audio_Speech_Actors_01-24"
dest_dir = r"C:\Users\Poorvi\emotion-detector\data\raw"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

count = 0
for root, dirs, files in os.walk(ravdess_dir):
    for file in files:
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    emotion_name = emotion_map[emotion_code]
                    source_path = os.path.join(root, file)
                    dest_folder = os.path.join(dest_dir, emotion_name)
                    
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    
                    dest_path = os.path.join(dest_folder, file)
                    
                    if not os.path.exists(dest_path):
                        shutil.copy2(source_path, dest_path)
                        count += 1

print(f"Successfully imported {count} files from RAVDESS dataset.")
